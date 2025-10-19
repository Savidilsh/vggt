import argparse
import copy
import sys
import glob
import os
import random
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument(
        "--scene_dir",
        type=str,
        default=str(ROOT_DIR / "examples" / "llff_fern"),
        help="Directory containing the scene images. Defaults to examples/llff_fern.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Select device for inference. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--img_load_resolution",
        type=int,
        default=768,
        help="Resolution used when loading images before resizing (square). Lower values use less memory.",
    )
    parser.add_argument(
        "--model_resolution",
        type=int,
        default=448,
        help="Resolution passed to VGGT after interpolation (square). Must be divisible by 14.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Keep every N-th frame to reduce sequence length and memory footprint.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional hard limit on number of frames processed (after stride).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="inference/weights/model.pt",
        help="Path to the locally stored VGGT weights.",
    )
    parser.add_argument(
        "--max_points_colmap",
        type=int,
        default=100000,
        help="Maximum number of 3D points kept for COLMAP-compatible export.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=(
            "Number of worker threads for image loading/preprocessing. "
            "Defaults to half of the available CPU cores; set 0 to disable threading."
        ),
    )
    parser.add_argument(
        "--disable_progress",
        action="store_true",
        help="Disable tqdm progress bars for non-interactive environments.",
    )
    parser.add_argument(
        "--enable_amp",
        action="store_true",
        help="Enable mixed-precision (AMP) inference on CUDA to trade a small accuracy drop for speed.",
    )
    parser.add_argument(
        "--fast_mode",
        action="store_true",
        help="Automatically downscale resolution and limit frames for quicker runs (~15 min target).",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def configure_torch_backends(device: torch.device):
    torch.backends.cudnn.enabled = device.type == "cuda"
    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch.backends.cudnn.deterministic = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def run_vggt(
    model,
    images,
    resolution=518,
    progress_callback=None,
    stage_callback=None,
    depth_chunk_size=None,
):
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with inference_ctx():
        images = images[None]
        last_progress = {"completed": 0, "total": None}
        head_steps = int(model.camera_head is not None) + int(model.depth_head is not None)
        agg_total = 0
        if progress_callback is not None and hasattr(model.aggregator, "set_progress_callback"):
            def wrapped_callback(completed, total):
                last_progress["completed"] = completed
                last_progress["total"] = total
                progress_callback(completed, total)

            model.aggregator.set_progress_callback(wrapped_callback)
        try:
            aggregated_tokens_list, ps_idx, layer_indices = model.aggregator(images)
        finally:
            if progress_callback is not None and hasattr(model.aggregator, "set_progress_callback"):
                model.aggregator.set_progress_callback(None)
                total = last_progress["total"]
                completed = last_progress["completed"]
                if total is not None and completed < total:
                    progress_callback(total, total)
        agg_total = last_progress["total"] or last_progress["completed"] or agg_total
        completed = last_progress["completed"]
        if stage_callback is not None:
            stage_callback(
                "aggregator_done",
                {"head_steps": head_steps, "agg_total": agg_total},
            )
        extended_total = agg_total
        if progress_callback is not None and head_steps > 0:
            extended_total = (agg_total or 0) + head_steps
            progress_callback(completed, extended_total)
        if stage_callback is not None and head_steps > 0:
            stage_callback("heads_running", {"head_steps": head_steps, "agg_total": agg_total})

        pose_enc_list = model.camera_head(aggregated_tokens_list, layer_indices=layer_indices)
        pose_enc = pose_enc_list[-1]
        if progress_callback is not None and head_steps > 0:
            completed = max(completed, agg_total or 0) + 1
            progress_callback(completed, extended_total)
        else:
            completed = completed or 0

        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_kwargs = {}
        if depth_chunk_size is not None:
            depth_kwargs["frames_chunk_size"] = depth_chunk_size
        depth_map, depth_conf = model.depth_head(
            aggregated_tokens_list, images, ps_idx, layer_indices=layer_indices, **depth_kwargs
        )
        if progress_callback is not None and head_steps > 0:
            completed += 1
            progress_callback(completed, extended_total)
        elif progress_callback is not None and head_steps == 0 and extended_total:
            progress_callback(extended_total, extended_total)
        if stage_callback is not None:
            stage_callback("heads_done", {"head_steps": head_steps, "agg_total": agg_total})

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    print("Arguments:", vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}", flush=True)

    device = resolve_device(args.device)
    configure_torch_backends(device)
    print(f"Using device: {device}", flush=True)
    print("Using dtype: float32 (mixed precision disabled for stability)", flush=True)

    amp_flag = args.enable_amp and device.type == "cuda"
    if args.enable_amp and device.type != "cuda":
        print("Warning: --enable_amp requested but CUDA device not available; running in FP32.", flush=True)
    if args.fast_mode and args.max_points_colmap == 100000:
        args.max_points_colmap = 40000
        print("Fast mode: reducing COLMAP point cap to 40k to speed up filtering.", flush=True)
    model = VGGT(amp_enabled=amp_flag)
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    if hasattr(model, "set_amp"):
        model.set_amp(amp_flag)
    required_layers = {model.aggregator.depth - 1}
    if model.depth_head is not None:
        required_layers.update(model.depth_head.intermediate_layer_idx)
    point_head = getattr(model, "point_head", None)
    if point_head is not None:
        required_layers.update(point_head.intermediate_layer_idx)
    track_head = getattr(model, "track_head", None)
    if track_head is not None:
        required_layers.update(track_head.feature_extractor.intermediate_layer_idx)
    model.aggregator.set_output_layers(sorted(required_layers))
    print(f"Model loaded from {model_path}", flush=True)

    progress_enabled = not args.disable_progress and tqdm is not None
    pipeline_bar = tqdm(total=4, desc="Preparing", leave=False) if progress_enabled else None

    def step_start(label: str):
        if pipeline_bar:
            pipeline_bar.set_description_str(label)

    def step_done():
        if pipeline_bar:
            pipeline_bar.update(1)

    try:
        image_dir = Path(args.scene_dir) / "images"
        image_path_list = sorted(glob.glob(str(image_dir / "*")))
        total_available = len(image_path_list)
        if args.fast_mode and total_available > 0:
            fast_stride = max(args.frame_stride, 2)
            if fast_stride != args.frame_stride:
                print(f"Fast mode: increasing frame_stride to {fast_stride}.", flush=True)
                args.frame_stride = fast_stride
        if args.frame_stride > 1:
            image_path_list = image_path_list[:: args.frame_stride]
        if args.fast_mode:
            fast_cap = min(4, len(image_path_list))
            max_frames = args.max_frames
            if max_frames is None or max_frames > fast_cap:
                max_frames = fast_cap
                print(f"Fast mode: limiting to first {max_frames} frames.", flush=True)
            image_path_list = image_path_list[: max_frames]
        elif args.max_frames is not None:
            image_path_list = image_path_list[: args.max_frames]
        if len(image_path_list) == 0:
            raise ValueError(f"No images found in {image_dir}")
        print(
            f"Selected {len(image_path_list)} frames (stride={args.frame_stride}, max_frames={args.max_frames})",
            flush=True,
        )

        vggt_fixed_resolution = args.model_resolution
        img_load_resolution = args.img_load_resolution
        if args.fast_mode:
            fast_model_res = min(args.model_resolution, 224)  # divisible by 14
            if vggt_fixed_resolution > fast_model_res:
                print(
                    f"Fast mode: lowering model_resolution from {vggt_fixed_resolution} to {fast_model_res}. "
                    "Expect slightly blurrier depth but faster inference.",
                    flush=True,
                )
                vggt_fixed_resolution = fast_model_res
            fast_load_res = max(fast_model_res, 320)
            if img_load_resolution > fast_load_res:
                print(
                    f"Fast mode: lowering img_load_resolution from {img_load_resolution} to {fast_load_res}. "
                    "This reduces image I/O and preprocessing cost.",
                    flush=True,
                )
                img_load_resolution = fast_load_res
            print("Fast mode: processing depth head in 2-frame chunks.", flush=True)
        if vggt_fixed_resolution % 14 != 0:
            raise ValueError("model_resolution must be divisible by 14 (VGGT patch size is 14).")

        cpu_count = os.cpu_count() or 1
        if args.num_workers is None:
            if cpu_count <= 1:
                num_workers = 0
            else:
                num_workers = min(8, max(2, cpu_count // 2))
        else:
            if args.num_workers < 0:
                raise ValueError("num_workers must be >= 0")
            num_workers = args.num_workers
        if num_workers > 0:
            num_workers = min(num_workers, cpu_count)

        step_start("Loading images")
        load_start = time.perf_counter()
        show_load_progress = progress_enabled
        images, original_coords = load_and_preprocess_images_square(
            image_path_list,
            img_load_resolution,
            show_progress=show_load_progress,
            desc="Loading images",
            num_workers=num_workers,
        )
        if device.type == "cuda":
            images = images.pin_memory()
        images = images.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.synchronize()
        load_time = time.perf_counter() - load_start
        step_done()
        if num_workers > 1:
            worker_msg = f"{num_workers} threads"
        elif num_workers == 1:
            worker_msg = "single-threaded"
        else:
            worker_msg = "no threading"
        print(f"Loaded {len(images)} images from {image_dir} in {load_time:.2f}s ({worker_msg}).", flush=True)

        step_start("Running VGGT")
        infer_start = time.perf_counter()
        total_steps = getattr(model.aggregator, "progress_total", None)
        if not total_steps:
            total_steps = getattr(model.aggregator, "depth", None)
        run_bar = None
        head_bar = None
        head_steps = int(model.camera_head is not None) + int(model.depth_head is not None)
        timing_info = {"agg_end": None, "head_start": None, "head_end": None}
        agg_total_steps = total_steps or 0
        progress_state = {"total": total_steps or 0}
        if progress_enabled and total_steps:
            run_bar = tqdm(total=total_steps, desc="Running VGGT", leave=False)
        elif not progress_enabled and total_steps:
            print(f"Running VGGT ({total_steps} transformer passes)...", flush=True)
        elif not progress_enabled:
            print("Running VGGT...", flush=True)

        def on_stage(stage: str, info: dict | None = None):
            nonlocal run_bar, head_bar, agg_total_steps
            info = info or {}
            if stage == "aggregator_done":
                agg_total_steps = info.get("agg_total") or progress_state["total"] or agg_total_steps
                timing_info["agg_end"] = time.perf_counter()
                agg_elapsed = timing_info["agg_end"] - infer_start
                if run_bar:
                    run_bar.close()
                    run_bar = None
                print(
                    f"Transformer stage complete in {agg_elapsed:.2f}s; decoding camera/depth heads...",
                    flush=True,
                )
                if progress_enabled and head_steps > 0:
                    head_bar = tqdm(total=head_steps, desc="Decoding heads", leave=False)
            elif stage == "heads_running":
                if timing_info["head_start"] is None:
                    timing_info["head_start"] = time.perf_counter()
                if not progress_enabled:
                    print("  · Running camera/depth heads...", flush=True)
            elif stage == "heads_done":
                timing_info["head_end"] = time.perf_counter()
                if head_bar:
                    head_bar.close()
                    head_bar = None
                print("Camera/depth heads finished.", flush=True)

        def on_layer_progress(completed, total):
            nonlocal head_bar, agg_total_steps
            if total:
                progress_state["total"] = total
            total = total or progress_state["total"] or total_steps
            if run_bar:
                if total and run_bar.total != total:
                    run_bar.total = total
                    run_bar.refresh()
                delta = completed - run_bar.n
                if delta > 0:
                    run_bar.update(delta)
            elif head_bar and head_steps > 0:
                head_completed = max(0, completed - agg_total_steps)
                if head_bar.total != head_steps:
                    head_bar.total = head_steps
                    head_bar.refresh()
                head_completed = min(head_steps, head_completed)
                delta = head_completed - head_bar.n
                if delta > 0:
                    head_bar.update(delta)
            elif not progress_enabled and total:
                # Print coarse progress updates every quarter of the passes
                checkpoint_size = max(1, total // 6)
                if completed % checkpoint_size == 0 or completed == total:
                    print(f"  · Transformer passes: {completed}/{total}", flush=True)

        try:
            extrinsic, intrinsic, depth_map, depth_conf = run_vggt(
                model,
                images,
                vggt_fixed_resolution,
                progress_callback=on_layer_progress,
                stage_callback=on_stage,
                depth_chunk_size=2 if args.fast_mode else None,
            )
        finally:
            if run_bar:
                run_bar.close()
            if head_bar:
                head_bar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - infer_start
        step_done()
        depth_finite_ratio = np.isfinite(depth_map).mean()
        agg_elapsed = None
        head_elapsed = None
        if timing_info["agg_end"] is not None:
            agg_elapsed = timing_info["agg_end"] - infer_start
        if timing_info["head_start"] is not None and timing_info["head_end"] is not None:
            head_elapsed = timing_info["head_end"] - timing_info["head_start"]
        if agg_elapsed is not None and head_elapsed is not None:
            print(
                f"Stage timings — transformer: {agg_elapsed:.2f}s, heads: {head_elapsed:.2f}s.",
                flush=True,
            )
        print(f"Ran VGGT in {infer_time:.2f}s. Depth finite ratio: {depth_finite_ratio:.4f}", flush=True)

        step_start("Filtering points")
        filter_start = time.perf_counter()
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

        conf_thres_value = args.conf_thres_value
        rgb_tensor = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (rgb_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)
        del rgb_tensor

        conf_mask = depth_conf >= conf_thres_value
        if not conf_mask.any():
            print(
                f"Warning: No points satisfied confidence threshold {conf_thres_value}. "
                "Falling back to keep all points.",
                flush=True,
            )
            conf_mask = np.ones_like(depth_conf, dtype=bool)
        conf_mask = randomly_limit_trues(conf_mask, args.max_points_colmap)
        if not conf_mask.any():
            print(
                "Warning: Confidence masking produced zero points after sampling; exporting without filtering.",
                flush=True,
            )
            conf_mask = np.ones_like(depth_conf, dtype=bool)

        points_3d = points_3d[conf_mask]
        points_rgb = points_rgb[conf_mask]
        depth_conf = depth_conf[conf_mask]

        finite_mask = np.isfinite(points_3d).all(axis=-1)
        if not finite_mask.any():
            raise ValueError("All points became non-finite (NaN/Inf) after filtering. Cannot export point cloud.")

        dropped = (~finite_mask).sum()
        if dropped:
            print(f"Filtered out {dropped} non-finite points before export.", flush=True)

        points_3d = points_3d[finite_mask]
        points_rgb = points_rgb[finite_mask]
        depth_conf = depth_conf[finite_mask]
        if device.type == "cuda":
            torch.cuda.synchronize()
        filter_time = time.perf_counter() - filter_start
        step_done()
        print(f"Filtering finished in {filter_time:.2f}s; keeping {points_3d.shape[0]} points.", flush=True)

        step_start("Exporting outputs")
        export_start = time.perf_counter()
        sparse_dir = Path(args.scene_dir) / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        point_cloud = trimesh.PointCloud(points_3d, colors=points_rgb)
        point_cloud.export(sparse_dir / "points.ply")

        scene = trimesh.Scene()
        scene.add_geometry(point_cloud)
        scene.export(sparse_dir / "scene.glb")

        scene_points = trimesh.Scene()
        scene_points.add_geometry(point_cloud.copy())
        scene_points.export(sparse_dir / "points.glb")

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        export_time = time.perf_counter() - export_start
        step_done()
        print(f"Saved outputs to {sparse_dir} in {export_time:.2f}s.", flush=True)
    finally:
        if pipeline_bar:
            pipeline_bar.close()

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    context_manager = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with context_manager():
        demo_fn(args)
