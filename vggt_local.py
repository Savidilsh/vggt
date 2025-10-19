"""
Run VGGT locally on CPU (or optionally GPU) using an images folder.

This mirrors the notebook flow:
 - Loads images from a folder (PNG/JPG)
 - Loads model weights from checkpoints/model.pt if present, otherwise from Hugging Face
 - Runs aggregator -> camera_head -> depth_head -> unprojects to 3D
 - Saves outputs: PLY point cloud, camera extrinsics/intrinsics, depth maps

Usage (PowerShell):
    cd 'd:\5th sem\Image proccessing\vggt'
    .venv\Scripts\Activate.ps1   # if you use a venv
    # By default, this script uses the examples/llff_fern/images folder below.
    python vggt_local.py --device cpu --max_images 5

Notes:
 - CPU is the default. For speed, start with 2–5 images.
 - If checkpoints/model.pt exists, it will be used to avoid downloading.
 - If missing, the script will attempt to load from_pretrained (needs Internet once).
"""

from __future__ import annotations

import argparse
import sys
import time
import shutil
from pathlib import Path
from typing import List

import numpy as np
import torch


def find_images(folder: Path, exts=(".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")) -> List[Path]:
    return sorted([p for p in folder.glob("*") if p.suffix in exts])


def write_ply(points_np: np.ndarray, colors_np: np.ndarray, out_path: Path) -> None:
    """Write point cloud to ASCII PLY.

    points_np: (M, 3) float32/float64
    colors_np: (M, 3) uint8
    """
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_np)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points_np)):
            x, y, z = points_np[i]
            r, g, b = colors_np[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT locally on CPU/GPU")
    parser.add_argument(
        "--image_folder",
        type=str,
        default=r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images",
        help="Folder containing PNG/JPG images (default: examples/llff_fern/images)",
    )
    parser.add_argument("--checkpoint", type=str, default=str(Path("checkpoints/model.pt")), help="Path to model.pt checkpoint")
    parser.add_argument("--output_prefix", type=str, default="vggt_reconstruction", help="Output filename prefix")
    parser.add_argument("--mode", type=str, choices=["crop", "pad"], default="crop", help="Preprocess mode for images")
    parser.add_argument("--max_images", type=int, default=0, help="Use only first N images (0 = all)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="cpu", help="Device to use (default cpu)")
    parser.add_argument("--force_from_pretrained", action="store_true", help="Ignore local checkpoint and load from HuggingFace")
    parser.add_argument("--cache_dir", type=str, default=str(Path(".hf_cache")), help="Cache directory for Hugging Face downloads")
    return parser.parse_args()


def select_device(choice: str) -> str:
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_folder)
    ckpt_path = Path(args.checkpoint)
    device = select_device(args.device)

    print("=" * 70)
    print("VGGT LOCAL RUNNER")
    print("=" * 70)
    print(f"Images folder: {image_dir.resolve()}")
    print(f"Checkpoint:    {ckpt_path.resolve()}" if ckpt_path.exists() else "Checkpoint:    (not found; will try from_pretrained)")
    print(f"Device:        {device}")
    print(f"Mode:          {args.mode}")
    if args.max_images > 0:
        print(f"Max images:    {args.max_images}")
    print()

    # 1) Collect images
    if not image_dir.exists():
        print(f"[ERROR] Image folder does not exist: {image_dir}")
        sys.exit(1)

    image_paths = find_images(image_dir)
    if len(image_paths) == 0:
        print("[ERROR] No PNG/JPG images found in the folder.")
        sys.exit(1)

    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    print(f"Found {len(image_paths)} images:")
    for p in image_paths[:10]:
        print(f" - {p.name}")
    if len(image_paths) > 10:
        print(f"   ... and {len(image_paths) - 10} more")
    print()

    # 2) Load model
    start_load = time.time()
    from vggt.models.vggt import VGGT
    # Optional: increase verbosity to see download logs
    try:
        from huggingface_hub.utils import logging as hf_logging
        hf_logging.set_verbosity_info()
    except Exception:
        pass

    if ckpt_path.exists() and not args.force_from_pretrained:
        size_gb = ckpt_path.stat().st_size / 1e9
        print(f"Loading model weights from local checkpoint (map_location='cpu')...")
        print(f" - File: {ckpt_path}")
        print(f" - Size: {size_gb:.1f} GB (reading can take several minutes on HDD/CPU)")
        print("   Please wait...\n")
        try:
            t_read = time.time()
            # Read the checkpoint into CPU memory. Prefer weights_only if available to reduce overhead.
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # PyTorch >=2.1
            except TypeError:
                state = torch.load(ckpt_path, map_location="cpu")  # Fallback for older versions
            print(f"✓ Checkpoint read in {time.time()-t_read:.1f}s")

            print("Initializing model and loading weights (this can also take a while)...")
            t_load = time.time()
            model = VGGT()
            missing, unexpected = model.load_state_dict(state, strict=False)
            # Free the state dict ASAP to release RAM
            del state
            import gc as _gc
            _gc.collect()
            if missing or unexpected:
                print(f"[WARN] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            print(f"✓ Weights loaded in {time.time()-t_load:.1f}s")
        except MemoryError as e:
            print("\n[ERROR] Ran out of RAM while loading the local checkpoint.")
            print("Falling back to from_pretrained (requires Internet on first run)...")
            model = VGGT.from_pretrained("facebook/VGGT-1B")
        except Exception as e:
            print(f"\n[ERROR] Failed to load local checkpoint: {e}")
            print("Falling back to from_pretrained (requires Internet on first run)...")
            model = VGGT.from_pretrained("facebook/VGGT-1B")
    else:
        if args.force_from_pretrained:
            print("--force_from_pretrained set; ignoring local checkpoint.")
        else:
            print("Local checkpoint not found.")
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Check free disk space
        du = shutil.disk_usage(str(cache_dir))
        free_gb = du.free / 1e9
        print("Trying from_pretrained (downloads on first run, cached afterwards)...")
        print(f" - Cache dir: {cache_dir.resolve()}")
        print(f" - Free space: {free_gb:.1f} GB (need ~6-10 GB)")
        try:
            t_dl = time.time()
            model = VGGT.from_pretrained(
                "facebook/VGGT-1B",
                cache_dir=str(cache_dir),
                force_download=False,
                resume_download=True,
                local_files_only=False,
            )
            print(f"✓ Download/init completed in {time.time()-t_dl:.1f}s")
        except Exception as e:
            print("\n[ERROR] from_pretrained failed:")
            print(f" - {e}")
            print("Possible causes: no Internet, firewall blocking Hugging Face, insufficient disk space, or gated model access.")
            print("If you have a local checkpoint, omit --force_from_pretrained and use it instead.")
            sys.exit(1)

    model = model.to(device).eval()
    print(f"Model ready in {time.time() - start_load:.1f}s\n")

    # 3) Load and preprocess images
    from vggt.utils.load_fn import load_and_preprocess_images

    print("Preprocessing images...")
    images = load_and_preprocess_images([str(p) for p in image_paths], mode=args.mode).to(device)
    print(f"Images tensor: {tuple(images.shape)}\n")

    # 4) Inference
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    print("Running inference (CPU can be slow; start with a few images)...")
    t0 = time.time()
    with torch.no_grad():
        batch = images.unsqueeze(0)
        print("[1/4] Aggregator ...", end=" ", flush=True)
        ts = time.time()
        tokens, ps_idx = model.aggregator(batch)
        print(f"✓ {time.time()-ts:.1f}s")

        print("[2/4] Camera head ...", end=" ", flush=True)
        ts = time.time()
        pose_enc = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
        print(f"✓ {time.time()-ts:.1f}s")

        print("[3/4] Depth head (slow on CPU) ...", end=" ", flush=True)
        ts = time.time()
        depth_map, depth_conf = model.depth_head(tokens, batch, ps_idx)
        print(f"✓ {time.time()-ts:.1f}s")

        print("[4/4] Unprojecting to 3D ...", end=" ", flush=True)
        ts = time.time()
        point_map = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )
        print(f"✓ {time.time()-ts:.1f}s")

    print(f"Total inference time: {(time.time()-t0)/60:.1f} min\n")

    # 5) Save outputs
    print("Saving outputs...")
    out_prefix = Path(args.output_prefix)
    ply_path = out_prefix.with_suffix(".ply")

    # Convert tensors to numpy
    points = point_map.cpu().numpy() if isinstance(point_map, torch.Tensor) else point_map
    colors = images.cpu().numpy() if isinstance(images, torch.Tensor) else images

    # Reshape and filter
    N, H, W, _ = points.shape
    total_points = N * H * W
    points = points.reshape(-1, 3)
    colors = colors.transpose(0, 2, 3, 1).reshape(-1, 3)

    valid = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid &= (np.linalg.norm(points, axis=1) < 100)
    points = points[valid]
    colors = (colors[valid] * 255).clip(0, 255).astype(np.uint8)

    write_ply(points, colors, ply_path)

    # Save .npy
    extr_np = extrinsic.cpu().numpy() if isinstance(extrinsic, torch.Tensor) else extrinsic
    intr_np = intrinsic.cpu().numpy() if isinstance(intrinsic, torch.Tensor) else intrinsic
    depth_np = depth_map.cpu().numpy() if isinstance(depth_map, torch.Tensor) else depth_map

    np.save(out_prefix.with_name(f"{out_prefix.name}_camera_extrinsics.npy"), extr_np)
    np.save(out_prefix.with_name(f"{out_prefix.name}_camera_intrinsics.npy"), intr_np)
    np.save(out_prefix.with_name(f"{out_prefix.name}_depth_maps.npy"), depth_np)

    print("Saved:")
    print(f" - {ply_path.resolve()}")
    print(f" - {out_prefix.with_name(f'{out_prefix.name}_camera_extrinsics.npy').resolve()}")
    print(f" - {out_prefix.with_name(f'{out_prefix.name}_camera_intrinsics.npy').resolve()}")
    print(f" - {out_prefix.with_name(f'{out_prefix.name}_depth_maps.npy').resolve()}")


if __name__ == "__main__":
    main()
