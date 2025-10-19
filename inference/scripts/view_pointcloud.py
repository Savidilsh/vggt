import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    OPEN3D_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    MATPLOTLIB_AVAILABLE = False

try:
    import trimesh
except ImportError as exc:  # pragma: no cover - trimesh is required
    raise SystemExit(
        "This viewer requires the 'trimesh' package. Install it with `pip install trimesh`."
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_POINT_CLOUD = ROOT_DIR / "examples" / "llff_fern" / "sparse" / "points.ply"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick viewer for VGGT point cloud/GLB outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_POINT_CLOUD,
        help="Path to the .ply or .glb file to visualise "
        "(defaults to examples/llff_fern/sparse/points.ply).",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Optional voxel downsampling size (only used when Open3D is available).",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=200000,
        help="Maximum number of points plotted when falling back to matplotlib.",
    )
    return parser.parse_args()


def load_points(path: Path) -> np.ndarray:
    """
    Load a point cloud into an (N, 3) numpy array using trimesh.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    # trimesh will interpret GLB as a scene; grab vertices from the first geometry
    if path.suffix.lower() in {".glb", ".gltf"}:
        scene = trimesh.load(str(path))
        if isinstance(scene, trimesh.Scene):
            # merge all geometries into a unified cloud
            combined = trimesh.util.concatenate(
                [g for g in scene.geometry.values() if hasattr(g, "vertices")]
            )
            points = combined.vertices if hasattr(combined, "vertices") else combined
        else:
            points = scene.vertices
    else:
        cloud = trimesh.load(str(path), process=False)
        if hasattr(cloud, "vertices"):
            points = cloud.vertices
        elif hasattr(cloud, "points"):
            points = cloud.points
        else:
            raise ValueError(f"Unable to extract vertices from {path}")

    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    return points.astype(np.float32)


def view_with_open3d(path: Path, voxel_size: float | None) -> None:
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        raise ValueError(f"{path} does not contain any valid points. Cannot render.")

    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
        if not pcd.has_points():
            raise ValueError("Voxel downsampling removed all points; try a smaller voxel size.")

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    print(f"Loaded {len(pcd.points)} points from {path}")
    print(f"Point cloud extent (XYZ): {extent}")

    # Recentre to improve the initial Open3D camera view
    centre = bbox.get_center()
    pcd.translate(-centre)

    # Avoid pathological scale by normalising if extent is degenerate
    max_extent = np.max(extent)
    if max_extent > 0:
        pcd.scale(1.0 / max_extent, center=(0, 0, 0))

    o3d.visualization.draw_geometries([pcd], window_name=str(path.name))


def view_with_matplotlib(path: Path, max_points: int) -> None:
    pts = load_points(path)
    if pts.shape[0] == 0:
        raise ValueError(f"{path} does not contain any vertices to visualise.")

    if pts.shape[0] > max_points:
        indices = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[indices]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=pts[:, 2], cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(path.name)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()

    if OPEN3D_AVAILABLE:
        view_with_open3d(input_path, args.voxel_size)
        return

    if MATPLOTLIB_AVAILABLE:
        view_with_matplotlib(input_path, args.max_points)
        return

    raise SystemExit(
        "Neither Open3D nor Matplotlib is available for visualisation. "
        "Install one of them, for example:\n"
        "  pip install open3d\n"
        "or\n"
        "  pip install matplotlib"
    )


if __name__ == "__main__":
    main()
