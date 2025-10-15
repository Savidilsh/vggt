import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

# ----------------------------------------------------------
# Ensure repo import path
# ----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# ----------------------------------------------------------
# Select images from fine_tuning/images
# ----------------------------------------------------------
images_dir = REPO_ROOT / "fine_tuning" / "images"
if not images_dir.exists():
    raise FileNotFoundError(f"Images directory not found: {images_dir}")

imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
if not imgs:
    raise FileNotFoundError(f"No images found in: {images_dir}")

selected_images = imgs[:5]  # Process up to 5 images
image_names = [str(p) for p in selected_images]
print(f"🖼️ Using {len(selected_images)} images from {images_dir}:")
for img in image_names:
    print(f"  • {Path(img).name}")

# ----------------------------------------------------------
# Device setup
# ----------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
else:
    dtype = torch.float32

# ----------------------------------------------------------
# Load images + model
# ----------------------------------------------------------
images = load_and_preprocess_images(image_names).to(device)
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

# ----------------------------------------------------------
# Inference
# ----------------------------------------------------------
with torch.no_grad():
    if device == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            preds = model(images)
    else:
        preds = model(images)

print("✅ Inference complete. Keys:", preds.keys())

# ----------------------------------------------------------
# Use correct field names for 3D
# ----------------------------------------------------------
if "world_points" in preds:
    point_maps = preds["world_points"]     # [N, 3, H, W]
    conf = preds.get("world_points_conf", None)
elif "points3d" in preds:
    point_maps = preds["points3d"]
else:
    raise KeyError("No valid 3D points field found. Available keys: " + str(preds.keys()))

# Process all frames
print("\n🔄 Processing point clouds...")
all_points = []
for i, frame in enumerate(point_maps):
    pts = frame.detach().cpu().numpy().reshape(3, -1).T  # (H*W, 3)
    print(f"  Frame {i+1}: {pts.shape} points")
    all_points.append(pts)

# Combine all points
combined_pts = np.concatenate(all_points, axis=0)
print(f"\n📊 Combined point cloud shape: {combined_pts.shape}")

# ----------------------------------------------------------
# Visualize with matplotlib
# ----------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points from each frame in different colors
colors = plt.cm.rainbow(np.linspace(0, 1, len(all_points)))
for i, (pts, color) in enumerate(zip(all_points, colors)):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c=[color], alpha=0.6, 
              label=f'Frame {i+1}')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"VGGT Predicted 3D Point Cloud ({len(all_points)} frames)")
ax.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Save to PLY (optional)
# ----------------------------------------------------------
if HAS_O3D:
    # Save combined point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pts)
    ply_path = REPO_ROOT / "vggt_reconstruction_combined.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"💾 Saved combined point cloud to: {ply_path}")
    
    # Save individual frames
    for i, pts in enumerate(all_points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        frame_ply = REPO_ROOT / f"vggt_reconstruction_frame{i+1}.ply"
        o3d.io.write_point_cloud(str(frame_ply), pcd)
        print(f"💾 Saved frame {i+1} to: {frame_ply}")
else:
    print("💡 Install open3d (`pip install open3d`) to export .ply files.")
