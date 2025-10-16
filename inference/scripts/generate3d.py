import sys
import torch
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1️⃣  Setup repo import path
# ----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# ----------------------------------------------------------
# 2️⃣  Select LLFF flower dataset images
# ----------------------------------------------------------
images_dir = REPO_ROOT / "examples" / "llff_flower" / "images"
if not images_dir.exists():
    raise FileNotFoundError(f"LLFF flower dataset not found: {images_dir}")

imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
if not imgs:
    raise FileNotFoundError(f"No images found in LLFF flower dataset: {images_dir}")

# Use 12 images from the dataset for a good coverage
selected_images = imgs[::2][:12]  # Take every 2nd image, up to 12 images
image_names = [str(p) for p in selected_images]
print(f"🖼️ Using {len(selected_images)} images from kitchen dataset:")
for img in selected_images:
    print(f"  • {img.name}")

# ----------------------------------------------------------
# 3️⃣  Device and memory setup
# ----------------------------------------------------------
import os

# Force CPU mode due to GPU memory constraints
device = "cpu"
dtype = torch.float32

print("\n💻 Device configuration:")
print("  • Using CPU mode (GPU memory insufficient)")
print("  • Using float32 precision")

if device == "cuda":
    # Clear any existing cached memory
    torch.cuda.empty_cache()
    # Enable memory optimization
    torch.backends.cudnn.benchmark = True

# ----------------------------------------------------------
# 4️⃣  Load model and images
# ----------------------------------------------------------
print("⏳ Loading and preprocessing images...")
# Process images in larger batches since we're using CPU
BATCH_SIZE = 5  # Process 5 images at a time on CPU
all_images = load_and_preprocess_images(image_names)
print(f"📐 Full dataset shape: {all_images.shape}")

# Ensure images are on CPU
all_images = all_images.cpu()

print("⏳ Initializing VGGT model...")
# Enable memory efficient options
torch.backends.cudnn.benchmark = False  # Disable benchmarking to save memory
torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 to save memory

model = VGGT(
    img_size=518,        # Fixed size used in official implementation
    patch_size=14,       # Default patch size
    embed_dim=1024,      # Large model configuration
    enable_camera=True,  # Enable all heads for full reconstruction
    enable_point=True,
    enable_depth=True,
    enable_track=True    # Enable tracking to match pretrained weights
)

# Enable gradient checkpointing to save memory
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

# Load weights from local file
weights_dir = Path(r"C:\Users\Savindu Dilshan\Desktop\Github\vggt\inference\weights")
weights_path = weights_dir / "model.pt"

# Verify the weights file exists and is a file (not a directory)
if not weights_path.is_file():
    raise FileNotFoundError(
        f"Weights file not found: {weights_path}\n"
        f"Please place the model.pt file in the weights directory"
    )

print(f"⏳ Loading weights from {weights_path}...")
# Load with weights_only=True for security and to avoid pickle-related warnings
state_dict = torch.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("✅ Model loaded and configured")

# ----------------------------------------------------------
# 5️⃣  Run inference in batches (CPU mode)
# ----------------------------------------------------------
print("⏳ Running model inference in batches...")
print("⚠️ Using CPU mode - this will be slower but more memory-efficient")
all_point_maps = []

try:
    with torch.no_grad():
        for i in range(0, len(all_images), BATCH_SIZE):
            # Process one image at a time
            batch_images = all_images[i:i+BATCH_SIZE].to(device, dtype=dtype)
            print(f"Processing image {i+1}/{len(all_images)}")
            
            batch_preds = model(batch_images)
            
            # Get the 3D points from predictions
            if "world_points" in batch_preds:
                points = batch_preds["world_points"]
            elif "points3d" in batch_preds:
                points = batch_preds["points3d"]
            else:
                raise KeyError(f"No 3D points found in model output: {list(batch_preds.keys())}")
            
            # Convert to numpy
            points_cpu = points.numpy()
            all_point_maps.append(points_cpu)
            
            # Clear memory
            del batch_images, batch_preds, points
            import gc
            gc.collect()
            
            print(f"  ✓ Image {i+1} complete")

    # Combine all batches
    point_maps = torch.from_numpy(np.concatenate(all_point_maps, axis=0))
    print("✅ Inference complete")

except RuntimeError as e:
    if "out of memory" in str(e):
        print("❌ GPU out of memory. Try reducing the batch size further.")
        raise e
    raise e

# ----------------------------------------------------------
# 6️⃣  Process point maps
# ----------------------------------------------------------
# point_maps was already collected during batch processing


# ----------------------------------------------------------
# 7️⃣  Build colorized point cloud (with error handling)
# ----------------------------------------------------------
rgb_images = []
for p in selected_images:
    img = cv2.imread(str(p))
    if img is None:
        print(f"❌ Error: Could not load image: {p}")
        exit(1)
    rgb_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

pts_list = []
cols_list = []
for idx, (frame, rgb) in enumerate(zip(point_maps, rgb_images)):
    arr = frame.detach().cpu().numpy()
    print(f"Frame {idx} shape: {arr.shape}")
    if arr.shape[0] == 3:
        # (3, H, W, ...)
        H, W = arr.shape[1], arr.shape[2]
        # Reshape to (H*W, 3)
        pts = arr.reshape(3, -1).T
    elif arr.shape[-1] == 3:
        # (..., H, W, 3)
        H, W = arr.shape[-3], arr.shape[-2]
        # Already in (..., 3) format, just reshape
        pts = arr.reshape(-1, 3)
    else:
        raise ValueError(f"Unexpected frame shape: {arr.shape}")

    # Ensure RGB image matches the model output spatial size
    if rgb.shape[0] != H or rgb.shape[1] != W:
        rgb_resized = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        print(f"⚠️ Resized image {selected_images[idx].name} from {rgb.shape[:2]} to {(H, W)}")
    else:
        rgb_resized = rgb
    
    # Convert colors to float and reshape
    cols = rgb_resized.reshape(-1, 3) / 255.0
    
    # Handle shape mismatches
    if pts.shape[0] != cols.shape[0]:
        print(f"⚠️ Shape mismatch: points {pts.shape[0]}, colors {cols.shape[0]} for {selected_images[idx].name}")
        min_len = min(pts.shape[0], cols.shape[0])
        pts = pts[:min_len]
        cols = cols[:min_len]
    
    pts_list.append(pts)
    cols_list.append(cols)

# Combine all points and colors
pts_all = np.concatenate(pts_list, axis=0)
cols_all = np.concatenate(cols_list, axis=0)

# ----------------------------------------------------------
# 8️⃣  Clean and visualize point cloud
# ----------------------------------------------------------
print("\n⏳ Cleaning point cloud data...")

# Process point cloud
print("\n⏳ Processing point cloud...")
print("  • Removing invalid points...")
mask = np.isfinite(pts_all).all(axis=1)
pts_clean = pts_all[mask]
cols_clean = cols_all[mask]

print("  • Computing point statistics...")
center = np.median(pts_clean, axis=0)
distances = np.linalg.norm(pts_clean - center, axis=1)

print("  • Removing outliers...")
# Use a tighter percentile for flower dataset (95% instead of 99%)
mask = distances < np.percentile(distances, 95)
pts_clean = pts_clean[mask]
cols_clean = cols_clean[mask]

print("  • Computing bounding box...")
bbox_min = pts_clean.min(axis=0)
bbox_max = pts_clean.max(axis=0)
bbox_center = (bbox_max + bbox_min) / 2
bbox_size = bbox_max - bbox_min

# Normalize points for better visualization
print("  • Normalizing point coordinates...")
scale = 1.5  # Adjust view scale for flower dataset
pts_final = (pts_clean - bbox_center) / max(bbox_size) * scale
cols_final = cols_clean

print(f"\n✅ Point cloud statistics:")
print(f"  • Initial points: {len(pts_all):,}")
print(f"  • After cleaning: {len(pts_clean):,}")
print(f"  • Bounding box size: {bbox_size}")

# Set up visualization
print("\n⏳ Creating point cloud visualization...")
print("  • Setting up plot style...")
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 12))

views = [
    (0, 30, "Front View (30°)"),
    (90, 30, "Right Side (30°)"),
    (-90, 30, "Left Side (30°)"),
    (0, 90, "Top View"),
    (45, 45, "Top-Right (45°)"),
    (-45, 45, "Top-Left (45°)")
]

print("  • Creating 6 views...")
for i, (azim, elev, title) in enumerate(views, 1):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    
    # Plot points with adjusted settings for flower dataset
    ax.scatter(pts_final[:, 0], pts_final[:, 1], pts_final[:, 2],
              c=cols_final, s=0.05, alpha=0.8)  # Smaller points, higher alpha
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Customize appearance
    ax.set_title(title, pad=12, fontsize=12)
    ax.set_xlabel('X', labelpad=8)
    ax.set_ylabel('Y', labelpad=8)
    ax.set_zlabel('Z', labelpad=8)
    
    # Set equal aspect ratio for better depth perception
    ax.set_box_aspect([1, 1, 1])
    
    # Add subtle grid
    ax.grid(True, alpha=0.3)
    
    # Remove background grid lines for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True)

plt.tight_layout()
output_path = "point_cloud_views.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {output_path}")
plt.close()
pts_all, cols_all = pts_all[mask], cols_all[mask]
dists = np.linalg.norm(pts_all - pts_all.mean(axis=0), axis=1)
keep = dists < np.percentile(dists, 98)
pts_all, cols_all = pts_all[keep], cols_all[keep]

# Normalize for viewer scale
pts_all -= pts_all.mean(axis=0)
scale = np.max(np.linalg.norm(pts_all, axis=1))
pts_all /= scale

print(f"📊 Final cloud: {len(pts_all):,} points")

# ----------------------------------------------------------
# 9️⃣  Visualize from 6 different angles
# ----------------------------------------------------------
# Increase sample size for better quality preview
sample_size = min(100000, len(pts_all))
idx = np.random.choice(len(pts_all), sample_size, replace=False)
pts_sample = pts_all[idx]
cols_sample = cols_all[idx]

# Create a figure with 6 static views
fig = plt.figure(figsize=(15, 10))
views = [
    (0, -90, "Front View"),      # Front
    (0, 0, "Side View (Right)"), # Right side
    (0, 180, "Side View (Left)"),# Left side
    (90, -90, "Top View"),       # Top down
    (45, -45, "Diagonal View 1"), # 45° from top-right
    (-45, 45, "Diagonal View 2"), # 45° from bottom-left
]

for i, (elev, azim, title) in enumerate(views, 1):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    ax.scatter(pts_sample[:, 0], pts_sample[:, 1], pts_sample[:, 2],
              s=0.1, c=cols_sample, alpha=1.0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_axis_off()

plt.suptitle("VGGT Kitchen Dataset 3D Reconstruction\n(6 Views)", y=0.95, fontsize=14)
plt.tight_layout()
plt.show()
