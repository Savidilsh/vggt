import sys
import torch
from pathlib import Path

# Ensure the repository root is on sys.path so local package imports (vggt.*) work
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Determine repository root (two levels up: scripts -> fine_tuning -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Candidate folders that contain example images in this repository
candidate_dirs = [
    REPO_ROOT / "examples" / "kitchen" / "images",
    REPO_ROOT / "examples" / "llff_fern" / "images",
    REPO_ROOT / "examples" / "llff_flower" / "images",
    REPO_ROOT / "examples" / "room" / "images",
    REPO_ROOT / "examples" / "single_cartoon" / "images",
    REPO_ROOT / "examples" / "single_oil_painting" / "images",
    REPO_ROOT / "fine_tuning" / "images",
]

# Find up to 3 images from the first candidate directory that exists
selected_images = []
for d in candidate_dirs:
    if d.exists() and d.is_dir():
        imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if imgs:
            selected_images = imgs[:3]
            break

# Fallback to a legacy relative folder if nothing found
if not selected_images:
    fallback = REPO_ROOT / "heritage_images"
    if fallback.exists() and fallback.is_dir():
        imgs = sorted([p for p in fallback.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        selected_images = imgs[:3]

if not selected_images:
    raise FileNotFoundError(
        "No example images found. Place images in one of the repository example folders (examples/*/images) or fine_tuning/images"
    )

image_names = [str(p) for p in selected_images]
print("Using images:", image_names)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Safe dtype choice: only query device capability when CUDA is available
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
else:
    dtype = torch.float32

images = load_and_preprocess_images(image_names).to(device)

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

with torch.no_grad():
    if device == "cuda":
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    else:
        predictions = model(images)
