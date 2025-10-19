# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def load_and_preprocess_images_square(
    image_path_list,
    target_size=1024,
    *,
    show_progress=False,
    desc="Loading images",
    num_workers=0,
):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 1024.
        show_progress (bool, optional): If True, display a tqdm progress bar during loading when available.
        desc (str, optional): Description shown alongside the tqdm progress bar.
        num_workers (int, optional): Number of worker threads for parallel image loading. 0 disables threading.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    def _process_single(image_path):
        # Open image and ensure it is closed promptly
        with Image.open(image_path) as img:
            # If there's an alpha channel, blend onto white background
            if img.mode == "RGBA":
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)

            # Convert to RGB
            img = img.convert("RGB")

            # Get original dimensions
            width, height = img.size

            # Make the image square by padding the shorter dimension
            max_dim = max(width, height)

            # Calculate padding
            left = (max_dim - width) // 2
            top = (max_dim - height) // 2

            # Calculate scale factor for resizing
            scale = target_size / max_dim

            # Calculate final coordinates of original image in target space
            x1 = left * scale
            y1 = top * scale
            x2 = (left + width) * scale
            y2 = (top + height) * scale

            # Create a new black square image and paste original
            square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
            square_img.paste(img, (left, top))

            # Resize to target size
            square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor outside the context manager to minimise time image file is open
        img_tensor = to_tensor(square_img)
        coords = np.array([x1, y1, x2, y2, width, height])
        return img_tensor, coords

    use_threads = (num_workers or 0) > 1
    if use_threads:
        max_workers = num_workers
        if max_workers is None or max_workers <= 0:
            cpu_count = os.cpu_count() or 1
            max_workers = min(8, cpu_count)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            iterator = executor.map(_process_single, image_path_list)
            if show_progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(image_path_list), desc=desc)
            elif show_progress:
                print(desc)
            for img_tensor, coords in iterator:
                images.append(img_tensor)
                original_coords.append(coords)
    else:
        iterable = image_path_list
        if show_progress and tqdm is not None:
            iterable = tqdm(iterable, total=len(image_path_list), desc=desc)
        elif show_progress:
            print(desc)
        for image_path in iterable:
            img_tensor, coords = _process_single(image_path)
            images.append(img_tensor)
            original_coords.append(coords)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images
