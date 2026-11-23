"""
Skin Lesion Image Preprocessing Pipeline for PyTorch Inference and Training.

Pipeline Steps:
1. Validate image format & integrity
2. Fix orientation using EXIF metadata
3. Center-crop to square aspect ratio
4. Resize to target dimension (224x224)
5. Convert color space to RGB
6. Normalize with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
7. Return channels-first PyTorch-ready array (shape: (3, 224, 224), float32)
"""

import io
from pathlib import Path
from typing import Union, Tuple
import numpy as np
from PIL import Image as PILImage, ImageOps

# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)


def validate_image(image_input: Union[str, Path, bytes, PILImage.Image, np.ndarray]) -> PILImage.Image:
    """
    Step 1: Validate the image input and load as PIL Image.
    Raises ValueError on empty, corrupt, or unsupported image input.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        try:
            img = PILImage.open(str(path))
            img.load()
        except Exception as e:
            raise ValueError(f"Corrupted or invalid image at {path}: {e}")
    elif isinstance(image_input, (bytes, bytearray)):
        if len(image_input) == 0:
            raise ValueError("Provided image byte buffer is empty.")
        try:
            img = PILImage.open(io.BytesIO(image_input))
            img.load()
        except Exception as e:
            raise ValueError(f"Failed to decode image from bytes: {e}")
    elif isinstance(image_input, PILImage.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        try:
            img = PILImage.fromarray(image_input)
        except Exception as e:
            raise ValueError(f"Failed to convert NumPy array to PIL Image: {e}")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if img.size[0] == 0 or img.size[1] == 0:
        raise ValueError(f"Image has invalid dimensions: {img.size}")

    return img


def fix_exif_orientation(img: PILImage.Image) -> PILImage.Image:
    """
    Step 2: Correct image orientation based on EXIF metadata tags.
    Handles smartphone camera rotations (0°, 90°, 180°, 270°, and flips).
    """
    try:
        # ImageOps.exif_transpose reads the EXIF orientation tag and rotates accordingly
        return ImageOps.exif_transpose(img)
    except Exception:
        # If EXIF reading fails or tag is missing, return original
        return img


def center_crop_square(img: PILImage.Image) -> PILImage.Image:
    """
    Step 3: Center-crop the image to a square aspect ratio (1:1).
    Preserves lesion framing without introducing geometric distortion.
    """
    width, height = img.size
    min_dim = min(width, height)

    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    return img.crop((left, top, right, bottom))


def resize_image(img: PILImage.Image, size: Tuple[int, int] = TARGET_IMAGE_SIZE) -> PILImage.Image:
    """
    Step 4: Resize cropped image to standard input dimensions (224x224).
    Uses high-quality bicubic interpolation.
    """
    return img.resize(size, resample=PILImage.Resampling.BICUBIC)


def convert_to_rgb(img: PILImage.Image) -> PILImage.Image:
    """
    Step 5: Convert image color mode to standard 3-channel RGB.
    Ensures alpha channels (RGBA) or grayscale modes are standardized.
    """
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def normalize_imagenet(img_array: np.ndarray) -> np.ndarray:
    """
    Step 6: Scale pixel values to [0.0, 1.0] and normalize using ImageNet mean & std.
    Input shape: (H, W, 3), range [0, 255]
    Output shape: (H, W, 3), normalized float32
    """
    # Scale to [0.0, 1.0]
    scaled = img_array.astype(np.float32) / 255.0
    # Normalize: (x - mean) / std
    normalized = (scaled - IMAGENET_MEAN) / IMAGENET_STD
    return normalized


def to_pytorch_channels_first(img_array: np.ndarray) -> np.ndarray:
    """
    Step 7: Transpose (H, W, C) -> (C, H, W) channels-first format for PyTorch tensors.
    Output shape: (3, 224, 224), float32 contiguous array
    """
    transposed = np.transpose(img_array, (2, 0, 1))
    return np.ascontiguousarray(transposed, dtype=np.float32)


def optimize_image_for_storage(
    image_input: Union[str, Path, bytes, PILImage.Image, np.ndarray],
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    quality: int = 85,
) -> Tuple[bytes, str]:
    """
    Optimizes and prepares an image for long-term storage in MinIO.
    - Centers and crops/resizes image to target dimensions (224x224)
    - Compresses to JPEG format with quality 85 (reducing storage footprints)

    Args:
        image_input: Image filepath, byte buffer, PIL Image, or NumPy array.
        target_size: Output resolution tuple (default: 224x224).
        quality: JPEG compression quality level (default: 85).

    Returns:
        Tuple[bytes, str]: (compressed_jpeg_bytes, "image/jpeg")
    """
    img = validate_image(image_input)
    img = fix_exif_orientation(img)
    img = center_crop_square(img)
    img = resize_image(img, size=target_size)
    img = convert_to_rgb(img)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue(), "image/jpeg"


def preprocess_image(
    image_input: Union[str, Path, bytes, PILImage.Image, np.ndarray],
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
) -> np.ndarray:
    """
    Complete end-to-end preprocessing pipeline for skin lesion classification.

    Args:
        image_input: Filepath, bytes, PIL Image, or NumPy array.
        target_size: Output resolution (default: (224, 224)).

    Returns:
        np.ndarray: PyTorch-ready array of shape (3, 224, 224), dtype float32.
    """
    # 1. Validate
    img = validate_image(image_input)

    # 2. Fix orientation
    img = fix_exif_orientation(img)

    # 3. Center-crop to square
    img = center_crop_square(img)

    # 4. Resize to target dimension
    img = resize_image(img, size=target_size)

    # 5. Convert to RGB
    img = convert_to_rgb(img)

    # Convert PIL Image to NumPy array (H, W, C)
    arr = np.array(img, dtype=np.float32)

    # 6. Normalize with ImageNet mean and std
    arr_normalized = normalize_imagenet(arr)

    # 7. Convert to channels-first (C, H, W)
    pytorch_array = to_pytorch_channels_first(arr_normalized)

    return pytorch_array

