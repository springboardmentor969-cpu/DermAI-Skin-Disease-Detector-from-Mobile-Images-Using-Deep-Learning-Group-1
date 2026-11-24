from PIL import Image, ImageOps
import numpy as np


# Standard input size for the initial baseline
IMAGE_SIZE = (224, 224)


def load_image(image_path):
    """
    Load an image and convert it to RGB.
    """
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    return image


def resize_image(image, size=IMAGE_SIZE):
    """
    Resize image to the required model input size.
    """
    return image.resize(size, Image.Resampling.LANCZOS)


def normalize_image(image):
    """
    Convert image pixels from [0, 255] to [0, 1].
    """
    image_array = np.asarray(image, dtype=np.float32)

    image_array = image_array / 255.0

    return image_array


def preprocess_image(image_path):
    """
    Complete preprocessing pipeline.

    Input:
        image_path - path to original image

    Output:
        normalized NumPy array
    """

    # 1. Load image
    image = load_image(image_path)

    # 2. Resize
    image = resize_image(image)

    # 3. Normalize pixels
    image_array = normalize_image(image)

    return image_array