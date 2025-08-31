import numpy as np
from PIL import Image

from .helpers import *


def spread_noise(image, max_offset=1, seed=None):
    """
    Apply a “spread” noise effect similar to GIMP's spread filter.
    Each output pixel is sampled from a random offset (in x and y)
    from the corresponding input pixel.
    
    Parameters:
      image (PIL.Image.Image): The input image.
      max_offset (int): Maximum pixel displacement in both directions.
                        For example, max_offset=1 allows a random offset
                        from -1 to +1 pixels horizontally and vertically.
      seed (int, optional): Random seed for reproducibility.
    
    Returns:
      PIL.Image.Image: The image with spread noise applied.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert the image to a NumPy array.
    arr = np.array(image)
    
    # Determine image dimensions.
    if arr.ndim == 2:  # Grayscale image
        height, width = arr.shape
        channels = 1
    else:             # Color image (e.g., RGB)
        height, width, channels = arr.shape

    # Create a coordinate grid for the image.
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Generate a random offset for each pixel (range: -max_offset to +max_offset).
    offset_y = np.random.randint(-max_offset, max_offset + 1, size=(height, width))
    offset_x = np.random.randint(-max_offset, max_offset + 1, size=(height, width))

    # Compute source coordinates by adding offsets and clamp them to image bounds.
    src_y = np.clip(y + offset_y, 0, height - 1)
    src_x = np.clip(x + offset_x, 0, width - 1)

    # Create the spread noise image by sampling the input image at the new coordinates.
    spread_arr = arr[src_y, src_x]

    # Convert the array back to a PIL image.
    return Image.fromarray(spread_arr.astype(np.uint8))


def sample_spread_params():
    """
    Returns a dictionary of parameters for the specified poisson noise distribution.
    """
    return {
        'transparency': np.random.uniform(0.1, 0.4),
        'max_offset': 1
    }


def apply_spread_noise(image):
    params = sample_spread_params()
    noisy_image = spread_noise(image, max_offset=1) # always use 1 as anything higher is too much
    return blend(image, noisy_image, params["transparency"])
