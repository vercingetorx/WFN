import random

import numpy as np

from scipy.ndimage import convolve
from PIL import Image


def sinc_filter(size=5, cutoff=0.5):
    """
    Create a 2D sinc filter kernel.
    :param size: The size of the kernel (must be odd).
    :param cutoff: The cutoff frequency (controls the filter's sharpness).  Values between 0 and 1 are the most useful, representing a fraction of the Nyquist frequency.
    :return: A 2D sinc filter kernel.
    """
    if size % 2 == 0:
        raise ValueError("Size must be an odd number.")

    half_size = size // 2
    x = np.linspace(-half_size, half_size, size)
    y = np.linspace(-half_size, half_size, size)

    X, Y = np.meshgrid(x, y)

    # **CRITICAL CHANGE 1: Scale the radius by the cutoff frequency**
    radius = np.sqrt(X**2 + Y**2) * cutoff  # Normalize by multiplying by the cutoff.

    # **CRITICAL CHANGE 2: Use a proper sinc function for 2D filtering**
    # Avoid division by zero.  The sinc(0) = 1 case is handled correctly by the where clause.
    kernel = np.sinc(radius)

    # **Optional but recommended: Apply a window function (e.g., Hamming)**
    #  This reduces ringing artifacts.  This is better than the Gaussian in the original.
    window = np.hamming(size)
    window_2d = np.outer(window, window)
    kernel = kernel * window_2d
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    return kernel


def apply_sinc_filter(img, size=5, cutoff=0.5):
    """
    Apply the 2D sinc filter to an image.
    :param img: The input image (PIL Image).
    :param size: Size of the sinc kernel.
    :param cutoff: The cutoff frequency for the sinc filter.
    :return: The filtered image.
    """
    # Convert PIL image to numpy array
    img_np = np.array(img)
    
    # If the image has 3 channels (RGB), apply the filter to each channel separately
    if img_np.ndim == 3:
        # Initialize an empty array to hold the filtered result
        img_filtered_np = np.zeros_like(img_np)
        
        # Apply the sinc filter to each channel
        for i in range(3):  # Iterate over R, G, and B channels
            img_filtered_np[:, :, i] = convolve(img_np[:, :, i], sinc_filter(size, cutoff), mode='reflect')
    else:
        # If the image is grayscale, apply the sinc filter to the single channel
        img_filtered_np = convolve(img_np, sinc_filter(size, cutoff), mode='reflect')
    
    # Convert back to PIL image
    return Image.fromarray(np.uint8(np.clip(img_filtered_np, 0, 255)))


def sample_sinc_filter_params(preset="light"):
    """
    Returns a dictionary of parameters for sinc_filter blur.
    """
    if preset == "heavy":
        return {
            'size': random.choice([5, 7, 11, 13, 15]),
            'cutoff': random.uniform(0.05, 0.2)
        }
    elif preset == "light":
        return {
            'size': random.choice([5, 7, 11, 13]),
            'cutoff': random.uniform(0.1, 0.2)
        }
    elif preset == "test":
        return {
            'size': 13,
            'cutoff': 0.1
        }
