import random

import numpy as np

from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter


def isotropic_gaussian_blur(img, radius=2.0):
    """
    Apply isotropic Gaussian blur to an image.
    :param img: The input image.
    :param radius: The radius of the blur (standard deviation of the Gaussian kernel).
    :return: Blurred image.
    """
    return img.filter(ImageFilter.GaussianBlur(radius))


def anisotropic_gaussian_blur(img, sigma_x=2.0, sigma_y=1.0):
    """
    Apply anisotropic Gaussian blur to an image.
    :param img: The input image.
    :param sigma_x: The standard deviation for the blur along the x-axis (horizontal).
    :param sigma_y: The standard deviation for the blur along the y-axis (vertical).
    :return: Blurred image.
    """
    # Convert PIL image to numpy array
    img_np = np.array(img)
    
    # Apply Gaussian filter with different sigmas for x and y directions
    img_blurred_np = gaussian_filter(img_np, sigma=[sigma_y, sigma_x, 0])
    
    # Convert back to PIL image
    return Image.fromarray(img_blurred_np)


def sample_isotropic_gaussian_params(preset="light"):
    """
    Returns a dictionary of parameters for isotropic gaussian blur.
    """
    if preset == "heavy":
        return {
            'radius': np.random.uniform(0.5, 3.0)
        }
    if preset == "light":
        return {
            'radius': np.random.uniform(0.5, 2.0)
        }
    if preset == "test":
        return {
            'radius': 2.0
        }


def sample_anisotropic_gaussian_params(preset="light"):
    """
    Returns a dictionary of parameters for anisotropic gaussian blur.
    """
    if preset == "heavy":
        return {
            'sigma_x': np.random.uniform(0.5, 3.0),
            'sigma_y': np.random.uniform(0.5, 3.0)
        }
    elif preset == "light":
        return {
            'sigma_x': np.random.uniform(0.5, 2.0),
            'sigma_y': np.random.uniform(0.5, 2.0)
        }
    elif preset == "test":
        return {
            'sigma_x': 0.5,
            'sigma_y': 2.0
        }
