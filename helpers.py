import numpy as np

from PIL import Image
from scipy.ndimage import gaussian_filter


def blend(image, noised_img, transparency=1.0):
    """
    Blend noised_img with the original image.

    Returns:
        PIL.Image.Image: Blended image.
    """
    blended_img = Image.blend(image, noised_img, alpha=transparency)
    return blended_img


def apply_intensity_scaling(noise, image_array, shadow_exponent):
    """
    Applies intensity-dependent scaling to the noise based on image luminance.

    Args:
        noise (np.ndarray): Generated noise (grayscale: [H, W], color: [H, W, 3]).
        image_array (np.ndarray): Input image normalized to [0, 1] (float32).
        shadow_exponent (float): Controls shadow emphasis (higher = darker areas get more noise).
        noise_strength (float): Overall strength of the noise.

    Returns:
        np.ndarray: Scaled noise with same shape as input noise.
    """
    if image_array.ndim == 3:
        # Compute luminance for color images (shape: [H, W])
        luminance = np.dot(image_array, [0.299, 0.587, 0.114])
    else:
        # Grayscale image (shape: [H, W])
        luminance = image_array

    # Intensity scaling factor (shape: [H, W])
    intensity_scale = (1 - luminance) ** shadow_exponent

    # Align dimensions between intensity_scale and noise
    if noise.ndim == 3 and image_array.ndim == 3:
        # Color noise: expand intensity_scale to [H, W, 3]
        intensity_scale = np.repeat(intensity_scale[..., np.newaxis], 3, axis=-1)
    elif noise.ndim == 3 and image_array.ndim == 2:
        # Grayscale image with color noise: unlikely case, treat as grayscale
        noise = noise.mean(axis=-1, keepdims=True)
    elif noise.ndim == 2 and image_array.ndim == 3:
        # Color image with grayscale noise: expand intensity_scale to [H, W, 1]
        intensity_scale = intensity_scale[..., np.newaxis]

    return noise * intensity_scale


def apply_correlation(noise, spatial_sigma=0, channel_correlation=0):
    """
    Applies spatial and channel correlation to a noise array.

    Args:
        noise (numpy.ndarray): The input noise array (H, W, C).
        spatial_sigma (float): Standard deviation for Gaussian blur (spatial correlation).
                                If 0, no spatial correlation is applied.
        channel_correlation (float):  Correlation factor between color channels (0 to 1).
                                    0: No correlation (fully independent channels).
                                    1: Perfect correlation (monochrome noise).

    Returns:
        numpy.ndarray: The noise array with applied correlations.
    """

    # Apply spatial correlation (Gaussian blur)
    if spatial_sigma > 0:
        noise = gaussian_filter(noise, sigma=(spatial_sigma, spatial_sigma, 0))

    # Apply channel correlation
    if channel_correlation > 0:
        alpha = channel_correlation
        mean_noise = noise.mean(axis=2, keepdims=True)
        noise = (1 - alpha) * noise + alpha * mean_noise

    return noise
