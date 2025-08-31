import random

import numpy as np

from PIL import Image

from .helpers import *


def add_gaussian_noise(
    pil_img: Image.Image,
    amplitude: float = 1.0,
    mode: str = "RGB",
    **kwargs
):
    """
    Adds 'film-like' grain to an RGB image:
      1) Creates random noise in each channel.
      2) Blurs it to introduce local spatial correlation.
      3) Mixes the channels slightly for color correlation.
      4) Adds the result onto the original image.

    :param pil_img:           (PIL.Image) Source image, mode='RGB' recommended
    :param amplitude:         (float) Strength of the noise (0..1)
    :return:                  (PIL.Image) Noisy image
    """

    # Convert to NumPy float32 array [H,W,3], range 0..1
    img = np.array(pil_img, dtype=np.float32) / 255.0
    h, w, c = img.shape

    # 1) Generate random noise for each channel
    # shape=(h,w,3)
    noise = np.random.randn(h, w, c).astype(np.float32)
    
    # 2) & 3) Apply correlation (spatial and channel)
    noise = apply_correlation(noise, spatial_sigma=kwargs.get("blur_sigma", 0), channel_correlation=kwargs.get("channel_correlation", 0))

    # 4) Add the correlated noise into the original image
    noisy_img = img + noise * amplitude
    
    # Clip to [0..255] and convert back to uint8
    noisy_img = np.clip(noisy_img, 0, 1)
    noisy_img = (noisy_img * 255).astype(np.uint8)
    
    return Image.fromarray(noisy_img, mode=mode)


def sample_gaussian_params(preset="light"):
    """
    Returns a set of randomized parameters for value noise,
    including multi-octave settings, suitable for data augmentation.
    """
    if preset == "heavy":
        return {
            'transparency': np.random.uniform(0.1, 1.0),
            'amplitude': np.random.uniform(0.01, 0.5),          # noise strength relative to [0,1]
            'blur_sigma': np.random.uniform(0.0, 1.0),          # base grid size factor
            'channel_correlation': np.random.uniform(0.0, 1.0)  # correlation factor between color channels
        }
    elif preset == "light":
        return {
            'transparency': np.random.uniform(0.1, 1.0),
            'amplitude': np.random.uniform(0.01, 0.1),          # noise strength relative to [0,1]
            'blur_sigma': np.random.uniform(0.0, 1.0),          # base grid size factor
            'channel_correlation': np.random.uniform(0.0, 1.0)  # correlation factor between color channels
        }
    elif preset == "test": # testing
        return {
            'transparency': 1.0,
            'amplitude': 0.1,          # noise strength relative to [0,1]
            'blur_sigma': 0.0,          # base grid size factor
            'channel_correlation': 0.0  # correlation factor between color channels
        }


def apply_gaussian_noise(image, preset="light"):
    # Sample parameters for augmentation
    params = sample_gaussian_params(preset)
    
    # Generate noise and add it to the image
    noisy_image = add_gaussian_noise(image, **params)

    return blend(image, noisy_image, params["transparency"])
