import random
import numpy as np

from PIL import Image

from .helpers import *


def add_poisson_noise(
    image: Image.Image,
    *args,
    amplitude: float = 1.0,
    mode: str = "RGB",
    **kwargs
):
    """
    Adds Poisson noise to an 8-bit image

    Standard Poisson for each pixel p is:
        sample = Poisson(p)
        new_pixel = sample
    which has mean p and stdev sqrt(p).

    Here we do:
        sample = Poisson(p)
        difference = (sample - p)
        new_pixel = p + amplitude * difference

    This ensures the output's expected value remains p
    but the variance is amplitude^2 * p.

    Args:
        image (PIL.Image):
            8-bit image ('RGB' or 'L') in [0..255].
        amplitude (float):
            - If 1.0 => standard Poisson noise (mean p, stdev sqrt(p)).
            - If >1 => amplify the noise around p.
            - If <1 => reduce the noise amplitude.
            Typical range might be [0.0..3.0], but can be anything.
        mode (str, optional):
            Output image mode. If None, defaults to image.mode.

    Returns:
        PIL.Image (8-bit):
            Noisy image, clipped to [0..255].
    """

    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    h, w, *c = img_array.shape
    c = c[0] if c else 1

    # Generate Poisson noise, centered around 0, in normalized [0,1] range.
    noise = np.random.poisson(img_array * 255.0) / 255.0 - img_array

    # Reshape noise to always have 3 dimensions.
    if noise.ndim == 2:
        noise = noise[:,:,np.newaxis]

    # Apply correlation
    noise = apply_correlation(noise, spatial_sigma=kwargs.get("blur_sigma", 0), channel_correlation=kwargs.get("channel_correlation", 0))

    # Scale the noise by the amplitude and add to the image.
    noisy_img = img_array + noise * amplitude

    # Clip and convert to uint8
    noisy_img = np.clip(noisy_img, 0, 1)
    noisy_img = (noisy_img * 255).astype(np.uint8)

    return Image.fromarray(noisy_img, mode=mode)


def sample_poisson_params(preset="light"):
    """
    Returns a dictionary of parameters for the specified poisson noise distribution.
    """
    if preset == "heavy":
        return {
            'transparency': np.random.uniform(0.1, 1.0),
            'amplitude': np.random.uniform(0.01, 10.0),         # noise amplitude relative to [0,1]
            'blur_sigma': np.random.uniform(0.0, 1.0),          # base grid size factor
            'channel_correlation': np.random.uniform(0.0, 1.0)  # correlation factor between color channels
        }
    elif preset == "light":
        return {
            'transparency': np.random.uniform(0.1, 1.0),
            'amplitude': np.random.uniform(0.01, 2.5),         # noise amplitude relative to [0,1]
            'blur_sigma': np.random.uniform(0.0, 1.0),          # base grid size factor
            'channel_correlation': np.random.uniform(0.0, 1.0)  # correlation factor between color channels
        }
    elif preset == "test": # testing
        return {
            'transparency': 1.0,
            'amplitude': 2.5,         # noise amplitude relative to [0,1]
            'blur_sigma': 0.0,          # base grid size factor
            'channel_correlation': 0.0  # correlation factor between color channels
        }


def apply_poisson_noise(image, preset="light"):
    # Sample parameters for augmentation
    params = sample_poisson_params(preset)
    
    # Generate noise and add it to the image
    noisy_image = add_poisson_noise(image, **params)

    return blend(image, noisy_image, params["transparency"])
