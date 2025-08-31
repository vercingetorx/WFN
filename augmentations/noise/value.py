import random
import numpy as np
from PIL import Image

from .helpers import *  # Assumes functions like blend() and apply_intensity_scaling() exist


def generate_value_noise_single(width, height, coarse=1, amplitude=1.0):
    """
    Generate a single-octave value noise field.
    
    Parameters:
        width (int): Output image width.
        height (int): Output image height.
        coarse (int): Controls the resolution of the noise grid.
                      Larger values yield larger “chunks” (i.e. lower frequency).
        amplitude (float): Scaling factor for the noise amplitude.
    
    Returns:
        A 2D numpy array (height x width) with values in [-amplitude, amplitude].
    """
    # Determine size of the low-res grid; add one to avoid edge issues.
    grid_h = height // coarse + 1
    grid_w = width // coarse + 1

    # Create a coarse grid of random values in [-1, 1].
    noise_coarse = np.random.uniform(-1, 1, (grid_h, grid_w)).astype(np.float32)
    
    # Convert the coarse grid to a PIL image (mode 'F' for 32-bit float) for interpolation.
    noise_img = Image.fromarray(noise_coarse, mode='F')
    
    # Upscale to the desired size using bicubic interpolation.
    noise_img = noise_img.resize((width, height), resample=Image.BICUBIC)
    noise = np.array(noise_img)
    
    # Normalize the upscaled noise to span [-1, 1]
    noise_min, noise_max = noise.min(), noise.max()
    noise = 2 * (noise - noise_min) / (noise_max - noise_min) - 1
    
    return noise * amplitude


def generate_value_noise(width, height, coarse=1, amplitude=1.0, octaves=1, persistence=0.5):
    """
    Generate multi-octave value noise by summing several noise layers.
    
    Parameters:
        width (int): Output image width.
        height (int): Output image height.
        coarse (int): Base resolution of the noise grid (for the first octave).
                      Larger values yield coarser (chunkier) noise.
        amplitude (float): Final amplitude (scaling) of the noise.
        octaves (int): Number of octaves (layers) to sum.
        persistence (float): Factor in (0, 1] by which the amplitude decreases for each octave.
                             (e.g., 0.5 halves the amplitude at each successive octave)
    
    Returns:
        A 2D numpy array (height x width) with values in [-amplitude, amplitude].
    """
    noise = np.zeros((height, width), dtype=np.float32)
    total_amp = 0.0
    base_coarse = coarse  # The coarse parameter for the first octave

    for i in range(octaves):
        # For higher octaves, use a finer grid.
        octave_coarse = max(1, int(base_coarse / (2 ** i)))
        current_amp = persistence ** i
        noise += generate_value_noise_single(width, height, coarse=octave_coarse, amplitude=1.0) * current_amp
        total_amp += current_amp

    # Normalize so that the sum of amplitudes is 1, then scale to the desired amplitude.
    noise = noise / total_amp
    return noise * amplitude


def generate_value_noise_color(width, height, coarse=1, amplitude=1.0, octaves=1, persistence=0.5):
    """
    Generate multi-octave value noise for color images.
    Each channel is generated independently.
    
    Parameters:
        width (int): Output image width.
        height (int): Output image height.
        coarse (int): Base noise grid resolution.
        amplitude (float): Final amplitude of the noise.
        octaves (int): Number of octaves to sum.
        persistence (float): Amplitude reduction factor for successive octaves.
    
    Returns:
        A 3D numpy array of shape (height, width, 3) with values in [-amplitude, amplitude].
    """
    noise_rgb = np.zeros((height, width, 3), dtype=np.float32)
    
    for c in range(3):
        noise_channel = np.zeros((height, width), dtype=np.float32)
        total_amp = 0.0
        for i in range(octaves):
            octave_coarse = max(1, int(coarse / (2 ** i)))
            current_amp = persistence ** i
            noise_channel += generate_value_noise_single(width, height, coarse=octave_coarse, amplitude=1.0) * current_amp
            total_amp += current_amp
        noise_channel = noise_channel / total_amp
        noise_rgb[..., c] = noise_channel * amplitude
    
    return noise_rgb


def add_value_noise(image, amplitude=0.01, coarse=1, octaves=1, persistence=0.5, **kwargs):
    """
    Add multi-octave value noise to an image.
    
    Parameters:
        image (PIL.Image.Image): Input image.
        noise_strength (float): Overall strength of the noise (relative to the [0, 1] pixel range).
        coarse (int): Base resolution factor for the noise grid.
        octaves (int): Number of octaves to sum.
        persistence (float): Amplitude decay factor per octave.
        color_noise (bool): If True, generate independent noise for each channel.
    
    Returns:
        A new PIL.Image with the noise added.
    """
    # Convert image to float array in [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    height, width, *c = img_array.shape
    # c = c[0] if c else 1
    
    noise = generate_value_noise_color(width, height, coarse=coarse, amplitude=amplitude,
                                       octaves=octaves, persistence=persistence)
    if img_array.ndim == 2:
        noise = noise[..., :1]  # Broadcast to all channels
    
    noise = apply_correlation(noise, spatial_sigma=kwargs.get("blur_sigma", 0), channel_correlation=kwargs.get("channel_correlation", 0))

    # Add the noise to the image and clip the result
    noisy_img = np.clip(img_array * (1 + noise), 0, 1)
    noisy_img = (noisy_img * 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def sample_value_params(preset="light"):
    """
    Returns a set of randomized parameters for value noise,
    including multi-octave settings, suitable for data augmentation.
    """
    if preset == "heavy":
        return {
            'transparency': np.random.uniform(0.1, 1.0),
            'amplitude': np.random.uniform(0.1, 1.0),  # noise strength relative to [0,1]
            'coarse': random.randint(1, 2),             # base grid size factor
            'octaves': random.randint(1, 6),            # number of octaves
            'persistence': np.random.uniform(0.3, 0.8),   # amplitude decay per octave
            'blur_sigma': np.random.uniform(0.0, 1.0),          # base grid size factor
            'channel_correlation': np.random.uniform(0.0, 1.0), # correlation factor between color channels
        }
    elif preset == "light":
        return {
            'transparency': np.random.uniform(0.1, 1.0),
            'amplitude': np.random.uniform(0.1, 0.4),  # noise strength relative to [0,1]
            'coarse': 1,             # base grid size factor
            'octaves': random.randint(1, 6),            # number of octaves
            'persistence': np.random.uniform(0.1, 0.5),   # amplitude decay per octave
            'blur_sigma': np.random.uniform(0.0, 1.0),          # base grid size factor
            'channel_correlation': np.random.uniform(0.0, 1.0), # correlation factor between color channels
        }
    elif preset == "test":
        return {
            'transparency': 1.0,
            'amplitude': 0.4,  # noise strength relative to [0,1]
            'coarse': 1,             # base grid size factor
            'octaves': 1,            # number of octaves
            'persistence': 0.5,   # amplitude decay per octave
            'blur_sigma': 0.0,          # base grid size factor
            'channel_correlation': 0.0, # correlation factor between color channels
        }


def apply_value_noise(image, preset="light"):
    # Sample parameters for augmentation
    params = sample_value_params(preset)
    
    # Generate noise and add it to the image
    noisy_image = add_value_noise(image, **params)

    return blend(image, noisy_image, params["transparency"])
