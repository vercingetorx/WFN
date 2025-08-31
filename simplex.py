import random
import numpy as np

from PIL import Image

from .helpers import *

###############################################################################
# 2D SIMPLEX NOISE CORE
###############################################################################

_GRAD2 = [
    ( 1,  1), (-1,  1), ( 1, -1), (-1, -1),
    ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1),
]

def _build_permutation_table(seed=0):
    rng = random.Random(seed)
    p = list(range(256))
    rng.shuffle(p)
    return p + p  # duplicate so we can index without extra modulus

def _simplex2D(x, y, perm):
    """
    2D Simplex noise at coordinate (x, y). Returns ~[-1..+1].
    """
    F2 = 0.3660254037844386  # (sqrt(3) - 1) / 2
    G2 = 0.21132486540518713 # (3 - sqrt(3)) / 6

    s = (x + y) * F2
    i = int(x + s)
    j = int(y + s)

    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t
    x0 = x - X0
    y0 = y - Y0

    if x0 > y0:
        i1, j1 = 1, 0
    else:
        i1, j1 = 0, 1

    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    ii = i & 255
    jj = j & 255
    g0 = perm[ii + perm[jj]] % 8
    g1 = perm[ii + i1 + perm[jj + j1]] % 8
    g2 = perm[ii + 1 + perm[jj + 1]] % 8

    def corner_contrib(tx, ty, grad_idx):
        tval = 0.5 - tx*tx - ty*ty
        if tval < 0:
            return 0.0
        tval *= tval
        gx, gy = _GRAD2[grad_idx]
        return tval * tval * (gx*tx + gy*ty)

    n0 = corner_contrib(x0, y0, g0)
    n1 = corner_contrib(x1, y1, g1)
    n2 = corner_contrib(x2, y2, g2)

    return 70.0 * (n0 + n1 + n2)


###############################################################################
# MULTI-OCTAVE (FRACTAL) NOISE GENERATION
###############################################################################

def _generate_simplex_noise(width, height,
                            scale=3.0, seed=0,
                            octaves=1, lacunarity=2.0, gain=0.5):
    """
    Generates a 2D array (H x W) of multi-octave Simplex noise in [-1..+1].
    If octaves=1, behaves like single-octave.
    
    :param width, height: Output dimensions
    :param scale:         Base scale (larger => bigger "blobs")
    :param seed:          Random seed
    :param octaves:       Number of noise layers to sum
    :param lacunarity:    Frequency multiplier each octave
    :param gain:          Amplitude multiplier each octave
    :return:              float32 array (H, W) in approx [-1..+1]
    """
    perm = _build_permutation_table(seed)
    noise = np.zeros((height, width), dtype=np.float32)

    amplitude = 1.0
    frequency = 1.0

    for _ in range(octaves):
        for y in range(height):
            for x in range(width):
                # Divide by (scale/frequency) => effectively x/(scale/freq)
                nx = x / (scale / frequency)
                ny = y / (scale / frequency)
                noise[y, x] += _simplex2D(nx, ny, perm) * amplitude

        # Increase frequency / reduce amplitude for next octave
        frequency *= lacunarity
        amplitude *= gain

    # Optionally normalize to exactly [-1..+1]
    # mn, mx = noise.min(), noise.max()
    # if mx > mn:
    #     noise = 2.0 * (noise - mn) / (mx - mn) - 1.0

    return noise


def _generate_simplex_noise_color(width, height,
                                  scale=3.0, seed=0,
                                  octaves=1, lacunarity=2.0, gain=0.5):
    """
    Generate a 2D color (RGB) array (H x W x 3), each channel in [-1..+1].
    Each channel uses a different seed: seed, seed+1, seed+2.
    Supports multi-octave fractal noise.
    """
    r = _generate_simplex_noise(width, height, scale=scale, seed=seed,
                                octaves=octaves, lacunarity=lacunarity, gain=gain)
    g = _generate_simplex_noise(width, height, scale=scale, seed=seed+1,
                                octaves=octaves, lacunarity=lacunarity, gain=gain)
    b = _generate_simplex_noise(width, height, scale=scale, seed=seed+2,
                                octaves=octaves, lacunarity=lacunarity, gain=gain)
    return np.stack([r, g, b], axis=-1)

###############################################################################
# ADDING NOISE TO A PIL IMAGE
###############################################################################

def add_simplex_noise(
    image,
    amplitude=0.1,
    scale=3.0,
    seed=9999,
    octaves=1,
    lacunarity=2.0,
    gain=0.5,
    **kwargs
):
    """
    Adds multi-octave Simplex noise to a PIL image (grayscale or color).
    
    :param image:      PIL.Image (mode 'L', 'RGB', etc.)
    :param amplitude:  Final strength factor for noise in [0..1] space.
    :param scale:      Base scale for noise frequency. 
    :param seed:       Random seed.
    :param color:      If True, each channel gets its own noise (for color images).
    :param octaves:    Number of noise layers (1 => single octave).
    :param lacunarity: Frequency multiplier per octave.
    :param gain:       Amplitude multiplier per octave.
    
    :return: A new PIL.Image with fractal Simplex noise added.
    """
    # Convert image to float array in [0..1]
    arr = np.array(image, dtype=np.float32) / 255.0
    height, width, *c = arr.shape
    c = c[0] if c else 1

    noise = _generate_simplex_noise_color(width, height,
                                          scale=scale, seed=seed,
                                          octaves=octaves,
                                          lacunarity=lacunarity,
                                          gain=gain)
    # If original image has multiple channels, broadcast
    if arr.ndim == 2:
        noise = noise[..., :1]  # shape => (H, W, 1)
    
    noise = apply_correlation(noise, spatial_sigma=kwargs.get("blur_sigma", 0), channel_correlation=kwargs.get("channel_correlation", 0))
    
    scaled_noise = apply_intensity_scaling(noise, arr, kwargs.get("shadow_exponent", 0))

    # Clip and convert back to uint8
    noisy = np.clip(arr * (1 + scaled_noise * amplitude), 0.0, 1.0)
    noisy_uint8 = (noisy * 255.0).astype(np.uint8)

    return Image.fromarray(noisy_uint8)


def sample_simplex_params(preset="light"):
    """
    Returns realistic Simplex noise parameters for robust denoising training.
    """
    if preset == "heavy":
        return {
            # Blend noise with original image (0=clean, 1=full noise)
            'transparency': np.random.uniform(0.1, 1.0),

            # Noise amplitude (relative to [0,1] pixel range)
            'amplitude': np.random.uniform(0.01, 2.0),  # 0.1 to 1.0

            'seed': 9999,

            # Feature scale (higher = larger patterns)
            'scale': np.random.uniform(0.1, 2.0),  # 0.1 to 3.0 (similar to Perlin's 'coarse')

            # Fractal parameters
            'octaves': random.randint(1, 6),       # 1-6 octaves
            'lacunarity': np.random.uniform(2, 5), # 2 to 5
            'gain': np.random.uniform(0.2, 0.4),  # Similar to Perlin's 'persistence'

            'blur_sigma': np.random.uniform(0.0, 1.0),          # base grid size factor
            'channel_correlation': np.random.uniform(0.0, 1.0), # correlation factor between color channels

            # Intensity dependence (0=uniform, higher=shadow emphasis)
            # 'shadow_exponent': np.random.uniform(0.0, 4.0)  # Extreme dark bias for low-light sim
            'shadow_exponent': 0.0
        }
    elif preset == "light":
        return {
            # Blend noise with original image (0=clean, 1=full noise)
            'transparency': np.random.uniform(0.1, 1.0),

            # Noise amplitude (relative to [0,1] pixel range)
            'amplitude': np.random.uniform(0.01, 0.3),  # 0.1 to 1.0

            'seed': 9999,

            # Feature scale (higher = larger patterns)
            'scale': np.random.uniform(0.1, 2.0),  # 0.1 to 3.0 (similar to Perlin's 'coarse')

            # Fractal parameters
            'octaves': random.randint(1, 6),       # 1-6 octaves
            'lacunarity': np.random.uniform(3, 8), # 2 to 5
            'gain': np.random.uniform(0.1, 0.4),  # Similar to Perlin's 'persistence'

            'blur_sigma': np.random.uniform(0.0, 1.0),          # base grid size factor
            'channel_correlation': np.random.uniform(0.0, 1.0), # correlation factor between color channels

            # Intensity dependence (0=uniform, higher=shadow emphasis)
            # 'shadow_exponent': np.random.uniform(0.0, 4.0)  # Extreme dark bias for low-light sim
            'shadow_exponent': 0.0
        }
    elif preset == "test":
        return {
            # Blend noise with original image (0=clean, 1=full noise)
            'transparency': 1.0,

            # Noise amplitude (relative to [0,1] pixel range)
            'amplitude': 0.09,  # 0.1 to 1.0

            'seed': 0,

            # Feature scale (higher = larger patterns)
            'scale': 2.0,  # 0.1 to 3.0 (similar to Perlin's 'coarse')

            # Fractal parameters
            'octaves': 3,       # 1-6 octaves
            'lacunarity': 3, # 2 to 5
            'gain': 0.5,  # Similar to Perlin's 'persistence'

            'blur_sigma': 0.0,          # base grid size factor
            'channel_correlation': 0.0, # correlation factor between color channels

            # Intensity dependence (0=uniform, higher=shadow emphasis)
            'shadow_exponent': 0.0
        }


def apply_simplex_noise(image, preset="light"):
    # Sample parameters for augmentation
    params = sample_simplex_params(preset)
    
    # Generate noise and add it to the image
    noisy_image = add_simplex_noise(image, **params)

    return blend(image, noisy_image, params["transparency"])
