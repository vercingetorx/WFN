import random
import numpy as np

from PIL import Image

from .helpers import *


# -----------------------------------------------------------------------------
# PERLIN NOISE IMPLEMENTATION (fractal brownian motion via multiple octaves)
# -----------------------------------------------------------------------------
# The classical Perlin noise algorithm (for 2D) works as follows:
#  1. For each grid point (integer coordinates) assign a pseudorandom gradient.
#  2. For any continuous (x, y) point, determine the cell it is in.
#  3. For each of the four corners of the cell, compute the dot product of
#     the corner’s gradient with the offset vector from the corner to (x, y).
#  4. Smoothly interpolate (using a “fade” function) between these values.
#
# In the code below a fixed permutation table is used to generate the gradient
# directions in a repeatable way.

# A fixed permutation table (a permutation of 0..255). It is duplicated below
# so that indices can “wrap‐around” without additional modulo operations.
_p = [151, 160, 137, 91, 90, 15,
      131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
      140, 36, 103, 30, 69, 142, 8, 99, 37, 240,
      21, 10, 23, 190, 6, 148, 247, 120, 234, 75,
      0, 26, 197, 62, 94, 252, 219, 203, 117, 35,
      11, 32, 57, 177, 33, 88, 237, 149, 56, 87,
      174, 20, 125, 136, 171, 168, 68, 175, 74,
      165, 71, 134, 139, 48, 27, 166, 77, 146, 158,
      231, 83, 111, 229, 122, 60, 211, 133, 230, 220,
      105, 92, 41, 55, 46, 245, 40, 244, 102, 143,
      54, 65, 25, 63, 161, 1, 216, 80, 73, 209,
      76, 132, 187, 208, 89, 18, 169, 200, 196, 135,
      130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
      186, 3, 64, 52, 217, 226, 250, 124, 123, 5,
      202, 38, 147, 118, 126, 255, 82, 85, 212, 207,
      206, 59, 227, 47, 16, 58, 17, 182, 189, 28,
      42, 223, 183, 170, 213, 119, 248, 152, 2, 44,
      154, 163, 70, 221, 153, 101, 155, 167, 43,
      172, 9, 129, 22, 39, 253, 19, 98, 108, 110,
      79, 113, 224, 232, 178, 185, 112, 104, 218,
      246, 97, 228, 251, 34, 242, 193, 238, 210,
      144, 12, 191, 179, 162, 241, 81, 51, 145,
      235, 249, 14, 239, 107, 49, 192, 214, 31,
      181, 199, 106, 157, 184, 84, 204, 176, 115,
      121, 50, 45, 127, 4, 150, 254, 138, 236, 205,
      93, 222, 114, 67, 29, 24, 72, 243, 141, 128,
      195, 78, 66, 215, 61, 156, 180]
# Duplicate the permutation table so that p[256:512] is the same as p[0:256]
perm = np.array(_p + _p, dtype=int)

# Pre‐define a gradient lookup table. For each integer hash value (mod 8)
# we associate one of eight unit vectors (or simple axis/diagonal directions).
grad_table = np.array([
    [1/np.sqrt(2),  1/np.sqrt(2)],
    [-1/np.sqrt(2), 1/np.sqrt(2)],
    [1/np.sqrt(2), -1/np.sqrt(2)],
    [-1/np.sqrt(2), -1/np.sqrt(2)],
    [1,  0],
    [-1, 0],
    [0,  1],
    [0, -1],
], dtype=np.float32)

def fade(t):
    """Perlin’s smoothing (or fade) function. This eases coordinate values
    so that they will ease towards integral values. (6t^5 - 15t^4 + 10t^3)"""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def lerp(a, b, t):
    """Linear interpolation between a and b with weight t."""
    return a + t * (b - a)


def grad(hash_val, x, y):
    """
    Given a hash value and the relative x, y coordinates,
    select a gradient vector from grad_table and compute the dot product.
    
    The ONLY change here is we handle arrays in 'hash_val' properly,
    so it doesn't break with "too many values to unpack."
    """
    h = (hash_val & 7)
    if isinstance(h, np.ndarray):
        # Vectorized approach for array h
        g = grad_table[h]
        return g[..., 0]*x + g[..., 1]*y
    else:
        # Scalar fallback for single integer h
        gx, gy = grad_table[h]
        return gx * x + gy * y


def perlin(x, y):
    """
    Compute 2D Perlin noise for coordinates x and y.
    x and y can be scalars or numpy arrays of the same shape.
    Returns noise values in roughly the range [-1, 1].
    """
    xi = np.floor(x).astype(int) & 255
    yi = np.floor(y).astype(int) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    
    # Compute the fade curves for x and y
    u = fade(xf)
    v = fade(yf)
    
    # Hash coordinates of the 4 square corners
    aa = perm[perm[xi    ] + yi    ]
    ab = perm[perm[xi    ] + yi + 1]
    ba = perm[perm[xi + 1] + yi    ]
    bb = perm[perm[xi + 1] + yi + 1]
    
    # And compute the gradient dot products for each corner.
    x1 = lerp(grad(aa, xf,     yf    ),
              grad(ba, xf - 1, yf    ), u)
    x2 = lerp(grad(ab, xf,     yf - 1),
              grad(bb, xf - 1, yf - 1), u)
    
    # Interpolate the two results along y to get the final noise value.
    return lerp(x1, x2, v)

# -----------------------------------------------------------------------------
# MULTI-OCTAVE PERLIN NOISE FUNCTIONS
# -----------------------------------------------------------------------------

def generate_perlin_noise(width, height, coarse=1, amplitude=1.0,
                          octaves=1, persistence=0.5, offset_x=0.0, offset_y=0.0):
    """
    Generate a 2D Perlin noise array using multiple octaves.
    
    Parameters:
        width (int): Output array width (number of columns).
        height (int): Output array height (number of rows).
        coarse (float): Controls the scale of the noise. Larger values produce
                        larger features (i.e. lower frequency noise).
        amplitude (float): Final amplitude (strength) of the noise.
        octaves (int): Number of noise layers (octaves) to sum.
        persistence (float): Determines how quickly the amplitude decreases for each octave.
        offset_x (float): Horizontal offset to shift the noise field.
        offset_y (float): Vertical offset to shift the noise field.
    
    Returns:
        A 2D numpy array of shape (height, width) with noise values in roughly [-amplitude, amplitude].
    """
    noise = np.zeros((height, width), dtype=np.float32)
    frequency = 1.0 / coarse
    current_amp = 1.0
    max_amp = 0.0

    for _ in range(octaves):
        lin_x = np.linspace(0, width * frequency, width, endpoint=False) + offset_x
        lin_y = np.linspace(0, height * frequency, height, endpoint=False) + offset_y
        xv, yv = np.meshgrid(lin_x, lin_y)
        noise += perlin(xv, yv) * current_amp
        max_amp += current_amp
        frequency *= 2
        current_amp *= persistence

    # Normalize to keep the noise within [-amplitude, amplitude]
    return (noise / max_amp) * amplitude


def generate_perlin_noise_color(width, height, coarse=1, amplitude=1.0,
                                octaves=1, persistence=0.5):
    """
    Generate a color (RGB) noise image using multi-octave Perlin noise.
    Each channel is generated with a slight offset to produce a distinct pattern.
    
    Parameters:
        width (int):  Output image width.
        height (int): Output image height.
        coarse (float): Controls the scale of the noise.
        amplitude (float): The final amplitude of the noise.
        octaves (int): Number of noise layers (octaves) to sum.
        persistence (float): Determines how quickly the amplitude decreases for each octave.
    
    Returns:
        A numpy array of shape (height, width, 3) with noise values.
    """
    noise_rgb = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        # Use a channel-specific offset so that each channel is different.
        offset = c * 0.397  
        noise_rgb[..., c] = generate_perlin_noise(
            width, height,
            coarse=coarse,
            amplitude=amplitude,
            octaves=octaves,
            persistence=persistence,
            offset_x=offset,
            offset_y=offset
        )
    return noise_rgb

# -----------------------------------------------------------------------------
# IMAGE NOISE FUNCTIONS
# -----------------------------------------------------------------------------

def add_perlin_noise(
    image,
    *args,
    amplitude=0.1,
    coarse=1,
    octaves=1,
    persistence=0.5,
    **kwargs
):
    """
    Add multi-octave Perlin noise.

    Parameters:
        image (PIL.Image.Image): Input image.
        amplitude (float): Overall noise strength.
        coarse (float): Noise scale.
        color_noise (bool): Independent noise per channel?
        octaves (int): Number of octaves.
        persistence (float): Amplitude decrease per octave.
        shadow_exponent (float): Controls intensity dependence (higher = more shadow noise).

    Returns:
        PIL.Image.Image: Noisy image.
    """
    img_array = np.array(image).astype(np.float32) / 255.0
    height, width, *c = img_array.shape
    c = c[0] if c else 1

    noise = generate_perlin_noise_color(
        width, height, coarse=coarse, amplitude=1.0,  # Amplitude will be scaled later
        octaves=octaves, persistence=persistence
    )
    if img_array.ndim == 2:
        noise = noise[..., :1]  # Make it (H, W, 1) for broadcasting

    noise = apply_correlation(noise, spatial_sigma=kwargs.get("blur_sigma", 0), channel_correlation=kwargs.get("channel_correlation", 0))

    # 2. Intensity scaling
    scaled_noise = apply_intensity_scaling(noise, img_array, kwargs.get("shadow_exponent", 0))

    # 5. Add the noise to the image
    noisy_img = np.clip(img_array * (1 + scaled_noise * amplitude), 0, 1)
    noisy_img = (noisy_img * 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def sample_perlin_params(preset="light"):
    """
    Returns realistic Perlin noise parameters for robust denoising training.
    Ranges derived from observational studies of natural/sensor noise.
    """
    if preset == "heavy":
        return {
            # Blend noise with original image (0=clean, 1=full noise)
            'transparency': np.random.uniform(0.1, 1.0),
            
            # Noise amplitude (relative to [0,1] pixel range)
            'amplitude': np.random.uniform(0.1, 2.0),  # 0.1 to 1.0
            
            # Feature scale (higher = larger patterns)
            'coarse': np.random.uniform(0.1, 2.0),  # 0.1 to 2.0 (mimicking sensor->environment scale)
            
            # Fractal parameters
            'octaves': random.randint(1, 6),       # 1-6 octaves (1=smooth, 6=textured)
            'persistence': np.random.uniform(0.2, 0.4),  # Strong decay to sustained detail

            'blur_sigma': np.random.uniform(0.0, 1.0),          # base grid size factor
            'channel_correlation': np.random.uniform(0.0, 1.0),  # correlation factor between color channels
            
            # Intensity dependence (0=uniform, higher=shadow emphasis)
            # 'shadow_exponent': np.random.uniform(0.0, 4.0)  # Extreme dark bias for low-light sim
            'shadow_exponent': 0.0
        }
    elif preset == "light":
        return {
            # Blend noise with original image (0=clean, 1=full noise)
            'transparency': np.random.uniform(0.1, 1.0),
            
            # Noise amplitude (relative to [0,1] pixel range)
            'amplitude': np.random.uniform(0.1, 0.75),  # 0.1 to 0.75
            
            # Feature scale (higher = larger patterns)
            'coarse': np.random.uniform(0.1, 2.0),  # 0.1 to 2.0 (mimicking sensor->environment scale)
            
            # Fractal parameters
            'octaves': random.randint(1, 6),       # 1-6 octaves (1=smooth, 6=textured)
            'persistence': np.random.uniform(0.3, 0.8),  # Strong decay to sustained detail

            'blur_sigma': np.random.uniform(0.0, 1.0),          # base grid size factor
            'channel_correlation': np.random.uniform(0.0, 1.0),  # correlation factor between color channels
            
            # Intensity dependence (0=uniform, higher=shadow emphasis)
            # 'shadow_exponent': np.random.uniform(0.0, 4.0)  # Extreme dark bias for low-light sim
            'shadow_exponent': 0.0
        }
    elif preset == "test": # testing
        return {
            # Blend noise with original image (0=clean, 1=full noise)
            'transparency': 1.0,

            # Noise amplitude (relative to [0,1] pixel range)
            'amplitude': 0.75,  # 0.1 to 1.0

            # Feature scale (higher = larger patterns)
            'coarse': 1.5,  # 0.1 to 2.0 (mimicking sensor->environment scale)

            # Fractal parameters
            'octaves': 3,       # 1-6 octaves (1=smooth, 6=textured)
            'persistence': 0.3,  # Strong decay to sustained detail

            'blur_sigma': 0.0,          # base grid size factor
            'channel_correlation': 0.0,  # correlation factor between color channels

            # Intensity dependence (0=uniform, higher=shadow emphasis)
            'shadow_exponent': 0.0
        }


def apply_perlin_noise(image, preset="light"):
    # Sample parameters for augmentation
    params = sample_perlin_params(preset)
    
    # Generate noise and add it to the image
    noisy_image = add_perlin_noise(image, **params)

    return blend(image, noisy_image, params["transparency"])
