import random

import numpy as np

from PIL import Image, ImageEnhance


def adjust_color_channels(img, red_factor=1.0, green_factor=1.0, blue_factor=1.0, warmth_factor=1.0, tone_factor=1.0):
    """Adjusts the individual color channels (RGB) and warmth/tone.

    Args:
        img: The input image (PIL Image object).
        red_factor: Multiplier for the red channel (1.0 is original, >1.0 enhances red, <1.0 reduces red).
        green_factor: Multiplier for the green channel.
        blue_factor: Multiplier for the blue channel.
        warmth_factor: Adjusts the warmth by modifying the red and blue balance (1.0 is neutral).
        tone_factor: Adjusts the tone (shifting colors towards cool or warm tones).

    Returns:
        A PIL Image object with adjusted color channels.
    """
    # Convert image to numpy array to manipulate channels
    img_array = np.array(img).astype(np.float32)

    # Adjust individual channels
    img_array[..., 0] *= red_factor   # Red channel
    img_array[..., 1] *= green_factor # Green channel
    img_array[..., 2] *= blue_factor  # Blue channel

    # Apply warmth by manipulating red and blue channels
    img_array[..., 0] *= warmth_factor  # Red channel for warmth
    img_array[..., 2] /= warmth_factor  # Blue channel for warmth (inverse effect)

    # Adjust the tone by slightly shifting the red, green, and blue channels
    img_array[..., 0] *= tone_factor   # Tone shift towards red
    img_array[..., 1] /= tone_factor   # Tone shift away from green
    img_array[..., 2] *= tone_factor   # Tone shift towards blue

    # Clip values to avoid overflow and convert back to uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    img = Image.fromarray(img_array)
    
    return img


def sample_color_adjust_params():
    """
    Returns a dictionary of parameters for anisotropic gaussian blur.
    """
    params = {
        'red_factor': np.random.uniform(0.95, 1.05),
        'green_factor': np.random.uniform(0.95, 1.05),
        'blue_factor': np.random.uniform(0.95, 1.05),
        'warmth_factor': np.random.uniform(0.95, 1.05),
        'tone_factor': np.random.uniform(0.95, 1.05)
    }
    return params


if __name__ == '__main__':
    input_image = Image.open("/home/xioren/Downloads/image.png").convert("RGB")

    # isotropic
    params = sample_color_adjust_params()
    adjusted_img = adjust_color_channels(input_image, **params)
    adjusted_img.save("/home/xioren/Downloads/image-color_adjusted.png")