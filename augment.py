import io
import random
import numpy as np

from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter

from augmentations.color_adjust import *
from augmentations.color_channel_adjust import *
from augmentations.denoise import *
from augmentations.gaussian_blur import *
from augmentations.sinc_filter import *
from augmentations.noise.gaussian import *
from augmentations.noise.perlin import *
from augmentations.noise.poisson import *
from augmentations.noise.simplex import *
from augmentations.noise.spread import *
from augmentations.noise.value import *


####################################################################
# Color
####################################################################

def apply_random_color(image):
    # channels
    params = sample_color_adjust_params()
    image = adjust_color_channels(image, **params)

    # vibrance
    vibrance = random.uniform(0.0, 0.5)
    return adjust_vibrance(image, vibrance=vibrance)

    # color
    brightness = round(random.uniform(0.7, 1.3), 1)
    contrast = round(random.uniform(0.8, 1.3), 1)
    color = round(random.uniform(0.7, 1.3), 1)
    return adjust_color(image, brightness=brightness, contrast=contrast, color=color)

####################################################################
# Noise and Denoise
####################################################################

def apply_random_noise(image, preset="light"):
    noise_mode = random.randint(0, 2)

    if noise_mode == 0: # gaussian
        return apply_gaussian_noise(image, preset=preset)
    elif noise_mode == 1: # poisson
        return apply_poisson_noise(image, preset=preset)
    elif noise_mode == 2:
        # choose one gradient type so as not to bias towards these three
        gradient_mode = random.randint(0, 2)
        
        if gradient_mode == 0: # perlin
            return apply_perlin_noise(image, preset=preset)
        elif noise_mode == 1: # simplex
            return apply_simplex_noise(image, preset=preset)
        elif noise_mode == 2: # value
            return apply_value_noise(image, preset=preset)


def apply_random_denoise(image, preset="light"):
    denoise_mode = random.choice(["bilateral", "nlmeans", "median"])

    params = sample_denoise_params(denoise_mode, preset=preset)
    return denoise_image(image, **params)

####################################################################
# Blur
####################################################################

def apply_random_blur(image, preset="light"):
    if random.random() > 0.5:
        # gaussian
        if random.random() > 0.5: # isotropic
            params = sample_isotropic_gaussian_params(preset)
            return isotropic_gaussian_blur(image, **params)
        else: # anisotropic
            params = sample_anisotropic_gaussian_params(preset)
            return anisotropic_gaussian_blur(image, **params)

    else:
        # sinc
        params = sample_sinc_filter_params(preset)
        return apply_sinc_filter(image, **params)

####################################################################
# Misc
####################################################################

def jpeg_compress(img, qlt=100, subsampling_value=0):
    # NOTE: in memory jpeg compression
    with io.BytesIO() as buffer:
        img.save(buffer, format="JPEG", quality=qlt, subsampling=subsampling_value)
        buffer.seek(0)  # Reset buffer pointer to the beginning
        with Image.open(buffer) as compressed_image:
            return compressed_image.copy()


def webp_compress(img, qlt=100):
    # NOTE: in memory webp compression
    with io.BytesIO() as buffer:
        img.save(buffer, format="WEBP", quality=qlt)
        buffer.seek(0)  # Reset buffer pointer to the beginning
        with Image.open(buffer) as compressed_image:
            return compressed_image.copy()


def dual_compress(img):
    qlt_j = random.randint(50, 99)
    qlt_w = random.randint(50, 99)
    if random.random() > 0.5:
        return webp_compress(jpeg_compress(img, qlt_j), qlt_w)
    else:
        return jpeg_compress(webp_compress(img, qlt_w), qlt_j)


def multi_jpeg_compress(img):
    count = random.randint(1, 3)
    for _ in range(count):
        qlt = random.randint(50, 99)
        subsampling_value = random.randint(0, 2)
        img = jpeg_compress(img, qlt=qlt, subsampling_value=subsampling_value)
    return img


def multi_webp_compress(img):
    count = random.randint(1, 3)
    for _ in range(count):
        qlt = random.randint(50, 99)
        img = webp_compress(img, qlt=qlt)
    return img


def convert_image(img, mode="RGB", as_rgb=False):
    img = img.convert(mode)
    if as_rgb:
        return img.convert("RGB")
    return img


def upsample_img(img, factor=2, mode=Image.LANCZOS):
    width, height = img.size
    return img.resize((width * factor, height * factor), mode)


def downsample_img(img, factor=2, mode=Image.LANCZOS):
    width, height = img.size
    return img.resize((width // factor, height // factor), mode)


def upsample_img_with_random_mode(img, factor=2):
    # List of downsampling methods
    upsampling_modes = [Image.LANCZOS, Image.BICUBIC, Image.BILINEAR, Image.NEAREST]
    
    # Randomly choose a downsampling method
    chosen_mode = random.choice(downsampling_modes)
    
    # Get current width and height
    width, height = img.size
    
    # Resize image with the chosen mode
    return img.resize((width * factor, height * factor), chosen_mode)


def downsample_img_with_random_mode(img, factor=2):
    # List of downsampling methods
    downsampling_modes = [Image.LANCZOS, Image.BICUBIC, Image.BILINEAR, Image.NEAREST]
    
    # Randomly choose a downsampling method
    chosen_mode = random.choice(downsampling_modes)
    
    # Get current width and height
    width, height = img.size
    
    # Resize image with the chosen mode
    return img.resize((width // factor, height // factor), chosen_mode)


def multi_step_downsample_img_with_random_mode(img, factor=2):
    return downsample_img_with_random_mode(downsample_img_with_random_mode(img, factor), factor)


def resample(img, factor=2):
    # randomly downsample and then upsample back
    return upsample_img_with_random_mode(downsample_img_with_random_mode(img, factor), factor)

####################################################################
# Augment Function
####################################################################

def augment_image(orig_img, augmentations):
    # --- Stage 0: Initial setup ---
    # Apply color augmentation to the original image before any degradation
    if augmentations.color and random.random() > 0.5:
        orig_img = apply_random_color(orig_img)

    # Start with a copy of the (potentially color-augmented) original image
    augmented_img = orig_img.copy()

    # 90% chance to apply the full degradation pipeline.
    # 10% chance to only apply downsampling (clean images).
    apply_degradations = random.random() < 0.9

    if apply_degradations:
        # --- Stage 1: Pre-Downsampling Degradations (Simulating "Original Source" Artifacts) ---
        if random.random() < 0.5: # 50% chance to apply pre-blur
            augmented_img = apply_random_blur(augmented_img, preset="light")

        if random.random() < 0.5: # 50% chance to apply pre-noise
            augmented_img = apply_spread_noise(augmented_img, amount=random.randint(1, 3))

        if random.random() < 0.7: # 70% chance of initial compression
            qlt = random.randint(85, 100)
            if random.random() < 0.5:
                augmented_img = jpeg_compress(augmented_img, qlt=qlt)
            else:
                augmented_img = webp_compress(augmented_img, qlt=qlt)

    # --- Stage 2: Downsampling (Applied to all images) ---
    if augmentations.downsample > 1:
        augmented_img = downsample_img_with_random_mode(augmented_img, augmentations.downsample)

    if apply_degradations:
        # --- Stage 3: Post-Downsampling Degradations (Simulating "In-the-Wild" Artifacts) ---
        if random.random() < 0.5: # 50% chance to apply post-blur
            augmented_img = apply_random_blur(augmented_img, preset=augmentations.preset)

        if random.random() < 0.5: # 50% chance to apply post-noise
            augmented_img = apply_random_noise(augmented_img, preset=augmentations.preset)

        if random.random() < 0.3: # 30% chance of denoising
            augmented_img = apply_random_denoise(augmented_img, preset=augmentations.preset)

        if random.random() < 0.8: # 80% chance of final compression
            if random.random() < 0.5:
                augmented_img = multi_jpeg_compress(augmented_img)
            else:
                augmented_img = multi_webp_compress(augmented_img)

        if random.random() < 0.1: # 10% chance of greyscale
            augmented_img = convert_image(augmented_img, mode="L", as_rgb=True)

    return orig_img, augmented_img

####################################################################
# Augmentations Class
####################################################################

class Augmentations:
    def __init__(
        self,
        spatial=False, # random by default
        color=False,
        greyscale=False,
        blur=False,
        noise=False,
        denoise = False,
        jpeg=False,
        webp=False,
        mixed_compression=False, # jpeg + webp
        downsample=1, # downsample 2 for 2x upscale, 4 for 4x etc.
        preset="light"
    ):
        self.spatial = spatial                                    # rotate and flip
        self.color = color                                        # random color adjust 50% of images
        self.greyscale = greyscale                                # convert to greyscale
        self.blur = blur                                          # blur 50% of images
        self.noise = noise                                        # add noise to 50% of images
        self.denoise = denoise                                    # denoise to 50% of images
        self.jpeg = jpeg                                          # jpeg compress 50% of images
        self.webp = webp                                          # webp compress 50% of images
        self.mixed_compression = mixed_compression                # randomly compress (jpeg, webp) 50% of images
        self.downsample = downsample                              # downsample by factor n
        self.preset = preset


if __name__ == '__main__':
    augs = Augmentations(
        noise=True,
        blur=True,
        mixed_compression=True
    )
    input_path = "/home/xioren/Downloads/image.png"
    image = Image.open(input_path).convert("RGB")
    _, aug_image = augment_image(image, augs)
    aug_image.save("/home/xioren/Downloads/image-augmented.png")