import io
import random
import numpy as np

from PIL import Image, ImageEnhance, ImageFilter
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
    params = sample_color_adjust_params()
    image = adjust_color_channels(image, **params)

    # vibrance (optional)
    if random.random() < 0.5:
        image = adjust_vibrance(image, vibrance=random.uniform(0.0, 0.5))

    # global brightness/contrast/color (optional)
    if random.random() < 0.5:
        image = adjust_color(
            image,
            brightness=random.uniform(0.7, 1.3),
            contrast  =random.uniform(0.8, 1.3),
            color     =random.uniform(0.7, 1.3),
        )
    return image

####################################################################
# Noise and Denoise
####################################################################

def apply_random_noise(image, preset="light"):
    noise_mode = random.randint(0, 2)
    if noise_mode == 0:   # Gaussian
        return apply_gaussian_noise(image, preset=preset)
    elif noise_mode == 1: # Poisson
        return apply_poisson_noise(image, preset=preset)
    else:                 # structured noises
        gradient_mode = random.randint(0, 3)
        if   gradient_mode == 0: return apply_perlin_noise(image, preset=preset)
        elif gradient_mode == 1: return apply_simplex_noise(image, preset=preset)
        elif gradient_mode == 2: return apply_value_noise(image, preset=preset)
        else:                    return apply_spread_noise(image)

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

def oversharpen(img):
    # USM: radius and percent tuned for visible halos
    r = random.uniform(1.0, 2.5)
    p = random.randint(100, 250)   # percent
    t = random.randint(0, 5)       # threshold
    return img.filter(ImageFilter.UnsharpMask(radius=r, percent=p, threshold=t))

def _compress_pil(img: Image.Image, fmt: str, **save_kwargs) -> Image.Image:
    """Save to memory and reopen (keeps your current behavior)."""
    with io.BytesIO() as buf:
        img.save(buf, format=fmt, **save_kwargs)
        buf.seek(0)
        with Image.open(buf) as comp:
            return comp.convert(img.mode)

def _with_block_offset(img: Image.Image, fmt: str, *, mcu=(8, 8), **save_kwargs) -> Image.Image:
    """
    Shift the encoder's DCT/macroblock grid by padding the image so that content
    starts at a random offset, compress, then crop back to original size.
    Padding uses edge replication to avoid seam artifacts.
    """
    w, h = img.size
    mx, my = mcu
    ox = random.randrange(mx)  # horizontal offset
    oy = random.randrange(my)  # vertical offset

    arr = np.array(img)
    if arr.ndim == 2:  # L / single channel
        arr = np.pad(arr, ((oy, my - oy), (ox, mx - ox)), mode="edge")
        padded = Image.fromarray(arr, mode=img.mode)
    else:              # RGB / RGBA
        arr = np.pad(arr, ((oy, my - oy), (ox, mx - ox), (0, 0)), mode="edge")
        padded = Image.fromarray(arr, mode=img.mode)

    comp = _compress_pil(padded, fmt, **save_kwargs)
    comp_arr = np.array(comp)

    if comp_arr.ndim == 2:
        crop = comp_arr[oy:oy + h, ox:ox + w]
    else:
        crop = comp_arr[oy:oy + h, ox:ox + w, :]

    return Image.fromarray(crop, mode=img.mode)


def jpeg_compress(img, qlt=100, subsampling_value=0, block_offset=False):
    save_kwargs = dict(quality=qlt, subsampling=subsampling_value)
    if block_offset:
        if subsampling_value == 0:
            mcu = (8, 8)
        elif subsampling_value == 1:
            mcu = (16, 8)
        else:  # 4:2:0
            mcu = (16, 16)
        return _with_block_offset(img, "JPEG", mcu=mcu, **save_kwargs)
    else:
        return _compress_pil(img, "JPEG", **save_kwargs)


def webp_compress(img, qlt=100, block_offset=False):
    # WebP uses 16×16 macroblocks internally
    save_kwargs = dict(quality=qlt)  # add method=4 if you like
    if block_offset:
        return _with_block_offset(img, "WEBP", mcu=(16, 16), **save_kwargs)
    else:
        return _compress_pil(img, "WEBP", **save_kwargs)


def dual_compress(img):
    qlt_j = random.randint(40, 99)
    qlt_w = random.randint(40, 99)
    if random.random() > 0.5:
        return webp_compress(jpeg_compress(img, qlt_j), qlt_w)
    else:
        return jpeg_compress(webp_compress(img, qlt_w), qlt_j)

def multi_jpeg_compress(img):
    count = random.randint(1, 3)
    for _ in range(count):
        qlt = random.randint(40, 99)
        subsampling_value = random.randint(0, 2)
        use_offset = random.random() < 0.75 # apply offset most of the time
        img = jpeg_compress(img, qlt=qlt, subsampling_value=subsampling_value, block_offset=use_offset)
        if random.random() < 0.2:
            img = resample(img)
    return img


def multi_webp_compress(img):
    count = random.randint(1, 3)
    for _ in range(count):
        qlt = random.randint(40, 99)
        use_offset = random.random() < 0.75
        img = webp_compress(img, qlt=qlt, block_offset=use_offset)
        if random.random() < 0.2:
            img = resample(img)
    return img


def convert_image(img, mode="RGB", as_rgb=False):
    img = img.convert(mode)
    if as_rgb:
        return img.convert("RGB")
    return img


def upsample_img(img, factor=2, mode=Image.Resampling.LANCZOS):
    width, height = img.size
    return img.resize((width * factor, height * factor), mode)


def downsample_img(img, factor=2, mode=Image.Resampling.LANCZOS):
    width, height = img.size
    return img.resize((width // factor, height // factor), mode)


def upsample_img_with_random_mode(img, factor=2):
    # List of downsampling methods
    upsampling_modes = [Image.Resampling.LANCZOS, Image.Resampling.BICUBIC, Image.Resampling.BILINEAR, Image.Resampling.NEAREST]
    
    # Randomly choose a downsampling method
    chosen_mode = random.choice(upsampling_modes)
    
    # Get current width and height
    width, height = img.size
    
    # Resize image with the chosen mode
    return img.resize((width * factor, height * factor), chosen_mode)


def downsample_img_with_random_mode(img, factor=2):
    # List of downsampling methods
    downsampling_modes = [Image.Resampling.LANCZOS, Image.Resampling.BICUBIC, Image.Resampling.BILINEAR, Image.Resampling.NEAREST]
    
    # Randomly choose a downsampling method
    chosen_mode = random.choice(downsampling_modes)
    
    # Get current width and height
    width, height = img.size
    
    # Resize image with the chosen mode
    return img.resize((width // factor, height // factor), chosen_mode)


def multi_step_downsample_img_with_random_mode(img, factor=2):
    # TODO: this downsamples twice by default and always uses a factor of 'factor'. Make this tunable.
    return downsample_img_with_random_mode(downsample_img_with_random_mode(img, factor), factor)


def resample(img, factor=2):
    # Always return to the original (w,h), even when not divisible by factor.
    w, h = img.size
    modes = [Image.Resampling.LANCZOS, Image.Resampling.BICUBIC,
             Image.Resampling.BILINEAR, Image.Resampling.NEAREST]
    down_w = max(1, round(w / factor))
    down_h = max(1, round(h / factor))
    down = img.resize((down_w, down_h), random.choice(modes))
    up   = down.resize((w, h),        random.choice(modes))
    return up

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

    # grescale
    if augmentations.greyscale:
        augmented_img = convert_image(augmented_img, mode="L", as_rgb=True)

    # 90% chance to apply the full degradation pipeline.
    # 10% chance to only apply downsampling (clean images).
    apply_degradations = random.random() < 0.9

    if apply_degradations:

        # --- Stage 1: Pre-Downsampling Degradations (Simulating "Original Source" Artifacts) ---
        if augmentations.blur and random.random() < 0.5: # 50% chance to apply pre-blur
            augmented_img = apply_random_blur(augmented_img, preset=augmentations.preset)

        if augmentations.noise and random.random() < 0.5: # 50% chance to apply pre-noise
            augmented_img = apply_poisson_noise(augmented_img, preset=augmentations.preset)
        
        # denoise
        if augmentations.denoise and random.random() < 0.3:
            augmented_img = apply_random_denoise(augmented_img)

        if random.random() < 0.7: # 70% chance of initial compression
            qlt = random.randint(40, 100)
            use_offset = random.random() < 0.75
            if random.random() < 0.5:
                augmented_img = jpeg_compress(augmented_img, qlt=qlt, block_offset=use_offset)
            else:
                augmented_img = webp_compress(augmented_img, qlt=qlt, block_offset=use_offset)
    
        if random.random() < 0.1:
            augmented_img = resample(augmented_img)

    # --- Stage 2: Downsampling (Applied to all images) ---
    if augmentations.downsample > 1:
        augmented_img = downsample_img_with_random_mode(augmented_img, augmentations.downsample)

    if apply_degradations:
        # --- Stage 3: Post-Downsampling Degradations (Simulating "In-the-Wild" Artifacts) ---
        if augmentations.blur and random.random() < 0.5: # 50% chance to apply post-blur
            augmented_img = apply_random_blur(augmented_img, preset=augmentations.preset)

        if augmentations.noise and random.random() < 0.5: # 50% chance to apply post-noise
            augmented_img = apply_random_noise(augmented_img, preset=augmentations.preset)

        if augmentations.denoise and random.random() < 0.3: # 30% chance of denoising
            augmented_img = apply_random_denoise(augmented_img, preset=augmentations.preset)

        if random.random() < 0.15:
            augmented_img = resample(augmented_img)

        if random.random() < 0.1:
            augmented_img = oversharpen(augmented_img)

        if augmentations.compress and random.random() < 0.8: # 80% chance of final compression
            if random.random() < 0.5:
                augmented_img = multi_jpeg_compress(augmented_img)
            else:
                augmented_img = multi_webp_compress(augmented_img)

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
        compress=False, # jpeg or webp
        mixed_compression=False, # jpeg and webp
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
        self.compress = compress
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
