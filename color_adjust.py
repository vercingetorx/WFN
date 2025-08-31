import numpy as np

from PIL import Image, ImageEnhance


def adjust_color(img, brightness=1.0, contrast=1.0, color=1.0, sharpness=1.0):
    """Adjusts brightness, contrast, color (saturation), and sharpness.
    Args:
        image_path: Path to the image.
        brightness: 1.0 is original, <1.0 darkens, >1.0 brightens.
        contrast: 1.0 is original, <1.0 reduces contrast, >1.0 increases contrast.
        color: 1.0 is original, 0.0 is grayscale, >1.0 increases saturation.
        sharpness: 1.0 is original, <1.0 blurs, >1.0 sharpens.

    Returns:
        A PIL Image object.
    
    NOTE:
        ranges: 0.0 - 2.0
    """
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)  # Saturation
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


#Vibrance example (requires conversion to HSV/HSL)
def adjust_vibrance(img, vibrance=0.5):
    img = img.convert("HSV")
    h, s, v = img.split()

    s_array = np.array(s, dtype=np.float32)
    s_array = s_array * (1 + vibrance)
    s_array = np.clip(s_array, 0, 255).astype(np.uint8)

    s = Image.fromarray(s_array)
    img = Image.merge("HSV", (h,s,v)).convert("RGB")
    return img


if __name__ == "__main__":
    #Example usage
    image_path = "/home/xioren/Downloads/image.png"
    with Image.open(image_path) as file:
        img = file.convert("RGB")

    adjust_color(img, brightness=1.2, contrast=1.1, color=1.3, sharpness=1.1).save("/home/xioren/Downloads/color_adjusted.png") #Vivid
    adjust_color(img, brightness=0.8, contrast=1.2, color=0.5, sharpness=1.0).save("/home/xioren/Downloads/color_adjusted_vintage.png") #Vintage Look

    adjust_vibrance(img, vibrance=0.5).save("/home/xioren/Downloads/vibrance_adjusted.png")
