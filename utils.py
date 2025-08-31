import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from typing import Tuple

from torch.utils.data import Dataset
from PIL import Image, ImageOps, UnidentifiedImageError

from augment import Augmentations, augment_image, downsample_img_with_random_mode



Image.MAX_IMAGE_PIXELS = None # YOLO
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


###### MODEL UTILS ######

class ImageDataset(Dataset):
    def __init__(self, *dirs, augments=None, crop_size=(0, 0)):
        self.dirs = dirs
        self.image_paths = self._load_image_paths()
        self.augmentations = augments
        self.crop_size = crop_size

    def _load_image_paths(self):
        image_paths = []
        for real_dir in self.dirs:
            for real_file in os.listdir(real_dir):
                real_path = os.path.join(real_dir, real_file)
                image_paths.append(real_path)
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        real_path = self.image_paths[idx]
        return load_and_augment_image(real_path, augmentations=self.augmentations, crop_size=self.crop_size)


def save_model(epoch, *args, path="checkpoint.pth", **kwargs):
    state = {
        "epoch": epoch
    }
    for key, value in kwargs.items():
        if value is not None:
            state[f"{key}_state_dict"] = value.state_dict()
    torch.save(state, path)


def load_model(path, *args, **kwargs):
    checkpoint = torch.load(path, weights_only=True)
    for key, value in kwargs.items():
        if value is not None:
            value.load_state_dict(checkpoint[f"{key}_state_dict"])
    return checkpoint["epoch"]


def count_parameters(model: nn.Module):
    """
    Calculate the total number of parameters and trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        dict: A dictionary with 'total' and 'trainable' parameter counts.
    """
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    return {"total": total_params, "trainable": trainable_params}


def export_to_onnx(model, input_size=(3, 64, 64), onnx_file_path="models/model.onnx"):
    """
    Exports a PyTorch model to ONNX format.

    Args:
        model: The PyTorch model to export.
        input_size: A tuple representing the input size.
        onnx_file_path: The path to save the ONNX model.
    """
    import torch.onnx

    # VERY IMPORTANT: Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor.  The shape should match your model's input.
    dummy_input = torch.randn(1, *input_size)  # Add a batch dimension (1)

    # Export the model
    torch.onnx.export(
        model,                         # The model to export
        dummy_input,                   # A dummy input for tracing
        onnx_file_path,                # The output file path
        export_params=True,            # Store the trained parameter weights
        opset_version=20,              # The ONNX operator set version
        do_constant_folding=True,      # Optimize constant folding for performance
        input_names = ['input'],       # Names for the input tensors (can be multiple)
        output_names = ['output'],     # Names for the output tensors (can be multiple)
        dynamic_axes={'input' : {0 : 'batch_size'},    # Handle dynamic batch size (optional, but often needed)
                      'output' : {0 : 'batch_size'}}
    )

    print(f"Model exported to {onnx_file_path}")

###### IMAGE UTILS ######

def extract_tensor_patches(tensor, patch_size=128, overlap=16):
    """
    Extracts overlapping patches using PyTorch's unfold method.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (int): The size of each patch.
        overlap (int): The overlap between patches.

    Returns:
        torch.Tensor: Patches of shape (B, num_patches, C, patch_size, patch_size).
    """
    B, C, H, W = tensor.shape
    stride = patch_size - overlap

    # Pad the tensor if necessary
    pad_height = (stride - (H - patch_size) % stride) % stride
    pad_width = (stride - (W - patch_size) % stride) % stride
    tensor = F.pad(tensor, (0, pad_width, 0, pad_height), mode="constant", value=0)

    # Use unfold to extract patches
    patches = tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)  # Shape: (B, num_patches, C, patch_size, patch_size)

    return patches


def recompile_tensor_patches(patches, original_shape, patch_size=128, overlap=16):
    """
    Reconstructs the tensor from overlapping patches using PyTorch's fold method.

    Args:
        patches (torch.Tensor): Patches of shape (B, num_patches, C, patch_size, patch_size).
        original_shape (tuple): Original shape of the tensor (B, C, H, W).
        patch_size (int): The size of each patch.
        overlap (int): The overlap between patches.

    Returns:
        torch.Tensor: Reconstructed tensor of shape (B, C, H, W).
    """
    B, num_patches, C, _, _ = patches.shape
    _, _, H, W = original_shape
    stride = patch_size - overlap

    # Calculate padded dimensions
    pad_height = (stride - (H - patch_size) % stride) % stride
    pad_width = (stride - (W - patch_size) % stride) % stride
    H_padded = H + pad_height
    W_padded = W + pad_width

    # Prepare for fold
    output_size = (H_padded, W_padded)
    patches = patches.permute(0, 2, 3, 4, 1).contiguous()  # (B, C, patch_size, patch_size, num_patches)
    patches = patches.view(B, C * patch_size * patch_size, num_patches)

    # Create weight matrix for averaging
    weight = torch.ones_like(patches)

    # Use fold to reconstruct
    reconstructed = F.fold(
        patches, output_size=output_size, kernel_size=patch_size, stride=stride
    )
    weight = F.fold(
        weight, output_size=output_size, kernel_size=patch_size, stride=stride
    )

    # Avoid division by zero
    weight[weight == 0] = 1

    # Normalize
    reconstructed = reconstructed / weight

    # Remove padding
    reconstructed = reconstructed[:, :, :H, :W]

    return reconstructed


def extract_image_patches(image, patch_size=128, overlap=16):
    """
    Extracts overlapping patches from a PIL image.

    Args:
        image (PIL.Image.Image): Input image.
        patch_size (int): The size of each patch.
        overlap (int): The overlap between patches.

    Returns:
        list of PIL.Image.Image: List of patches as PIL images.
        tuple: Shape of the original image.
    """
    # Convert the image to a NumPy array
    image_np = np.array(image)
    original_shape = image_np.shape
    H, W, C = original_shape
    stride = patch_size - overlap

    # Pad the array if necessary
    pad_height = (stride - (H - patch_size) % stride) % stride
    pad_width = (stride - (W - patch_size) % stride) % stride
    image_np = np.pad(image_np, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant", constant_values=0)
    H_padded, W_padded, _ = image_np.shape

    # Extract patches
    patches = []
    for i in range(0, H_padded - patch_size + 1, stride):
        for j in range(0, W_padded - patch_size + 1, stride):
            patch = image_np[i:i + patch_size, j:j + patch_size]
            patches.append(Image.fromarray(patch))

    return patches, original_shape


def recompile_image_patches(patches, original_shape, patch_size=128, overlap=16):
    """
    Reconstructs the original image from overlapping patches.

    Args:
        patches (list of PIL.Image.Image): List of patches as PIL images.
        original_shape (tuple): Original shape of the image (H, W, C).
        patch_size (int): The size of each patch.
        overlap (int): The overlap between patches.

    Returns:
        PIL.Image.Image: Reconstructed image as a PIL image.
    """
    H, W, C = original_shape
    stride = patch_size - overlap

    # Calculate padded dimensions
    pad_height = (stride - (H - patch_size) % stride) % stride
    pad_width = (stride - (W - patch_size) % stride) % stride
    H_padded = H + pad_height
    W_padded = W + pad_width

    # Initialize an array for reconstruction
    reconstructed = np.zeros((H_padded, W_padded, C), dtype=np.float32)
    weight = np.zeros((H_padded, W_padded, C), dtype=np.float32)

    # Reassemble patches into the full image
    patch_idx = 0
    for i in range(0, H_padded - patch_size + 1, stride):
        for j in range(0, W_padded - patch_size + 1, stride):
            patch = np.array(patches[patch_idx])
            reconstructed[i:i + patch_size, j:j + patch_size] += patch
            weight[i:i + patch_size, j:j + patch_size] += 1
            patch_idx += 1

    # Normalize to avoid overlapping patch artifacts
    reconstructed = reconstructed / np.maximum(weight, 1)

    # Remove padding and convert back to a PIL image
    reconstructed = reconstructed[:H, :W]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)  # Ensure valid pixel range
    return Image.fromarray(reconstructed)


def extract_patches_from_dir(root, patch_size=128, overlap=16, every=1, mode="RGB"):
    """Extract patches from source images in a directory to be used as HR training images.

    Args:
        root (str): Directory containing images.
        patch_size (int): Size of each patch.
        overlap (int): Overlap between patches.
        every (int): Save every nth patch.
    """
    output_dir = os.path.join(root, f"patches_every{every}")
    os.makedirs(output_dir, exist_ok=True)

    for img_idx, filename in enumerate(sorted(os.listdir(root))):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            file_path = os.path.join(root, filename)
            try:
                with Image.open(file_path) as img:
                    original_image = img.convert(mode)
            except (OSError, UnidentifiedImageError) as e:
                print(f"{e}: {filename}")
                continue
            
            # Extract all patches and original shape for potential reference
            patches, original_shape = extract_image_patches(original_image, patch_size=patch_size, overlap=overlap)

            # Save every nth patch
            for patch_idx, patch in enumerate(patches):
                if patch_idx % every == 0:  # Keep every nth patch based on `every` parameter
                    clean_patch =  Image.new(mode, patch.size)
                    clean_patch.putdata(list(patch.getdata()))
                    patch_path = os.path.join(output_dir, f"{str(img_idx).zfill(4)}-{str(patch_idx).zfill(4)}.png")
                    clean_patch.save(patch_path)


def pad_to_min_size(
    image: Image.Image,
    min_size: Tuple[int, int],
    mode: str = "constant",                 # "constant" is portable; see note
    fill: int | Tuple[int, int, int] = 0,
    center: bool = True
) -> Image.Image:
    """
    Pad *image* so that width ≥ min_size[0] and height ≥ min_size[1].

    If *center* is True (default) the extra pixels are split as evenly
    as possible left/right and top/bottom, keeping the original content
    centred. Otherwise all padding is added to the right and bottom.
    """
    tgt_w, tgt_h = min_size
    img_w, img_h = image.size

    # nothing to do ----------------------------------------------------------
    if img_w >= tgt_w and img_h >= tgt_h:
        return image

    # pixels still missing in each dimension ---------------------------------
    pad_w = max(tgt_w - img_w, 0)
    pad_h = max(tgt_h - img_h, 0)

    if center:
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left      # ensures sum == pad_w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
    else:
        pad_left = pad_top = 0
        pad_right, pad_bottom = pad_w, pad_h

    border = (pad_left, pad_top, pad_right, pad_bottom)

    # Pillow’s ImageOps.expand natively supports constant-colour padding.
    # For “edge” or “reflect”, use torchvision or implement manually.
    if mode != "constant":
        raise ValueError("Only mode='constant' is supported in this minimal helper.")
    return ImageOps.expand(image, border=border, fill=fill)


def get_random_crop(
    image: Image.Image,
    crop_size: Tuple[int, int],
    pad_mode: str = "constant",
    pad_fill: int | Tuple[int, int, int] = 0
) -> Image.Image:
    """
    Return a random crop of exact `crop_size`.
    If `image` is too small, it is padded first via `pad_to_min_size`.

    Parameters
    ----------
    image : PIL.Image
    crop_size : (width, height) of the desired output
    pad_mode, pad_fill : passed straight through to `pad_to_min_size`

    Returns
    -------
    PIL.Image of shape `crop_size`.
    """
    crop_w, crop_h = crop_size
    # Ensure the image is big enough
    image = pad_to_min_size(
        image,
        crop_size,
        mode=pad_mode,
        fill=pad_fill,
        center=True,
    )

    img_w, img_h = image.size
    # Sample a random top-left corner
    left = random.randint(0, img_w - crop_w)
    top = random.randint(0, img_h - crop_h)

    return image.crop((left, top, left + crop_w, top + crop_h))


def get_random_crop_pair(
    image_A: Image.Image,
    image_B: Image.Image,
    crop_size: Tuple[int, int],
    pad_mode: str = "constant",
    pad_fill: int | Tuple[int, int, int] = 0
) -> Image.Image:
    """
    Return a random crop of exact `crop_size`.
    If `image` is too small, it is padded first via `pad_to_min_size`.

    Parameters
    ----------
    image : PIL.Image
    crop_size : (width, height) of the desired output
    pad_mode, pad_fill : passed straight through to `pad_to_min_size`

    Returns
    -------
    PIL.Image of shape `crop_size`.
    """
    crop_w, crop_h = crop_size
    # Ensure the images are big enough
    image_A = pad_to_min_size(
        image_A,
        crop_size,
        mode=pad_mode,
        fill=pad_fill,
        center=True,
    )

    image_B = pad_to_min_size(
        image_B,
        crop_size,
        mode=pad_mode,
        fill=pad_fill,
        center=True,
    )

    img_w, img_h = image_A.size
    # Sample a random top-left corner
    left = random.randint(0, img_w - crop_w)
    top = random.randint(0, img_h - crop_h)

    return image_A.crop((left, top, left + crop_w, top + crop_h)), image_B.crop((left, top, left + crop_w, top + crop_h))


def select_norm(mode):
    if mode == "RGB":
        n = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif mode == "L":
        n = transforms.Normalize([0.5], [0.5])
    else:
        n = transforms.Lambda(lambda x: x)
    return n


def unnormalize(normalized_tensor):
    # NOTE: unnormalized_tensor * 255  # scale from [0, 1] to [0, 255]
    return (normalized_tensor + 1) / 2 # scale from -1, 1 to 0, 1


def tensor_to_pil(tensor, normalize=True):
    if normalize:
        return TF.to_pil_image(unnormalize(tensor))
    else:
        return TF.to_pil_image(tensor)


def spatial_transformation():
    """
    Returns a set of deterministic augmentations for a given call.
    The augmentations will be applied consistently to paired images.
    """

    # Randomly determine augmentation parameters
    do_hflip = random.random() > 0.5  # Random horizontal flip
    do_vflip = random.random() > 0.5  # Random vertical flip
    rotation_angle = random.choice([0, 90, 180, 270])  # Random rotation

    # Build the transformation
    transform_list = []
    if do_hflip:
        transform_list.append(transforms.RandomHorizontalFlip(p=1.0))  # Always flip if chosen
    if do_vflip:
        transform_list.append(transforms.RandomVerticalFlip(p=1.0))    # Always flip if chosen
    transform_list.append(transforms.RandomRotation(degrees=(rotation_angle, rotation_angle)))  # Deterministic angle

    return transforms.Compose(transform_list)


def pil_to_tensor(img, mode="RGB", normalize=True, transform=None):
    """
    Applies the provided transform, then converts an image to a tensor.
    """
    transform_list = []

    # Apply shared transformations if provided
    if transform:
        transform_list.append(transform)

    # Convert to Tensor
    transform_list.append(transforms.ToTensor())

    # Optional normalization
    if normalize:
        transform_list.append(select_norm(mode))

    # Compose the transformations
    composed_transform = transforms.Compose(transform_list)

    return composed_transform(img)


def pil_to_tensor_with_augments(source_img, mode="RGB", normalize=True, augmentations=None, crop_size=(0,0)):
    """
    Applies the provided transform, then converts an image to a tensor.
    """
    transform_list = []

    # Get random crop
    if crop_size != (0,0):
        source_img = get_random_crop(source_img, crop_size=crop_size)
    
    # Apply augmentations (which now include downsampling)
    if augmentations:
        source_img, augmented_img = augment_image(source_img, augmentations)
    
    
    # Apply shared transformations if provided
    if augmentations and augmentations.spatial:
        transform_list.append(spatial_transformation())

    # Convert to Tensor
    transform_list.append(transforms.ToTensor())

    # Optional normalization
    if normalize:
        transform_list.append(select_norm(mode))

    # Compose the transformations
    composed_transform = transforms.Compose(transform_list)

    return composed_transform(source_img), composed_transform(augmented_img)


def load_and_augment_image(image_path, mode="RGB", normalize=True, augmentations=None, crop_size=(0,0)):
    """Loads and augments the input image along with an augmented and degraded copy"""
    try:
        with Image.open(image_path) as image:
            return pil_to_tensor_with_augments(image.convert(mode), mode=mode, normalize=normalize, augmentations=augmentations, crop_size=crop_size)
    except (OSError, UnidentifiedImageError):
        return None


def load_image(image_path, mode="RGB", normalize=True):
    """Loads and augments the input image along with an augmented and degraded copy"""
    try:
        with Image.open(image_path) as image:
            return pil_to_tensor(image.convert(mode), mode=mode, normalize=True)
    except (OSError, UnidentifiedImageError):
        return None


def save_image_pairs(gen_imgs, real_imgs, filename, num_pairs=8, nrow=4, normalize=True):
    """
    Saves a grid of paired generated and real images for visual comparison.

    This function creates a side-by-side grid of generated images and their corresponding real images,
    allowing for easy comparison. Useful for visualizing training progress in tasks like super-resolution,
    image generation, or denoising.

    Args:
        gen_imgs (torch.Tensor): A tensor of generated images with shape (N, C, H, W).
        real_imgs (torch.Tensor): A tensor of real images with shape (N, C, H, W).
        filename (str): Path to save the output image grid.
        num_pairs (int): The number of image pairs to display. Defaults to 8.
        nrow (int): The number of rows in the grid. Defaults to 4.
        normalize (bool): If True, normalizes the tensor images for display. Defaults to True.
    
    Returns:
        None: The composed grid image is saved to the specified filename.
    """
    total_pairs = min(num_pairs, gen_imgs.size(0), real_imgs.size(0))
    indices = random.sample(range(gen_imgs.size(0)), total_pairs)

    images = []
    for idx in indices:
        # Normalize or unnormalize as needed
        gen_img = tensor_to_pil(gen_imgs[idx].cpu(), normalize=normalize)
        real_img = tensor_to_pil(real_imgs[idx].cpu(), normalize=normalize)
        images.append((gen_img, real_img))

    # Calculate dimensions for the grid
    img_width, img_height = images[0][0].size
    ncol = (total_pairs + nrow - 1) // nrow  # Calculate columns needed based on the number of rows
    total_width = img_width * 2 * ncol  # Two images (gen and real) side by side per column
    total_height = img_height * nrow

    # Create a new blank image for the grid
    new_image = Image.new("RGB", (total_width, total_height))

    # Place images in the grid
    for idx, (gen_img, real_img) in enumerate(images):
        x_offset = (idx % ncol) * img_width * 2  # Calculate the horizontal offset for the current image
        y_offset = (idx // ncol) * img_height   # Calculate the vertical offset for the current image
        new_image.paste(gen_img, (x_offset, y_offset))
        new_image.paste(real_img, (x_offset + img_width, y_offset))

    # Save the final composed image (order: lr_img -> gen_img -> hr_img)
    new_image.save(filename)


def save_image_triplets(lr_imgs, gen_imgs, real_imgs, filename, num_groups=8, nrow=4, normalize=True):
    """
    Saves a grid of paired generated and real images for visual comparison.

    This function creates a side-by-side grid of generated images and their corresponding low res and real images,
    allowing for easy comparison. Useful for visualizing training progress in tasks like super-resolution,
    image generation, or denoising.

    Args:
        lr_imgs (torch.Tensor): A tensor of generated images with shape (N, C, H, W).
        gen_imgs (torch.Tensor): A tensor of generated images with shape (N, C, H, W).
        real_imgs (torch.Tensor): A tensor of real images with shape (N, C, H, W).
        filename (str): Path to save the output image grid.
        num_groups (int): The number of image groups to display. Defaults to 8.
        nrow (int): The number of rows in the grid. Defaults to 4.
        normalize (bool): If True, normalizes the tensor images for display. Defaults to True.
    
    Returns:
        None: The composed grid image is saved to the specified filename.
    """
    total_groups = min(num_groups, lr_imgs.size(0), gen_imgs.size(0), real_imgs.size(0))
    indices = random.sample(range(gen_imgs.size(0)), total_groups)
    _, img_width, img_height = real_imgs[-1].shape

    images = []
    for idx in indices:
        # Normalize or unnormalize as needed
        lr_img = tensor_to_pil(lr_imgs[idx].cpu(), normalize=normalize)
        lr_img_width, lr_img_height = lr_img.size
        if lr_img_width != img_width or lr_img_width != lr_img_height:
            lr_img = lr_img.resize((img_width, img_height), Image.Resampling.BICUBIC)
        gen_img = tensor_to_pil(gen_imgs[idx].cpu(), normalize=normalize)
        real_img = tensor_to_pil(real_imgs[idx].cpu(), normalize=normalize)
        images.append((lr_img, gen_img, real_img))

    # Calculate dimensions for the grid
    ncol = (total_groups + nrow - 1) // nrow  # Calculate columns needed based on the number of rows
    total_width = img_width * 3 * ncol  # Three images (lr, gen and real) side by side per column
    total_height = img_height * nrow

    # Create a new blank image for the grid
    new_image = Image.new("RGB", (total_width, total_height))

    # Place images in the grid
    for idx, (lr_img, gen_img, real_img) in enumerate(images):
        x_offset = (idx % ncol) * img_width * 3  # Calculate the horizontal offset for the current image
        y_offset = (idx // ncol) * img_height   # Calculate the vertical offset for the current image
        new_image.paste(lr_img, (x_offset, y_offset))
        new_image.paste(gen_img, (x_offset + img_width, y_offset))
        new_image.paste(real_img, (x_offset + img_width * 2, y_offset))

    # Save the final composed image
    new_image.save(filename)


def save_pil_image(image, image_path, qlt=100):
    _, ext = os.path.splitext(image_path)
    if ext in (".jpg", ".jpeg"):
        image.save(image_path, format="JPEG", quality=qlt)
    elif ext == ".png":
        image.save(image_path, format="PNG")
    else:
        image.save(image_path)


###### MISC UTILS ######


def format_time(seconds):
    """Convert seconds to a more readable format."""
    seconds = int(seconds)  # Ensure integer conversion for cleaner output
    if seconds < 60:
        return f"{seconds} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes} minutes, {remaining_seconds} seconds"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours} hours, {minutes} minutes"


def calculate_psnr(img1, img2, data_range=1.0):
    """Peak Signal-to-Noise Ratio"""
    # 1.0 for [0, 1], 2.0 for [-1, 1]
    assert img1.shape == img2.shape, "Images must have the same shape"
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match case
    
    epsilon = 1e-8
    psnr = 10 * torch.log10((data_range ** 2) / (mse + epsilon))
    return psnr.item()


if __name__ == "__main__":
    pass
