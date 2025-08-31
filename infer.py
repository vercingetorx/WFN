from sys import argv

from utils import *
from generator import WaveFusionNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(
    in_channels=3,
    base_channels=64,
    embed_dim=128,
    num_ll_blocks=4,
    num_hf_indep_blocks=2,
    num_hf_fused_blocks=2,
    num_global_blocks=3,
    num_heads=4,
    mlp_ratio=2.0,
    upscale_factor=1,
    device="cpu",
    model_path="models/checkpoint.pth"
):
    # Load the trained generator model
    model = WaveFusionNet(
        in_channels=in_channels,
        base_channels=base_channels,
        embed_dim=embed_dim,
        num_ll_blocks=num_ll_blocks,
        num_hf_indep_blocks=num_hf_indep_blocks,
        num_hf_fused_blocks=num_hf_fused_blocks,
        num_global_blocks=num_global_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        upscale_factor=upscale_factor,
        device=device,
    )
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()
    model.to(device)
    return model


def inference(
    model,
    input_image_path,
    batch_size=128,
    patch_size=64,
    overlap=8
):
    upscale_factor = model.upscale_factor

    # Load and preprocess the input image
    image_tensor = load_image(input_image_path)
    if image_tensor is not None:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    else:
        print("error loading image:", input_image_path)
        return None
    original_shape = image_tensor.shape  # (B, C, H, W)
    B, C, H, W = original_shape

    # Extract patches from the image
    patches = extract_tensor_patches(
        image_tensor, patch_size=patch_size, overlap=overlap
    )  # Shape: (B, num_patches, C, patch_size, patch_size)
    B, num_patches, C, _, _ = patches.shape

    # Reshape patches to (num_patches_total, C, patch_size, patch_size) for processing
    patches = patches.view(-1, C, patch_size, patch_size)

    # Process patches through the model in batches
    processed_patches = []
    num_patches_total = patches.shape[0]
    for i in range(0, num_patches_total, batch_size):
        batch = patches[i : i + batch_size].to(device)
        with torch.no_grad():
            processed_batch, _ = model(batch)
        processed_patches.append(processed_batch.cpu())

    # Concatenate processed patches and reshape back to (B, num_patches, C, upscaled_patch_size, upscaled_patch_size)
    processed_patches = torch.cat(processed_patches, dim=0)
    upscaled_patch_size = patch_size * upscale_factor
    processed_patches = processed_patches.view(
        B, num_patches, C, upscaled_patch_size, upscaled_patch_size
    )

    # Reconstruct the upscaled image from processed patches
    upscaled_overlap = overlap * upscale_factor
    upscaled_shape = (
        B,
        C,
        H * upscale_factor,
        W * upscale_factor,
    )  # (B, C, H_upscaled, W_upscaled)

    reconstructed_image = recompile_tensor_patches(
        processed_patches,
        upscaled_shape,
        patch_size=upscaled_patch_size,
        overlap=upscaled_overlap,
    )  # Shape: (B, C, H_upscaled, W_upscaled)

    # Remove batch dimension and convert to PIL image
    output_image = tensor_to_pil(reconstructed_image.squeeze(0), normalize=True)

    # Create output path
    base, end = os.path.split(input_image_path)
    fn, ext = os.path.splitext(end)
    output_image_path = os.path.join(base, f"{fn}-{upscale_factor}x.png")

    # Save the output image
    save_pil_image(output_image, output_image_path)
    print(f"Saved generated image to {output_image_path}")


if __name__ == "__main__":
    if len(argv) > 1:
        path = argv[1]
    else:
        path = "/home/rory/Downloads/image2.jpeg"

    model = init_model(
        base_channels=128,
        embed_dim=256,
        num_ll_blocks=10,
        num_hf_indep_blocks=10,
        num_hf_fused_blocks=10,
        num_global_blocks=10,
        num_heads=32,
        upscale_factor=2,
        device=device
    )
    try:
        inference(model, path)
    except KeyboardInterrupt:
        pass