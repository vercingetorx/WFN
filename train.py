import torch.optim as optim

from loss import *
from utils import *
from generator import WaveFusionNet
from discriminator import WaveFusionDiscriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    dataloader,
    generator, discriminator,
    optimizer_G, optimizer_D,
    scheduler_G, scheduler_D,
    num_epochs=100,
    save_interval=25,
    sample_interval=5,
    accumulation_steps=1,
    checkpoint_path="models/checkpoint.pth"
):
    # loss functions
    criterion_GAN       = RelativisticLoss(loss_type="RaLSGAN", gradient_penalty=True)
    criterion_pixelwise = CharbonnierLoss()
    criterion_SSIM      = SSIM(window_size=11, size_average=True).to(device)
    criterion_spectral  = MultiBandSpectralLoss().to(device)
    criterion_gradient  = GradientLoss()

    # training parameters
    start_epoch     = 1
    lambda_pixel    = 1.0
    lambda_SSIM     = 0.20
    lambda_spectral = 0.10
    lambda_gradient = 0.05
    lambda_GAN      = LambdaRamp(
        ramp_epochs=25, start_epoch=10,
        start_weight=0.01, end_weight=0.10
    )

    # load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        start_epoch = load_model(
            checkpoint_path,
            generator=generator, discriminator=discriminator,
            optimizer_G=optimizer_G, optimizer_D=optimizer_D,
            scheduler_G=scheduler_G, scheduler_D=scheduler_D
        )
        print(f"Resuming from epoch {start_epoch}")
        start_epoch += 1

    generator.train()

    t_s = time.time()
    # training loop
    for epoch in range(start_epoch, num_epochs + 1):
        # Initialize epoch-level trackers
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_psnr = 0.0
        
        # Zero gradients once before starting the accumulation for the epoch
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        for i, (hr_imgs, lr_imgs) in enumerate(dataloader):
            hr_imgs = hr_imgs.to(device)
            lr_imgs = lr_imgs.to(device)

            # --------------------------------
            #  PHASE 1: Train Discriminator
            # --------------------------------
            
            # Freeze generator parameters to save computation
            for p in generator.parameters(): p.requires_grad = False
            for p in discriminator.parameters(): p.requires_grad = True

            # Use torch.no_grad() for generator's forward pass in D's training phase
            with torch.no_grad():
                gen_imgs, d_vec = generator(lr_imgs)

            real_output = discriminator(hr_imgs, d_vec.detach())
            fake_output = discriminator(gen_imgs.detach(), d_vec.detach())
            
            loss_D, _ = criterion_GAN(
                real_output, fake_output,
                discriminator,
                hr_imgs, gen_imgs.detach(), d_vec.detach()
            )
            loss_D = loss_D / accumulation_steps
            loss_D.backward() 
            epoch_loss_D += loss_D.item()

            # --------------------------------
            #  PHASE 2: Train Generator
            # --------------------------------

            # Freeze discriminator parameters so only generator's gradients are computed
            for p in generator.parameters(): p.requires_grad = True
            for p in discriminator.parameters(): p.requires_grad = False

            # Run a fresh forward pass for the generator to build the computation graph
            gen_imgs, d_vec = generator(lr_imgs)
            
            fake_for_G = discriminator(gen_imgs, d_vec)
            real_for_G = discriminator(hr_imgs, d_vec)

            _, loss_GAN = criterion_GAN(real_for_G, fake_for_G)
            loss_pixel = criterion_pixelwise(gen_imgs, hr_imgs)
            loss_SSIM = criterion_SSIM(gen_imgs, hr_imgs)
            loss_spectral = criterion_spectral(gen_imgs, hr_imgs)
            loss_gradient = criterion_gradient(gen_imgs, hr_imgs)

            loss_G = (
                lambda_GAN * loss_GAN +
                lambda_pixel * loss_pixel +
                lambda_SSIM * loss_SSIM +
                lambda_spectral * loss_spectral +
                lambda_gradient * loss_gradient
            )
            
            loss_G = loss_G / accumulation_steps
            loss_G.backward() 
            epoch_loss_G += loss_G.item()
            
            # --------------------------------
            #  PHASE 3: Update Weights
            # --------------------------------
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer_D.step()
                optimizer_G.step()
                
                optimizer_D.zero_grad()
                optimizer_G.zero_grad()

            # --- PSNR Calculation ---
            batch_psnr = 0.0
            for j in range(len(hr_imgs)):
                # Assuming images are in [-1, 1] or [0, 1] range
                batch_psnr += calculate_psnr(gen_imgs[j], hr_imgs[j], data_range=2.0)
            epoch_psnr += batch_psnr / len(hr_imgs)
        
        if scheduler_G: scheduler_G.step()
        if scheduler_D: scheduler_D.step()

        lambda_GAN.step()

        avg_loss_G = epoch_loss_G / len(dataloader) * accumulation_steps
        avg_loss_D = epoch_loss_D / len(dataloader) * accumulation_steps
        avg_psnr = epoch_psnr / len(dataloader)

        # save and sample checkpoints
        if epoch % sample_interval == 0:
            t_n = time.time()
            t_d = t_n - t_s
            t_s = t_n
            print(f"[Epoch {epoch}/{num_epochs}] [D loss: {round(avg_loss_D, 6)}] [G loss: {round(avg_loss_G, 6)}] [PSNR: {round(avg_psnr, 6)}] [Duration: {format_time(t_d)}]")
            save_image_triplets(
                lr_imgs, gen_imgs.detach(), hr_imgs,
                f"samples/epoch[{str(epoch).zfill(4)}]-d_loss[{round(avg_loss_D, 6)}]-g_loss[{round(avg_loss_G, 6)}]-psnr[{round(avg_psnr, 6)}].png"
            )

        if epoch % save_interval == 0:
            # rename last model
            if epoch > save_interval and os.path.exists(checkpoint_path):
                path, ext = os.path.splitext(checkpoint_path)
                os.rename(checkpoint_path, f"{path}-{epoch-save_interval}{ext}")
            save_model(
                epoch,
                path=checkpoint_path,
                generator=generator, discriminator=discriminator,
                optimizer_G=optimizer_G, optimizer_D=optimizer_D,
                scheduler_G=scheduler_G, scheduler_D=scheduler_D
            )


if __name__ == "__main__":
    # instantiate models
    generator = WaveFusionNet(
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
    # print(count_parameters(generator))
    discriminator = WaveFusionDiscriminator(
        base_channels=128,
        deg_dim=256,
        num_heads=32
    )

    # device configuration
    generator.to(device)
    discriminator.to(device)

    # instantiate data loader
    dataset = ImageDataset(
        "/media/Applications/data/train/div2k/source",
        "/media/Applications/data/train/next_png/source",
        "/media/Applications/data/train/ffhq_small/complete",
        augments=Augmentations(
            spatial=True,
            color=True,
            blur=True,
            noise=True,
            mixed_compression=True,
            downsample=2,
            preset="heavy"
        ),
        crop_size=(256, 256)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=12)

    # optimizers
    optimizer_G = optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=5e-5, betas=(0.9, 0.999))
    # optimizer_G = ApolloM(generator.parameters())
    # optimizer_D = ApolloM(discriminator.parameters())

    # schedulers
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=200, eta_min=1e-6)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=200, eta_min=1e-6)
    # scheduler_G = None
    # scheduler_D = None

    try:
        train(
            dataloader,
            generator, discriminator,
            optimizer_G, optimizer_D,
            scheduler_G, scheduler_D,
            num_epochs=200,
            save_interval=10,
            sample_interval=1,
            accumulation_steps=12
        )
    except KeyboardInterrupt:
        pass
