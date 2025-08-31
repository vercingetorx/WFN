import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from typing import Iterable, Optional, Dict, Any, Tuple

from torch.nn.utils import clip_grad_norm_

####### LOSS #######


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, value_range=2):
        """
        SSIM module to calculate SSIM with flexible value ranges.
        :param window_size: Size of the Gaussian window.
        :param size_average: Whether to average the SSIM over the batch.
        :param value_range: The dynamic range of the image values (e.g., 2 for range -1 to 1, 1 for range 0 to 1).
        
        Notes:
            - needs to be on the same device as model
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.value_range = value_range
        self.channel = 1
        self.window = None

    @staticmethod
    def gaussian_window(size, sigma):
        """
        Generates a 1-D Gaussian window for computing local means and variances.
        """
        coords = torch.arange(size).float() - size // 2
        gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        """
        Create a 2D Gaussian window for SSIM calculation.
        """
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
        Calculate the SSIM index between two images.
        """
        mu1 = nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = (0.01 * self.value_range) ** 2 
        C2 = (0.03 * self.value_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window is not None:
            pass
        else:
            self.channel = channel
            self.window = self.create_window(self.window_size, self.channel).to(img1.device)

        return 1 - self.ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG-19 feature extractor.
    This loss compares high-level feature representations from a pre-trained VGG-19 model
    to evaluate perceptual similarity between generated and target images.

    Notes:
        - needs to be on the same device as model
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load VGG-19 with default weights and use first 20 layers as the feature extractor
        weights = models.VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights).features[:20].eval()
        self.vgg = vgg
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters for feature extraction only

    def forward(self, generated, target):
        """
        Calculates perceptual loss between generated and target images.
        :param generated: Generated image tensor
        :param target: Target image tensor
        :return: L1 loss between VGG feature representations of the generated and target images
        """
        # Compute L1 loss on feature representations
        loss = nn.functional.l1_loss(self.vgg(generated), self.vgg(target))
        return loss


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss to reduce high-frequency noise.
    This loss encourages spatial smoothness by penalizing pixel intensity differences
    between neighboring pixels.

    Notes:
        - does NOT need to be on the same device as model
    """
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        """
        Calculates Total Variation loss for an image tensor.
        :param img: Image tensor
        :return: TV loss, which is the mean of absolute differences between neighboring pixels
        """
        # Compute vertical and horizontal pixel intensity differences
        loss = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])) + \
               torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return loss


class PSNRLoss(nn.Module):
    def __init__(self, data_range=1.0):
        """
        PSNR Loss class.
        
        Args:
            data_range (float): The data range of the images (e.g. 1.0 for [0, 1] normalized images, 2.0 for [-1, 1]) or 255 for 8-bit images.
        
        Notes:
            - does NOT need to be on the same device as model
        """
        super(PSNRLoss, self).__init__()
        self.data_range = data_range
        self.epsilon = 1e-8

    def forward(self, pred, target):
        """
        Computes the PSNR loss between the predicted and target images.
        
        Args:
            pred (torch.Tensor): Predicted (generated) image of shape (batch_size, channels, height, width).
            target (torch.Tensor): Ground truth image of the same shape as pred.
        
        Returns:
            torch.Tensor: PSNR loss.
        """
        mse = nn.functional.mse_loss(pred, target, reduction='mean')
        psnr = 10 * torch.log10((self.data_range ** 2) / torch.sqrt(mse + self.epsilon))
        return -psnr  # Negative PSNR for minimization


class SpectralLoss(nn.Module):
    def __init__(self, loss_type='rmse'):
        """
        Spectral loss module.

        Args:
            loss_type: Type of loss to use in frequency domain ('l1', 'l2', or 'rmse').
        
        Notes:
            - needs to be on the same device as model
        """
        super(SpectralLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, generated_image, target_image):
        """
        Calculates the spectral loss between generated and target images.

        Args:
            generated_image: Generated image tensor (B, C, H, W).
            target_image: Target image tensor (B, C, H, W).

        Returns:
            The spectral loss value.
        """
        gen_fft = torch.fft.fft2(generated_image, dim=(-2, -1))
        target_fft = torch.fft.fft2(target_image, dim=(-2, -1))

        if self.loss_type == 'l1':
            loss = torch.mean(torch.abs(gen_fft - target_fft))
        elif self.loss_type == 'l2':
            loss = torch.mean(torch.pow(torch.abs(gen_fft - target_fft), 2))
        elif self.loss_type == 'rmse':
            loss = torch.sqrt(torch.mean(torch.pow(torch.abs(gen_fft - target_fft), 2)))
        else:
            raise ValueError("Invalid loss_type. Choose from 'l1', 'l2', or 'rmse'.")

        return loss


class MultiBandSpectralLoss(nn.Module):
    def __init__(self, loss_type='rmse', low_weight=1.0, mid_weight=1.0, high_weight=1.0):
        """
        Spectral loss module with multi-frequency analysis.

        Args:
            loss_type: Type of loss to use in frequency domain ('l1', 'l2', or 'rmse').
            low_weight: Weight for low-frequency band.
            mid_weight: Weight for mid-frequency band.
            high_weight: Weight for high-frequency band.
        
        Notes:
            - needs to be on the same device as model
        """
        super(MultiBandSpectralLoss, self).__init__()
        self.loss_type = loss_type
        self.low_weight = low_weight
        self.mid_weight = mid_weight
        self.high_weight = high_weight

    def band_loss(self, gen_fft, target_fft, band_mask):
        """
        Compute loss for a specific frequency band.

        Args:
            gen_fft: FFT of generated image.
            target_fft: FFT of target image.
            band_mask: Mask for the frequency band.

        Returns:
            Loss for the frequency band.
        """
        gen_band = gen_fft * band_mask
        target_band = target_fft * band_mask
        
        if self.loss_type == 'l1':
            loss = torch.mean(torch.abs(gen_band - target_band))
        elif self.loss_type == 'l2':
            loss = torch.mean(torch.pow(torch.abs(gen_band - target_band), 2))
        elif self.loss_type == 'rmse':
            loss = torch.sqrt(torch.mean(torch.pow(torch.abs(gen_band - target_band), 2)))
        else:
            raise ValueError("Invalid loss_type. Choose from 'l1', 'l2', or 'rmse'.")
        
        return loss

    def forward(self, generated_image, target_image):
        """
        Calculates the spectral loss between generated and target images.

        Args:
            generated_image: Generated image tensor (B, C, H, W).
            target_image: Target image tensor (B, C, H, W).

        Returns:
            The spectral loss value.
        """
        # FFT of generated and target images
        gen_fft = torch.fft.fftshift(torch.fft.fft2(generated_image, dim=(-2, -1)))
        target_fft = torch.fft.fftshift(torch.fft.fft2(target_image, dim=(-2, -1)))

        # Create frequency masks
        B, C, H, W = gen_fft.size()
        freq_x, freq_y = torch.meshgrid(torch.linspace(-0.5, 0.5, H, device=gen_fft.device),
                                        torch.linspace(-0.5, 0.5, W, device=gen_fft.device),
                                        indexing="ij")
        freq_radius = torch.sqrt(freq_x**2 + freq_y**2)

        low_mask = (freq_radius <= 0.1).float()
        mid_mask = ((freq_radius > 0.1) & (freq_radius <= 0.3)).float()
        high_mask = (freq_radius > 0.3).float()

        # Compute losses for each band
        low_loss = self.band_loss(gen_fft, target_fft, low_mask)
        mid_loss = self.band_loss(gen_fft, target_fft, mid_mask)
        high_loss = self.band_loss(gen_fft, target_fft, high_mask)

        # Combine losses with weights
        total_loss = (self.low_weight * low_loss +
                      self.mid_weight * mid_loss +
                      self.high_weight * high_loss)

        return total_loss


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, generated_patches, target_patches):
        """
        Computes the gradient loss between generated and target patches.

        Args:
            generated_patches: Tensor of shape (N, C, H, W) representing generated patches.
            target_patches: Tensor of shape (N, C, H, W) representing target patches.

        Returns:
            loss: Scalar tensor representing the gradient loss.
        
        Notes:
            - does NOT need to be on the same device as model
        """
        # Compute gradients of generated patches
        gen_dx = generated_patches[:, :, :, 1:] - generated_patches[:, :, :, :-1]
        gen_dy = generated_patches[:, :, 1:, :] - generated_patches[:, :, :-1, :]

        # Compute gradients of target patches
        tgt_dx = target_patches[:, :, :, 1:] - target_patches[:, :, :, :-1]
        tgt_dy = target_patches[:, :, 1:, :] - target_patches[:, :, :-1, :]

        # Compute L1 loss between gradients
        loss_dx = F.l1_loss(gen_dx, tgt_dx)
        loss_dy = F.l1_loss(gen_dy, tgt_dy)

        # Total gradient loss
        gradient_loss = (loss_dx + loss_dy)

        return gradient_loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (a smooth approximation of L1 loss) for image reconstruction tasks.
    
    This loss is defined as:
        L(x, y) = sqrt((x - y)^2 + epsilon^2)
    It is robust to outliers and helps preserve finer details in image restoration tasks.

    Args:
        epsilon (float): A small constant added for numerical stability. Default is 1e-6.
        reduction (str): Specifies the reduction to apply to the output:
                         'mean' (default) | 'sum' | 'none'.
    
    Usage:
        criterion = CharbonnierLoss(epsilon=1e-6, reduction='mean')
        loss = criterion(generated_patches, target_patches)
    """
    def __init__(self, epsilon=1e-6, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon  # Numerical stability constant
        self.reduction = reduction  # Reduction method: mean, sum, or none

    def forward(self, generated_patches, target_patches):
        """
        Computes the Charbonnier Loss between the generated patches and target patches.

        Args:
            generated_patches (torch.Tensor): Predicted output from the model.
            target_patches (torch.Tensor): Ground truth target values.

        Returns:
            torch.Tensor: The computed Charbonnier Loss (scalar if reduced, tensor otherwise).
        
        Notes:
            - does NOT need to be on the same device as model
        """
        # Compute the difference between the predicted and ground truth tensors
        diff = generated_patches - target_patches
        
        # Compute the Charbonnier Loss element-wise
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        
        # Apply the specified reduction method
        if self.reduction == 'mean':
            return torch.mean(loss)  # Average the loss across all elements
        elif self.reduction == 'sum':
            return torch.sum(loss)  # Sum the loss across all elements
        else:
            return loss  # Return the element-wise loss without reduction


class HingeLossG(nn.Module):
    def __init__(self):
        super(HingeLossG, self).__init__()

    def forward(self, fake_output):
        return -torch.mean(fake_output)  # Maximize discriminator's output for fake images


class HingeLossD(nn.Module):
    def __init__(self):
        super(HingeLossD, self).__init__()

    def forward(self, real_output, fake_output):
        real_loss = torch.mean(F.relu(1.0 - real_output))  # Max(0, 1 - real_output)
        fake_loss = torch.mean(F.relu(1.0 + fake_output))  # Max(0, 1 + fake_output)
        return real_loss + fake_loss


class RelativisticLoss(nn.Module):
    def __init__(self, loss_type='RaLSGAN', gradient_penalty=False, gp_weight=10.0):
        super().__init__()
        assert loss_type in ('RaGAN','RaLSGAN','RaHingeGAN')
        self.loss_type = loss_type
        self.gradient_penalty = gradient_penalty
        self.gp_weight = gp_weight

    # --------- D PHASE ----------
    def d_loss(self, real_pred, fake_pred, discriminator=None, real_data=None, fake_data=None, d_vec=None):
        r_mean = real_pred.mean()
        f_mean = fake_pred.mean()

        if self.loss_type == 'RaLSGAN':
            loss = 0.5 * ((real_pred - f_mean - 1) ** 2).mean() + \
                   0.5 * ((fake_pred - r_mean + 1) ** 2).mean()
        elif self.loss_type == 'RaGAN':
            loss = F.softplus(-(real_pred - f_mean)).mean() + \
                   F.softplus( (fake_pred - r_mean)).mean()
        else:  # RaHingeGAN
            loss = F.relu(1.0 - (real_pred - f_mean)).mean() + \
                   F.relu(1.0 + (fake_pred - r_mean)).mean()

        if self.gradient_penalty:
            assert discriminator is not None and real_data is not None and fake_data is not None
            d_const = None if d_vec is None else d_vec.detach()
            gp = self.compute_gradient_penalty(discriminator, real_data, fake_data, d_const)
            loss = loss + self.gp_weight * gp
        return loss

    # --------- G PHASE ----------
    def g_loss(self, real_pred_const, fake_pred):
        # Real logits must be constants for G
        real_pred_const = real_pred_const.detach()
        r_mean = real_pred_const.mean()
        f_mean = fake_pred.mean()

        if self.loss_type == 'RaLSGAN':
            loss = 0.5 * ((real_pred_const - f_mean + 1) ** 2).mean() + \
                   0.5 * ((fake_pred - r_mean - 1) ** 2).mean()
        elif self.loss_type == 'RaGAN':
            loss = F.softplus( (real_pred_const - f_mean)).mean() + \
                   F.softplus(-(fake_pred - r_mean)).mean()
        else:  # RaHingeGAN
            loss = F.relu(1.0 + (real_pred_const - f_mean)).mean() + \
                   F.relu(1.0 - (fake_pred - r_mean)).mean()
        return loss

    # --------- GP helper ----------
    def compute_gradient_penalty(self, discriminator, real_data, fake_data, d_vec_const):
        b = real_data.size(0)
        alpha = torch.rand(b, 1, 1, 1, device=real_data.device)
        interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        pred = discriminator(interpolated, d_vec_const)
        grad = torch.autograd.grad(
            outputs=pred, inputs=interpolated,
            grad_outputs=torch.ones_like(pred),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad = grad.view(b, -1)
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()



####### MISC #######


def r1_regularization(real_images, real_logits):
    gradients = torch.autograd.grad(
        outputs=real_logits.sum(), inputs=real_images, create_graph=True
    )[0]
    penalty = gradients.view(gradients.size(0), -1).pow(2).sum(1).mean()
    return penalty


class LambdaRamp:
    def __init__(self, ramp_epochs, start_weight, end_weight, start_epoch=0, mode="linear"):
        """
        Allows ramping of loss lambdas in either direction (up or down).

        Args:
            ramp_epochs (int): Number of epochs over which to ramp.
            start_weight (float): Starting weight value.
            end_weight (float): Ending weight value.
            start_epoch (int): The starting epoch for ramping.
            mode (str): "linear", "quadratic", or "exponential" ramping mode.
        """
        self.ramp_epochs = ramp_epochs
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.start_epoch = start_epoch
        self.mode = mode
        self.current_epoch = start_epoch
        self.current_weight = self._calculate_weight()

    def __str__(self):
        return f"{self.current_weight:.5f}"
    
    def __float__(self):
        return float(self.current_weight)
    
    def __add__(self, n):
        return self.current_weight + n
    
    def __sub__(self, n):
        return self.current_weight - n
    
    def __mul__(self, n):
        return self.current_weight * n
    
    def __truediv__(self, n):
        return self.current_weight / n
    
    def _calculate_weight(self):
        """
        Calculate the current weight based on the epoch and ramping mode.
        """
        if self.ramp_epochs == 1:
            return self.end_weight
        if self.current_epoch >= self.ramp_epochs:
            return self.end_weight

        progress = self.current_epoch / (self.ramp_epochs - 1)
        if self.mode == "linear":
            weight = self.start_weight + (self.end_weight - self.start_weight) * progress
        elif self.mode == "quadratic":
            weight = self.start_weight + (self.end_weight - self.start_weight) * (progress ** 2)
        elif self.mode == "exponential":
            weight = self.start_weight * (self.end_weight / self.start_weight) ** progress
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'linear', 'quadratic', or 'exponential'.")
        
        return weight

    def step(self):
        """
        Updates the weight based on the current epoch.
        """
        if self.current_epoch < self.ramp_epochs - 1:
            self.current_epoch += 1
            self.current_weight = self._calculate_weight()
        
        return self
            

class ApolloM(torch.optim.Optimizer):
    """
    ApolloM: Apollo with Momentum.

    This version integrates a standard momentum component (Adam's `beta1`) to
    stabilize the update *direction*, while retaining Apollo's novel `rho`
    controller to dynamically set the update *magnitude*.

    This is a direct test of the hypothesis that Apollo's primary missing
    ingredient was directional smoothing.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        beta1: float = 0.9,         ### NEW ###: The momentum factor
        E_max: float = 0.02,
        E_min: float = 1e-6,
        gamma: float = 2.0,
        beta2: float = 0.99,
        soft_clip: float = 1.5,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        eps: float = 1e-9,
    ):
        if not 0.0 <= beta1 < 1.0:  ### NEW ###
            raise ValueError("beta1 must be in [0,1)")
        if not (0.0 < E_min <= E_max):
            raise ValueError("Need 0 < E_min ≤ E_max")
        if not 0.0 < beta2 < 1.0:
            raise ValueError("beta2 must be in (0,1)")
            
        defaults = dict(
            beta1=beta1,            ### NEW ###
            E_max=E_max, E_min=E_min, gamma=gamma,
            beta2=beta2, clip=soft_clip,
            wd=weight_decay, gclip=max_grad_norm, eps=eps,
        )
        super().__init__(params, defaults)

        # Initialize state for all parameters
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                st["step"] = 0
                st["prev_g"] = torch.zeros_like(p)
                st["diag_cov"] = torch.ones_like(p)
                st["momentum_buffer"] = torch.zeros_like(p) ### NEW ###

    @staticmethod
    def _hilbert_imag_vec(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        N = x.shape[dim]
        if N < 2: return torch.zeros_like(x)
        Xf = torch.fft.rfft(x, dim=dim)
        h = torch.zeros_like(Xf)
        if N % 2 == 0: h.narrow(dim, 1, N // 2 - 1).fill_(2.0)
        else: h.narrow(dim, 1, (N - 1) // 2).fill_(2.0)
        Xf.mul_(h)
        return torch.fft.irfft(Xf, n=N, dim=dim)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["gclip"] > 0:
                clip_grad_norm_(group["params"], group["gclip"])

            beta1 = group["beta1"] ### NEW ###
            E_max, E_min, gamma = group["E_max"], group["E_min"], group["gamma"]
            beta2, clip_val, wd, eps = group["beta2"], group["clip"], group["wd"], group["eps"]

            for p in group["params"]:
                if p.grad is None: continue
                g_raw = p.grad
                if g_raw.is_sparse: raise RuntimeError("ApolloM does not support sparse grads")

                st = self.state[p]
                st["step"] += 1

                if wd > 0: p.mul_(1.0 - wd)

                # --- 1. Pre-conditioning (Same as Apollo) ---
                diag_cov = st["diag_cov"]
                if st["step"] == 1:
                    diag_cov.copy_(g_raw.pow(2))
                else:
                    diag_cov.mul_(beta2).addcmul_(g_raw, g_raw, value=1 - beta2)
                denom = diag_cov.sqrt().add_(eps)
                pg = g_raw / denom

                # --- 2. Momentum for Directional Smoothing (THE NEW PART) ---
                m = st["momentum_buffer"]
                m.mul_(beta1).add_(pg, alpha=1 - beta1)
                
                # Bias correction for momentum (important in early steps)
                bias_correction1 = 1.0 - beta1 ** st["step"]
                m_hat = m / bias_correction1
                
                # --- 3. Self-Scheduling LR (Same as Apollo) ---
                prev_g = st["prev_g"]
                if p.dim() < 1 or prev_g.abs().sum() == 0:
                    rho = torch.tensor(0.0, device=p.device, dtype=p.dtype)
                else:
                    if p.dim() == 1:
                        flat_now, flat_prev = g_raw.unsqueeze(0), prev_g.unsqueeze(0)
                    else:
                        flat_now, flat_prev = g_raw.flatten(1), prev_g.flatten(1)

                    h_now = self._hilbert_imag_vec(flat_now, dim=1)
                    h_prev = self._hilbert_imag_vec(flat_prev, dim=1)
                    num = torch.abs((flat_now * h_prev - h_now * flat_prev).sum(dim=1))
                    den = flat_now.norm(dim=1) * flat_prev.norm(dim=1) + eps
                    rho = (num / den).mean().clamp_(0.0, 1.0)

                E = E_min + (E_max - E_min) * (1.0 - rho).pow(gamma)

                # --- 4. The Final Update Rule (MODIFIED) ---
                # The direction is now from the momentum buffer `m_hat`.
                # The magnitude is still explicitly normalized to be exactly `E`.
                direction_vec = m_hat ### MODIFIED ###
                dir_norm = direction_vec.norm()
                step = direction_vec.mul(-E / (dir_norm + eps))

                # The soft clip is a final safety rail
                if clip_val > 0:
                    step_norm = step.norm()
                    scale = clip_val / (step_norm + clip_val + eps)
                    if scale < 1.0:
                        step.mul_(scale)

                p.add_(step)
                st["prev_g"].copy_(g_raw)

        return loss
