import random

import cv2
import numpy as np

from PIL import Image


def denoise_image(image, method='bilateral', bilateral_params=(9, 75, 75), nlmeans_params=(10, 7, 21), gaussian_sigma=0, median_size=0):
    """Denoises an image using various methods, including bilateral filtering,
    Non-Local Means denoising, Gaussian blur, and median filtering, with optional combinations.

    Args:
        image: PIL.Image.
        method: Denoising method ('bilateral', 'nlmeans', 'gaussian', 'median', 'combined_bilateral', 'combined_median').
        bilateral_params: Tuple (diameter, sigma_color, sigma_space) for bilateral filtering.
        nlmeans_params: Tuple (h, templateWindowSize, searchWindowSize) for NLMeans denoising.
        gaussian_sigma: Sigma for Gaussian blur (used alone or in combination).
        median_size: Size of the median filter (used alone or in combination).

    Returns:
        A PIL Image object with the denoising applied.
    """
    img = np.array(image)

    if method == 'bilateral':
        denoised_img = cv2.bilateralFilter(img, *bilateral_params)
    elif method == 'nlmeans':
        h, templateWindowSize, searchWindowSize = nlmeans_params
        if len(img.shape) == 3:
            denoised_img = cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, searchWindowSize)
        else:
            denoised_img = cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)
    elif method == 'gaussian':
        # OpenCV uses kernel size, not radius.  Convert sigma to kernel size.
        ksize = int(6 * gaussian_sigma + 1)  # Rule of thumb: ksize ≈ 6*sigma
        ksize = ksize if ksize % 2 != 0 else ksize + 1  # Ensure ksize is odd
        denoised_img = cv2.GaussianBlur(img, (ksize, ksize), gaussian_sigma)
    elif method == 'median':
        denoised_img = cv2.medianBlur(img, median_size)
    elif method == 'combined_bilateral':
        bilateral_filtered = cv2.bilateralFilter(img, *bilateral_params)
        if gaussian_sigma > 0:
          ksize = int(6 * gaussian_sigma + 1)
          ksize = ksize if ksize % 2 != 0 else ksize + 1
          denoised_img = cv2.GaussianBlur(bilateral_filtered, (ksize, ksize), gaussian_sigma)
        else:
          denoised_img = bilateral_filtered
    elif method == 'combined_median':
        median_filtered = cv2.medianBlur(img, median_size)
        if gaussian_sigma > 0:
            ksize = int(6*gaussian_sigma + 1)
            ksize = ksize if ksize%2 != 0 else ksize + 1
            denoised_img = cv2.GaussianBlur(median_filtered, (ksize, ksize), gaussian_sigma)
        else:
            denoised_img = median_filtered
    else:
        raise ValueError("Invalid method. Choose from 'bilateral', 'nlmeans', 'gaussian', 'median', 'combined_bilateral', 'combined_median'")

    return Image.fromarray(denoised_img, mode="RGB")


def sample_denoise_params(method, preset="light"):
    """
    Returns a dictionary of parameters for denoising.
    """
    params = {
        'method': method
    }
    if preset == "heavy":
        if method == "bilateral":
            diameter = random.choice([3, 5, 7, 9])
            sigma_color = random.uniform(10, 100)
            sigma_space = random.uniform(10, 100)
            params["bilateral_params"] = (diameter, sigma_color, sigma_space)
        elif method == "nlmeans":
            h = random.randint(3, 10)
            templateWindowSize = random.choice([3, 5, 7, 9])
            searchWindowSize = random.choice([7, 9, 11, 15, 17, 19, 21])
            params["nlmeans_params"] = (h, templateWindowSize, searchWindowSize)
        elif method == "median":
            ksize = random.choice([3, 5, 7, 9])
            params["median_size"] = ksize
    if preset == "light":
        if method == "bilateral":
            diameter = random.choice([3, 5, 7, 9])
            sigma_color = random.uniform(10, 50)
            sigma_space = random.uniform(10, 50)
            params["bilateral_params"] = (diameter, sigma_color, sigma_space)
        elif method == "nlmeans":
            h = random.randint(3, 7)
            templateWindowSize = random.choice([3, 5, 7, 9])
            searchWindowSize = random.choice([7, 9, 11, 15])
            params["nlmeans_params"] = (h, templateWindowSize, searchWindowSize)
        elif method == "median":
            ksize = random.choice([3, 5, 7, 9])
            params["median_size"] = ksize
    if preset == "test":
        if method == "bilateral":
            diameter = 9
            sigma_color = 50
            sigma_space = 50
            params["bilateral_params"] = (diameter, sigma_color, sigma_space)
        elif method == "nlmeans":
            h = 7
            templateWindowSize = 9
            searchWindowSize = 13
            params["nlmeans_params"] = (h, templateWindowSize, searchWindowSize)
        elif method == "median":
            ksize = 9
            params["median_size"] = ksize
    return params
