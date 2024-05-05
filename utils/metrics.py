from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io
import numpy as np

def calculate_metrics(original_image_path, denoised_image_path):
    """
    Calculate the SSIM and PSNR between an original image and a denoised image, both provided by their paths.

    Args:
        original_image_path (str): The path to the original image.
        denoised_image_path (str): The path to the denoised image.

    Returns:
        tuple: A tuple containing the SSIM and PSNR values.
    """
    # Load the original and denoised images from the provided paths
    original_image = io.imread(original_image_path, as_gray=True)
    denoised_image = io.imread(denoised_image_path, as_gray=True)

    # Ensure both images are of the same shape
    if original_image.shape != denoised_image.shape:
        raise ValueError("The original and denoised images must have the same dimensions.")

    # Convert images to floating-point type in the range [0, 1] for SSIM calculation
    original_image_float = original_image.astype(np.float64) / 255
    denoised_image_float = denoised_image.astype(np.float64) / 255

    # Calculate SSIM
    ssim_value = ssim(original_image_float, denoised_image_float, data_range=1)

    # Calculate PSNR
    psnr_value = psnr(original_image_float, denoised_image_float, data_range=1)

    # Print the metrics with explanations
    print(f"SSIM: {ssim_value:.4f} (measures image similarity, 1 is perfect similarity).")
    print(f"PSNR: {psnr_value:.2f} dB (indicates peak signal-to-noise ratio, higher is better, commonly used to measure the quality of reconstruction).")
    return ssim_value, psnr_value
