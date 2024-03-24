import os
import numpy as np
from PIL import Image
from skimage import io

def denoise_image_with_svd(image_path: str, main_folder: str, type_of_noise: str, variance: float, n_components: int) -> None:
    """
    Apply SVD to denoise a single image and save it in a specified main folder.
    The function decomposes the image using SVD and reconstructs it using a limited number of singular values.

    Args:
        image_path (str): Path to the noisy image.
        main_folder (str): The main folder where the denoised image folder will be created.
        type_of_noise (str): The type of noise applied to the original image.
        variance (float): The variance used when the noise was added.
        n_components (int): The number of singular values to keep.

    This function saves the denoised image in a subfolder named 'denoise_images_{type_of_noise}_{variance}_{n_components}'.
    """
    # Create the output directory based on the noise type and variance
    output_folder = os.path.join(main_folder, f'denoise_images_{type_of_noise}_{variance}_{n_components}')
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    image = io.imread(image_path, as_gray=True)
    original_shape = image.shape

    # Perform SVD decomposition
    U, S, VT = np.linalg.svd(image, full_matrices=False)

    # Truncate the singular values
    S_truncated = np.zeros_like(S)
    S_truncated[:n_components] = S[:n_components]

    # Reconstruct the image using the truncated singular values
    S_matrix = np.diag(S_truncated)
    denoised_image = np.dot(U, np.dot(S_matrix, VT))

    # Clip values to valid pixel range
    denoised_image = np.clip(denoised_image, 0, 255).astype('uint8')

    # Save the denoised image
    denoised_image_path = os.path.join(output_folder, os.path.basename(image_path))
    Image.fromarray(denoised_image).save(denoised_image_path)
    print(f"Denoised image saved as {denoised_image_path}")
