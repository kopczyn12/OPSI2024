import os
import numpy as np
from PIL import Image
from sklearn.decomposition import FastICA
from skimage import io

def denoise_image_with_ica(image_path: str, main_folder: str, type_of_noise: str, variance: float, n_components: int) -> None:
    """
    Apply ICA to denoise a single image and save it in a specified main folder.
    The function treats each row of the image as a separate sample for ICA.

    Args:
        image_path (str): Path to the noisy image.
        main_folder (str): The main folder where the denoised image folder will be created.
        type_of_noise (str): The type of noise applied to the original image.
        variance (float): The variance used when the noise was added.
        n_components (int): The number of independent components to keep.

    This function saves the denoised image in a subfolder named 'denoise_images_{type_of_noise}_{variance}_{n_components}'.
    """
    # Create the output directory based on the noise type and variance
    output_folder = os.path.join(main_folder, f'denoise_images_{type_of_noise}_{variance}_{n_components}')
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    image = io.imread(image_path, as_gray=True)
    original_shape = image.shape

    # Treat each row of the image as a sample
    image_reshaped = image.reshape(-1, original_shape[1])

    # Ensure n_components is within the valid range
    n_components = min(n_components, original_shape[1])
    if n_components <= 0:
        raise ValueError("n_components must be greater than 0 and less than or equal to the image width.")

    # Initialize ICA with the specified number of components
    ica = FastICA(n_components=n_components, random_state=42)

    # Apply ICA and inverse transform to reconstruct the image
    transformed_data = ica.fit_transform(image_reshaped)
    image_reconstructed = ica.inverse_transform(transformed_data).reshape(original_shape)

    # Clip values to valid pixel range
    denoised_image = np.clip(image_reconstructed, 0, 255).astype('uint8')

    # Save the denoised image
    denoised_image_path = os.path.join(output_folder, os.path.basename(image_path))
    Image.fromarray(denoised_image).save(denoised_image_path)
    print(f"Denoised image saved as {denoised_image_path}")
