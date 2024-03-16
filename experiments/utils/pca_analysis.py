import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA, KernelPCA
from skimage import io

def denoise_image_with_pca(image_path: str, main_folder: str, type_of_noise: str, variance: float, n_components: int, use_kernel_pca: bool = False) -> None:
    """
    Apply PCA or Kernel PCA to denoise a single image and save it in a specified main folder.
    The function treats each row of the image as a separate sample for PCA.

    Args:
        image_path (str): Path to the noisy image.
        main_folder (str): The main folder where the denoised image folder will be created.
        type_of_noise (str): The type of noise applied to the original image.
        variance (float): The variance used when the noise was added.
        n_components (int): The number of principal components to keep.
        use_kernel_pca (bool): If True, use Kernel PCA instead of standard PCA.

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

    # Choose between PCA and Kernel PCA with the fit_inverse_transform parameter set appropriately
    if use_kernel_pca:
        pca = KernelPCA(
        n_components=400,
        kernel="rbf",
        gamma=1e-3,
        fit_inverse_transform=True,
        alpha=5e-3,
        random_state=42,
        )
    else:
        pca = PCA(n_components=n_components)

    # Apply PCA and inverse transform to reconstruct the image
    transformed_data = pca.fit_transform(image_reshaped)
    image_reconstructed = pca.inverse_transform(transformed_data).reshape(original_shape)

    # Clip values to valid pixel range
    denoised_image = np.clip(image_reconstructed, 0, 255).astype('uint8')

    # Save the denoised image
    denoised_image_path = os.path.join(output_folder, os.path.basename(image_path))
    Image.fromarray(denoised_image).save(denoised_image_path)
    print(f"Denoised image saved as {denoised_image_path}")
