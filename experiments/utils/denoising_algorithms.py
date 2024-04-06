import os
import numpy as np
from PIL import Image
from sklearn.decomposition import FastICA, NMF, PCA, KernelPCA, DictionaryLearning
import numpy as np
from skimage.color import rgb2gray
from skimage import io


def denoise_image_with_ica(image_path: str, main_folder: str,
                           type_of_noise: str, variance: float,
                           n_components: int) -> None:
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
    output_folder = os.path.join(
        main_folder,
        f'ICA_denoise_images_{type_of_noise}_{variance}_{n_components}')
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    image = io.imread(image_path, as_gray=True)
    original_shape = image.shape

    # Treat each row of the image as a sample
    image_reshaped = image.reshape(-1, original_shape[1])

    # Ensure n_components is within the valid range
    n_components = min(n_components, original_shape[1])
    if n_components <= 0:
        raise ValueError(
            "n_components must be greater than 0 and less than or equal to the image width."
        )

    # Initialize ICA with the specified number of components
    ica = FastICA(n_components=n_components, random_state=42)

    # Apply ICA and inverse transform to reconstruct the image
    transformed_data = ica.fit_transform(image_reshaped)
    image_reconstructed = ica.inverse_transform(transformed_data).reshape(
        original_shape)

    # Ensure the data is in the correct range and type
    denoised_image = np.clip(image_reconstructed, 0, 1)
    denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)

    # Save the denoised image
    denoised_image_path = os.path.join(output_folder, os.path.basename(image_path))
    Image.fromarray(denoised_image_uint8).save(denoised_image_path)
    print(f"Denoised image saved as {denoised_image_path}")
    return denoised_image_path


def denoise_image_with_low_rank(image_path: str, main_folder: str,
                                type_of_noise: str, variance: float,
                                n_components: int) -> None:
    """
    Apply Low-Rank Approximation using SVD to denoise a single image and save it in a specified main folder.
    The function computes the SVD of the image matrix and keeps only the first n_components singular values/components.

    Args:
        image_path (str): Path to the noisy image.
        main_folder (str): The main folder where the denoised image folder will be created.
        type_of_noise (str): The type of noise applied to the original image.
        variance (float): The variance used when the noise was added.
        n_components (int): The number of singular values/components to keep.

    This function saves the denoised image in a subfolder named 'denoise_images_{type_of_noise}_{variance}_{n_components}'.
    """
    # Create the output directory based on the noise type and variance
    output_folder = os.path.join(
        main_folder,
        f'Low_rank_denoise_images_{type_of_noise}_{variance}_{n_components}')
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    image = io.imread(image_path, as_gray=True)
    if len(image.shape) == 3:  # Convert to grayscale if it's a color image
        image = rgb2gray(image) * 255
    original_shape = image.shape

    # Flatten the image to 2D if it's grayscale
    image_flattened = image.reshape(original_shape[0], -1)

    # Perform SVD
    U, S, VT = np.linalg.svd(image_flattened, full_matrices=False)

    # Keep only the first n_components components
    S[n_components:] = 0

    # Reconstruct the image
    denoised_image_flattened = np.dot(U, np.dot(np.diag(S), VT))

    denoised_image_reshaped = denoised_image_flattened.reshape(original_shape)

    # Ensure the data is in the correct range and type
    denoised_image = np.clip(denoised_image_reshaped, 0, 1)
    denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)

    # Save the denoised image
    denoised_image_path = os.path.join(output_folder, os.path.basename(image_path))
    Image.fromarray(denoised_image_uint8).save(denoised_image_path)
    print(f"Denoised image saved as {denoised_image_path}")
    return denoised_image_path


def denoise_image_with_nmf(image_path: str, main_folder: str,
                           type_of_noise: str, variance: float,
                           n_components: int) -> None:
    """
    Apply NMF to denoise a single image and save it in a specified main folder.
    The function treats each row of the image as a separate sample for NMF.

    Args:
        image_path (str): Path to the noisy image.
        main_folder (str): The main folder where the denoised image folder will be created.
        type_of_noise (str): The type of noise applied to the original image.
        variance (float): The variance used when the noise was added.
        n_components (int): The number of components to keep.

    This function saves the denoised image in a subfolder named 'denoise_images_{type_of_noise}_{variance}_{n_components}'.
    """
    # Create the output directory based on the noise type and variance
    output_folder = os.path.join(
        main_folder,
        f'NMF_denoise_images_{type_of_noise}_{variance}_{n_components}')
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    image = io.imread(image_path, as_gray=True)
    if len(image.shape) == 3:  # Convert to grayscale if it's a color image
        image = rgb2gray(image) * 255
    original_shape = image.shape

    # Treat each row of the image as a separate sample
    image_reshaped = image.reshape(-1, original_shape[1])

    # Ensure n_components is within the valid range
    n_components = min(n_components, original_shape[1])
    if n_components <= 0:
        raise ValueError(
            "n_components must be greater than 0 and less than or equal to the image width."
        )

    # Initialize NMF with the specified number of components
    nmf = NMF(n_components=n_components, init='random', random_state=42)

    # Apply NMF and reconstruct the image
    W = nmf.fit_transform(image_reshaped)
    H = nmf.components_
    image_reconstructed = np.dot(W, H).reshape(original_shape)

    # Ensure the data is in the correct range and type
    denoised_image = np.clip(image_reconstructed, 0, 1)
    denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)

    # Save the denoised image
    denoised_image_path = os.path.join(output_folder, os.path.basename(image_path))
    Image.fromarray(denoised_image_uint8).save(denoised_image_path)
    print(f"Denoised image saved as {denoised_image_path}")
    return denoised_image_path


def denoise_image_with_pca(image_path: str,
                           main_folder: str,
                           type_of_noise: str,
                           variance: float,
                           n_components: int,
                           use_kernel_pca: bool = False) -> None:
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
    output_folder = os.path.join(
        main_folder,
        f'PCA_denoise_images_{type_of_noise}_{variance}_{n_components}')
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    image = io.imread(image_path, as_gray=True)
    original_shape = image.shape

    # Treat each row of the image as a sample
    image_reshaped = image.reshape(-1, original_shape[1])

    # Ensure n_components is within the valid range
    n_components = min(n_components, original_shape[1])
    if n_components <= 0:
        raise ValueError(
            "n_components must be greater than 0 and less than or equal to the image width."
        )

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
    image_reconstructed = pca.inverse_transform(transformed_data).reshape(
        original_shape)

    # Ensure the data is in the correct range and type
    denoised_image = np.clip(image_reconstructed, 0, 1)
    denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)

    # Save the denoised image
    denoised_image_path = os.path.join(output_folder, os.path.basename(image_path))
    Image.fromarray(denoised_image_uint8).save(denoised_image_path)
    print(f"Denoised image saved as {denoised_image_path}")
    return denoised_image_path


def denoise_image_with_dictionary_learning(image_path: str, main_folder: str,
                                           type_of_noise: str, variance: float,
                                           n_components: int) -> None:
    """
    Apply Dictionary Learning to denoise a single image and save it in a specified main folder.
    The function treats each row of the image as a separate sample for Dictionary Learning.

    Args:
        image_path (str): Path to the noisy image.
        main_folder (str): The main folder where the denoised image folder will be created.
        type_of_noise (str): The type of noise applied to the original image.
        variance (float): The variance used when the noise was added.
        n_components (int): The number of dictionary atoms to use.

    This function saves the denoised image in a subfolder named 'denoise_images_{type_of_noise}_{variance}_{n_components}'.
    """
    # Create the output directory based on the noise type and variance
    output_folder = os.path.join(
        main_folder,
        f'DL_denoise_images_{type_of_noise}_{variance}_{n_components}')
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    image = io.imread(image_path, as_gray=True)
    if len(image.shape) == 3:  # Convert to grayscale if it's a color image
        image = rgb2gray(image) * 255
    original_shape = image.shape

    # Flatten the image to 2D if it's grayscale
    image_flattened = image.reshape(original_shape[0], -1)

    # Initialize Dictionary Learning with the specified number of components
    dict_learner = DictionaryLearning(n_components=n_components,
                                      transform_algorithm='lasso_lars',
                                      random_state=42)

    # Fit to the image data and transform it into a sparse code
    code = dict_learner.fit_transform(image_flattened)

    # Reconstruct the image from the code and dictionary
    image_reconstructed = np.dot(code, dict_learner.components_)

    # Ensure the data is in the correct range and type
    denoised_image = np.clip(image_reconstructed, 0, 1)
    denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)

    # Save the denoised image
    denoised_image_path = os.path.join(output_folder, os.path.basename(image_path))
    Image.fromarray(denoised_image_uint8).save(denoised_image_path)
    print(f"Denoised image saved as {denoised_image_path}")
    return denoised_image_path
