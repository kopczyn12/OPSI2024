import os
import numpy as np
from skimage import io, transform
from skimage.util import random_noise
from PIL import Image
from omegaconf import DictConfig

def add_noise_and_save(cfg: DictConfig, 
                       image_path: str,
                       main_folder: str,
                       noise_type: str,
                       var: float = 0.01) -> None:
    """
    Adds specified type of noise to an image and saves it to a subfolder within a specified main folder,
    appending the variance value to the file name.

    Args:
        cfg (DictConfig): The configuration object.
        image_path (str): The path to the input image.
        main_folder (str): The base folder where the 'noise_photos' folder will be created.
        noise_type (str): The type of noise to add. Options are 'gaussian', 'salt_pepper', and 'poisson'.
        var (float): The variance of the noise (used for gaussian and salt & pepper noise).

    The function creates a subfolder named after the noise type in the 'noise_photos' subfolder of the main folder
    and saves the noisy image there, appending the variance value to the filename. It does not return any value but prints updates about its progress.
    """

    # Load the image
    # Check if the image size is as expected
    image = io.imread(image_path)
    if image.shape[1] != cfg.pipeline.figures.size_width or image.shape[0] != cfg.pipeline.figures.size_height:
        # Resize the image
        image = transform.resize(image, (cfg.pipeline.figures.size_height, cfg.pipeline.figures.size_width), anti_aliasing=True)
        io.imsave(image_path, (image * 255).astype(np.uint8))
    print(f"Loaded image from {image_path}")

    # Apply the specified type of noise to the image
    if noise_type == 'gaussian':
        noisy_image = random_noise(image, mode='gaussian', var=var)
    elif noise_type == 'salt_pepper':
        noisy_image = random_noise(image, mode='s&p', amount=var)
    elif noise_type == 'poisson':
        noisy_image = random_noise(image, mode='poisson')
    elif noise_type == 'speckle':
        noisy_image = random_noise(image, mode='speckle', var=var)
    else:
        raise ValueError(
            "Unsupported noise type. Choose 'gaussian', 'salt_pepper', or 'poisson'."
        )
    print(f"Applied {noise_type} noise to the image.")

    # Convert the noisy image to the correct format
    noisy_image = (255 * noisy_image).astype(np.uint8)

    # Create the output directory if it doesn't exist
    noise_folder_path = os.path.join(main_folder, 'noise_photos', noise_type)
    os.makedirs(noise_folder_path, exist_ok=True)
    print(f"Directory {noise_folder_path} is ready for saving images.")

    # Modify the filename to include the variance value before the file extension
    filename = os.path.basename(image_path)
    base_filename, file_extension = os.path.splitext(filename)
    new_filename = f"{base_filename}_{var}{file_extension}"
    save_path = os.path.join(noise_folder_path, new_filename)

    # Save the noisy image in the specified directory
    Image.fromarray(noisy_image).save(save_path)
    print(f"Noisy image saved as {new_filename} in {save_path}")
    return save_path