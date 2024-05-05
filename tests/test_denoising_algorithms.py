import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import sys
from skimage import io
sys.path.append('.')
sys.path.append('..')
from utils.denoising_algorithms import denoise_image_with_ica, denoise_image_with_dictionary_learning, denoise_image_with_low_rank, denoise_image_with_nmf, denoise_image_with_pca
from utils.metrics import calculate_metrics
from utils.noise_generation import add_noise_and_save
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@pytest.fixture
def noisy_image(tmpdir):
    """Create a noisy image and save it to a temporary file."""
    image_path = tmpdir.join("noisy_image.png")
    noisy_image = (np.random.rand(100, 100) * 255).astype(np.uint8)  # scale to 0-255 and convert to uint8
    io.imsave(str(image_path), noisy_image)  # save the image
    return str(image_path), noisy_image

# Test case 1: Test denoise_image_with_ica function
@ignore_warnings(category=ConvergenceWarning)
def test_denoise_image_with_ica(noisy_image):
    image_path, original_image = noisy_image
    temp_dir = os.path.dirname(image_path)

    # Call the denoising function
    denoised_image_path = denoise_image_with_ica(
        image_path=image_path,
        main_folder=temp_dir,
        type_of_noise="gaussian",
        variance=0.1,
        n_components=10
    )

    # Verify the output file exists
    assert os.path.isfile(denoised_image_path), "Denoised image file should exist"

    # Load and verify the denoised image
    denoised_image = io.imread(denoised_image_path)
    assert denoised_image.shape == original_image.shape, "Denoised image should have the same shape as the original"

    # Check pixel value ranges
    assert denoised_image.min() >= 0, "Pixel values should be >= 0"
    assert denoised_image.max() <= 255, "Pixel values should be <= 255"

    # Check if the denoised image is different from the original (this might depend on the effectiveness of the denoising algorithm)
    assert not np.array_equal(denoised_image, original_image), "Denoised image should differ from the noisy image"

# Test case 2: Test denoise_image_with_pca function
@ignore_warnings(category=ConvergenceWarning)
def test_denoise_image_with_pca(noisy_image):
    image_path, original_image = noisy_image
    temp_dir = os.path.dirname(image_path)

    # Call the denoising function
    denoised_image_path = denoise_image_with_pca(
        image_path=image_path,
        main_folder=temp_dir,
        type_of_noise="gaussian",
        variance=0.1,
        n_components=10
    )

    # Verify the output file exists
    assert os.path.isfile(denoised_image_path), "Denoised image file should exist"

    # Load and verify the denoised image
    denoised_image = io.imread(denoised_image_path)
    assert denoised_image.shape == original_image.shape, "Denoised image should have the same shape as the original"

    # Check pixel value ranges
    assert denoised_image.min() >= 0, "Pixel values should be >= 0"
    assert denoised_image.max() <= 255, "Pixel values should be <= 255"

    # Check if the denoised image is different from the original (this might depend on the effectiveness of the denoising algorithm)
    assert not np.array_equal(denoised_image, original_image), "Denoised image should differ from the noisy image"

# Test case 3: Test denoise_image_with_nmf function
@ignore_warnings(category=ConvergenceWarning)
def test_denoise_image_with_nmf(noisy_image):
    image_path, original_image = noisy_image
    temp_dir = os.path.dirname(image_path)

    # Call the denoising function
    denoised_image_path = denoise_image_with_nmf(
        image_path=image_path,
        main_folder=temp_dir,
        type_of_noise="gaussian",
        variance=0.1,
        n_components=10
    )

    # Verify the output file exists
    assert os.path.isfile(denoised_image_path), "Denoised image file should exist"

    # Load and verify the denoised image
    denoised_image = io.imread(denoised_image_path)
    assert denoised_image.shape == original_image.shape, "Denoised image should have the same shape as the original"

    # Check pixel value ranges
    assert denoised_image.min() >= 0, "Pixel values should be >= 0"
    assert denoised_image.max() <= 255, "Pixel values should be <= 255"

    # Check if the denoised image is different from the original (this might depend on the effectiveness of the denoising algorithm)
    assert not np.array_equal(denoised_image, original_image), "Denoised image should differ from the noisy image"

# Test case 4: Test denoise_image_with_low_rank function
@ignore_warnings(category=ConvergenceWarning)
def test_denoise_image_with_low_rank(noisy_image):
    image_path, original_image = noisy_image
    temp_dir = os.path.dirname(image_path)

    # Call the denoising function
    denoised_image_path = denoise_image_with_low_rank(
        image_path=image_path,
        main_folder=temp_dir,
        type_of_noise="gaussian",
        variance=0.1,
        n_components=10
    )

    # Verify the output file exists
    assert os.path.isfile(denoised_image_path), "Denoised image file should exist"

    # Load and verify the denoised image
    denoised_image = io.imread(denoised_image_path)
    assert denoised_image.shape == original_image.shape, "Denoised image should have the same shape as the original"

    # Check pixel value ranges
    assert denoised_image.min() >= 0, "Pixel values should be >= 0"
    assert denoised_image.max() <= 255, "Pixel values should be <= 255"

    # Check if the denoised image is different from the original (this might depend on the effectiveness of the denoising algorithm)
    assert not np.array_equal(denoised_image, original_image), "Denoised image should differ from the noisy image"

# Test case 5: Test denoise_image_with_dictionary_learning function
@ignore_warnings(category=ConvergenceWarning)
def test_denoise_image_with_dictionary_learning(noisy_image):
    image_path, original_image = noisy_image
    temp_dir = os.path.dirname(image_path)

    # Call the denoising function
    denoised_image_path = denoise_image_with_dictionary_learning(
        image_path=image_path,
        main_folder=temp_dir,
        type_of_noise="gaussian",
        variance=0.1,
        n_components=10
    )

    # Verify the output file exists
    assert os.path.isfile(denoised_image_path), "Denoised image file should exist"

    # Load and verify the denoised image
    denoised_image = io.imread(denoised_image_path)
    assert denoised_image.shape == original_image.shape, "Denoised image should have the same shape as the original"

    # Check pixel value ranges
    assert denoised_image.min() >= 0, "Pixel values should be >= 0"
    assert denoised_image.max() <= 255, "Pixel values should be <= 255"

    # Check if the denoised image is different from the original (this might depend on the effectiveness of the denoising algorithm)
    assert not np.array_equal(denoised_image, original_image), "Denoised image should differ from the noisy image"

@pytest.fixture
def fake_image():
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)
# Test case 6: Test add_noise_and_save function
def test_add_noise_and_save(tmpdir, fake_image):
    image_path = tmpdir.join("original_image.png")
    main_folder = tmpdir.mkdir("output")
    noise_type = "gaussian"
    variance = 0.05

    # Save the fake image to simulate an existing file
    Image.fromarray(fake_image).save(str(image_path))

    with patch('skimage.io.imread', return_value=fake_image) as mock_read,\
         patch('PIL.Image.fromarray') as mock_fromarray,\
         patch('os.makedirs') as mock_makedirs,\
         patch('numpy.random.normal', return_value=fake_image) as mock_noise:
        noisy_image_path = add_noise_and_save(
            image_path=str(image_path),
            main_folder=str(main_folder),
            noise_type=noise_type,
            var=variance
        )

        # Check if imread was called correctly
        mock_read.assert_called_once_with(image_path)

        # Check if the directory was created
        expected_noise_folder_path = os.path.join(str(main_folder), 'noise_photos', noise_type)
        mock_makedirs.assert_called_with(expected_noise_folder_path, exist_ok=True)

        # Check if the image was saved with the correct file name
        mock_fromarray.return_value.save.assert_called()
        call_args = mock_fromarray.return_value.save.call_args
        assert call_args[0][0].startswith(expected_noise_folder_path), "Image should be saved in the correct directory"

        # Confirm the file path includes the variance
        assert f"_{variance}." in call_args[0][0], "Filename should include the variance"

        # Check the returned path is correct
        base_filename = os.path.basename(str(image_path))
        new_filename = f"{os.path.splitext(base_filename)[0]}_{variance}{os.path.splitext(base_filename)[1]}"
        expected_path = os.path.join(expected_noise_folder_path, new_filename)
        assert noisy_image_path == expected_path, "Function should return the correct file path"

# Test 7: Test calculate_metrics function
def test_calculate_metrics():
    # Create mock images
    original_image = np.random.rand(100, 100)  # 100x100 random pixels
    denoised_image = np.random.rand(100, 100)  # Another 100x100 random pixels

    # Patch 'io.imread' to return these mock images
    with patch('skimage.io.imread', side_effect=[original_image, denoised_image]):
        ssim_value, psnr_value = calculate_metrics('dummy_path_original.png', 'dummy_path_denoised.png')

        # Assert that SSIM and PSNR are calculated and are float types
        assert isinstance(ssim_value, float), "SSIM should be a float value"
        assert isinstance(psnr_value, float), "PSNR should be a float value"
        assert 0 <= ssim_value <= 1, "SSIM should be between 0 and 1"
        assert psnr_value >= 0, "PSNR should be non-negative"