import os
import logging
from utils.noise_generation import add_noise_and_save
from utils.metrics import calculate_metrics
from omegaconf import DictConfig
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd
from utils.denoising_algorithms import (
    denoise_image_with_dictionary_learning,
    denoise_image_with_ica,
    denoise_image_with_pca,
    denoise_image_with_low_rank,
    denoise_image_with_nmf
)

def run_denoising_experiment(cfg: DictConfig) -> pd.DataFrame:
    """
    Conducts a denoising experiment based on configurations provided.

    This function processes each image in a specified directory by applying various noise types
    and variances, denoises them using multiple denoising methods, and then calculates performance
    metrics for each denoising operation.

    Args:
        cfg (DictConfig): Configuration object that includes settings for the experiment, such as
        image folder, noise types, variances, denoising methods, and where to save results.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the experiments including metrics such
        as SSIM and PSNR for each image, noise type, variance, and denoising method.
    """
    logger = logging.getLogger(__name__)

    # Collect all image paths from the specified directory with specified extensions
    images_folder = cfg.pipeline.images_folder
    image_paths = [
        os.path.join(images_folder, f) for f in os.listdir(images_folder)
        if f.endswith(('jpg', 'png', 'jpeg'))
    ]
    logger.info(f"Found {len(image_paths)} images in directory {images_folder}.")

    # Set up the k_range for matrix rank approximations
    k_start = cfg.pipeline.k_range.start
    k_stop = cfg.pipeline.k_range.stop
    k_step = cfg.pipeline.k_range.step
    k_range = range(k_start, k_stop, k_step)
    logger.debug(f"k_range set to start at {k_start}, stop at {k_stop}, step by {k_step}.")

    # Define methods based on configuration flags
    methods = {}
    if cfg.pipeline.methods.get('dictionary_learning', True):
        methods["Dictionary Learning"] = denoise_image_with_dictionary_learning
    if cfg.pipeline.methods.get('ICA', True):
        methods["ICA"] = denoise_image_with_ica
    if cfg.pipeline.methods.get('PCA', True):
        methods["PCA"] = denoise_image_with_pca
    if cfg.pipeline.methods.get('Low Rank', True):
        methods["Low Rank"] = denoise_image_with_low_rank
    if cfg.pipeline.methods.get('NMF', True):
        methods["NMF"] = denoise_image_with_nmf
    logger.info("Denoising methods configured.")

    results_list = []

    # Main processing loop for each image and noise/denoising configuration
    for image_path in image_paths:
        for noise_type in cfg.pipeline.noise_types:
            for var in cfg.pipeline.variances:
                logger.info(f"Processing {os.path.basename(image_path)} with {noise_type} noise at variance {var}.")

                # Add noise and save the noisy image
                noisy_image_path = add_noise_and_save(image_path, cfg.pipeline.main_folder, noise_type, var)
                logger.info("Noisy image saved at: " + noisy_image_path)

                for method_name, method_function in methods.items():
                    logger.info(f"Applying denoising method: {method_name}")
                    for k_rank in k_range:
                        logger.info(f"Using rank {k_rank} for denoising.")

                        # Denoise the image and save the denoised version
                        denoised_image_path = method_function(
                            noisy_image_path, cfg.pipeline.results_folder, noise_type, var, k_rank)
                        logger.info("Denoised image saved.")

                        # Calculate and log performance metrics
                        ssim_value, psnr_value = calculate_metrics(image_path, denoised_image_path)
                        logger.info(f"Metrics calculated - SSIM: {ssim_value}, PSNR: {psnr_value}.")

                        # Append results to the list for later analysis
                        results_list.append({
                            "Image": os.path.basename(image_path),
                            "Noise Type": noise_type,
                            "Variance": var,
                            "Method": method_name,
                            "K Rank": k_rank,
                            "SSIM": ssim_value,
                            "PSNR": psnr_value,
                        })
                        logger.debug("Results appended for image.")

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results_list)
    logger.info("All experiments completed and results compiled.")
    results_df.to_csv(os.path.join(cfg.pipeline.main_folder, 'results.csv'), index=False)
    return results_df

def plot_results(cfg: DictConfig, results_df: pd.DataFrame) -> None:
    """
    Plots SSIM and PSNR metrics for each denoising method by rank, noise type, and variance.

    This function iterates through each unique method, noise type, variance, and image in the results
    dataframe, filters the results, and generates plots for SSIM and PSNR metrics which are saved to
    the specified directory.

    Args:
        cfg (DictConfig): Configuration object containing experiment settings, including the directory
        where plots should be saved.
        results_df (pd.DataFrame): Dataframe containing denoising results with columns for method, noise type,
        variance, image name, K Rank, SSIM, and PSNR.

    Returns:
        Plots are saved to disk in the directory specified in the configuration.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting to plot SSIM and PSNR for each method by K Rank.")

    # Ensure the plots directory exists
    os.makedirs(cfg.pipeline.plots_dir, exist_ok=True)

    # Iterate over each method, noise type, variance, and specific image
    for method in results_df['Method'].unique():
        for noise_type in results_df['Noise Type'].unique():
            for variance in results_df['Variance'].unique():
                for image_name in results_df['Image'].unique():
                    # Filter the DataFrame for the current method, noise type, variance, and image
                    df_filtered = results_df[
                        (results_df['Method'] == method) &
                        (results_df['Noise Type'] == noise_type) &
                        (results_df['Variance'] == variance) &
                        (results_df['Image'] == image_name)
                    ].sort_values(by='K Rank')

                    if not df_filtered.empty:
                        # Plot for SSIM
                        plt.figure(figsize=(10, 6))
                        plt.plot(df_filtered['K Rank'], df_filtered['SSIM'], marker='o', linestyle='-', color='blue')
                        plt.title(f'{method} - SSIM by K Rank\nImage: {image_name}, Noise: {noise_type}, Variance: {variance}')
                        plt.xlabel('K Rank')
                        plt.ylabel('SSIM')
                        plt.grid(True)
                        ssim_plot_path = os.path.join(cfg.pipeline.plots_dir, f'{method}_{image_name}_{noise_type}_{variance}_SSIM.png')
                        plt.savefig(ssim_plot_path)
                        plt.close()

                        # Plot for PSNR
                        plt.figure(figsize=(10, 6))
                        plt.plot(df_filtered['K Rank'], df_filtered['PSNR'], marker='o', linestyle='-', color='red')
                        plt.title(f'{method} - PSNR by K Rank\nImage: {image_name}, Noise: {noise_type}, Variance: {variance}')
                        plt.xlabel('K Rank')
                        plt.ylabel('PSNR')
                        plt.grid(True)
                        psnr_plot_path = os.path.join(cfg.pipeline.plots_dir, f'{method}_{image_name}_{noise_type}_{variance}_PSNR.png')
                        plt.savefig(psnr_plot_path)
                        plt.close()

                        # Log the paths where plots are saved
                        logger.info(f"SSIM plot saved at {ssim_plot_path}.")
                        logger.info(f"PSNR plot saved at {psnr_plot_path}.")

    logger.info("Completed plotting all metrics.")