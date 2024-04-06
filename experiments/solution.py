import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from utils.noise_generation import add_noise_and_save
from utils.denoising_algorithms import (
    denoise_image_with_dictionary_learning,
    denoise_image_with_ica,
    denoise_image_with_pca,
    denoise_image_with_low_rank,
    denoise_image_with_nmf
)
from utils.metrics import calculate_metrics

# Initialize logging with both file and console handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logger's level

# Create a file handler
file_handler = logging.FileHandler('denoising_evaluation.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Starting pipeline for examination of rank matrix approximation influence on image denoising results.")
# Configuration
HOME = os.getenv("HOME")
logger.info(f"Working directory is {HOME}.")
images_folder = os.path.join(HOME, 'data/original_photos')
logger.info(f"Images folder is {images_folder}.")
main_folder = os.path.join(HOME, 'data')
logger.info(f"Main folder is {main_folder}.")
noise_types = ["gaussian", "salt_pepper", "poisson"]
logger.info(f"Noise types are {noise_types}.")
variances = [0.01, 0.05, 0.1]
logger.info(f"Variance values are {variances}.")
k_range = range(1, 40, 2)  # Ranks for approximation matrix
logger.info(f"Rank values are {k_range}.")
methods = {
    "Dictionary Learning": denoise_image_with_dictionary_learning,
    "ICA": denoise_image_with_ica,
    "PCA": denoise_image_with_pca,
    "Low Rank": denoise_image_with_low_rank,
    "NMF": denoise_image_with_nmf,
}
logger.info(f"Methods are {methods.keys()}.")

# List to collect results
results_list = []

logger.info("Starting evaluation loop.")
# Automatically list all images in the given directory
image_paths = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith(('jpg', 'png', 'jpeg'))]

# Main loop
for image_path in image_paths:
    for noise_type in noise_types:
        for var in variances:
            logger.info(
                f"Processing image {os.path.basename(image_path)} with {noise_type} noise and variance {var}."
            )
            # Add noise and save the noisy image
            noisy_image_path = add_noise_and_save(image_path, main_folder,
                                                  noise_type, var)
            logger.info("Noisy image saved.")
            for method_name, method_function in methods.items():
                logger.info(f"Processing method {method_name}.")
                for k_rank in k_range:
                    logger.info(f"Processing rank {k_rank}.")
                    logger.info("Processing denoising.")
                    # Denoise the image and save the denoised version
                    denoised_image_path = method_function(
                        noisy_image_path, main_folder, noise_type, var, k_rank)
                    logger.info("Denoised image saved.")

                    logger.info("Calculating metrics.")
                    # Calculate metrics
                    ssim_value, psnr_value = calculate_metrics(
                        image_path, denoised_image_path)
                    logger.info("Metrics calculated.")

                    logger.info("Appending results.")
                    # Collect results
                    results_list.append({
                        "Image": os.path.basename(image_path),
                        "Noise Type": noise_type,
                        "Variance": var,
                        "Method": method_name,
                        "K Rank": k_rank,
                        "SSIM": ssim_value,
                        "PSNR": psnr_value,
                    })
                    logger.info("Results appended.")

logger.info("Evaluation loop complete.")

# Convert the list of results to a DataFrame
results_df = pd.DataFrame(results_list)

# Calculate the overall mean SSIM and PSNR for each method
overall_means = results_df.groupby("Method")[["SSIM", "PSNR"]].mean().rename(columns={"SSIM": "Overall Mean SSIM through all ranks", "PSNR": "Overall Mean PSNR through all ranks"}).reset_index()

# Now, we will iterate over each method to assign the overall mean SSIM and PSNR to each corresponding row
# Initialize the columns for the overall means
results_df["Overall Mean SSIM through all ranks"] = 0
results_df["Overall Mean PSNR through all ranks"] = 0

# Assign the calculated overall means to each row based on its method
for index, row in overall_means.iterrows():
    method = row["Method"]
    results_df.loc[results_df["Method"] == method, "Overall Mean SSIM through all ranks"] = row["Overall Mean SSIM through all ranks"]
    results_df.loc[results_df["Method"] == method, "Overall Mean PSNR through all ranks"] = row["Overall Mean PSNR through all ranks"]

logger.info("Overall mean SSIM and PSNR calculated and assigned for each method.")

# Save the results DataFrame with the overall means included
results_df.to_csv("denoising_final_results_with_overall_means.csv", index=False)

logger.info(f"Results with overall means saved to CSV at {os.path.join(main_folder, 'denoising_final_results_with_overall_means.csv')}.")

logger.info("Plotting SSIM and PSNR for each method by K Rank.")


# Ensure the plots directory exists
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Iterate over each method, noise type, variance, and now specifically image
for method in results_df['Method'].unique():
    for noise_type in results_df['Noise Type'].unique():
        for variance in results_df['Variance'].unique():
            for image_name in results_df['Image'].unique():
                # Filter the DataFrame for the current method, noise type, variance, and specific image
                df_filtered = results_df[(results_df['Method'] == method) & 
                                         (results_df['Noise Type'] == noise_type) & 
                                         (results_df['Variance'] == variance) & 
                                         (results_df['Image'] == image_name)].sort_values(by='K Rank')

                if not df_filtered.empty:
                    # Plot for SSIM
                    plt.figure(figsize=(10, 6))
                    plt.plot(df_filtered['K Rank'], df_filtered['SSIM'], marker='o', linestyle='-', color='blue')
                    plt.title(f'{method} - SSIM by K Rank\nImage: {image_name}, Noise: {noise_type}, Variance: {variance}')
                    plt.xlabel('K Rank')
                    plt.ylabel('SSIM')
                    plt.grid(True)
                    ssim_plot_path = os.path.join(plots_dir, f'{method}_{image_name}_{noise_type}_{variance}_SSIM.png')
                    plt.savefig(ssim_plot_path)
                    plt.close()  # Ensure the plot is closed after saving

                    # Plot for PSNR
                    plt.figure(figsize=(10, 6))
                    plt.plot(df_filtered['K Rank'], df_filtered['PSNR'], marker='o', linestyle='-', color='red')
                    plt.title(f'{method} - PSNR by K Rank\nImage: {image_name}, Noise: {noise_type}, Variance: {variance}')
                    plt.xlabel('K Rank')
                    plt.ylabel('PSNR')
                    plt.grid(True)
                    psnr_plot_path = os.path.join(plots_dir, f'{method}_{image_name}_{noise_type}_{variance}_PSNR.png')
                    plt.savefig(psnr_plot_path)
                    plt.close()  # Ensure the plot is closed after saving

                    logger.info(f"SSIM plot for {method}, Image: {image_name}, {noise_type}, Variance: {variance} saved to {ssim_plot_path}.")
                    logger.info(f"PSNR plot for {method}, Image: {image_name}, {noise_type}, Variance: {variance} saved to {psnr_plot_path}.")