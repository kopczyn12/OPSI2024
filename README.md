# OPSI2024 - Image Denoising Experiment

## Overview
This repository hosts the necessary experiments and a production pipeline for the academic project titled "The Effect of the Rank of the Approximation Matrix on the Quality of Image Denoising". This project has been developed as part of a college credit requirement. It evaluates the performance of several denoising algorithms to understand how different ranks of approximation matrices impact the quality of image denoising.

## Algorithms Tested
The project tests five prominent denoising algorithms:
- **Dictionary Learning**: Utilizes sparse representations over learned dictionaries.
- **PCA (Principal Component Analysis)**: Reduces dimensionality by extracting principal components.
- **ICA (Independent Component Analysis)**: Identifies independent components in the image data.
- **NMF (Non-negative Matrix Factorization)**: Decomposes multivariate data by maintaining non-negativity constraints.
- **Low-Rank Approximation (SVD)**: Employs singular value decomposition to approximate matrices at lower ranks.

## Getting Started

### Prerequisites
To run this project, you need Python 3.8 or later and `pip`. It's recommended to use a virtual environment to avoid conflicts with existing packages.

### Setting Up Your Environment
1. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
2. **Install Required Packages**
    ```pip install -r requirements.txt```

### Configuration
Before running the experiments, configure the settings in config/denoising_experiment.yaml within the config subfolder to meet your experiment's specific need

### Running the Experiment
Execute the pipeline with the following command:
```python3 solution_pipeline.py +pipeline=denoising_experiment```

### Unit tests

To verify the correctness of the implemented code, execute the command ```pytest``` following the installation of the required dependencies. This command will initiate the unit tests, which evaluate the functionality of metrics calculations, denoising functions, and noise generation. You can review the tests located in the tests folder. Currently, all tests are passing successfully.

### Results
Upon completion, the pipeline outputs:

- CSV Files: Containing detailed metrics like SSIM and PSNR for each tested configuration.
- Plots: Visualizations of SSIM and PSNR across different ranks.
- Denoised Images: For each input image, the results folder will contain a visual representation of the denoised output.
  
Explore the results folder to review the denoised images and quantitative metrics. This comprehensive output allows for a thorough analysis of the algorithm performances under various conditions.
