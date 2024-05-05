import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from utils.logger import initialize_logger, logger
from utils.results_handling import run_denoising_experiment, plot_results

@hydra.main(config_path="config", config_name="default", version_base="1.2")
def pipeline(cfg: DictConfig) -> None:
    initialize_logger(cfg)
    logger.info("Starting pipeline for examination of rank matrix approximation influence on image denoising results.")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    if cfg.pipeline.name=="denoising_experiment":
        logger.info("Running denoising experiment.")
        results_df = run_denoising_experiment(cfg)
        logger.info("Denosing experiment completed.")
        logger.info("Plotting results.")
        plot_results(cfg, results_df)
        logger.info("Results plotted.")
    else:
        logger.error(f"Pipeline {cfg.pipeline.name} not recognized.")
        raise ValueError(f"Pipeline {cfg.pipeline.name} not recognized.")

if __name__ == "__main__":
    pipeline()
