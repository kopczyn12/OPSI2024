import hydra
import pandas as pd
import os
from omegaconf import DictConfig, OmegaConf
from utils.logger import initialize_logger, logger
from utils.results_handling import run_denoising_experiment, plot_results
from utils.draw_figures import draw_triangle, draw_circle, draw_square, draw_parallelogram, draw_rhombus, draw_shape

@hydra.main(config_path="config", config_name="default", version_base="1.2")
def pipeline(cfg: DictConfig) -> None:
    initialize_logger(cfg)
    logger.info("Starting pipeline for examination of rank matrix approximation influence on image denoising results.")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    if cfg.pipeline.name=="denoising_experiment":
        if cfg.pipeline.figures.draw_figures:
            if not os.path.exists(cfg.pipeline.images_folder):
                os.makedirs(cfg.pipeline.images_folder, exist_ok=True)
            logger.info("Drawing figures.")
            draw_shape(draw_triangle, cfg.pipeline.figures.triangle_path, (cfg.pipeline.figures.size_width, cfg.pipeline.figures.size_height))
            draw_shape(draw_circle, cfg.pipeline.figures.circle_path, (cfg.pipeline.figures.size_width, cfg.pipeline.figures.size_height))
            draw_shape(draw_square, cfg.pipeline.figures.square_path, (cfg.pipeline.figures.size_width, cfg.pipeline.figures.size_height))
            draw_shape(draw_parallelogram, cfg.pipeline.figures.parallelogram_path, (cfg.pipeline.figures.size_width, cfg.pipeline.figures.size_height))
            draw_shape(draw_rhombus, cfg.pipeline.figures.rhombus_path, (cfg.pipeline.figures.size_width, cfg.pipeline.figures.size_height))
            logger.info("Figures drawn.")

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
