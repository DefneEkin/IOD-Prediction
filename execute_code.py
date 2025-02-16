

import wandb
import config
import os
import sys

models_folder = os.path.join(os.path.dirname(__file__), "models")
sys.path.append(models_folder)

if __name__ == "__main__":
    wandb.init(project="project")
    cfg = config.Config()
    selected_code_filename = cfg.selected_code

    try:
        selected_code_module = __import__(selected_code_filename[:-3])  # Remove ".py" extension
        train_function = selected_code_module.train
    except ImportError:
        raise ImportError(f"Failed to import {selected_code_filename}")

    with wandb.init(config=cfg):
        train_function(cfg)

    wandb.finish()