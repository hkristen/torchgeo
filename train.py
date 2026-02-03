import sys

import importlib
import torch
import yaml
from lightning.pytorch import Trainer

from torchgeo.datamodules import HabitAlp2DataModule


def main():
    """Run training from config file."""
    # Parse command line arguments
    if len(sys.argv) < 4 or sys.argv[1] != "fit" or sys.argv[2] != "--config":
        print("Usage: python train.py fit --config <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[3]
    
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Load datamodule
    init_cfg = cfg["data"]["init_args"]
    dict_cfg = cfg["data"].get("dict_kwargs", {})

    dm = HabitAlp2DataModule(**init_cfg, **dict_cfg, download=True)
    
    # Load model task
    model_cfg = cfg["model"]["init_args"]
    task_library_name = cfg["model"]["class_path"]

    # Dynamically load class from a string path
    module_path, class_name = task_library_name.rsplit(".", 1)
    module = importlib.import_module(module_path)

    Task = getattr(module, class_name)
    model = Task(**model_cfg)
    
    # Create trainer
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=cfg.get("max_epochs", 30),
        log_every_n_steps=5,
    )
    
    # Run training
    trainer.fit(model, datamodule=dm)
    
    # Run testing
    trainer.test(model, datamodule=dm)

    # Save model
    torch.save(model.model.state_dict(), "model_state_dict.pt")


if __name__ == "__main__":
    main()
