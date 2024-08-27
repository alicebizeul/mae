import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import logging
import numpy as np
import random
import matplotlib.pyplot as plt 
import csv
import medmnist
import numpy

import model
from model.module import ViTMAE
from model.module_eval import ViTMAE_eval
from model.vit_mae import ViTMAEForPreTraining
from dataset.dataloader import DataModule
import transformers
from transformers import ViTMAEConfig
from utils import (
    print_config,
    setup_wandb,
    get_git_hash
)

# Configure logging
log = logging.getLogger(__name__)
git_hash = get_git_hash()
OmegaConf.register_new_resolver("compute_lr", lambda base_lr, batch_size: base_lr * (batch_size / 256))

# Main function
@hydra.main(version_base="1.2", config_path="config", config_name="train_defaults.yaml")
def main(config: DictConfig) -> None:

    # Setup 
    print_config(config)
    pl.seed_everything(config.seed)
    hydra_core_config = HydraConfig.get()
    wandb_logger = setup_wandb(
        config, log, git_hash, {"job_id": hydra_core_config.job.name}
    )

    # Creating data 
    datamodule = instantiate(
        config.datamodule,
        data = config.datasets,
        masking = config.masking,
    )
    
    # Creating model
    vit_config = instantiate(config.module_config)
    vit = instantiate(config.module,vit_config)
    model_train = instantiate(
        config.pl_module, 
        model=vit,
        datamodule = datamodule,
        save_dir=config.local_dir
        )
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,  # Directory where to save the checkpoints
        filename='{epoch:02d}-{train_loss:.2f}',  # Filename format
        save_top_k=-1,  # Save all checkpoints
        save_weights_only=False,  # Save the full model (True for weights only)
        every_n_epochs=100  # Save every epoch
    )

    # Runing training (with eval on masked data to track behavior/convergence)
    trainer_configs = OmegaConf.to_container(config.trainer, resolve=True)
    trainer = pl.Trainer(
            **trainer_configs,
            logger=wandb_logger,
            enable_checkpointing = True,
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback],
            check_val_every_n_epoch=config.pl_module.eval_freq
        )
    trainer.fit(model_train, datamodule=datamodule)


    # Final evaluation: original data, no pixel or pc masking, MAE eval protocol
    eval_configs = OmegaConf.to_container(config.evaluator, resolve=True)
    datamodule = instantiate(
        config.datamodule_eval,
        masking = {"type":"pixel"},
        data = config.datasets,
    )
    model_eval = instantiate(
        config=config.pl_module_eval,
        model=model_train.model,
        datamodule=datamodule,
        save_dir=config.local_dir
    )
    del model_train, trainer, vit
    evaluator = pl.Trainer(
            **eval_configs,
            logger=wandb_logger,
            enable_checkpointing = False,
            num_sanity_val_steps=0
        )
    evaluator.fit(model_eval, datamodule=datamodule)

if __name__ == "__main__":
    main()

