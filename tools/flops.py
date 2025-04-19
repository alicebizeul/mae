import sys
sys.path.append("/cluster/home/abizeul/mae")

import torch
import torch.nn as nn
import torch.optim as optim
from lightning.fabric.utilities.throughput import measure_flops
from model.module import ViTMAE
from dataset.dataloader import DataModule
from model.vit_mae import ViTMAEForPreTraining
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
import torch.nn as nn
import torchvision
import torchvision.datasets
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, Timer

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
from model.module_lin import ViTMAE_lin
from model.module_knn import ViTMAE_knn
from model.vit_mae import ViTMAEForPreTraining
from dataset.dataloader import DataModule
# from dataset.CLEVRCustomDataset import CLEVRCustomDataset
import transformers
from transformers import ViTMAEConfig
from utils import (
    print_config,
    setup_wandb,
    get_git_hash,
    load_checkpoints,
    Normalize
)
import line_profiler
# Configure logging
log = logging.getLogger(__name__)
git_hash = get_git_hash()
def create_lambda_transform(mean, std):
    return torchvision.transforms.Lambda(lambda sample: (sample - mean) / std)
OmegaConf.register_new_resolver('divide', lambda a, b: int(int(a)/b))
OmegaConf.register_new_resolver('multiply', lambda a, b: int(int(a)*b))
OmegaConf.register_new_resolver("compute_lr", lambda base_lr, batch_size: base_lr * (batch_size / 256))
OmegaConf.register_new_resolver("decimal_2_percent", lambda decimal: int(100*decimal) if decimal is not None else decimal)
OmegaConf.register_new_resolver("convert_str", lambda number: "_"+str(number))
OmegaConf.register_new_resolver("substract_one", lambda number: number-1)
OmegaConf.register_new_resolver('to_tuple', lambda a, b, c: (a,b,c))
OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))

@hydra.main(version_base="1.2", config_path="../config", config_name="train_defaults.yaml")
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
        extra_data = config.extradata,
    )
    
    # Creating model
    vit_config = instantiate(config.module_config)
    vit = instantiate(config.module,vit_config)
    model = instantiate(
        config.pl_module, 
        model=vit,
        datamodule = datamodule,
        save_dir=config.local_dir
        )
    trainer_configs = OmegaConf.to_container(config.trainer, resolve=True)
    trainer = pl.Trainer(
            **trainer_configs,
        )

    # Create dummy data (make sure it matches your model's expected input shape)
    nb_pc = 1000
    dummy_input = [torch.randn(1,config.data.channels,config.data.resolution,config.data.resolution),torch.randint(low=0,high=9,size=[1]),torch.randint(low=0,high=10000,size=[nb_pc])]   # for example, batch size 32, 784 features

    # Measure FLOPs by running the training_step function multiple times
    # 'iterations' sets the number of runs to average over.
    training_step = lambda: model.training_step(dummy_input,1)
    loss_fn = lambda x: x
    flops = measure_flops(model,training_step,loss_fn)
    print("Estimated FLOPs per training step:", flops)



if __name__ == "__main__":
    main()
