import pytorch_lightning as pl
import torchmetrics
from torch import Tensor
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from model_zoo.scattering_network import Scattering2dResNet
from torchvision.models import resnet18
from torch import Tensor
import wandb
import os 
import matplotlib.pyplot as plt
import numpy as np
from utils import save_reconstructed_images, save_attention_maps, save_attention_maps_batch
from plotting import plot_loss, plot_performance

class ViTMAE(pl.LightningModule):

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        base_learning_rate: float =1e-3,
        weight_decay: float = 0.05,
        betas: list =[0.9,0.95],
        optimizer_name: str = "adamw",
        warmup: int =10,
        datamodule: Optional[pl.LightningDataModule] = None,
        eval_freq: int =100,
        save_dir: str =None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.optimizer_name = optimizer_name
        self.datamodule = datamodule
        self.num_classes = datamodule.num_classes
        self.image_size = datamodule.image_size
        self.classifier_lr = learning_rate
        self.warm_up = warmup
        self.eval_freq = eval_freq

        self.model = model
        self.classifier = nn.Linear(model.config.hidden_size, self.num_classes)

        self.online_classifier_loss = nn.CrossEntropyLoss()
        self.online_train_accuracy = torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.num_classes, top_k=1
        )
        self.online_val_accuracy = torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.num_classes, top_k=1
        ) 
        self.save_dir = save_dir
        self.train_losses = []
        self.avg_train_losses = []
        self.online_losses = []
        self.avg_online_losses = []
        self.performance = {}

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch: Tensor, stage: str = "train", batch_idx: int =None):
        if stage == "train":
            img, target, y = batch

            # mae training
            outputs, cls = self.model(img,return_rep=False)
            reconstruction = self.model.unpatchify(outputs.logits)
            mask = outputs.mask.unsqueeze(-1).repeat(1, 1, self.model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
            mask = self.model.unpatchify(mask)
            loss_mae = self.model.forward_loss(target,outputs.logits,outputs.mask)
            self.log(
                f"{stage}_mae_loss", 
                loss_mae, 
                prog_bar=True,
                sync_dist=True,
                on_step=False,
                on_epoch=True
                )
            
            self.train_losses.append(loss_mae.item())
            self.avg_train_losses.append(np.mean(self.train_losses))

            if (self.current_epoch+1)%self.eval_freq==0 and batch_idx==0:
                plot_loss(self.avg_train_losses,name_loss="MSE",save_dir=self.save_dir,name_file="_train")
                plot_loss(self.avg_online_losses,name_loss="X-Ent",save_dir=self.save_dir,name_file="_train_online_cls")

                if self.model.config.mask_ratio > 0:
                    save_reconstructed_images((-1*(mask[:10]-1))*img[:10],mask[:10]*target[:10], reconstruction[:10], self.current_epoch+1, self.save_dir,"train")
                else:
                    save_reconstructed_images(img[:10], target[:10], reconstruction[:10], self.current_epoch+1, self.save_dir,"train")

            del mask, reconstruction

            # online classifier
            logits_cls = self.classifier(cls.detach())
            loss_ce = self.online_classifier_loss(logits_cls,y.squeeze())
            self.log(f"{stage}_classifier_loss", loss_ce, sync_dist=True)
            self.online_losses.append(loss_ce.item())
            self.avg_online_losses.append(np.mean(self.online_losses))

            accuracy_metric = getattr(self, f"online_{stage}_accuracy")
            accuracy_metric(F.softmax(logits_cls, dim=-1), y.squeeze())
            self.log(
                f"online_{stage}_accuracy",
                accuracy_metric,
                prog_bar=False,
                sync_dist=True,
            )
            del logits_cls 

            if (self.current_epoch+1)%self.eval_freq==0 and batch_idx==0:
                plot_loss(self.avg_online_losses,name_loss="X-Ent",save_dir=self.save_dir,name_file="_train_online_cls")

            return loss_mae + loss_ce

        else:
            img, y = batch
            cls, _ = self.model(img,return_rep=True)
            logits = self.classifier(cls.detach())

            accuracy_metric = getattr(self, f"online_{stage}_accuracy")
            accuracy_metric(F.softmax(logits, dim=-1), y.squeeze())
            self.log(
                f"online_{stage}_accuracy",
                accuracy_metric,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
            )

            if batch_idx == 0:
                if self.current_epoch+1 not in list(self.performance.keys()): 
                    self.performance[self.current_epoch+1]=[]

            self.performance[self.current_epoch+1].append(sum(1*(torch.argmax(logits, dim=-1)==y.squeeze())).item())  

            return None

    # def on_train_epoch_start(self):
    #     # Check the current epoch and switch regimes every 100 epochs
    #     if self.current_epoch % (self.eval_freq+self.eval_duration) < self.eval_freq:
    #         self.classifier_training = False
    #     else:
    #         self.classifier_training = True    

    def on_validation_epoch_end(self):
        self.performance[self.current_epoch+1] = sum(self.performance[self.current_epoch+1])/self.datamodule.num_val_samples
        plot_performance(list(self.performance.keys()),list(self.performance.values()),self.save_dir,name="val")

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="train", batch_idx=batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="val", batch_idx=batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="test", batch_idx=batch_idx)
        return loss

    def configure_optimizers(self):
        print("This is the learning rate",self.learning_rate)

        def warmup(current_step: int):
            return 1 / (10 ** (float(num_warmup_epochs - current_step)))

        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.trainer.max_epochs, verbose=False
            )
        elif self.optimizer_name == "adamw_warmup":
            num_warmup_epochs = self.warm_up
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.betas
            )


            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=warmup
            )

            train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.trainer.max_epochs, verbose=False
            )

            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, train_scheduler], [num_warmup_epochs]
            )

        else:
            raise ValueError(f"{self.optimizer_name} not supported")
        
        return [optimizer], [lr_scheduler]

