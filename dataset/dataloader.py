import pytorch_lightning as pl
import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from typing import Optional
from torchvision import transforms
from hydra.utils import instantiate
from torchvision.transforms.functional import InterpolationMode
import random
import torchvision.datasets
import time

USER_NAME = os.environ.get("USER")

class PairedDataset(Dataset):
    def __init__(self, dataset, masking, extra_data):
        """
        Initialize the dataset with two root directories and an optional transform.

        :param root1: Root directory for the first dataset.
        :param root2: Root directory for the second dataset.
        :param transform: Transformations to apply to the images.
        """
        self.dataset = dataset
        self.masking = masking
        if self.masking.type == "pc":
            assert "pcamodule" in list(extra_data.keys())
            assert "eigenratiomodule" in list(extra_data.keys())

            self.pc_matrix   = torch.Tensor(extra_data.pcamodule)
            self.eigenvalues = torch.Tensor(extra_data.eigenratiomodule)

            self.find_threshold = lambda eigenvalues ,ratio: np.argmin(np.abs(np.cumsum(eigenvalues) - ratio))
            self.get_pcs_index  = np.arange

            if self.masking.strategy == "tvb": 
                # what we keep
                threshold = self.find_threshold(self.eigenvalues,self.masking.pc_ratio)
                self.pc_mask = self.get_pcs_index(threshold)

                # what we drop
                self.pc_anti_mask = self.get_pcs_index(threshold,self.eigenvalues.shape[0])

            elif self.masking.strategy == "sampling":
                self.pc_mask_options = {}
                self.nb_shuffle = 20
                for i in range(self.nb_shuffle):
                    self.pc_mask_options[i] = self.get_pcs_index(self.find_threshold(random.shuffle(self.eigenvalues),self.masking.pc_ratio))
                

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # Load the images
        img1, y = self.dataset[idx]
        original_shape = img1.shape

        if self.masking.type == "pixel":
            img2 = img1
        elif self.masking.type == "pc":
            if self.masking.strategy == "sampling":
                # option_nb = random.randint(0,self.nb_shuffle)
                # self.pc_mask = self.get_pcs_index(self.find_threshold(self.pc_mask_options[option_nb],self.masking.pc_ratio))
                # self.pc_anti_mask = 
                raise NotImplementedError

            elif self.masking.strategy == "tvb_dynamic":
                dynamic_ratio = random.randint(int(np.ceil(100*self.eigenvalues[0])), 100)/100
                threshold     = self.find_threshold(self.eigenvalues,dynamic_ratio)

                self.pc_mask      = self.get_pcs_index(threshold)
                self.pc_anti_mask = self.get_pcs_index(threshold,self.eigenvalues.shape[0])

            P   = self.pc_matrix[:,self.pc_anti_mask]
            img2 = (img1.reshape(-1) @ P @ P.T).reshape(original_shape)

            P = self.pc_matrix[:,self.pc_mask]
            img1 = (img1.reshape(-1) @ P @ P.T).reshape(original_shape)

        else: raise NotImplementedError

        return img1, img2, y

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data,
        masking, 
        extra_data =None,
        batch_size: int = 512,
        num_workers: int = 8,
        classes: int =10,
        channels: int =3,
        resolution: int =32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = classes
        self.input_channels = channels
        self.image_size = resolution
        self.masking = masking
        self.extra_data = extra_data

        self.datasets = data

    def setup(self, stage):
        self.train_dataset = PairedDataset(
            dataset=self.datasets["train"],
            masking=self.masking,
            extra_data=self.extra_data)

        self.val_dataset = self.datasets["val"]
        self.num_val_samples = len(self.val_dataset)
        self.test_dataset = self.datasets["test"]

    def train_dataloader(self) -> DataLoader:
        training_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        return training_loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )
        return loader