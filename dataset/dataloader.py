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

        self.dataset = dataset
        self.masking = masking
        if self.masking.type == "pixel":
            self.pc_mask = 0
        elif self.masking.type == "pc":
            assert "eigenratiomodule" in list(extra_data.keys())
            assert "pcamodule" in list(extra_data.keys())

            self.eigenvalues = torch.Tensor(extra_data.eigenratiomodule)

            self.find_threshold = lambda eigenvalues ,ratio: np.argmin(np.abs(np.cumsum(eigenvalues) - ratio))
            self.get_pcs_index  = np.arange

            if self.masking.strategy == "tvb" or self.masking.strategy == "bvt": 
                threshold = self.find_threshold(self.eigenvalues,self.masking.pc_ratio)
                if self.masking.strategy == "bvt": self.pc_mask = self.get_pcs_index(threshold)
                if self.masking.strategy == "tvb": self.pc_mask = self.get_pcs_index(threshold,self.eigenvalues.shape[0])
            else: 
                self.pc_mask = None
        elif self.masking.type == "segmentation":
            self.pc_mask = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # Load the images
        img1, y = self.dataset[idx]
        pc_mask = self.pc_mask
        if self.masking.type == "pc":
            if self.masking.strategy == "sampling_ratio":
                pc_ratio      = float(np.random.randint(np.ceil(100*(self.eigenvalues[0]+self.eigenvalues[1])),99,1)[0]/100)
                threshold     = self.find_threshold(self.eigenvalues,pc_ratio)
                top_vs_bottom = np.random.randint(0,2,1)[0]
                if top_vs_bottom == 0:
                    pc_mask = self.get_pcs_index(threshold)
                else:
                    pc_mask = self.get_pcs_index(threshold,self.eigenvalues.shape[0])

            elif self.masking.strategy == "sampling_pc":
                nb_pc = np.random.randint(1,self.eigenvalues.shape[0],1)[0]
                index = torch.randperm(self.eigenvalues.shape[0]).numpy()
                pc_mask = index[:nb_pc]
            elif self.masking.strategy == "sampling_pc_block":
                nb_block = np.random.randint(1,self.eigenvalues.shape[0]//(8*8),1)[0]
                index = torch.randperm(self.eigenvalues.shape[0]//(8*8)).numpy()
                blocks = index[:nb_block]
                pc_mask = np.linspace(0,self.eigenvalues.shape[0]-(8*8),self.eigenvalues.shape[0]//(8*8),dtype=int)[blocks]
                pc_mask = np.concatenate([np.arange(x,x+(8*8)) for x in pc_mask])
        elif self.masking.type == "pixel":
            if self.masking.strategy == "sampling":
                pc_mask = float(np.random.randint(50,90,1)[0]/100)            
        elif self.masking.type == "segmentation":
            pc_mask = y[1]
            pc_mask = 0.75
            y = y[0]
        return img1, y, pc_mask

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
            extra_data=self.extra_data
            )

        self.val_dataset = self.datasets["val"]
        self.num_val_samples = len(self.val_dataset)
        self.test_dataset = self.datasets["test"]

    def collate_fn(self,batch):
        """
        Custom collate function to handle variable-sized pc_mask.
        Pads the pc_mask to the size of the largest pc_mask in the batch.
        """

        # Unpack the batch (which is a list of tuples)
        imgs, labels, pc_masks = zip(*batch)
        # Find the maximum length of pc_mask in this batch
        max_len = max([pc_mask.size for pc_mask in pc_masks])

        # Pad pc_masks to the same size
        padded_pc_masks = [torch.nn.functional.pad(torch.tensor(pc_mask), (0, max_len - pc_mask.size),value=-1) for pc_mask in pc_masks]
        # Stack images, labels, and padded pc_masks
        imgs = torch.stack(imgs)  # Assuming images are tensors and can be stacked directly
        labels = torch.tensor(labels)  # Convert labels to tensor
        padded_pc_masks = torch.stack(padded_pc_masks)  # Stack the padded pc_masks

        return imgs, labels, padded_pc_masks

    def train_dataloader(self) -> DataLoader:
        training_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, collate_fn=self.collate_fn if self.masking.type == "pc" and self.masking.strategy in ["sampling_pc","sampling_ratio","sampling_pc_block"] else None
        )
        return training_loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers
        )
        return loader
