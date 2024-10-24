import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
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
from dataset.CLEVRCustomDataset import CLEVRCustomDataset

USER_NAME = os.environ.get("USER")

class PairedDataset(Dataset):
    def __init__(self, dataset, masking, eigenvalues, patch_sizes, patch_nb):

        self.dataset = dataset
        self.masking = masking
        self.patch_nb = patch_nb
        # if self.masking.type == "pixel":
        #     self.pc_mask = 0
        if self.masking.type == "pc":
            # assert "eigenratiomodule" in list(extra_data.keys())
            # assert "pcamodule" in list(extra_data.keys())

            # self.eigenvalues = torch.Tensor(extra_data.eigenratiomodule)
            # self.pca         = torch.Tensor(extra_data.pcamodule)

            # shuffling        = torch.permute(torch.arange(self.eigenvalues.shape[0]))

            # self.eigenvalues = self.eigenvalues[shuffling]
            # self.pca         = self.pca[shuffling]

            
            
            # self.index_patches      = torch.arange(1,eigenvalues.shape[0])[torch.diff(torch.cumsum(eigenvalues)//(1/self.patch_nb))==-1]
            # self.patch_sizes       = [self.index_patches[0]] + list(torch.diff(self.index_patches).numpy())
            self.eigenvalues        = torch.split(eigenvalues, patch_sizes)
            print("splitted",len(self.eigenvalues))
            self.eigenvalues_cumsum = [torch.cumsum(x,dim=0).item() for x in eigenvalues]
            self.find_threshold     = lambda eigenvalues ,ratio: np.argmin(np.abs(np.cumsum(eigenvalues) - ratio))
            self.get_pcs_index      = np.arange

            # if self.masking.strategy == "tvb" or self.masking.strategy == "bvt": 
            #     threshold = self.find_threshold(self.eigenvalues,self.masking.pc_ratio)
            #     if self.masking.strategy == "bvt": self.pc_mask = self.get_pcs_index(threshold)
            #     if self.masking.strategy == "tvb": self.pc_mask = self.get_pcs_index(threshold,self.eigenvalues.shape[0])
            # else: 
            # self.pc_mask = None
        # elif self.masking.type == "segmentation":
        #     self.pc_mask = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # Load the images
        img1, y = self.dataset[idx]
        # pc_mask = self.pc_mask

        if isinstance(y,list) and len(y)==2:
            pc_mask = y[1]
            y = y[0]

        if self.masking.type == "pc":
            if self.masking.strategy == "sampling_pc":
                index     = torch.randperm(self.patch_nb).numpy()
                pc_ratio  = np.random.randint(10,90,1)[0]/100
                threshold = self.find_threshold(self.eigenvalues_cumsum[index],pc_ratio)
                pc_mask   = index[:threshold]
            elif self.masking.strategy == "pc":
                index = torch.randperm(self.patch_nb)
                threshold = self.find_threshold(self.eigenvalues_cumsum[index],self.masking.pc_ratio)
                pc_mask = index[:threshold]

        elif self.masking.type == "pixel":
            if self.masking.strategy == "sampling":
                pc_mask = float(np.random.randint(10,90,1)[0]/100)            
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
        patch_size: int =8,
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
        self.patch_nb = (resolution//patch_size)**2

        if masking.type == "pc":
            self.eigenvalues = torch.Tensor(extra_data.eigenratiomodule)
            self.pca         = torch.Tensor(extra_data.pcamodule)
            shuffling        = torch.randperm(self.eigenvalues.shape[0])
            self.eigenvalues = self.eigenvalues[shuffling]
            self.pca         = self.pca[shuffling] # look at this
        
            index_patches      = torch.arange(1,self.eigenvalues.shape[0])[torch.diff(torch.cumsum(self.eigenvalues,dim=0)//(1/self.patch_nb))==1]
            self.patch_sizes   = [index_patches[0].item()+1] + list(torch.diff(index_patches).numpy())

    def setup(self, stage):
        self.train_dataset = PairedDataset(
            dataset=self.datasets["train"],
            masking=self.masking,
            eigenvalues=self.eigenvalues,
            patch_sizes=self.patch_sizes,
            patch_nb=self.patch_nb
            )

        self.val_dataset = self.datasets["val"]
        self.num_val_samples = len(self.val_dataset)
        self.test_dataset = self.datasets["test"]

    def collate_fn(self,batch):
        """
        Custom collate function to handle variable-sized pc_mask.
        Pads the pc_mask to the size of the largest pc_mask in the batch.
        """

        imgs, labels, pc_masks = zip(*batch)
        max_len = max([pc_mask.size for pc_mask in pc_masks])

        padded_pc_masks = [torch.nn.functional.pad(torch.tensor(pc_mask), (0, max_len - pc_mask.size),value=-1) for pc_mask in pc_masks]
        imgs = torch.stack(imgs)  # Assuming images are tensors and can be stacked directly
        # if isinstance(labels,tuple):
        #     labels = torch.stack(labels)
        # else:
        labels = torch.tensor(labels)  # Convert labels to tensor
        padded_pc_masks = torch.stack(padded_pc_masks)  # Stack the padded pc_masks

        return imgs, labels, padded_pc_masks

    def train_dataloader(self) -> DataLoader:
        training_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, collate_fn=self.collate_fn if (self.masking.type == "pc" and self.masking.strategy in ["sampling_pc","sampling_ratio","sampling_pc_block","pc"]) else None
        )
        return training_loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers
        )
        return loader
