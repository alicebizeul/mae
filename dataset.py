import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torch.utils.data import Dataset
import torch
from jax import jit
import time
from torchvision.utils import save_image
import random
import matplotlib.pyplot as plt

# @jit
# def reconstruct_image(components, pcs, mean):
#     return jnp.dot(components, pcs) + mean


# create mask given a kernel size, image size and probability 
def create_mask(image_res,kernel_size,proba,show=False):
    assert kernel_size<=image_res
    ratio = int(np.ceil(image_res/kernel_size))
    nb_events = int(ratio * ratio)
    random_events = round(proba*nb_events)*[0] + round((1-proba)*nb_events)*[1]
    random.shuffle(random_events)
    mask = np.reshape(random_events,[int(ratio),int(ratio)])
    mask = np.kron(mask, np.ones((kernel_size,kernel_size)))

    if show:
        plt.imshow(mask,cmap="gray")
        plt.show()
    return mask

def select_pixels(image_res,kernel_size,proba,show=False):
    assert kernel_size<=image_res
    ratio = int(np.ceil(image_res/kernel_size))
    nb_events = int(ratio * ratio)
    random_events = round(proba*nb_events)*[0] + round((1-proba)*nb_events)*[1]
    random.shuffle(random_events)
    mask = np.reshape(random_events,[int(ratio),int(ratio)])
    mask = np.kron(mask, np.ones((kernel_size,kernel_size)))
    mask = (mask-1)*(-1)
    return mask


class PCA(torch.nn.Module):
    def __init__(self,projection,pcs,mean,nb_pc,shape):
        super().__init__()
        self.projection = projection
        self.pcs = pcs
        self.mean = mean
        self.nb_pc = nb_pc
        self.original_shape = shape

        print()

    def forward(self, index):  # we assume inputs are always structured like this

        # change projection
        # components = np.zeros_like(self.projection)
        # print(index,-self.nb_pc,components.shape,components[:, -self.nb_pc:],self.projection[:, -self.nb_pc:].shape)
        zeros = np.zeros([self.projection.shape[0],self.projection.shape[1]-self.nb_pc])
        to_keep = self.projection[:, -self.nb_pc:]
        
        components = np.concatenate([zeros,to_keep],axis=1)
        
        # components[:,:-self.nb_pc] = np.zeros([self.projection.shape[0],self.projection.shape[1][:-self.nb_pc]])
        components = components[index]

        # reconstruct
        # t0=time.time()
        image_reconstructed = np.dot(components, self.pcs) + self.mean
        # print(time.time()-t0)
        image_reconstructed = image_reconstructed.reshape(self.original_shape)

        return torch.tensor(image_reconstructed)


class IndexDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_label=None):
        self.dataset = dataset
        # self.transform_label = transform_label
        # self.pc_projection = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]  # Ignore the original label
        # path, _ = self.dataset.samples[index]

        # if index not in list(self.pc_projection.keys()):
        #     if self.transform_label:
        #         image = self.transform_label(index)
        #     print(path)
        #     save_image(image, path)
        # else:
        #     image = self.pc_projection[index]
        mask = torch.tensor(create_mask(64,2,0.75))
        mask=mask.unsqueeze(0).repeat(image.shape[0],1,1)
        # masked = (mask*image).to(torch.float32)

        return image, mask # Replace the label with the index

class LabelSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels_to_include):
        self.dataset = dataset
        self.labels_to_include = labels_to_include
        self.indices = self._get_indices()

    def _get_indices(self):
        # Get indices of the samples with the specified labels
        indices = [i for i, (_, label) in enumerate(self.dataset) if label in self.labels_to_include]
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_idx = self.indices[idx]
        return self.dataset[dataset_idx]

class RepDataset(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        x = self.x_list[idx]
        y = self.y_list[idx]
        return x, y