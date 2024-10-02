# Principled Masked Autoencoders

In this repository we explore more principled methods for image masking 

## Installation 

In your environment of choice, install the necessary requirements

    !pip install -r requirements.txt 

Create a config file that suits your machine:

    cd ./config/user
    cp myusername_mymachine.yaml <myusername>_<mymachine>.yaml

Adjust the paths to point to the directory you would like to use for storage of results and for fetching the data

### Training
To launch experiments, you can find a good example for training at  ```./script/jobs_pcmae_random.sh```.

### Evaluation on linear probing
To evaluate a checkpoint, you can gain inspiration from ```./script/jobs_eval_pcmae_random.sh``` and the yaml file ```./config/user/<myusername>_<mymachine>.yaml``` where runs are stored. 

### Information needed to run PMAE

PMAE relies on PCA hence the code runs only if the PC matrix, list of percentages of explained variance for each eigenvalues, and the mean and std of the data set to be evaluated; The path to these elements should be stored in ```./config/dataset/mydataset.yaml``` . The following code gives an overview of how these components are obtained: 

```
#%%

import os 
import glob 
import sklearn
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA, IncrementalPCA
from torch.utils.data import DataLoader
import random
from torch.utils.data import DataLoader, Dataset, Subset
transform = transforms.Compose([
    transforms.ToTensor()
])


data_folder = "/local/home/data"

# CIFAR10 - Download and load training dataset

folder = f'{data_folder}/cifar-10-batches-py'
trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

# # Fetch the entire dataset in one go
data_iter = iter(trainloader)
images, labels = next(data_iter)

# # Step 2: Convert the dataset to NumPy arrays
images_np = images.numpy()
labels_np = labels.numpy()

# # Reshape the images to (num_samples, height * width * channels)
num_samples = images_np.shape[0]
original_shape = images_np.shape
images_flat = images_np.reshape(num_samples, -1)

# # Standardize
mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
images_flat = (images_flat - mean) / std

# # Step 4: Perform PCA
pca = PCA()  # You can adjust the number of components
pca.fit(images_flat)

np.save(f'{folder}/mean.npy',mean)
np.save(f'{folder}/std.npy',std)
np.save(f'{folder}/pc_matrix.npy',pca.components_)
np.save(f'{folder}/eigenvalues_ratio.npy',pca.explained_variance_ratio_)

```

