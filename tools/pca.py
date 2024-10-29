import sys
sys.path.append("../")
import os 
import glob 
import sklearn
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
# from dataset import TinyImageNetDataset
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from torch.utils.data import DataLoader, Dataset, Subset
import medmnist
from medmnist import DermaMNIST, PathMNIST, BloodMNIST
random.seed(42)

nb_aug=100
resolution=224
name="imagenet"
# data_fn = torchvision.datasets.CIFAR10
data_fn = torchvision.datasets.ImageFolder
# data_fn = DermaMNIST
# folder = '/cluster/project/sachan/callen/data_alice/cifar-10-batches-py'
# folder = "/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train"
# folder = "/cluster/project/sachan/callen/data_alice/medmnist/"
folder = "/cluster/project/sachan/callen/data_alice/ILSVRC2012_img/train"

########
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
transform_aug = transforms.Compose([
    torchvision.transforms.RandomResizedCrop(resolution,scale=[0.2,1.0],interpolation= 3),
    torchvision.transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #for imagenet
])

# Download and load training dataset
# trainset = data_fn(root='/cluster/project/sachan/callen/data_alice', train=True, download=True, transform=transform)
trainset = data_fn(root=folder, transform=transform)
# trainset = data_fn(root=folder, transform=transform, split='train', size=resolution)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(len(trainset),3*resolution*resolution), shuffle=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

data_iter = iter(trainloader)
images, labels = next(data_iter)

images_np = images.numpy()
labels_np = labels.numpy()
print("This is the batch size",int((3*resolution*resolution)/4),flush=True)

# Fetch the entire dataset in one go
pca = IncrementalPCA(n_components=12288,batch_size=6144,copy=False)  # You can adjust the number of components

# Reshape the images to (num_samples, height * width * channels)
num_samples = images_np.shape[0]
original_shape = images_np.shape
images_flat = images_np.reshape(num_samples, -1)

# Standardize
# mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
# images_flat = (images_flat - mean) / std

# Step 4: Perform PCA
pca.fit(images_flat)

# # trainset = data_fn(root='/cluster/project/sachan/callen/data_alice', train=True, download=True, transform=transform_aug)
# trainset = data_fn(root=folder, transform=transform_aug)
# # trainset = data_fn(root=folder, transform=transform_aug, split='train', size=resolution)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(len(trainset),3*resolution*resolution), shuffle=False)
# print("This is the batch size",min(len(trainset),3*resolution*resolution),flush=True)
# for i in range(nb_aug):
#     print("we are processing aug",i)
#     # Fetch the entire dataset in one go

    # for batch in trainloader:
    #     images, labels = batch 

#     # Step 2: Convert the dataset to NumPy arrays
#     images_np = images.numpy()
#     labels_np = labels.numpy()

#     # Reshape the images to (num_samples, height * width * channels)
#     num_samples = images_np.shape[0]
#     original_shape = images_np.shape
#     images_flat = images_np.reshape(num_samples, -1)
#     images_flat = (images_flat - mean) / std
#     pca.fit(images_flat)

# np.save(f'{folder}/{name}_pc_matrix_augmented.npy',pca.components_)
# np.save(f'{folder}/{name}_eigenvalues_augmented.npy',pca.explained_variance_)
# np.save(f'{folder}/{name}_eigenvalues_ratio_augmented.npy',pca.explained_variance_ratio_)

np.save(f'{folder}/pc_matrix_augmented.npy',pca.components_)
np.save(f'{folder}/eigenvalues_augmented.npy',pca.explained_variance_)
np.save(f'{folder}/eigenvalues_ratio_augmented.npy',pca.explained_variance_ratio_)