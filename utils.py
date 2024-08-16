<<<<<<< HEAD
import os 
import torchvision
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define the function to save images and their reconstructions
def save_reconstructed_images(input, target, reconstructed, epoch, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    input_grid = torchvision.utils.make_grid(input[:8].cpu(), nrow=4, normalize=True)
    target_grid = torchvision.utils.make_grid(target[:8].cpu(), nrow=4, normalize=True)
    reconstructed_grid = torchvision.utils.make_grid(reconstructed[:8].cpu(), nrow=4, normalize=True)
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_grid.permute(1, 2, 0))
    axes[0].set_title('Input Images')
    axes[0].axis('off')

    axes[1].imshow(target_grid.permute(1, 2, 0))
    axes[1].set_title('Input Images')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed_grid.permute(1, 2, 0))
    axes[2].set_title('Reconstructed Images')
    axes[2].axis('off')
    
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_{name}.png'))
    plt.close()

def get_eigenvalues(data):
    pca = PCA()  # You can adjust the number of components

    if len(data.shape)!=2:
        data = data.reshape(data.shape[0],*data.shape[1:])
    pca.fit(data)

    return pca.explained_variance_

class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.base_lr = 0.0
        self.annealing_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, total_epochs - warmup_epochs, eta_min=0)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.annealing_scheduler.step(epoch - self.warmup_epochs)

class PairedImageDataset(Dataset):
    def __init__(self, folder_A, folder_B, transform=None):
        self.folder_A = folder_A
        self.folder_B = folder_B
        self.transform = transform

        # Get list of image names in folder_A
        self.image_names = os.listdir(folder_A)
        if os.path.isdir(os.path.join(folder_A,self.image_names[0])):
            self.image_names_extended = []
            for dir in self.image_names:
                self.image_names_extended.extend([os.path.join(dir,x) for x in os.listdir(os.path.join(folder_A,dir))])
        # Optionally sort to ensure consistent ordering
        self.image_names= self.image_names_extended
        self.image_names.sort()
        del self.image_names_extended

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        img_A_path = os.path.join(self.folder_A, img_name)
        img_B_path = os.path.join(self.folder_B, img_name)

        image_A = Image.open(img_A_path).convert('RGB')
        image_B = Image.open(img_B_path).convert('RGB')

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B
||||||| empty tree
=======
import os 
import torchvision
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Iterable, Optional, Callable, Tuple
from torch import nn

# Define the function to save images and their reconstructions
def save_reconstructed_images(input, target, reconstructed, epoch, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    input_grid = torchvision.utils.make_grid(input[:8].cpu(), nrow=4, normalize=True)
    target_grid = torchvision.utils.make_grid(target[:8].cpu(), nrow=4, normalize=True)
    reconstructed_grid = torchvision.utils.make_grid(reconstructed[:8].cpu(), nrow=4, normalize=True)
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_grid.permute(1, 2, 0))
    axes[0].set_title('Input Images')
    axes[0].axis('off')

    axes[1].imshow(target_grid.permute(1, 2, 0))
    axes[1].set_title('Target Images')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed_grid.permute(1, 2, 0))
    axes[2].set_title('Reconstructed Images')
    axes[2].axis('off')
    
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_{name}.png'))
    plt.close()

# Define the function to save images and their reconstructions
def save_attention_maps(input, attention, epoch, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    input_grid = torchvision.utils.make_grid(input[:8].cpu(), nrow=4, normalize=True)
    target_grid = torchvision.utils.make_grid(attention[:8].cpu(), nrow=4, normalize=True)
    
    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(input_grid.permute(1, 2, 0))
    axes[0].set_title('Input Images')
    axes[0].axis('off')

    axes[1].imshow(input_grid.permute(1, 2, 0))
    axes[1].imshow(target_grid.permute(1, 2, 0),cmap='jet', alpha=0.5)
    axes[1].set_title('Attention Maps')
    axes[1].axis('off')

    
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_{name}_attention.png'))
    plt.close()

def get_eigenvalues(data):
    pca = PCA()  # You can adjust the number of components

    if len(data.shape)!=2:
        data = data.reshape(data.shape[0],*data.shape[1:])
    pca.fit(data)

    return pca.explained_variance_

class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.base_lr = 0.0
        self.annealing_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, total_epochs - warmup_epochs, eta_min=0)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.annealing_scheduler.step(epoch - self.warmup_epochs)

class PairedImageDataset(Dataset):
    def __init__(self, folder_A, folder_B, transform=None):
        self.folder_A = folder_A
        self.folder_B = folder_B
        self.transform = transform

        # Get list of image names in folder_A
        self.image_names = os.listdir(folder_A)
        if os.path.isdir(os.path.join(folder_A,self.image_names[0])):
            self.image_names_extended = []
            for dir in self.image_names:
                self.image_names_extended.extend([os.path.join(dir,x) for x in os.listdir(os.path.join(folder_A,dir))])
        # Optionally sort to ensure consistent ordering
        self.image_names= self.image_names_extended
        self.image_names.sort()
        del self.image_names_extended

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        img_A_path = os.path.join(self.folder_A, img_name)
        img_B_path = os.path.join(self.folder_B, img_name)

        image_A = Image.open(img_A_path).convert('RGB')
        image_B = Image.open(img_B_path).convert('RGB')

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B
    
class Lars(Optimizer):
    r"""Implements the LARS optimizer from `"Large batch training of convolutional networks"
    <https://arxiv.org/pdf/1708.03888.pdf>`_.
    Code taken from: https://github.com/NUS-HPC-AI-Lab/InfoBatch/blob/master/examples/lars.py 
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eeta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr=1e-3,
            momentum=0,
            eeta=1e-3,
            weight_decay=0,
            epsilon=0.0
    ) -> None:
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eeta <= 0:
            raise ValueError("Invalid eeta value: {}".format(eeta))
        if epsilon < 0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, eeta=eeta, epsilon=epsilon, lars=True)

        super().__init__(params, defaults)

    def set_decay(self,weight_decay):
        for group in self.param_groups:
            group['weight_decay'] = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eeta = group['eeta']
            lr = group['lr']
            lars = group['lars']
            eps = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                decayed_grad = p.grad
                scaled_lr = lr
                if lars:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(p.grad)
                    trust_ratio = torch.where(
                        w_norm > 0 and g_norm > 0,
                        eeta * w_norm / (g_norm + weight_decay * w_norm + eps),
                        torch.ones_like(w_norm)
                    )
                    trust_ratio.clamp_(0.0, 50)
                    scaled_lr *= trust_ratio.item()
                    if weight_decay != 0:
                        decayed_grad = decayed_grad.add(p, alpha=weight_decay)
                decayed_grad = torch.clamp(decayed_grad, -10.0, 10.0)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            decayed_grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(decayed_grad)
                    decayed_grad = buf

                p.add_(decayed_grad, alpha=-scaled_lr)

        return loss
>>>>>>> ef3aa814808eee5583d56279a2e9b29fc9a39ffa
