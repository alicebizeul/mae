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