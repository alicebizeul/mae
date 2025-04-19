import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# Load PC matrix
pc_matrix = torch.tensor(np.load("/cluster/scratch/abizeul/cifar10/pc_matrix.npy"), dtype=torch.float32).T  # shape (d, d)
print(pc_matrix.shape)
assert pc_matrix.shape[0] == pc_matrix.shape[1], "PC matrix should be square"

# Parameters
batch_size = 1
img_shape = (3, 32, 32)
d = np.prod(img_shape)

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])
dataset = datasets.CIFAR10(root="/cluster/scratch/abizeul/cifar10", train=False, download=False, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Test projection + reconstruction
for images, _ in loader:
    images_flat = images.view(-1, d)  # shape (B, d)

    # Project to PC space and back
    projected = images_flat @ pc_matrix[:,:100]      # (B, d)
    reconstructed = projected @ pc_matrix[:,:100].T  # (B, d)

    # Compute reconstruction error
    # error = torch.norm(images_flat - reconstructed, dim=1).mean()
    # print(f"Reconstruction error (L2): {error.item():.4f}")

    # Optional: visualize reconstruction
    print(reconstructed.shape)
    reconstructed = reconstructed.view(-1, *img_shape).clamp(0, 1)
    images_flat = images_flat.view(-1, *img_shape).clamp(0, 1)
    print(reconstructed.shape)
    for i, img in enumerate(reconstructed):
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # Convert to HWC, uint8
        img_pil = Image.fromarray(img_np)
        img_pil.save(f"./reconstructed.png")

    for i, img in enumerate(images_flat):
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # Convert to HWC, uint8
        img_pil = Image.fromarray(img_np)
        img_pil.save(f"./original.png")

    break  # Just one batch for demo
