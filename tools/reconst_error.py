import os 
import numpy as np 
import torchvision 
import transformers
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Normalize, load_checkpoints
from model.vit_mae import ViTMAEForPreTraining
from model.module import ViTMAE
from dataset.dataloader import DataModule
from types import SimpleNamespace
import matplotlib.pyplot as plt
# dataset 
mytrans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),Normalize(np.load("/cluster/scratch/abizeul/cifar-10-batches-py/mean_reshaped.npy"),np.load("/cluster/scratch/abizeul/cifar-10-batches-py/std_reshaped.npy"))])
dataset = torchvision.datasets.CIFAR10(root="/cluster/scratch/abizeul",transform=mytrans,train=False)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=64)
datamodule = DataModule(data={"train":dataset,"val":dataset,"test":dataset},batch_size=64,classes=10,channels=3,resolution=32,masking=SimpleNamespace(**{"type":"pixel","strategy":"pixel","pixel_ratio":0.75,"norm_pix_loss":False,"ratio":75}))
datamodule.setup(stage="val")
# model 
vit_config  = transformers.ViTMAEConfig(hidden_size=192,num_attention_head=3,intermediate_size=768,norm_pix_loss=False,attn_implementation="eager",mask_ratio=0.75,patch_size=4,image_size=32)
vit         = ViTMAEForPreTraining(vit_config)
checkpoint  = "/cluster/scratch/abizeul/mae_alice/logs/abizeul/mae_cifar10_pixel_0.75_lossA/2025-01-28_10-28-25/checkpoints/epoch=999-train_loss=0.00.ckpt"
model_train = ViTMAE(model=vit,datamodule=datamodule)
model_train.load_state_dict(torch.load(checkpoint)["state_dict"],strict=False)

# # 
loss=0
for i, (batch,_) in enumerate(datamodule.val_dataloader()):

    outputs, cls = model_train.model(batch,return_rep=False)
    reconstruction = model_train.model.unpatchify(outputs.logits)
    mask = outputs.mask.unsqueeze(-1).repeat(1, 1, 4**2 *3)  # (N, H*W, p*p*3)
    mask = model_train.model.unpatchify(mask)
    loss+=torch.nn.MSELoss()(mask*batch,mask*reconstruction)

    if i ==0:
        # Select 6 images
        num_images = 6
        batch_images = batch[:num_images]  # Select first 6 images
        reconstructions = reconstruction[:num_images]
        masked_images = ((-1*(mask-1)) * batch)[:num_images]
        masked_images2 = (mask * batch)[:num_images]

        # Function to convert tensor images to numpy arrays
        def tensor_to_image(tensor):
            image = tensor.cpu().detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            image = np.clip((image+1)/2, 0, 1)  # Ensure pixel values are within [0, 1]
            return image

        # Create the figure
        fig, axes = plt.subplots(4, num_images, figsize=(num_images * 4, 9))

        for j in range(num_images):
            # Original image
            original_img = tensor_to_image(batch_images[j])
            axes[0,j].imshow(original_img)
            if j==0: axes[0,j].set_ylabel("Original")
            axes[0,j].axis("off")

            # Reconstruction
            recon_img = tensor_to_image(masked_images2[j])
            axes[3,j].imshow(recon_img)
            if j==0:axes[2,j].set_ylabel("Reconstruction")
            axes[2,j].axis("off")

            # Masked Reconstruction
            recon_img = tensor_to_image(mask[j]*reconstructions[j])
            axes[2,j].imshow(recon_img)
            if j==0:axes[3,j].set_ylabel("Target prediction")
            axes[3,j].axis("off")

            # Masked image
            masked_img = tensor_to_image(masked_images[j])
            axes[1,j].imshow(masked_img)
            if j==0:axes[1,j].set_ylabel("Model Input")
            axes[1,j].axis("off")

        # Adjust layout and save the image
        plt.tight_layout(rect=[0, 0, 0,0])  # Leave space for the title
        output_path = "/cluster/home/abizeul/mae/tools/reconstruction_comparison.png"
        plt.savefig(output_path, dpi=300)
        print(f"Saved image to {output_path}")
    break

loss/=len(dataset)
print("Loss",loss)

