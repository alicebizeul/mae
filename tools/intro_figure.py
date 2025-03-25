import torchvision
import os
import numpy as np
from torchvision.transforms import ToPILImage
import torch
from PIL import Image
import random
# Define a folder to save the images
output_dir = "./output_images/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

denormalize = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
])
seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SingleClassImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        # idx = "178_"
        idx="og_0_0"
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.lower().endswith((f'{idx}.png', f'{idx}.jpg', f'{idx}.jpeg', f'{idx}.bmp', f'{idx}.tiff'))]
        print(self.image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure 3-channel RGB

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return image and a single class label (e.g., 0 for all images)
        label = 0  # Single class
        return image, label
        
# Transform to convert tensor to PIL image
to_pil = ToPILImage()
# data loader
resize = torchvision.transforms.RandomResizedCrop(size=224,scale=[0.8,1.0],interpolation=3)
# resize = torchvision.transforms.Resize(256),
# crop   = torchvision.transforms.CenterCrop(224), 
tensor = torchvision.transforms.ToTensor()
normalize = torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
mytransform = torchvision.transforms.Compose([tensor,normalize])
# dataset = SingleClassImageDataset("/cluster/scratch/abizeul/imagenet/train/n07753275/",transform=mytransform)
dataset = SingleClassImageDataset("/cluster/home/abizeul",transform=mytransform)

# dataset = SingleClassImageDataset("/cluster/scratch/abizeul/imagenet/train/n02113023/",transform=mytransform)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=1)

def patchify(pixel_values, interpolate_pos_encoding = False):
    """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.
        interpolate_pos_encoding (`bool`, *optional*, default `False`):
            interpolation flag passed during the forward pass.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
            Patchified pixel values.
    """
    patch_size, num_channels = 16, 3
    # sanity checks
    if not interpolate_pos_encoding and (
        pixel_values.shape[2] != pixel_values.shape[3] or pixel_values.shape[2] % patch_size != 0
    ):
        raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
    if pixel_values.shape[1] != num_channels:
        raise ValueError(
            "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
        )

    # patchify
    batch_size = pixel_values.shape[0]
    num_patches_h = pixel_values.shape[2] // patch_size
    num_patches_w = pixel_values.shape[3] // patch_size
    patchified_pixel_values = pixel_values.reshape(
        batch_size, num_channels, num_patches_h, patch_size, num_patches_w, patch_size
    )
    patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
    patchified_pixel_values = patchified_pixel_values.reshape(
        batch_size, num_patches_h * num_patches_w, patch_size**2 * num_channels
    )
    return patchified_pixel_values

def unpatchify(patchified_pixel_values, original_image_size= None):
    """
    Args:
        patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
            Patchified pixel values.
        original_image_size (`Tuple[int, int]`, *optional*):
            Original image size.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
            Pixel values.
    """
    patch_size, num_channels = 16, 3
    original_image_size = (
        original_image_size
        if original_image_size is not None
        else (224,224)
    )
    original_height, original_width = original_image_size
    num_patches_h = original_height // patch_size
    num_patches_w = original_width // patch_size

    # sanity check
    if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
        raise ValueError(
            f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
        )

    # unpatchify
    batch_size = patchified_pixel_values.shape[0]
    patchified_pixel_values = patchified_pixel_values.reshape(
        batch_size,
        num_patches_h,
        num_patches_w,
        patch_size,
        patch_size,
        num_channels,
    )
    patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
    pixel_values = patchified_pixel_values.reshape(
        batch_size,
        num_channels,
        num_patches_h * patch_size,
        num_patches_w * patch_size,
    )
    return pixel_values

def random_masking(sequence, noise=None):
    """
    Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
    noise.

    Args:
        sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
        noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
            mainly used for testing purposes to control randomness and maintain the reproducibility
    """
    batch_size, seq_length, dim = sequence.shape
    len_keep = int(seq_length * (1 - 0.75))

    if noise is None:
        noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

    # keep the first subset
    ids_keep, ids_drop = ids_shuffle[:, :len_keep], ids_shuffle[:, len_keep:]
    sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
    sequence_masked = torch.gather(sequence, dim=1, index=ids_drop.unsqueeze(-1).repeat(1, 1, dim))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([batch_size, seq_length], device=sequence.device)
    mask[:, :len_keep] = 0

    anti_mask = torch.ones([batch_size, seq_length], device=sequence.device)
    anti_mask[:, len_keep:] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    anti_mask = torch.gather(anti_mask, dim=1, index=ids_drop)

    return mask, anti_mask

print("loade",flush=True)
pc_matrix = np.load("/cluster/scratch/abizeul/pc_matrix_ipca.npy")
# pc_matrix = np.load("/cluster/scratch/abizeul/tiny/pc_matrix.npy")
# eigenvalues = np.load("/cluster/scratch/abizeul/tiny/eigenvalues_ratio.npy")
eigenvalues = np.load("/cluster/scratch/abizeul/eigenvalues_ratio_ipca.npy")


for i, (batch,_) in enumerate(dataloader):
    print("Starting",flush=True)
    for s in range(5):
        # index = torch.randperm(eigenvalues.shape[0]).numpy()
        index = torch.arange(eigenvalues.shape[0]).numpy()

        find_threshold = lambda eigenvalues ,ratio: np.argmin(np.abs(np.cumsum(eigenvalues) - ratio))
        threshold = find_threshold(eigenvalues[index],0.84)
        threshold2 = find_threshold(eigenvalues[index],0.94)
        print(threshold,threshold2)
        pc_mask, pc_unmask = pc_matrix[index[threshold:threshold2+1],:].T, pc_matrix[index[threshold2+1:],:].T

        mask,unmask = random_masking(patchify(batch))
        mask= unpatchify(mask.unsqueeze(-1).repeat(1, 1, 16*16*3))
        masked, unmasked = (-1*(mask-1))*batch,mask*batch

        # Save each image in the batch
        for j in range(unmasked.size(0)):  # Handle batch size
            unmasked_pil = to_pil(unmasked[j].cpu())  # Convert to PIL image
            masked_pil = to_pil(masked[j].cpu())  # Convert to PIL image

            # Save the images
            unmasked_pil.save(os.path.join(output_dir, f"unmasked_{i}_{j}.png"))
            masked_pil.save(os.path.join(output_dir, f"masked_{i}_{j}.png"))

        print(f"Batch {i} saved.",flush=True)  # Feedback to track progress

        #
        masked = torch.reshape(batch,[batch.shape[0],-1]) @ pc_mask @ pc_mask.T
        unmasked = torch.reshape(batch,[batch.shape[0],-1]) @ pc_unmask @ pc_unmask.T

        masked, unmasked = torch.reshape(masked,[masked.shape[0],3,224,224]),torch.reshape(unmasked,[unmasked.shape[0],3,224,224])
        sum_m_u = masked+unmasked
        # masked, unmasked = torchvision.transforms.Resize(size=224)(masked), torchvision.transforms.Resize(size=224)(unmasked)

        masked = torch.clamp(denormalize(masked), 0, 1)
        print(torch.max(denormalize(masked)),torch.min(denormalize(masked)),flush=True)

        unmasked = torch.clamp(denormalize(unmasked), 0, 1)
        summed = torch.clamp(denormalize(sum_m_u), 0, 1)
        print(torch.max(denormalize(summed)),torch.min(denormalize(summed)),flush=True)

        # masked, unmasked = (1+(masked-torch.mean(masked))/torch.std(masked)), (1+(unmasked-torch.mean(unmasked))/torch.std(unmasked))
        for j in range(unmasked.size(0)):  # Handle batch size
            og_pil = to_pil(batch[j].cpu())
            unmasked_pil = to_pil(unmasked[j].cpu())  # Convert to PIL image
            masked_pil = to_pil(masked[j].cpu())  # Convert to PIL image
            summed_pil = to_pil(summed[j].cpu())
            # Save the images
            unmasked_pil.save(os.path.join(output_dir, f"pc_unmasked_{i}_{j}_{s}.png"))
            masked_pil.save(os.path.join(output_dir, f"pc_masked_{i}_{j}_{s}.png"))
            summed_pil.save(os.path.join(output_dir, f"pc_summed_{i}_{j}_{s}.png"))
            og_pil.save(os.path.join(output_dir,f"og_{i}_{j}.png"))

        print(f"Batch {i} saved.")  # Feedback to track progress
    

