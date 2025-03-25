import os
import shutil
import json

# Paths
val_images_dir = "/cluster/scratch/abizeul/val"
ground_truth_file = "/cluster/scratch/abizeul/ILSVRC2012_validation_ground_truth.txt"
imagenet_class_index = "/cluster/scratch/abizeul/imagenet_class_index.json"

# Load class mapping from index to class folder names
with open(imagenet_class_index, 'r') as f:
    class_idx = json.load(f)

# Create dict from index (int) to folder name (nXXXXXXXX)
idx_to_class = {int(k): v[0] for k, v in class_idx.items()}

# Read ground truth labels (indices)
with open(ground_truth_file, 'r') as f:
    labels = [int(line.strip()) for line in f.readlines()]

# List and sort image filenames to match order in labels file
image_files = sorted([fname for fname in os.listdir(val_images_dir) if fname.endswith(".JPEG")])

assert len(image_files) == len(labels), "Mismatch in number of images and labels."

# Move images to correct class folders
for fname, label in zip(image_files, labels):
    class_folder = idx_to_class[label - 1]  # labels are 1-indexed
    target_folder = os.path.join(val_images_dir, class_folder)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    source = os.path.join(val_images_dir, fname)
    destination = os.path.join(target_folder, fname)
    print(source,destination)
    shutil.move(source, destination)

print("Validation set successfully organized into class-specific folders.")
