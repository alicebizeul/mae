import os
import shutil

# Paths (modify as needed)
val_images_dir = "/cluster/scratch/abizeul/imagenet/val_fixed/ILSVRC2012"  # Path to validation images
val_map_file = "/cluster/scratch/abizeul/imagenet/val_fixed/val_map.txt"           # Path to val_map.txt
output_dir = "/cluster/scratch/abizeul/imagenet/val"              # Path where sorted images will be stored

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read validation image mappings
with open(val_map_file, "r") as f:
    lines = f.readlines()

# Process each image
for line in lines:
    print(line)
    img_name, class_idx = line.strip().split()
    img_name = img_name.split("/")[-1]  # Extract image name
    class_idx = int(class_idx)  # Ensure class index is an integer

    # Create class folder if it doesn't exist
    class_folder = os.path.join(output_dir, f"{class_idx:04d}")  # Padded class index (optional)
    os.makedirs(class_folder, exist_ok=True)

    # Move image to its corresponding class folder
    src_path = os.path.join(val_images_dir, img_name)
    dst_path = os.path.join(class_folder, img_name)

    print(src_path,dst_path)

    if os.path.exists(src_path):  # Ensure the file exists before moving
        shutil.move(src_path, dst_path)
    else:
        print(f"Warning: {src_path} not found.")

print("Done! Images have been sorted into class folders.")
