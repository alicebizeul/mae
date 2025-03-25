import os
import tarfile

# Set the path to your folder containing .tar files
tar_folder = "/cluster/scratch/abizeul/imagenet/train"
output_folder = "/cluster/scratch/abizeul/imagenet/train_untar"  # Set output folder for extracted files

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all .tar files in the folder
for filename in os.listdir(tar_folder):
    if filename.endswith(".tar"):  # Check if file is a .tar archive
        tar_path = os.path.join(tar_folder, filename)
        extract_path = os.path.join(output_folder, filename.replace(".tar", ""))  # Create subfolder for each .tar

        # Extract the .tar file
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_path)
            print(f"Extracted: {tar_path} → {extract_path}")

print("All .tar files have been extracted!")
