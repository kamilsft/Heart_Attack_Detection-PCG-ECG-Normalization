import os
import shutil

# This script copies all folders containing "HEA" in their names from the source directory to the target directory.
# Example of target root path: HAD-Normalization\HAD-PCG-ECG-normalization\CycleGAN\data\coimbra_healthy
# Example of source root path: HAD-Normalization\HAD-PCG-ECG-normalization\coimbra
source_root = r"YOur_PATH_HERE"  # Replace with your source root path
target_root = r"YOUR_PATH_HERE"  # Replace with your target root path

# Looping through all folders in the source directory 
for folder_name in os.listdir(source_root):
    full_path = os.path.join(source_root, folder_name)
    if os.path.isdir(full_path) and "HEA" in folder_name.upper(): # change the HEA to CVD if you wanna copy unhealthy folders 
        target_path = os.path.join(target_root, folder_name)
        print(f"Copying: {folder_name}")
        shutil.copytree(full_path, target_path, dirs_exist_ok=True)

print("Done.")

