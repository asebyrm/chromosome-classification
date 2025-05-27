import os
import random
import shutil

# Source directory: where per-class folders are located
source_folder = "./data/processed/chromosomes"

# Target directories
train_folder = "./data/processed/train"
test_folder = "./data/processed/test"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Split ratio
train_ratio = 0.8

# Process each chromosome class folder
for chromosome_class in os.listdir(source_folder):
    class_path = os.path.join(source_folder, chromosome_class)
    if not os.path.isdir(class_path):
        continue

    # List and shuffle files
    files = os.listdir(class_path)
    random.shuffle(files)

    # Compute split
    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # Class name from folder name (e.g., chromosome_12 → 12)
    class_name = chromosome_class.split("_")[-1]

    # Create class folders in train/test
    train_class_dir = os.path.join(train_folder, class_name)
    test_class_dir = os.path.join(test_folder, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy files
    for file in train_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(train_class_dir, file))
    for file in test_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(test_class_dir, file))

print("[INFO] Dataset successfully split into train and test folders.")
