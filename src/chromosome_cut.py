import numpy as np
import cv2
import os

# Define folders
train_folder = "./data/raw/train"
chromosomes_folder = "./data/processed/chromosomes"

# Iterate over image files
for filename in os.listdir(train_folder):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(train_folder, filename)
    txt_path = os.path.splitext(image_path)[0] + ".txt"

    if not os.path.exists(txt_path):
        print(f"[WARNING] Annotation file not found for: {filename}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        continue

    # Read annotations
    chromosomes = []
    with open(txt_path, "r") as file:
        for line in file:
            chromosomes.append(line.strip().split())

    for chromosome in chromosomes:
        chromosome_number = chromosome[0]
        angle = float(chromosome[1])
        h, w, _ = image.shape

        # Convert normalized coordinates to absolute pixel coordinates
        coordinates = np.array([
            [float(chromosome[i]) * w, float(chromosome[i + 1]) * h]
            for i in range(2, len(chromosome), 2)
        ], dtype=np.float32)

        # Grayscale + Laplacian filter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        laplacian = np.uint8(np.absolute(laplacian))

        # Mask background
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [coordinates.astype(np.int32)], 255)
        masked_image = image.copy()
        masked_image[mask == 0] = 200  # Gray background for outside area

        # Rotate the image
        center = np.mean(coordinates, axis=0)
        rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        rotated_image = cv2.warpAffine(masked_image, rotation_matrix, (w, h))

        # Transform coordinates with rotation matrix
        rotated_coords = cv2.transform(np.array([coordinates]), rotation_matrix)[0]
        x_min, y_min = np.min(rotated_coords, axis=0).astype(int)
        x_max, y_max = np.max(rotated_coords, axis=0).astype(int)

        # Clamp coordinates within image bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        # Crop the rotated chromosome region
        cropped = rotated_image[y_min:y_max, x_min:x_max]

        # Save cropped image into chromosome-specific folder
        save_folder = os.path.join(chromosomes_folder, f"chromosome_{chromosome_number}")
        os.makedirs(save_folder, exist_ok=True)

        file_base = os.path.splitext(filename)[0]
        counter = 1
        save_path = os.path.join(save_folder, f"{file_base}_{counter}.jpg")
        while os.path.exists(save_path):
            counter += 1
            save_path = os.path.join(save_folder, f"{file_base}_{counter}.jpg")

        if cropped is not None and cropped.size > 0:
            cv2.imwrite(save_path, cropped)
            print(f"[INFO] Saved: {save_path}")
        else:
            print(f"[ERROR] Failed to crop chromosome {chromosome_number} in {filename}")
