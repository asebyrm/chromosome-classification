import cv2
import numpy as np
import os

# Define paths for images and annotations
train_folder = "./data/raw/train"
annotation_folder = "./outputs/annotations/"
os.makedirs(annotation_folder, exist_ok=True)

for filename in os.listdir(train_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(train_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[ERROR] Could not read image: {image_path}")
            continue

        # Get the corresponding annotation file (.txt)
        annotation_path = os.path.splitext(image_path)[0] + ".txt"
        if not os.path.exists(annotation_path):
            print(f"[WARNING] Annotation not found for: {filename}. Skipping.")
            continue

        # Read annotation data
        with open(annotation_path, "r") as file:
            annotations = file.readlines()

        h, w, _ = image.shape

        for line in annotations:
            data = line.strip().split()
            chromosome_id = data[0]
            angle = float(data[1])  # Not used
            coords = np.array(
                [(float(data[i]) * w, float(data[i + 1]) * h) for i in range(2, len(data), 2)],
                dtype=np.int32
            ).reshape((-1, 1, 2))

            # Draw polygon outline for chromosome
            cv2.polylines(image, [coords], isClosed=True, color=(255, 0, 0), thickness=2)

            # Draw chromosome ID label
            x, y = coords[0][0]
            cv2.putText(image, chromosome_id, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Save annotated image
        output_path = os.path.join(annotation_folder, f"{os.path.splitext(filename)[0]}_annotated.jpg")
        cv2.imwrite(output_path, image)
        print(f"[INFO] Saved: {output_path}")
