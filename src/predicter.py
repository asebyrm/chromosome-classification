import os
import torch
from torchvision import transforms
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from PIL import Image
import numpy as np
import cv2

# Load model
def load_best_model(model_name, model_path, num_classes):
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif model_name == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
    else:
        raise ValueError("Model must be 'resnet18' or 'resnet34'")

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Predict class from image
def predict(image, model, transform):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Annotate image with chromosome outlines
def annotate_image(image, annotations):
    h, w, _ = image.shape
    for ann in annotations:
        parts = ann.strip().split()
        chromosome_id = parts[0]
        coords = np.array([
            (float(parts[i]) * w, float(parts[i + 1]) * h)
            for i in range(2, len(parts), 2)
        ], dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(image, [coords], isClosed=True, color=(255, 0, 0), thickness=2)
        x, y = coords[0][0]
        cv2.putText(image, chromosome_id, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return image

# Process all images and annotate predictions
def update_chromosome_predictions(image_folder, output_txt_folder, output_img_folder):
    os.makedirs(output_txt_folder, exist_ok=True)
    os.makedirs(output_img_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(image_folder, filename)
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        output_txt_path = os.path.join(output_txt_folder, os.path.splitext(filename)[0] + ".txt")
        output_img_path = os.path.join(output_img_folder, os.path.splitext(filename)[0] + "_annotated.jpg")

        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            continue

        if not os.path.exists(txt_path):
            print(f"[ERROR] Missing annotation: {txt_path}")
            continue

        chromosomes = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    chromosomes.append(parts)
                else:
                    print(f"[WARNING] Skipping malformed line: {line.strip()}")

        updated_chromosomes = []
        for chrom in chromosomes:
            angle = float(chrom[1])
            h, w, _ = image.shape
            coords = np.array([
                [float(chrom[i]) * w, float(chrom[i + 1]) * h]
                for i in range(2, len(chrom), 2)
            ], dtype=np.float32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [coords.astype(np.int32)], 255)
            masked_image = image.copy()
            masked_image[mask == 0] = 200

            center = np.mean(coords, axis=0)
            rot_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
            rotated = cv2.warpAffine(masked_image, rot_matrix, (w, h))
            rotated_coords = cv2.transform(np.array([coords]), rot_matrix)[0]

            x_min, y_min = np.min(rotated_coords, axis=0).astype(int)
            x_max, y_max = np.max(rotated_coords, axis=0).astype(int)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            cropped = rotated[y_min:y_max, x_min:x_max]

            if cropped is None or cropped.size == 0:
                print(f"[ERROR] Cropped image is empty. Skipping.")
                continue

            predicted_class = predict(cropped, model, transform)
            updated_line = [str(predicted_class)] + chrom[1:]
            updated_chromosomes.append(" ".join(updated_line))

        with open(output_txt_path, "w") as f:
            for line in updated_chromosomes:
                f.write(line + "\n")

        annotated = annotate_image(image.copy(), updated_chromosomes)
        cv2.imwrite(output_img_path, annotated)
        print(f"[INFO] Saved: {output_txt_path}, {output_img_path}")

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 24
model_path = "./models/resnet18_best_epoch3.pth"
model_name = "resnet18"

# Load model
model = load_best_model(model_name, model_path, num_classes)

# Define folders
image_folder = "./data/raw/test"
prediction_folder = "./outputs/predicted_labels"
annotation_folder = "./outputs/annotated_images"

# Run prediction + annotation
update_chromosome_predictions(image_folder, prediction_folder, annotation_folder)
