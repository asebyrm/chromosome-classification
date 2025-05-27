import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Compare ground truth and predicted chromosome labels from .txt files
def compare_chromosome_numbers(original_folder, predicted_folder):
    y_true = []
    y_pred = []

    for filename in os.listdir(original_folder):
        if filename.endswith(".txt"):
            orig_path = os.path.join(original_folder, filename)
            pred_path = os.path.join(predicted_folder, filename)

            if not os.path.exists(pred_path):
                print(f"[WARNING] Prediction missing for: {filename}. Skipping.")
                continue

            with open(orig_path, "r") as f_orig, open(pred_path, "r") as f_pred:
                orig_lines = f_orig.readlines()
                pred_lines = f_pred.readlines()

                if len(orig_lines) != len(pred_lines):
                    print(f"[WARNING] Line count mismatch in: {filename}. Skipping.")
                    continue

                for o_line, p_line in zip(orig_lines, pred_lines):
                    o_parts = o_line.strip().split()
                    p_parts = p_line.strip().split()
                    if len(o_parts) < 1 or len(p_parts) < 1:
                        print(f"[WARNING] Malformed line in: {filename}. Skipping line.")
                        continue

                    y_true.append(int(o_parts[0]))
                    y_pred.append(int(p_parts[0]))

    return y_true, y_pred

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="viridis", colorbar=True)
    plt.title("Chromosome Classification - Confusion Matrix")
    plt.grid(False)
    plt.show()

# Define folders (adjust to your structure)
original_folder = "./outputs/true_labels"
predicted_folder = "./outputs/predicted_labels"

# Run evaluation
y_true, y_pred = compare_chromosome_numbers(original_folder, predicted_folder)

# Label range: 0–23 (for chromosomes 1–22, X, Y)
class_labels = list(range(24))

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, class_labels)
