import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model save directory
os.makedirs("models", exist_ok=True)

# Dataset directories
train_folder = "./data/processed/train"
test_folder = "./data/processed/test"

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset loading
train_dataset = datasets.ImageFolder(train_folder, transform=transform)
# test_dataset = datasets.ImageFolder(test_folder, transform=transform)

# Split into train/val
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128)
# test_loader = DataLoader(test_dataset, batch_size=128)

# Model init
def initialize_model(model_name, num_classes, feature_extract=True):
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
    else:
        raise ValueError("Unsupported model")

    for param in model.parameters():
        param.requires_grad = feature_extract

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Training loop
def train_model(model, train_loader, val_loader, epochs=3, save_path="best_model.pth", save_all_models=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_accuracy = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_preds = []
        train_all_labels = []

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", ncols=100)
        for _, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_all_preds.extend(preds.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({"Loss": loss.item()})

        train_accuracy = train_correct / train_total
        train_precision = precision_score(train_all_labels, train_all_preds, average='macro')
        train_recall = recall_score(train_all_labels, train_all_preds, average='macro')
        train_f1 = f1_score(train_all_labels, train_all_preds, average='macro')

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        val_accuracy = val_correct / val_total
        val_precision = precision_score(val_all_labels, val_all_preds, average='macro')
        val_recall = recall_score(val_all_labels, val_all_preds, average='macro')
        val_f1 = f1_score(val_all_labels, val_all_preds, average='macro')

        print(f"[Epoch {epoch+1}] Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        if save_all_models:
            epoch_path = save_path.replace(".pth", f"_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_path)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Best model updated at epoch {epoch+1}")

# Optional evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(f"Precision: {precision_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"F1 Score:  {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")

# Run training
num_classes = len(train_dataset.classes)
resnet18_model = initialize_model("resnet18", num_classes).to(device)
resnet34_model = initialize_model("resnet34", num_classes).to(device)

train_model(resnet18_model, train_loader, val_loader, save_path="models/resnet18_best.pth")
train_model(resnet34_model, train_loader, val_loader, save_path="models/resnet34_best.pth")

# To evaluate:
# test_dataset = datasets.ImageFolder(test_folder, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=128)
# resnet18_model.load_state_dict(torch.load("models/resnet18_best.pth"))
# evaluate_model(resnet18_model, test_loader)
