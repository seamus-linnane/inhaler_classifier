import os
import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

# Paths
base_path = "./data/processed/"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "validate")
model_save_path = "./models/"
classes_file = "./data/classes.txt"

# Ensure directories exist
if not os.path.exists(train_path) or not os.path.exists(val_path):
    print("Error: Train or validation data not found. Please ensure prep_data.py and standardise_images.py have been run.")
    exit()

if not os.path.exists(classes_file):
    print(f"Error: '{classes_file}' not found. Please run 'class_counter.py' first.")
    exit()

# Load class details from classes.txt
try:
    with open(classes_file, "r") as file:
        class_details = dict(line.strip().split(": ") for line in file)
except ValueError:
    print("Error: Unexpected formatting in classes.txt. Please verify the file.")
    exit()

# Convert numeric values to integers where applicable
for key, value in class_details.items():
    if value.isdigit():
        class_details[key] = int(value)

# Get the number of classes from classes.txt
num_classes = class_details.get("class_number")
if not num_classes:
    print("Error: 'class_number' not found in classes.txt.")
    exit()

# Print the class details for verification
print(f"\nNumber of classes: {num_classes}")
for i in range(1, num_classes + 1):
    class_name = class_details.get(f"class_{i}_name")
    class_count = class_details.get(f"class_{i}_count")
    print(f"Class {i}: {class_name} ({class_count} items)")

# Set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    print("MPS device not found.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the last layer of the model
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
print(f"Loaded ResNet-18 model with {num_classes} output classes.")

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the train and validation datasets
train_dataset = ImageFolder(train_path, transform=transform)
val_dataset = ImageFolder(val_path, transform=transform)

# Print dataset sizes
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Create data loaders for the train and validation datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define the training function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # ✅ FIX: Ensure float32 for MPS compatibility
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.float() / len(train_dataset)  # ✅ Use .float()
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Validation phase
        model.eval()
        running_loss, running_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.float() / len(val_dataset)  # ✅ Use .float()
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    # Plot losses and accuracies
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy", marker="o")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Val Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Move the model to the specified device
model.to(device)

# Fine-tune the last layer for a few epochs
train(model, train_loader, val_loader, criterion, optimizer, num_epochs=6)

# Unfreeze all layers for full fine-tuning
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Train the entire model
train(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Save the fine-tuned model with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_file = os.path.join(model_save_path, f"finetuned_model_{timestamp}.pth")
torch.save(model.state_dict(), model_save_file)
print(f"Model saved to {model_save_file}")