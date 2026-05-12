import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


DATA_DIR = "data/splits"
MODEL_SAVE_PATH = "models/individual_classifier_resnet50_augmented.pth"
CONFUSION_MATRIX_PATH = "models/individual_classifier_confusion_matrix_augmented.png"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224
RANDOM_SEED = 42


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Transforms

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# LOAD fixed split datasets

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transforms
)

valid_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "valid"),
    transform=eval_transforms
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "test"),
    transform=eval_transforms
)

class_names = train_dataset.classes

print("\nClasses:")
print(class_names)

print("\nClass counts:")

for split_name, dataset in [
    ("Train", train_dataset),
    ("Valid", valid_dataset),
    ("Test", test_dataset)
]:
    print(f"\n{split_name}:")
    targets = dataset.targets
    for class_index, class_name in enumerate(class_names):
        print(f"{class_name}: {targets.count(class_index)}")

print("\nSplit sizes:")
print(f"Train: {len(train_dataset)}")
print(f"Valid: {len(valid_dataset)}")
print(f"Test:  {len(test_dataset)}")

# Dataloaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Model

model = models.resnet50(weights="DEFAULT")

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model = model.to(device)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Evaluation function

def evaluate_model(model, dataloader, split_name):
    model.eval()

    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = correct / total if total > 0 else 0

    print(f"\n{split_name} Accuracy: {accuracy:.4f}")

    print(f"\n{split_name} Classification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    ))

    return all_labels, all_preds, accuracy

# Training loop

for epoch in range(NUM_EPOCHS):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
    print(f"Training Loss: {avg_loss:.4f}")

    evaluate_model(model, valid_loader, "Validation")

# Final evaluation on test set

test_labels, test_preds, test_accuracy = evaluate_model(
    model,
    test_loader,
    "Test"
)

# Confusion matrix

cm = confusion_matrix(test_labels, test_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Individual Seal Classifier - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
plt.close()

print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_PATH}")

# Save model

torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"\nModel saved to: {MODEL_SAVE_PATH}")