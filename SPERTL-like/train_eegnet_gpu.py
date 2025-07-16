# train_eegnet.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import json
import os

from eeg_model import EEGNet
from eeg_dataset import EEGDatasetH5

# --- Config ---
h5_path = 'data/eeg_file.h5'
batch_size = 64
epochs = 100
learning_rate = 1e-3
val_split = 0.2
save_path = 'best_model.pth'
report_path = 'evaluation_report.json'
patience = 10  # Early stopping patience

# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Load Dataset ---
dataset = EEGDatasetH5(h5_path)
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- Model, Loss, Optimizer ---
model = EEGNet(input_channels=20).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Training & Validation ---
best_val_acc = 0.0
early_stop_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_eeg, batch_labels in progress_bar:
        batch_eeg, batch_labels = batch_eeg.to(device), batch_labels.to(device)

        outputs = model(batch_eeg)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    # Validation Accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for eeg, labels in val_loader:
            eeg, labels = eeg.to(device), labels.to(device)
            outputs = model(eeg)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print("✅ Saved Best Model")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# --- Load Best Model ---
model.load_state_dict(torch.load(save_path))
model.eval()

# --- Evaluation ---
y_true = []
y_pred = []

with torch.no_grad():
    for eeg, labels in val_loader:
        eeg = eeg.to(device)
        outputs = model(eeg)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# --- Metrics ---
report = classification_report(y_true, y_pred, digits=4, target_names=["No Seizure", "Seizure"], output_dict=True)
print("\nClassification Report:")
print(json.dumps(report, indent=2))

# Save report to file
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
    print(f"📄 Evaluation report saved to {report_path}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Seizure", "Seizure"], yticklabels=["No Seizure", "Seizure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("🖼️ Confusion matrix saved as 'confusion_matrix.png'")
