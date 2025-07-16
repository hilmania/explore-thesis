# train_eegnet_multi.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import json
import os

from eeg_model import EEGNet
from eeg_dataset_multi import EEGDatasetMulti

# --- Config ---
data_dir = 'data'  # Directory containing all HDF5 and CSV files
subjects = ['BM10', 'BM11', 'BM12']  # List of subjects
batch_size = 64
epochs = 100
learning_rate = 1e-3
save_path = 'best_model_multi.pth'
report_path = 'evaluation_report_multi.json'
patience = 10  # Early stopping patience

# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Load Datasets ---
print("Loading datasets...")
train_dataset = EEGDatasetMulti(data_dir, subjects, split='train')
val_dataset = EEGDatasetMulti(data_dir, subjects, split='validation')
test_dataset = EEGDatasetMulti(data_dir, subjects, split='testing')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# --- Model, Loss, Optimizer ---
# Get input shape from first sample
sample_eeg, _ = train_dataset[0]
input_channels = sample_eeg.shape[0]
print(f"Input channels: {input_channels}")

model = EEGNet(input_channels=input_channels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Training & Validation ---
best_val_acc = 0.0
early_stop_counter = 0
train_losses = []
val_accuracies = []

print("Starting training...")
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
    train_losses.append(avg_loss)
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
    val_accuracies.append(val_acc)
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

# --- Load Best Model for Testing ---
model.load_state_dict(torch.load(save_path))
model.eval()

# --- Test Evaluation ---
print("\nEvaluating on test set...")
y_true = []
y_pred = []
y_proba = []

with torch.no_grad():
    for eeg, labels in test_loader:
        eeg = eeg.to(device)
        outputs = model(eeg)
        probabilities = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())
        y_proba.extend(probabilities.cpu().numpy())

# --- Metrics ---
report = classification_report(y_true, y_pred, digits=4, target_names=["No Seizure", "Seizure"], output_dict=True)
print("\nTest Classification Report:")
print(json.dumps(report, indent=2))

# Calculate per-subject performance
print("\nPer-subject performance:")
test_subject_indices = []
for i in range(len(test_dataset)):
    subject = test_dataset.get_subject_info(i)
    test_subject_indices.append(subject)

for subject in subjects:
    subject_mask = [i for i, s in enumerate(test_subject_indices) if s == subject]
    if subject_mask:
        subject_y_true = [y_true[i] for i in subject_mask]
        subject_y_pred = [y_pred[i] for i in subject_mask]
        subject_acc = sum([1 for i, j in zip(subject_y_true, subject_y_pred) if i == j]) / len(subject_y_true)
        print(f"{subject}: {len(subject_mask)} samples, Accuracy: {subject_acc:.4f}")

# Save comprehensive report
final_report = {
    'test_classification_report': report,
    'best_validation_accuracy': best_val_acc,
    'training_config': {
        'subjects': subjects,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs_trained': epoch + 1,
        'input_channels': input_channels
    },
    'training_history': {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }
}

with open(report_path, 'w') as f:
    json.dump(final_report, f, indent=2)
    print(f"📄 Comprehensive evaluation report saved to {report_path}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Seizure", "Seizure"],
            yticklabels=["No Seizure", "Seizure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Multi-Subject Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix_multi.png", dpi=300)
plt.show()

# --- Training History Plot ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig("training_history_multi.png", dpi=300)
plt.show()

print("🖼️ Plots saved: confusion_matrix_multi.png, training_history_multi.png")
print(f"🎯 Final Test Accuracy: {sum([1 for i, j in zip(y_true, y_pred) if i == j]) / len(y_true):.4f}")
