"""
Cross-validation script for multi-subject EEG seizure prediction
Supports subject-independent and subject-dependent evaluation strategies
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import json
import os
from itertools import combinations

from eeg_model import EEGNet
from eeg_dataset_multi import EEGDatasetMulti

class SubjectCrossValidator:
    def __init__(self, data_dir, subjects, device='auto'):
        self.data_dir = data_dir
        self.subjects = subjects
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.results = {}

    def subject_independent_cv(self, batch_size=64, epochs=50, lr=1e-3):
        """
        Leave-One-Subject-Out Cross Validation
        Train on 2 subjects, test on 1 subject
        """
        print("🔄 Subject-Independent Cross-Validation (LOSO)")
        print("=" * 60)

        results = {}

        for test_subject in self.subjects:
            train_subjects = [s for s in self.subjects if s != test_subject]

            print(f"\n📋 Fold: Test on {test_subject}, Train on {train_subjects}")

            # Load datasets
            train_data = self._combine_datasets(train_subjects, ['train', 'validation'])
            test_data = self._load_dataset([test_subject], 'testing')

            # Train model
            model, history = self._train_model(train_data, test_data, epochs, lr, batch_size)

            # Evaluate
            metrics = self._evaluate_model(model, test_data)

            results[test_subject] = {
                'train_subjects': train_subjects,
                'test_subject': test_subject,
                'metrics': metrics,
                'training_history': history
            }

            print(f"✅ {test_subject} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

        # Calculate average performance
        avg_accuracy = np.mean([r['metrics']['accuracy'] for r in results.values()])
        avg_auc = np.mean([r['metrics']['auc'] for r in results.values()])

        print(f"\n🎯 Average Performance:")
        print(f"   Accuracy: {avg_accuracy:.4f} ± {np.std([r['metrics']['accuracy'] for r in results.values()]):.4f}")
        print(f"   AUC: {avg_auc:.4f} ± {np.std([r['metrics']['auc'] for r in results.values()]):.4f}")

        self.results['subject_independent'] = {
            'individual_results': results,
            'average_accuracy': avg_accuracy,
            'average_auc': avg_auc,
            'std_accuracy': np.std([r['metrics']['accuracy'] for r in results.values()]),
            'std_auc': np.std([r['metrics']['auc'] for r in results.values()])
        }

        return results

    def subject_dependent_evaluation(self, batch_size=64, epochs=50, lr=1e-3):
        """
        Subject-Dependent Evaluation
        Train and test on the same subject using train/val/test splits
        """
        print("\n🔄 Subject-Dependent Evaluation")
        print("=" * 60)

        results = {}

        for subject in self.subjects:
            print(f"\n📋 Subject: {subject}")

            # Load subject-specific datasets
            train_dataset = EEGDatasetMulti(self.data_dir, [subject], split='train')
            val_dataset = EEGDatasetMulti(self.data_dir, [subject], split='validation')
            test_dataset = EEGDatasetMulti(self.data_dir, [subject], split='testing')

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Train model
            model, history = self._train_model_with_validation(
                train_loader, val_loader, epochs, lr
            )

            # Evaluate on test set
            metrics = self._evaluate_model(model, test_loader)

            results[subject] = {
                'metrics': metrics,
                'training_history': history,
                'dataset_sizes': {
                    'train': len(train_dataset),
                    'validation': len(val_dataset),
                    'test': len(test_dataset)
                }
            }

            print(f"✅ {subject} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

        # Calculate average performance
        avg_accuracy = np.mean([r['metrics']['accuracy'] for r in results.values()])
        avg_auc = np.mean([r['metrics']['auc'] for r in results.values()])

        print(f"\n🎯 Average Subject-Dependent Performance:")
        print(f"   Accuracy: {avg_accuracy:.4f} ± {np.std([r['metrics']['accuracy'] for r in results.values()]):.4f}")
        print(f"   AUC: {avg_auc:.4f} ± {np.std([r['metrics']['auc'] for r in results.values()]):.4f}")

        self.results['subject_dependent'] = {
            'individual_results': results,
            'average_accuracy': avg_accuracy,
            'average_auc': avg_auc,
            'std_accuracy': np.std([r['metrics']['accuracy'] for r in results.values()]),
            'std_auc': np.std([r['metrics']['auc'] for r in results.values()])
        }

        return results

    def _combine_datasets(self, subjects, splits):
        """Combine multiple subjects and splits into one dataset"""
        all_data = []
        all_labels = []

        for split in splits:
            dataset = EEGDatasetMulti(self.data_dir, subjects, split=split)
            for i in range(len(dataset)):
                data, label = dataset[i]
                all_data.append(data)
                all_labels.append(label)

        return list(zip(all_data, all_labels))

    def _load_dataset(self, subjects, split):
        """Load dataset as list of (data, label) tuples"""
        dataset = EEGDatasetMulti(self.data_dir, subjects, split=split)
        return [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))]

    def _train_model(self, train_data, val_data, epochs, lr, batch_size):
        """Train model with given data"""
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        return self._train_model_with_validation(train_loader, val_loader, epochs, lr)

    def _train_model_with_validation(self, train_loader, val_loader, epochs, lr):
        """Train model with validation"""
        # Get input shape
        sample_data, _ = next(iter(train_loader))
        input_channels = sample_data.shape[1]

        # Initialize model
        model = EEGNet(input_channels=input_channels).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training history
        history = {'train_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(epochs):
            # Training
            model.train()
            total_loss = 0
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    outputs = model(batch_data)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == batch_labels).sum().item()
                    total += batch_labels.size(0)

            val_acc = correct / total
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Load best model
        model.load_state_dict(best_model_state)
        return model, history

    def _evaluate_model(self, model, data_loader_or_list):
        """Evaluate model on test data"""
        model.eval()

        # Handle both DataLoader and list inputs
        if isinstance(data_loader_or_list, list):
            # Convert list to DataLoader
            data_loader = DataLoader(data_loader_or_list, batch_size=64, shuffle=False)
        else:
            data_loader = data_loader_or_list

        y_true = []
        y_pred = []
        y_proba = []

        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(self.device)
                outputs = model(batch_data)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                y_true.extend(batch_labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_proba.extend(probs.cpu().numpy()[:, 1])  # Probability of seizure class

        # Calculate metrics
        accuracy = sum([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)
        auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else 0.5

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        return {
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

    def save_results(self, filename='cv_results.json'):
        """Save all results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Results saved to {filename}")

    def plot_results(self):
        """Create visualizations of results"""
        if 'subject_independent' in self.results:
            self._plot_subject_independent_results()

        if 'subject_dependent' in self.results:
            self._plot_subject_dependent_results()

    def _plot_subject_independent_results(self):
        """Plot subject-independent CV results"""
        results = self.results['subject_independent']['individual_results']

        subjects = list(results.keys())
        accuracies = [results[s]['metrics']['accuracy'] for s in subjects]
        aucs = [results[s]['metrics']['auc'] for s in subjects]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy plot
        bars1 = ax1.bar(subjects, accuracies, color='skyblue', alpha=0.7)
        ax1.axhline(np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.3f}')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Subject-Independent CV - Accuracy')
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

        # AUC plot
        bars2 = ax2.bar(subjects, aucs, color='lightcoral', alpha=0.7)
        ax2.axhline(np.mean(aucs), color='red', linestyle='--', label=f'Mean: {np.mean(aucs):.3f}')
        ax2.set_ylabel('AUC')
        ax2.set_title('Subject-Independent CV - AUC')
        ax2.legend()
        ax2.set_ylim(0, 1)

        # Add value labels on bars
        for bar, auc in zip(bars2, aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('subject_independent_cv_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_subject_dependent_results(self):
        """Plot subject-dependent results"""
        results = self.results['subject_dependent']['individual_results']

        subjects = list(results.keys())
        accuracies = [results[s]['metrics']['accuracy'] for s in subjects]
        aucs = [results[s]['metrics']['auc'] for s in subjects]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy plot
        bars1 = ax1.bar(subjects, accuracies, color='lightgreen', alpha=0.7)
        ax1.axhline(np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.3f}')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Subject-Dependent Evaluation - Accuracy')
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

        # AUC plot
        bars2 = ax2.bar(subjects, aucs, color='gold', alpha=0.7)
        ax2.axhline(np.mean(aucs), color='red', linestyle='--', label=f'Mean: {np.mean(aucs):.3f}')
        ax2.set_ylabel('AUC')
        ax2.set_title('Subject-Dependent Evaluation - AUC')
        ax2.legend()
        ax2.set_ylim(0, 1)

        # Add value labels on bars
        for bar, auc in zip(bars2, aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('subject_dependent_results.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data"
    SUBJECTS = ['BM10', 'BM11', 'BM12']

    print("🧠 EEG Multi-Subject Cross-Validation")
    print("=" * 60)

    # Initialize cross-validator
    cv = SubjectCrossValidator(DATA_DIR, SUBJECTS)

    # Run subject-independent cross-validation
    print("Starting Subject-Independent Cross-Validation...")
    cv.subject_independent_cv(batch_size=32, epochs=30, lr=1e-3)

    # Run subject-dependent evaluation
    print("Starting Subject-Dependent Evaluation...")
    cv.subject_dependent_evaluation(batch_size=32, epochs=30, lr=1e-3)

    # Save results and create plots
    cv.save_results('comprehensive_cv_results.json')
    cv.plot_results()

    print("\n✅ Cross-validation completed!")
