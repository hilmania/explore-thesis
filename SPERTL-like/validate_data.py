"""
Data preparation and validation script for multi-subject EEG data
This script helps validate your data structure and provides utilities for data preparation
"""

import os
import h5py
import pandas as pd
import numpy as np
import json

def validate_data_structure(data_dir, subjects, splits=['train', 'validation', 'testing']):
    """
    Validate that all required files exist and have consistent structure

    Args:
        data_dir (str): Directory containing the data files
        subjects (list): List of subject IDs (e.g., ['BM10', 'BM11', 'BM12'])
        splits (list): List of data splits to check

    Returns:
        dict: Validation report
    """
    report = {
        'missing_files': [],
        'file_info': {},
        'data_shapes': {},
        'label_distributions': {},
        'total_samples': {}
    }

    print("🔍 Validating data structure...")

    for split in splits:
        report['total_samples'][split] = 0
        report['data_shapes'][split] = []
        report['label_distributions'][split] = {}

        for subject in subjects:
            h5_file = os.path.join(data_dir, f"{subject}_{split}.h5")
            csv_file = os.path.join(data_dir, f"{subject}_{split}.csv")

            # Check if files exist
            if not os.path.exists(h5_file):
                report['missing_files'].append(h5_file)
                print(f"❌ Missing: {h5_file}")
                continue

            if not os.path.exists(csv_file):
                report['missing_files'].append(csv_file)
                print(f"❌ Missing: {csv_file}")
                continue

            # Validate HDF5 file
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'eeg' not in f:
                        print(f"❌ HDF5 file {h5_file} missing 'eeg' dataset")
                        continue

                    eeg_shape = f['eeg'].shape
                    report['data_shapes'][split].append({
                        'subject': subject,
                        'eeg_shape': eeg_shape
                    })

                    print(f"✅ {subject}_{split}.h5: EEG shape {eeg_shape}")

            except Exception as e:
                print(f"❌ Error reading {h5_file}: {e}")
                continue

            # Validate CSV file
            try:
                df = pd.read_csv(csv_file)

                # Find label column
                label_col = None
                if 'label' in df.columns:
                    label_col = 'label'
                elif 'labels' in df.columns:
                    label_col = 'labels'
                else:
                    label_col = df.columns[0]
                    print(f"⚠️  Using first column '{label_col}' as labels for {csv_file}")

                labels = df[label_col].values
                num_samples = len(labels)
                report['total_samples'][split] += num_samples

                # Label distribution
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
                report['label_distributions'][split][subject] = {
                    'distribution': label_dist,
                    'total_samples': num_samples
                }

                print(f"✅ {subject}_{split}.csv: {num_samples} samples, labels: {label_dist}")

                # Check if EEG and label counts match
                if eeg_shape[0] != num_samples:
                    print(f"⚠️  Sample count mismatch: EEG {eeg_shape[0]} vs Labels {num_samples}")

            except Exception as e:
                print(f"❌ Error reading {csv_file}: {e}")
                continue

    # Summary
    print("\n📊 Summary:")
    for split in splits:
        total = report['total_samples'][split]
        print(f"{split.capitalize()}: {total} total samples")

        # Overall label distribution for this split
        all_labels = []
        for subject_data in report['label_distributions'][split].values():
            for label, count in subject_data['distribution'].items():
                all_labels.extend([label] * count)

        if all_labels:
            unique, counts = np.unique(all_labels, return_counts=True)
            split_dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"  Label distribution: {split_dist}")

    if report['missing_files']:
        print(f"\n❌ {len(report['missing_files'])} missing files found!")
    else:
        print("\n✅ All files present and validated!")

    return report

def create_data_summary(data_dir, subjects, output_file='data_summary.json'):
    """Create a comprehensive data summary"""
    splits = ['train', 'validation', 'testing']
    report = validate_data_structure(data_dir, subjects, splits)

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 Data summary saved to {output_file}")
    return report

def prepare_sample_structure():
    """Create sample directory structure and instructions"""
    print("\n📁 Expected data structure:")
    print("data/")

    subjects = ['BM10', 'BM11', 'BM12']
    splits = ['train', 'validation', 'testing']

    for subject in subjects:
        for split in splits:
            print(f"  ├── {subject}_{split}.h5     # EEG data in HDF5 format")
            print(f"  ├── {subject}_{split}.csv    # Labels in CSV format")

    print("\n📋 HDF5 file requirements:")
    print("  - Must contain 'eeg' dataset")
    print("  - Shape: [num_samples, time_points, channels] or [num_samples, channels, time_points]")
    print("  - Data type: float32 recommended")

    print("\n📋 CSV file requirements:")
    print("  - Must contain labels for each sample")
    print("  - Column name: 'label' or 'labels' (or first column will be used)")
    print("  - Values: 0 (no seizure) and 1 (seizure) or similar binary classification")

    return True

if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data"
    SUBJECTS = ['BM10', 'BM11', 'BM12']

    print("🧠 EEG Multi-Subject Data Validation")
    print("=" * 50)

    # Show expected structure
    prepare_sample_structure()

    # Validate data if directory exists
    if os.path.exists(DATA_DIR):
        print(f"\n🔍 Validating data in '{DATA_DIR}'...")
        report = create_data_summary(DATA_DIR, SUBJECTS)
    else:
        print(f"\n📁 Data directory '{DATA_DIR}' not found.")
        print("Please create the directory and place your files according to the structure above.")

        # Create directory
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Created directory: {DATA_DIR}")
