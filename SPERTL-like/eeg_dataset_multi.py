import h5py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os

class EEGDatasetMulti(Dataset):
    def __init__(self, data_dir, subjects, split='train'):
        """
        Args:
            data_dir (str): Directory containing HDF5 and CSV files
            subjects (list): List of subject IDs (e.g., ['BM10', 'BM11', 'BM12'])
            split (str): 'train', 'validation', or 'testing'
        """
        self.data_dir = data_dir
        self.subjects = subjects
        self.split = split

        self.eeg_data = []
        self.labels = []
        self.file_indices = []  # To track which file each sample comes from

        self._load_all_data()

    def _load_all_data(self):
        """Load data from all subjects for the specified split"""
        current_idx = 0

        for subject in self.subjects:
            # File paths
            h5_file = os.path.join(self.data_dir, f"{subject}_{self.split}.h5")
            csv_file = os.path.join(self.data_dir, f"{subject}_{self.split}.csv")

            if not os.path.exists(h5_file) or not os.path.exists(csv_file):
                print(f"Warning: Files for {subject} {self.split} not found")
                continue

            # Load HDF5 data
            with h5py.File(h5_file, 'r') as f:
                eeg = f['eeg'][:]  # Load all data into memory
                self.eeg_data.append(eeg)

            # Load CSV labels
            df = pd.read_csv(csv_file)
            if 'label' in df.columns:
                labels = df['label'].values
            elif 'labels' in df.columns:
                labels = df['labels'].values
            else:
                # Assume first column is labels if no 'label' column found
                labels = df.iloc[:, 0].values

            self.labels.append(labels)

            # Track file indices for each sample
            num_samples = len(labels)
            self.file_indices.extend([subject] * num_samples)

            print(f"Loaded {subject} {self.split}: {eeg.shape[0]} samples")

        # Concatenate all data
        if self.eeg_data:
            self.eeg_data = np.concatenate(self.eeg_data, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
        else:
            raise ValueError(f"No data found for split: {self.split}")

        print(f"Total {self.split} samples: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg_data = self.eeg_data[idx]  # Shape depends on your data format

        # Adjust shape based on your data format
        # If data is [1280, 20], permute to [20, 1280]
        if len(eeg_data.shape) == 2:
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32).permute(1, 0)
        else:
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return eeg_data, label

    def get_subject_info(self, idx):
        """Get subject information for a given index"""
        return self.file_indices[idx]

# Alternative: Lazy loading version for large datasets
class EEGDatasetMultiLazy(Dataset):
    def __init__(self, data_dir, subjects, split='train'):
        """
        Lazy loading version - keeps files open and loads data on demand
        Better for very large datasets that don't fit in memory
        """
        self.data_dir = data_dir
        self.subjects = subjects
        self.split = split

        self.h5_files = {}
        self.csv_data = {}
        self.cumulative_lengths = [0]

        self._prepare_files()

    def _prepare_files(self):
        """Prepare file handles and calculate cumulative lengths"""
        total_length = 0

        for subject in self.subjects:
            h5_file = os.path.join(self.data_dir, f"{subject}_{self.split}.h5")
            csv_file = os.path.join(self.data_dir, f"{subject}_{self.split}.csv")

            if not os.path.exists(h5_file) or not os.path.exists(csv_file):
                print(f"Warning: Files for {subject} {self.split} not found")
                continue

            # Open HDF5 file
            self.h5_files[subject] = h5py.File(h5_file, 'r')

            # Load CSV labels
            df = pd.read_csv(csv_file)
            if 'label' in df.columns:
                labels = df['label'].values
            elif 'labels' in df.columns:
                labels = df['labels'].values
            else:
                labels = df.iloc[:, 0].values

            self.csv_data[subject] = labels

            # Update cumulative length
            total_length += len(labels)
            self.cumulative_lengths.append(total_length)

            print(f"Prepared {subject} {self.split}: {len(labels)} samples")

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find which subject this index belongs to
        subject_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:]):
            if idx < cum_len:
                subject_idx = i
                break

        subject = self.subjects[subject_idx]
        local_idx = idx - self.cumulative_lengths[subject_idx]

        # Load data
        eeg_data = self.h5_files[subject]['eeg'][local_idx]

        # Adjust shape
        if len(eeg_data.shape) == 2:
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32).permute(1, 0)
        else:
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32)

        label = torch.tensor(self.csv_data[subject][local_idx], dtype=torch.long)

        return eeg_data, label

    def __del__(self):
        """Close all HDF5 files when dataset is destroyed"""
        for h5_file in self.h5_files.values():
            h5_file.close()
