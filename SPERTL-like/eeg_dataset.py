import h5py
import torch
from torch.utils.data import Dataset

class EEGDatasetH5(Dataset):
    def __init__(self, h5_path):
        self.h5file = h5py.File(h5_path, 'r')
        self.eeg = self.h5file['eeg']          # Shape: [7200, 1280, 20]
        self.labels = self.h5file['labels']    # Shape: [7200]

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        eeg_data = self.eeg[idx]               # [1280, 20]
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32).permute(1, 0)  # → [20, 1280]

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return eeg_data, label
