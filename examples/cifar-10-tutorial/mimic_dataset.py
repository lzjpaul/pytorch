from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MIMICDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, feature_csv_file, label_csv_file):
        """
        Args:
            csv_file (string): Path to the csv file.
        """
        self.mimic_features = np.genfromtxt(feature_csv_file, dtype=np.float32, delimiter = ',')
        self.mimic_labels = np.genfromtxt(label_csv_file, dtype=np.float32, delimiter = ',')

    def __len__(self):
        return len(self.mimic_features)

    def __getitem__(self, idx):
        sample = {'features': self.mimic_features[idx], 'label': self.mimic_labels[idx]}
        return sample

    def feature_dim(self):
        return self.mimic_features.shape[1]

    def label_dim(self):
        return self.mimic_labels.shape[1]
