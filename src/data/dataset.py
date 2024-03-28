"""PyTorch dataset for windowed sensor data."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SensorWindowDataset(Dataset):
    """Dataset of sliding windows over sensor readings.

    Args:
        windows: array of shape (n, window_size, n_features)
        rul_labels: optional RUL for each window
        flatten: if True, flatten each window to 1D
    """

    def __init__(
        self,
        windows: np.ndarray,
        rul_labels: np.ndarray | None = None,
        flatten: bool = True,
    ):
        self.windows = torch.FloatTensor(windows)
        self.rul = (
            torch.FloatTensor(rul_labels)
            if rul_labels is not None and len(rul_labels) > 0
            else None
        )
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        if self.flatten:
            w = w.reshape(-1)

        item = {"input": w}
        if self.rul is not None:
            item["rul"] = self.rul[idx]
        return item
