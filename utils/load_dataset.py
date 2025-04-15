import os
import torch
from torch.utils.data import Dataset
import numpy as np


class SatelliteDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if f.endswith(".npy") and not f.endswith("_mask.npy")
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(
            self.mask_dir, self.image_files[idx].replace(".npy", "_mask.npy")
        )

        # Load data
        image = np.load(image_path)
        mask = np.load(mask_path)

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # Optional transforms (e.g. normalization, augmentation)
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
