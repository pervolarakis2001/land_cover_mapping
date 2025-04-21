import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random


class SatelliteDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        # one-hot encoding
        self.label_mapping = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.int64)

        if self.transform:
            image = torch.from_numpy(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)

        for k, v in self.label_mapping.items():
            mask[mask == k] = v
        mask = torch.from_numpy(mask)

        return image, mask


class Normalize13Band:
    def __call__(self, x):
        return (
            x / 10000.0
        )  # normalization the same as the base pretrained model (ResNet18_Weights.SENTINEL2_ALL_MOCO)


class AddGaussianNoise:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


class RandomHorizontalFlip:
    def __call__(self, x, y):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
        return x, y


class Compose:
    def __init__(self, transforms, with_mask=False):
        self.transforms = transforms
        self.with_mask = with_mask

    def __call__(self, x, y=None):
        if self.with_mask:
            for t in self.transforms:
                if isinstance(t, RandomHorizontalFlip):
                    x, y = t(x, y)
                else:
                    x = t(x)
            return x, y
        else:
            for t in self.transforms:
                x = t(x)
            return x


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x, y)

    def __len__(self):
        return len(self.subset)
