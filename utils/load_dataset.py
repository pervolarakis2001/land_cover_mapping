import torch
from torch.utils.data import Dataset
import numpy as np
import random
import torchvision.transforms.functional as TF


class SatelliteDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.label_mapping = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load raw data
        image = np.load(self.image_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.int64)

        invalid_mask = image == -9999

        image[invalid_mask] = 0.0

        # Apply label remapping
        for k, v in self.label_mapping.items():
            mask[mask == k] = v

        # Set mask to 255 (ignore) where all bands are nodata
        if image.ndim == 3:
            mask[np.all(invalid_mask, axis=0)] = 255

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long()

        if self.transform:
            image = self.transform(image)

        return image, mask


class Normalize13Band:
    def __call__(self, x):
        # normalization the same as the base pretrained model (ResNet18_Weights.SENTINEL2_ALL_MOCO)
        return x / 10000.0


class AddGaussianNoise:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


class RandomRotation:
    def __call__(self, x, y):
        angle = random.choice([0, 90, 180, 270])
        x = TF.rotate(x, angle)
        y = TF.rotate(y.unsqueeze(0), angle).squeeze(0)
        return x, y


class RandomHorizontalFlip:
    def __call__(self, x, y):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
        return x, y


class RandomGaussianBlur:
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, x):
        if random.random() < 0.5:
            return TF.gaussian_blur(x, kernel_size=self.kernel_size)
        return x


class Compose:
    def __init__(self, transforms, with_mask=False):
        self.transforms = transforms
        self.with_mask = with_mask

    def __call__(self, x, y=None):
        if self.with_mask:
            for t in self.transforms:
                if isinstance(
                    t,
                    (
                        RandomHorizontalFlip,
                        RandomRotation,
                    ),
                ):
                    x, y = t(x, y)
                else:
                    x = t(x)
            return x, y
        else:
            for t in self.transforms:
                x = t(x)
            return x, y


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x, y)

    def __len__(self):
        return len(self.subset)
