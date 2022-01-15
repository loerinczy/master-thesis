#!/usr/bin/env python3

# Imports
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import numpy as np

# Dataset
class OCTDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        assert Path(
            data_dir
            ).exists(), f"The directory {data_dir} does not exists!"
        super(OCTDataset, self).__init__()
        self.data_dir = Path(data_dir)
        self.len = len(list(Path(data_dir).glob("img_*")))
        self.image_name = lambda idx: f"img_{idx}.png"
        self.mask_name = lambda idx: f"mask_{idx}.png"
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_file = self.data_dir / self.image_name(idx)
        mask_file = self.data_dir / self.mask_name(idx)
        img = torch.from_numpy(np.array(Image.open(img_file)))
        mask = torch.from_numpy(np.array(Image.open(mask_file)))
        if self.transform is not None:
            transformed = self.transform(image=img.numpy(), mask=mask.numpy())
            img = torch.from_numpy(transformed["image"])
            mask = torch.from_numpy(transformed["mask"])
        return img, mask


class RetLSTMDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.img = lambda idx: f"img_{idx}.png"
        self.transform = transform
        with open(self.root / "boundary_indices.json", "r") as bfile:
            self.boundary_dict = json.load(bfile)

    def __len__(self):
        return len(list(self.root.glob("img_*")))

    def __getitem__(self, idx):
        img_path = self.root / self.img(idx)
        img = np.array(Image.open(img_path))
        boundary_indices = self.boundary_dict[str(idx)]
        boundary_indices = np.array(boundary_indices).T
        if self.transform:
            transformed = self.transform(image=img, mask=boundary_indices)
            img = transformed["image"]
            boundary_indices = transformed["mask"].T
        img = torch.from_numpy(img).T
        boundary_indices = torch.from_numpy(boundary_indices)
        return img, boundary_indices
