#!/usr/bin/env python3

# Imports
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
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
        self.mask_name = lambda idx: f"mask_{idx[0]}_r{idx[1]}.png"
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_file = self.data_dir / self.image_name(idx)
        img = torch.from_numpy(np.array(Image.open(img_file)))
        masks = []
        for mask_idx in range(4):
            mask_file = self.data_dir / self.mask_name([idx, mask_idx])
            mask = list(np.array(Image.open(mask_file)))
            masks.append(mask)
        mask = torch.tensor(masks).max(dim=0).indices
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask