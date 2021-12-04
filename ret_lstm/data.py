import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import json

class RetLSTMDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.img = lambda idx: f"img_{idx}.png"
        with open(self.root / "boundary_indices.json", "r") as bfile:
            self.boundary_dict = json.load(bfile)

    def __len__(self):
        return len(list(self.root.glob("img_*")))

    def __getitem__(self, idx):
        img_path = self.root / self.img(idx)
        boundary_indices = self.boundary_dict[str(idx)]
        boundary_indices = torch.tensor(boundary_indices)
        img = np.array(Image.open(img_path))
        img = torch.from_numpy(img).T
        return img, boundary_indices



# class RetLSTMDataset(Dataset):
#     def __init__(self, root: str):
#         self.root = Path(root)
#         self.img = lambda idx: f"img_{idx}.png"
#         self.mask = lambda idx: f"mask_{idx}.png"
#
#     def __len__(self):
#         return len(list(self.root.glob("img_*")))
#
#     def __getitem__(self, idx):
#         img_path = self.root / self.img(idx)
#         mask_path = self.root / self.mask(idx)
#         img = np.array(Image.open(img_path))
#         mask = np.array(Image.open(mask_path))
#         boundary_mask = np.pad(np.diff(mask, axis=0), ((0, 1), (0, 0)), constant_values=(0,))
#         img = torch.from_numpy(img)
#         boundary_mask = torch.from_numpy(boundary_mask)
#         return img, boundary_mask