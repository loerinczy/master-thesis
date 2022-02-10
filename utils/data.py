#!/usr/bin/env python3

# Imports
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

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
        img = np.array(Image.open(img_file), dtype=float)
        img /= 255
        mask = np.array(Image.open(mask_file))
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
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
        img = np.array(Image.open(img_path), dtype=float)
        img /= 255
        boundary_indices = self.boundary_dict[str(idx)]
        boundary_indices = np.array(boundary_indices, dtype=float)
        boundary_indices /= img.shape[-2]
        if self.transform:
            img = self.transform[0][0](image=img)["image"]
            boundary_indices = self.transform[0][1](image=boundary_indices)["image"]
            if self.transform[1]:
                boundary_indices = boundary_indices.T
                transformed = self.transform(image=img, mask=boundary_indices)
                img = transformed["image"]
                boundary_indices = transformed["mask"].T
        img = torch.from_numpy(img).T
        boundary_indices = torch.from_numpy(boundary_indices)
        return img, boundary_indices


def create_patches(img, lyr, patch_width):
    idx = np.where((np.isnan(lyr)).any(axis=0))[0]
    diff = np.diff(idx)
    useful_parts = diff >= patch_width
    useful_lengths = diff[useful_parts]
    useful_start_idx = idx[np.pad(useful_parts, (0, 1), constant_values=[False])]
    for ustart, ulength in zip(useful_start_idx, useful_lengths):
        number_of_shifts = (ulength - 1) // patch_width
        for shift_idx in range(number_of_shifts):
            img_patch = img[:, ustart + 1 + shift_idx * patch_width:ustart + 1 + (
                          shift_idx + 1) * patch_width]
            lyr_patch = lyr[:, ustart + 1 + shift_idx * patch_width:ustart + 1 + (
                          shift_idx + 1) * patch_width]
            mask = create_layer_mask(lyr_patch, img_patch.shape[0])
            yield img_patch, mask


def generate_dme_dataset(input_dir, output_dir, patch_width):
    files = list(Path(input_dir).glob("*"))
    output_dir = Path(output_dir)
    output_dir.mkdir()
    cnt = 0
    for file in tqdm(files, desc="data generation"):
        data = loadmat(file)
        layers = data["manualLayers1"].transpose((2, 0, 1))
        images = data["images"].transpose((2, 0, 1))
        for idx, (image, layer) in enumerate(zip(images, layers)):
            patch_generator = create_patches(image, layer, patch_width)
            for img, mask in patch_generator:
                Image.fromarray(img).save(output_dir / f"img_{cnt}.png")
                Image.fromarray(mask).save(output_dir / f"mask_{cnt}.png")
                cnt += 1


def create_boundary_mask(boundary_array, height):
    mask = np.zeros((height, boundary_array.shape[1]), dtype="uint8")
    for col_idx, col in enumerate(boundary_array.T):
        if ~np.isnan(col).any():
            for boundary in col:
                mask[int(boundary), col_idx] = 1
    return mask


def create_layer_mask(boundary_array, height, fluid=None):
    mask = np.zeros((height, boundary_array.shape[1]), dtype="uint8")
    for col_idx, col in enumerate(boundary_array.T):
        prev_boundary = 0
        for boundary_idx, boundary in enumerate(col):
            mask[prev_boundary:int(boundary) + 1, col_idx] = boundary_idx
            prev_boundary = int(boundary) + 1
        mask[prev_boundary:, col_idx] = boundary_idx + 1
    fluid_class = len(col) + 1
    if isinstance(fluid, np.ndarray):
        mask[fluid != 0] = fluid_class
    return mask


def create_patches(img, lyr, patch_width, fluid):
    idx = np.where((np.isnan(lyr)).any(axis=0))[0]
    diff = np.diff(idx)
    useful_parts = diff >= patch_width
    useful_lengths = diff[useful_parts]
    useful_start_idx = idx[np.pad(useful_parts, (0, 1), constant_values=[False])]
    for ustart, ulength in zip(useful_start_idx, useful_lengths):
        number_of_shifts = (ulength - 1) // patch_width
        for shift_idx in range(number_of_shifts):
            indices = (slice(None), slice(
                      ustart + 1 + shift_idx * patch_width,
                      ustart + 1 + (shift_idx + 1) * patch_width
            ))
            img_patch = img[indices]
            lyr_patch = lyr[indices]
            fluid_patch = fluid[indices] if isinstance(fluid, np.ndarray) else fluid
            mask = create_layer_mask(lyr_patch, img_patch.shape[0], fluid_patch)
            yield img_patch, mask, lyr_patch.T.tolist()


def generate_dme_dataset(input_dir, output_dir, patch_width, use_fluid=False):
    files = list(Path(input_dir).glob("*"))
    output_dir = Path(output_dir)
    output_dir.mkdir()
    cnt = 0
    boundary_indices_dict = {}
    layer_widths_dict = {}
    for file in tqdm(files, desc="data generation"):
        data = loadmat(file)
        layers = data["manualLayers1"].transpose((2, 0, 1))
        images = data["images"].transpose((2, 0, 1))
        fluids = data["manualFluid1"].transpose((2, 0, 1))
        for idx, (image, layer, fluid) in enumerate(zip(images, layers, fluids)):
            fluid = fluid if (~np.isnan(fluid)).any() and use_fluid else None
            patch_generator = create_patches(image, layer, patch_width, fluid)
            for img, mask, boundary_indices_list in patch_generator:
                Image.fromarray(img).save(output_dir / f"img_{cnt}.png")
                Image.fromarray(mask).save(output_dir / f"mask_{cnt}.png")
                boundary_indices_dict[cnt] = boundary_indices_list
                cnt += 1
    with open(output_dir / "boundary_indices.json", "w") as boundary_file:
        json.dump(boundary_indices_dict, boundary_file)
