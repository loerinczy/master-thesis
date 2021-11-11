#!/usr/bin/env python3

from PIL import Image
from scipy.io import loadmat
import numpy as np
from pathlib import Path
from tqdm import tqdm
import itertools
from scipy.ndimage.morphology import distance_transform_edt
from collections import Counter


def show_layers_from_boundary_array(img_array, layer_array, fluid=None):
    err_msg = "layer boundaries not compatible with image width"
    assert img_array.shape[1] == layer_array.shape[0], err_msg
    dme_colorcode = {
        1: (170, 160, 250),
        2: (120, 200, 250),
        3: (80, 200, 250),
        4: (50, 230, 250),
        5: (20, 230, 250),
        6: (0, 230, 250),
        7: (0, 230, 100),
        8: (180, 255, 255)  # fluid
    }
    amd_colorcode = {
        1: (180, 200, 250),
        2: (120, 200, 250),
    }
    zeros = np.zeros_like(img_array, dtype="uint8")
    hue = np.zeros_like(img_array, dtype="uint8")
    saturation = np.zeros_like(img_array, dtype="uint8")
    value = np.zeros_like(img_array, dtype="uint8")
    mask = np.zeros(img_array.shape, dtype="uint8")
    for w in range(img_array.shape[1]):
        if ~np.isnan(layer_array[w, :]).any():
            last_boundary = int(layer_array[w, 0])
            for idx, h in enumerate(layer_array[w, 1:]):
                curr_boundary = int(h) + 1
                mask[last_boundary:curr_boundary, w] = idx + 1
                last_boundary = curr_boundary
    if fluid is not None:
        mask[fluid != 0] = 8
    if layer_array.shape[1] == 8:
        for klass, hsv in dme_colorcode.items():
            hue[mask == klass] = hsv[0]
            saturation[mask == klass] = hsv[1]
            value[mask == klass] = hsv[2]
    if layer_array.shape[1] == 3:
        for klass, hsv in amd_colorcode.items():
            hue[mask == klass] = hsv[0]
            saturation[mask == klass] = hsv[1]
            value[mask == klass] = hsv[2]
    alpha = np.zeros_like(img_array, dtype="uint8")
    alpha[mask != 0] = 255
    img_stack = np.array([zeros, zeros, img_array]).transpose((1, 2, 0))
    img = Image.fromarray(img_stack, mode="HSV")
    colored_mask_stack = np.array([hue, saturation, value]).transpose((1, 2, 0))
    colored_mask = Image.fromarray(colored_mask_stack, mode="HSV")
    alphaimg = Image.fromarray(alpha)
    img_with_boundaries = Image.composite(colored_mask, img, alphaimg)
    return img_with_boundaries


def boundary_length_distribution(folder):
    lengths = []
    for file in folder:
        data = loadmat(file)
        for bscan in data["layerMaps"]:
            x_inds, _ = np.where(~np.isnan(bscan))
            if x_inds.any():
                lengths.append(x_inds[-1] - x_inds[0])
    return lengths


def boundary_middle_distribution(folder):
    middles = []
    for file in folder:
        data = loadmat(file)
        for bscan in data["layerMaps"]:
            x_inds, _ = np.where(~np.isnan(bscan))
            if x_inds.any():
                middles.append(x_inds[len(x_inds) // 2])
    return middles


def show_boundary_from_boundary_array(img_array, layer_array):
    err_msg = "layer boundaries not compatible with image width"
    assert img_array.shape[1] == layer_array.shape[0], err_msg
    mask = np.zeros(img_array.shape, dtype="uint8")
    for w_idx in range(img_array.shape[1]):
        if ~np.isnan(layer_array[w_idx, :]).any():
            for h_idx in layer_array[w_idx, :]:
                mask[int(h_idx), w_idx] = 255
    empty = mask.copy()
    ones = np.ones(img_array.shape, dtype="uint8") * 255
    img_stack = np.array([empty, empty, img_array]).transpose((1, 2, 0))
    img = Image.fromarray(img_stack, mode="HSV")
    colored_mask_stack = np.array([empty, ones, ones]).transpose((1, 2, 0))
    colored_mask = Image.fromarray(colored_mask_stack, mode="HSV")
    maskimg = Image.fromarray(mask)
    img_with_boundaries = Image.composite(colored_mask, img, maskimg)
    return img_with_boundaries


def show_layers_from_mask_array(img, mask):
    err_msg = "image and mask do not have the same dimensions"
    assert img.shape == mask.shape, err_msg
    dme_colorcode = {
        2: (170, 160, 250),
        3: (120, 200, 250),
        4: (80, 200, 250),
        5: (50, 230, 250),
        6: (20, 230, 250),
        7: (0, 230, 250),
        8: (0, 230, 100),
        10: (180, 255, 255)  # fluid
    }
    zeros = np.zeros_like(mask, dtype="uint8")
    hue = zeros.copy()
    saturation = zeros.copy()
    value = zeros.copy()
    alpha = zeros.copy()
    for klass, hsv in dme_colorcode.items():
        hue[mask == klass] = hsv[0]
        saturation[mask == klass] = hsv[1]
        value[mask == klass] = hsv[2]
        alpha[mask == klass] = 255
    img_stack = np.array([zeros, zeros, img]).transpose((1, 2, 0))
    img_img = Image.fromarray(img_stack, mode="HSV")
    colored_mask_stack = np.array([hue, saturation, value]).transpose((1, 2, 0))
    mask_img = Image.fromarray(colored_mask_stack, mode="HSV")
    alpha_img = Image.fromarray(alpha)
    print(colored_mask_stack.max())
    img_w_layers = Image.composite(mask_img, img_img, alpha_img)
    return img_w_layers


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


def create_layer_mask(boundary_array, height):
    mask = np.zeros((height, boundary_array.shape[1]), dtype="uint8")
    for col_idx, col in enumerate(boundary_array.T):
        prev_boundary = 0
        for boundary_idx, boundary in enumerate(col):
            mask[prev_boundary:int(boundary) + 1, col_idx] = boundary_idx + 1
            prev_boundary = int(boundary) + 1
        mask[prev_boundary:, col_idx] = boundary_idx + 2
    return mask