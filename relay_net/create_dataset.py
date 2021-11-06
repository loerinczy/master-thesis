#!/usr/bin/env python3

from PIL import Image
from scipy.io import loadmat
import numpy as np
from pathlib import Path
from tqdm import tqdm
import itertools


def generate_dataset(out_dir, amd, control):
    amd_files = Path(amd).glob("*")
    control_files = Path(control).glob("*")
    raw_files = list(itertools.chain(amd_files, control_files))
    Path(out_dir).mkdir()
    img_cnt = 0
    for file_ind in tqdm(range(len(raw_files)), desc="data generation"):
        file = raw_files[file_ind]
        img_dict = loadmat(file)
        layers = img_dict["layerMaps"]
        images = img_dict["images"].transpose((2, 0, 1))
        for image, layer in zip(images, layers):
            x_ind, _ = np.where(~np.isnan(layer))
            if len(x_ind):
                x_min = x_ind.min()
                x_max = x_ind.max()
                width = x_max - x_min
                if width > 64:
                    number_of_slices = width // 64
                    for num in range(number_of_slices):
                        img = image[:, x_min + num * 64: x_min + (num + 1) * 64]
                        lyr = layer[x_min + num * 64: x_min + (num + 1) * 64, :]
                        mask_rbr = np.zeros((image.shape[0], 64), dtype="uint8")
                        mask_r1 = np.zeros((image.shape[0], 64), dtype="uint8")
                        mask_r2 = np.zeros((image.shape[0], 64), dtype="uint8")
                        mask_rar = np.zeros((image.shape[0], 64), dtype="uint8")
                        for col_idx, col in enumerate(lyr):
                            border1 = int(col[0])
                            border2 = int(col[1])
                            border3 = int(col[2])
                            mask_rbr[:border1 + 1, col_idx] = 255
                            mask_r1[border1 + 1:border2 + 1, col_idx] = 255
                            mask_r2[border2 + 1:border3 + 1, col_idx] = 255
                            mask_rar[border3 + 1:, col_idx] = 255
                        Image.fromarray(img).save(f"{out_dir}/img_{img_cnt}.png")
                        Image.fromarray(mask_rbr).save(f"{out_dir}/mask_{img_cnt}_r0.png")
                        Image.fromarray(mask_r1).save(f"{out_dir}/mask_{img_cnt}_r1.png")
                        Image.fromarray(mask_r2).save(f"{out_dir}/mask_{img_cnt}_r2.png")
                        Image.fromarray(mask_rar).save(f"{out_dir}/mask_{img_cnt}_r3.png")
                        img_cnt += 1


def show_boundaries(img_array, layer_array):
    err_msg = "layer boundaries not compatible with image width"
    assert img_array.shape[1] == layer_array.shape[0], err_msg
    mask = np.zeros(img_array.shape, dtype="uint8")
    for w_idx in range(img_array.shape[1]):
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

def show_layers_from_layer_array(img_array, layer_array):
    err_msg = "layer boundaries not compatible with image width"
    assert img_array.shape[1] == layer_array.shape[0], err_msg
    dme_colorcode = {
        1: (180, 200, 250),
        2: (120, 200, 250),
        3: (80, 200, 250),
        4: (50, 230, 250),
        5: (20, 230, 250),
        6: (0, 230, 250),
        7: (0, 230, 100)
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
    if layer_array.shape[1] == 8:
        for klass in range(1, 8):
            hue[mask == klass] = dme_colorcode[klass][0]
            saturation[mask == klass] = dme_colorcode[klass][1]
            value[mask == klass] = dme_colorcode[klass][2]
    alpha = np.zeros_like(img_array, dtype="uint8")
    alpha[mask != 0] = 255
    img_stack = np.array([zeros, zeros, img_array]).transpose((1, 2, 0))
    img = Image.fromarray(img_stack, mode="HSV")
    colored_mask_stack = np.array([hue, saturation, value]).transpose((1, 2, 0))
    colored_mask = Image.fromarray(colored_mask_stack, mode="HSV")
    alphaimg = Image.fromarray(alpha)
    img_with_boundaries = Image.composite(colored_mask, img, alphaimg)
    return img_with_boundaries


if __name__ == "__main__":
    generate_dataset("dataset", "raw_dataset/AMD", "raw_dataset/Control")