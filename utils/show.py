import numpy as np
import torch
from PIL import Image


def show_layers_from_mask(img, mask):
    err_msg = "image and mask do not have the same dimensions"
    assert img.shape == mask.shape, err_msg
    if type(img) == torch.Tensor:
        img = img.cpu().numpy()
    if type(mask) == torch.Tensor:
        mask = mask.cpu().numpy()
    dme_colorcode = {
        1: (170, 160, 250),
        2: (120, 200, 250),
        3: (80, 200, 250),
        4: (50, 230, 250),
        5: (20, 230, 250),
        6: (0, 230, 250),
        7: (0, 230, 100),
        9: (180, 255, 255)  # fluid
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
    img_w_layers = Image.composite(mask_img, img_img, alpha_img)
    return img_w_layers


def show_layers_from_boundary(img_array, layer_array, mean_std=None, a_scan_length=496, fluid=None):
    err_msg = "layer boundaries not compatible with image width"
    assert img_array.shape[1] == layer_array.shape[0], err_msg
    if type(img_array) == torch.Tensor:
      img_array = img_array.numpy()
    if type(layer_array) == torch.Tensor:
      layer_array = layer_array.numpy()
    if mean_std:
        img_array = img_array * mean_std[1] + mean_std[0]
        layer_array = layer_array * mean_std[3] + mean_std[2]
    img_array = np.asarray(img_array * 255, "uint8")
    layer_array = np.asarray(layer_array * a_scan_length, "uint8")
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


@torch.no_grad()
def show_prediction(model, data, target, colab=True):
    if len(data.shape) == 2:
        data.unsqueeze_(0).unsqueeze_(0)
    model = model.cpu()
    prediction = model(data.float())
    prediction = prediction.max(1).indices
    img_pred = show_layers_from_mask((data.squeeze() * 255).byte(), prediction.squeeze())
    img_target = show_layers_from_mask((data.squeeze() * 255).byte(), target.squeeze())
    if colab:
        return img_pred, img_target
    else:
        img_pred.show()
        img_target.show()