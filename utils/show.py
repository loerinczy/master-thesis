import numpy as np
import torch
from PIL import Image

colorcode = {
    1: (359, 5, 10),
    2: (328, 90, 5),
    3: (118, 58, 69),
    4: (292, 52, 80),
    5: (30, 20, 10),
    6: (60, 10, 5),
    7: (22, 76, 100),
    9: (220, 80, 60)  # fluid
}

colorcode = {key: (np.array(value) * np.array([255 / 359, 255, 255])).astype(int) for key, value in colorcode.items()}

def show_layers_from_mask(img, mask, mean_std=None, normed=False):
    err_msg = "image and mask do not have the same dimensions"
    assert img.shape == mask.shape, err_msg
    if type(img) == torch.Tensor:
        img = img.cpu().numpy()
    if type(mask) == torch.Tensor:
        mask = mask.cpu().numpy()
    if mean_std:
        img = img * mean_std[1].numpy() + mean_std[0].numpy()
        normed = True
    if normed:
        img = np.asarray(img * 255, "uint8")
    zeros = np.zeros_like(mask, dtype="uint8")
    hue = zeros.copy()
    saturation = zeros.copy()
    value = zeros.copy()
    alpha = zeros.copy()
    for klass, hsv in colorcode.items():
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


def show_layers_from_boundary(img_array, layer_array, mean_std=None, a_scan_length=496, fluid=None, normed=False):
    err_msg = "layer boundaries not compatible with image width"
    if len(img_array.shape) == 3:
        img_array.squeeze_()
    if len(layer_array.shape) == 3:
        layer_array.squeeze_()
    assert img_array.shape[1] == layer_array.shape[1], err_msg
    if type(img_array) == torch.Tensor:
        img_array = img_array.detach().cpu().numpy()
    if type(layer_array) == torch.Tensor:
        layer_array = layer_array.detach().cpu().numpy()
    if mean_std:
        img_array = img_array * mean_std[1].numpy() + mean_std[0].numpy()
        layer_array = layer_array * mean_std[3].numpy() + mean_std[2].numpy()
        normed = True
    if normed:
        img_array = np.asarray(img_array * 255, "uint8")
        layer_array = np.asarray(layer_array * a_scan_length, "uint8")
    zeros = np.zeros_like(img_array, dtype="uint8")
    hue = np.zeros_like(img_array, dtype="uint8")
    saturation = np.zeros_like(img_array, dtype="uint8")
    value = np.zeros_like(img_array, dtype="uint8")
    mask = np.zeros(img_array.shape, dtype="uint8")
    layer_array = layer_array.T
    for w in range(img_array.shape[1]):
        if ~np.isnan(layer_array[w, :]).any():
            last_boundary = int(layer_array[w, 0])
            for idx, h in enumerate(layer_array[w, 1:]):
                curr_boundary = int(h) + 1
                mask[last_boundary:curr_boundary, w] = idx + 1
                last_boundary = curr_boundary
    if fluid is not None:
        mask[fluid != 0] = 9
    if layer_array.shape[1] == 8:
        for klass, hsv in colorcode.items():
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
def show_prediction(model, data, target, mean_std, standardized=False, colab=True):
    if len(data.shape) == 2:
        data.unsqueeze_(0).unsqueeze_(0)
    dev = next(model.parameters()).device
    model.cpu()
    if not standardized:
      data = (data - mean_std[0]) / mean_std[1]
    prediction = model(data.float())
    prediction = prediction.max(1).indices
    data = (data.squeeze() * mean_std[1] + mean_std[0]) * 255
    img_pred = show_layers_from_mask(data.byte(), prediction.squeeze())
    img_target = show_layers_from_mask(data.byte(), target.squeeze())
    model.to(dev)
    if colab:
        return img_pred, img_target
    else:
        img_pred.show()
        img_target.show()