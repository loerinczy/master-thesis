import torch
from PIL import Image
import numpy as np
from torch import nn


def dice_coefficient(prediction, target, num_classes):
    """
    Computes the dice coefficient.
    :param prediction: torch.Tensor of shape N x C x H x W
    :param target: torch.Tensor of shape N x H x W
    :return: torch.Tensor of shape N x C
    """

    target = get_layer_channels(target, num_classes)
    prediction = prediction / prediction.sum(1).unsqueeze(1)
    intersection = 2 * (prediction * target).sum((-1, -2))
    denominator = (prediction + target).sum((-1, -2))
    dice_coeff = intersection / denominator
    return dice_coeff


def get_layer_channels(data: torch.Tensor, num_classes):
    """
    Creates the channels for each layer.
    :param data: torch.Tensor of shape N x H x W
    :return: torch.Tensor of shape N x 4 x H x W
    """

    data = data.unsqueeze(1)
    zeros = torch.zeros(data.shape[0], num_classes, *data.shape[2:]).to(data.device)
    zeros.scatter_(1, data, 1)
    return zeros


def show_layers_from_mask_array(img, mask):
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


@torch.no_grad()
def show_prediction(model, data, target, colab=True):
    if len(data.shape) == 2:
        data.unsqueeze_(0).unsqueeze_(0)
    model = model.cpu()
    prediction = model(data.float())
    prediction = prediction.max(1).indices
    img_pred = show_layers_from_mask_array(data.squeeze(), prediction.squeeze())
    img_target = show_layers_from_mask_array(data.squeeze(), target.squeeze())
    if colab:
        return img_pred, img_target
    else:
        img_pred.show()
        img_target.show()


def contour_error(pred, target):
    classes = sorted(list(set(target.flatten().tolist())))
    ce_dict = {}
    mse_fn = nn.MSELoss()
    for klass in classes[1:]:
        pred_mask = (pred == klass).int()
        pred_diff = (pred_mask[:, 1:] - pred_mask[:, :-1])
        pred_idx = pred_diff.max(1).indices
        target_mask = (target == klass).int()
        target_diff = target_mask[:, 1:] - target_mask[:, :-1]
        target_idx = target_diff.max(1).indices
        mse = mse_fn(pred_idx.float(), target_idx.float())
        ce_dict[klass] = mse.item()
    return ce_dict

def mad_lt(pred, target):
    classes = sorted(list(set(target.flatten().tolist())))
    mad_dict = {}
    for klass in classes[1:-1]:
        pred_mask = (pred == klass).int()
        pred_diff = pred_mask[:, 1:] - pred_mask[:, :-1]
        pred_widths = pred_diff.min(1).indices - pred_diff.max(1).indices
        target_mask = (target == klass).int()
        target_diff = target_mask[:, 1:] - target_mask[:, :-1]
        target_widths = target_diff.min(1).indices - target_diff.max(1).indices
        mad = (pred_widths - target_widths).abs().float().mean()
        mad_dict[klass] = mad.item()
    return mad_dict


def dice_acc(pred, target, num_classes):
    dice = dice_coefficient(pred, target, num_classes).mean(0)
    dice_dict = {klass: dice[klass].item() for klass in range(dice.shape[0])}
    return dice_dict



