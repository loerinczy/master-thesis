import torch
from PIL import Image
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from .data import OCTDataset



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
        ce_dict[klass - 1] = mse.item() / pred.shape[0]
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
        mad_dict[klass - 1] = mad.item()
    return mad_dict


def dice_acc(pred, target, num_classes):
    dice = dice_coefficient(pred, target, num_classes).mean(0)
    dice_dict = {klass: dice[klass].item() for klass in range(dice.shape[0])}
    return dice_dict


class Metric:
    def __init__(self, class_min, class_max):
        self.metric = {i: 0 for i in range(class_min, class_max + 1)}
        self.counter = 0

    def update(self, curr_metric, return_avg=False):
        for key, value in curr_metric.items():
            self.metric[key] += value
        self.counter += 1
        if return_avg:
            avg = sum(value for value in curr_metric.values()) / len(curr_metric.values())
            return avg

    def normalize(self):
        for key, value in self.metric.items():
            self.metric[key] /= self.counter

    def __str__(self):
        s = ""
        for key, value in self.metric.items():
            s += f"{key}: {round(value, 3)}; "
        return s

    def mean(self):
        s = sum(value for value in self.metric.values())
        mean = s / len(self.metric.values())
        return round(mean, 3)


def get_loaders(
          data_dir,
          batch_size,
          train_transform,
          num_workers,
          train_val_ratio
):
    train_ds = OCTDataset(data_dir, transform=train_transform)
    valid_ds = OCTDataset(data_dir, transform=None)
    train_length = int(len(train_ds) * train_val_ratio / (1 + train_val_ratio))
    indices = list(
        SubsetRandomSampler(
            torch.arange(0, len(train_ds))
            )
        )
    train_indices = indices[:train_length]
    valid_indices = indices[train_length:]
    train_ds = Subset(train_ds, train_indices)
    valid_ds = Subset(valid_ds, valid_indices)
    train_loader = DataLoader(
              train_ds,
              batch_size=batch_size,
              num_workers=num_workers,
              shuffle=True
    )
    valid_loader = DataLoader(
              valid_ds,
              batch_size=batch_size,
              num_workers=num_workers,
              shuffle=False
    )

    return train_loader, valid_loader


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, loss_fn, num_classes):
    """
    Validates the model.
    :param model: torch.nn.Module
    :param loader: torch.utils.data.DataLoader
    :param loss_fn: Function
    :return: torch.Tensor, torch.Tensor
    """
    model.eval()
    loader = tqdm(loader, desc="Validation", leave=False)
    device = next(model.parameters()).device
    ce = Metric(0, num_classes - 2)
    mad = Metric(0, num_classes - 3)
    dice = Metric(0, num_classes - 1)
    for batch_idx, (data, target) in enumerate(loader):
        data = data.unsqueeze(1).float().to(device)
        prediction = model(data)
        target = target.long().to(device)
        prediction_mask = prediction.max(dim=1).indices
        layer_mask = get_layer_channels(prediction_mask, num_classes)
        dice_avg = dice.update(dice_acc(layer_mask, target, num_classes), return_avg=True)
        ce_avg = ce.update(contour_error(prediction_mask, target), return_avg=True)
        mad_avg = mad.update(mad_lt(prediction_mask, target), return_avg=True)
        loader.set_postfix(dice=dice_avg, mad=mad_avg, ce=ce_avg)
    model.train()
    dice.normalize()
    ce.normalize()
    mad.normalize()
    return dice, ce, mad


def normalize(x, y, x_mean, x_std, y_mean, y_std):
  x_mean = torch.tile(x_mean, (x.shape[0], x.shape[1], 1))
  x_std = torch.tile(x_std, (x.shape[0], x.shape[1], 1))
  y_mean = torch.tile(y_mean, (x.shape[0], x.shape[1], 1))
  y_std = torch.tile(y_std, (x.shape[0], x.shape[1], 1))
  x_normalized = (x - x_mean) / x_std
  y_normalized = (y - y_mean) / y_std
  return x_normalized, y_normalized

@torch.no_grad()
def denormalize(y, y_mean, y_std):
  y_mean = torch.tile(y_mean, (y.shape[0], y.shape[1], 1))
  y_std = torch.tile(y_std, (y.shape[0], y.shape[1], 1))
  y = (y * y_std.to(y.device) + y_mean.to(y.device))
  return y


@torch.no_grad()
def validate_retlstm(model: torch.nn.Module, loader: DataLoader, mean_std, num_classes):
    """
    Validates the model.
    :param model: torch.nn.Module
    :param loader: torch.utils.data.DataLoader
    :param loss_fn: Function
    :return: torch.Tensor, torch.Tensor
    """
    model.eval()
    loader = tqdm(loader, desc="Validation", leave=False)
    device = next(model.parameters()).device
    ce = Metric(0, num_classes - 2)
    mad = Metric(0, num_classes - 3)
    dice = Metric(0, num_classes - 1)
    for batch_idx, (data, target) in enumerate(loader):
        data, _ = normalize(data, target, *mean_std)
        data, target = data.swapaxes(0, 1).to(device), target.swapaxes(0, 1).to(device)
        prediction = model(data)
        prediction = denormalize(prediction, *mean_std[2:])
        mse = ((target - prediction)**2).sum(1).mean(0)
        ce_avg = ce.update({idx: round(value, 3) for idx, value in enumerate(mse.tolist())}, return_avg=True)
        mad_dict = {i: (prediction[:, :, i] - target[:, :, i]).abs().mean().item() for i in range(num_classes - 2)}
        mad_avg = mad.update(mad_dict, return_avg=True)
        target = get_layer_mask_from_boundaries(target.swapaxes(0, 1), data.shape[-1]).long()
        prediction_mask = get_layer_mask_from_boundaries(prediction.swapaxes(0, 1), data.shape[-1]).long()
        layer_mask = get_layer_channels(prediction_mask, num_classes)
        dice_avg = dice.update(dice_acc(layer_mask, target, num_classes), return_avg=True)
        loader.set_postfix(dice=dice_avg, mad=mad_avg, ce=ce_avg)
    model.train()
    dice.normalize()
    ce.normalize()
    mad.normalize()
    return dice, ce, mad


def get_layer_mask_from_boundaries(y, a_scan_length=496):
    idx_tensor = torch.tile(torch.arange(0, a_scan_length, device=y.device), dims=(*y.shape[:-1], 1))
    out = torch.zeros_like(idx_tensor)
    for idx, row in enumerate(y.permute((-1, 0, 1))):
        out[idx_tensor > row.unsqueeze(-1)] = idx + 1
    return out


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


def get_mean_std(loader):
  x, y = next(iter(loader))
  x_mean = torch.zeros((x.shape[-1],))
  x_square_mean = torch.zeros((x.shape[-1],))
  y_mean = torch.zeros((y.shape[-1],))
  y_square_mean = torch.zeros((y.shape[-1],))
  num_a_scans = 0
  for x, y in loader:
    x_mean += x.sum((0, 1))
    x_square_mean += (x**2).sum((0, 1))
    y_mean += y.sum((0, 1))
    y_square_mean += (y**2).sum((0, 1))
    num_a_scans += x.shape[1] * x.shape[0]

  x_mean /= num_a_scans
  x_square_mean /= num_a_scans
  y_mean /= num_a_scans
  y_square_mean /= num_a_scans
  x_std = (x_square_mean - x_mean**2).abs().sqrt()
  y_std = (y_square_mean - y_mean**2).abs().sqrt()
  return x_mean, x_std, y_mean, y_std