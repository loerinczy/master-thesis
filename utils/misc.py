import numpy as np
import torch
from utils.data import OCTDataset, RetLSTMDataset
from torch.utils.data import DataLoader, Subset
import albumentations as A
from pathlib import Path


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


def get_mean_std(ds, retlstm=False):
    x_mean = 0
    x_square_mean = 0
    if retlstm:
        y_mean = 0
        y_square_mean = 0
    for x, y in ds:
        x_mean += x.sum()
        x_square_mean += (x**2).sum()
        if retlstm:
            y_mean += y.sum()
            y_square_mean += (y**2).sum()
    x_mean /= len(ds) * np.prod(x.shape)
    x_square_mean /= len(ds) * np.prod(x.shape)
    if retlstm:
        y_mean /= len(ds) * np.prod(y.shape)
        y_square_mean /= len(ds) * np.prod(y.shape)
    x_std = (x_square_mean - x_mean**2).abs().sqrt()
    if retlstm:
        y_std = (y_square_mean - y_mean**2).abs().sqrt()
        return x_mean, x_std, y_mean, y_std
    return x_mean, x_std


def get_loaders(
          data_dir,
          patch_width,
          batch_size,
          train_transform,
          num_workers,
):
    data_dir = Path(data_dir)
    train_ds = OCTDataset(data_dir / "training" / f"DME_{patch_width}")
    mean_std = get_mean_std(Subset(train_ds, range(len(train_ds))))
    norm = A.Normalize((mean_std[0],), (mean_std[1],), max_pixel_value=1., always_apply=True)
    transf = A.Compose((norm, train_transform))
    train_ds = OCTDataset(data_dir / "training" / f"DME_{patch_width}", transform=transf)
    valid_ds = OCTDataset(data_dir / "validation" / f"DME_{patch_width}", transform=norm)

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
    return train_loader, valid_loader, mean_std


def get_loaders_retlstm(
          data_dir,
          batch_size,
          train_transform,
          num_workers,
          train_val_ratio,
          indices=None,
          relaynet_setup=True,
          retlstm=False
):
    dataset_class = RetLSTMDataset if retlstm else OCTDataset
    train_ds = dataset_class(data_dir, transform=None)
    if indices:
        train_indices, valid_indices = indices
    else:
        train_length = int(len(train_ds) * train_val_ratio / (1 + train_val_ratio))
        train_indices = range(train_length)
        valid_indices = range(train_length, len(train_ds))
    train_ds = Subset(train_ds, train_indices)
    mean_std = get_mean_std(train_ds, retlstm)
        norm = [
            A.Normalize((mean_std[0],), (mean_std[1],), max_pixel_value=1., always_apply=True),
            A.Normalize((mean_std[2],), (mean_std[3],), max_pixel_value=1., always_apply=True)
        ]
        train_ds = dataset_class(data_dir, transform=(norm, train_transform))
        valid_ds = dataset_class(data_dir, transform=norm)
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
    return train_loader, valid_loader, mean_std


def get_layer_mask_from_boundaries(y, a_scan_length=496):
    idx_tensor = torch.tile(torch.arange(0, a_scan_length, device=y.device), dims=(*y.shape[:-1], 1))
    out = torch.zeros_like(idx_tensor)
    for idx, row in enumerate(y.permute((-1, 0, 1))):
        out[idx_tensor > row.unsqueeze(-1)] = idx + 1
    return out


def denormalize(t, mean_std, max_val):
    t = t * mean_std[1] + mean_std[0]
    t *= max_val
    return t



# Deprecated functions

# def get_mean_std_retlstm(loader):
#     x, y = next(iter(loader))
#     x_mean = torch.zeros((x.shape[-1],))
#     x_square_mean = torch.zeros((x.shape[-1],))
#     y_mean = torch.zeros((y.shape[-1],))
#     y_square_mean = torch.zeros((y.shape[-1],))
#     num_a_scans = 0
#     for x, y in loader:
#         x_mean += x.sum((0, 1))
#         x_square_mean += (x**2).sum((0, 1))
#         y_mean += y.sum((0, 1))
#         y_square_mean += (y**2).sum((0, 1))
#         num_a_scans += x.shape[1] * x.shape[0]
#
#     x_mean /= num_a_scans
#     x_square_mean /= num_a_scans
#     y_mean /= num_a_scans
#     y_square_mean /= num_a_scans
#     x_std = (x_square_mean - x_mean**2).abs().sqrt()
#     y_std = (y_square_mean - y_mean**2).abs().sqrt()
#     return x_mean, x_std, y_mean, y_std

# def normalize(x, y, x_mean, x_std, y_mean, y_std):
#     x_mean = torch.tile(x_mean, (x.shape[0], x.shape[1], 1))
#     x_std = torch.tile(x_std, (x.shape[0], x.shape[1], 1))
#     y_mean = torch.tile(y_mean, (x.shape[0], x.shape[1], 1))
#     y_std = torch.tile(y_std, (x.shape[0], x.shape[1], 1))
#     x_normalized = (x - x_mean) / x_std
#     y_normalized = (y - y_mean) / y_std
#     return x_normalized, y_normalized

# @torch.no_grad()
# def denormalize(y, y_mean, y_std):
#     y_mean = torch.tile(y_mean, (y.shape[0], y.shape[1], 1))
#     y_std = torch.tile(y_std, (y.shape[0], y.shape[1], 1))
#     y = (y * y_std.to(y.device) + y_mean.to(y.device))
#     return y