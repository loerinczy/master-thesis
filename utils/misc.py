import numpy as np
import torch
from utils.data import OCTDataset, RetLSTMDataset
from torch.utils.data import DataLoader, Subset
import albumentations as A
from pathlib import Path
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def dice_coefficient(prediction, target, num_classes):
    """
    Computes the dice coefficient.
    :param prediction: torch.Tensor of shape N x C x H x W
    :param target: torch.Tensor of shape N x H x W
    :return: torch.Tensor of shape N x C
    """

    target = get_layer_channels(target, num_classes)
    prediction = prediction / prediction.sum(1, keepdims=True)
    intersection = 2 * (prediction * target).sum((-1, -2))
    denominator = (prediction + target + 1e-12).sum((-1, -2))
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


def get_fluid_boundary(fluid):
    shifted = fluid[:, :, 1:] - fluid[:, :, :-1]
    x, y, z = torch.where(shifted == -1)
    z = z - 1
    shifted[shifted != 1] = 0
    shifted[x, y, z] = 1
    mask1 = torch.nn.functional.pad(shifted, (1, 0, 0, 0), "constant", 0)
    shifted = fluid[:, 1:, :] - fluid[:, :-1, :]
    x, y, z = torch.where(shifted == -1)
    y = y - 1
    shifted[shifted != 1] = 0
    shifted[x, y, z] = 1
    mask2 = torch.nn.functional.pad(shifted, (0, 0, 1, 0), "constant", 0)
    mask = torch.logical_or(mask1, mask2).int()
    return mask


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
          fluid,
          patch_width,
          batch_size,
          train_transform,
          num_workers,
          shuffle_training=True
):
    data_dir = Path(data_dir)
    train_ds = OCTDataset(data_dir / "training" / f"DME_{patch_width}", fluid)
    mean_std = get_mean_std(Subset(train_ds, range(len(train_ds))))
    norm = A.Normalize((mean_std[0],), (mean_std[1],), max_pixel_value=1., always_apply=True)
    if fluid:
        transf = A.Compose((norm, train_transform), additional_targets={"fluid": "mask"})
    else:
        transf = A.Compose((norm, train_transform))
    train_ds = OCTDataset(data_dir / "training" / f"DME_{patch_width}", fluid, transform=transf)
    valid_ds = OCTDataset(data_dir / "validation" / f"DME_{patch_width}", fluid, transform=norm)

    train_loader = DataLoader(
              train_ds,
              batch_size=batch_size,
              num_workers=num_workers,
              shuffle=shuffle_training
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
          patch_width,
          batch_size,
          train_transform,
          num_workers,
):
    data_dir = Path(data_dir)
    train_ds = RetLSTMDataset(data_dir / "training" / f"DME_{patch_width}")
    mean_std = get_mean_std(Subset(train_ds, range(len(train_ds))), retlstm=True)
    norm = [
        A.Normalize((mean_std[0],), (mean_std[1],), max_pixel_value=1., always_apply=True),
        A.Normalize((mean_std[2],), (mean_std[3],), max_pixel_value=1., always_apply=True)
    ]
    train_ds = RetLSTMDataset(data_dir / "training" / f"DME_{patch_width}", transform=(norm, train_transform))
    valid_ds = RetLSTMDataset(data_dir / "validation" / f"DME_{patch_width}", transform=(norm, None))

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


def tflog2pandas(path):
    """Function implemented by Adnan Ali
       source: https://stackoverflow.com/questions/71239557/export-tensorboard-with-pytorch-data-into-csv-with-python
    """
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data