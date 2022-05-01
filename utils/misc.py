import numpy as np
import os
import torch
from utils.data import OCTDataset, RetLSTMDataset
from torch.utils.data import DataLoader, Subset
import albumentations as A
from pathlib import Path
import traceback
import pandas as pd
import subprocess
from torch import nn
import pickle
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def dice_coefficient(prediction, target, num_classes):
    """
    Computes the dice coefficient.
    :param prediction: torch.Tensor of shape N x C x H x W
    :param target: torch.Tensor of shape N x H x W
    :return: torch.Tensor of shape N x C
    """

    target = get_layer_channels(target, num_classes)
    intersection = 2 * (prediction * target).sum((-1, -2))
    denominator = (prediction**2 + target**2 + 1e-12).sum((-1, -2))
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
    num_pixels = 0
    if retlstm:
        y_mean = 0
        y_square_mean = 0
    for x, y, corner in ds:
        x_not_corner = x[corner != 1]
        x_mean += x_not_corner.sum()
        x_square_mean += (x_not_corner**2).sum()
        if retlstm:
            y = y[~y.isnan()]
            y_mean += y.sum()
            y_square_mean += (y**2).sum()
        num_pixels += len(x_not_corner)
    x_mean /= num_pixels
    x_square_mean /= num_pixels
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
          shuffle_training=True,
):
    data_dir = Path(data_dir)
    train_ds = OCTDataset(data_dir / "training" / f"DME_{patch_width}", fluid)
    mean_std = get_mean_std(Subset(train_ds, range(len(train_ds))))
    norm = A.Normalize((mean_std[0],), (mean_std[1],), max_pixel_value=1., always_apply=True)
    if fluid:
        transf = A.Compose((norm, train_transform), additional_targets={"fluid": "mask", "lyr": "mask"})
    else:
        transf = A.Compose((norm, train_transform))
    train_ds = OCTDataset(data_dir / "training" / f"DME_{patch_width}", fluid, transform=transf, use_lyr=False)
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
          normalize_img=True,
          normalize_lyr=True,
          standardize=False,
          boundary_center=False,
          num_workers=2,
):
    data_dir = Path(data_dir)
    train_ds = RetLSTMDataset(
              data_dir / "training" / f"DME_{patch_width}",
              normalize_img=normalize_img,
              normalize_lyr=normalize_lyr
    )
    mean_std = get_mean_std(Subset(train_ds, range(len(train_ds))), retlstm=True)
    if standardize:
        if boundary_center:
            mean_std = list(mean_std)
            mean_std[2] = 0.5
        stand = [
            A.Normalize((mean_std[0],), (mean_std[1],), max_pixel_value=1., always_apply=True),
            A.Normalize((mean_std[2],), (mean_std[3],), max_pixel_value=1., always_apply=True)
        ]
        train_ds_transf = (stand, train_transform)
        valid_ds_transf = (stand, train_transform)
    else:
        train_ds_transf = (None, train_transform)
        valid_ds_transf = (None, train_transform)
    train_ds = RetLSTMDataset(
              data_dir / "training" / f"DME_{patch_width}",
              normalize_img=normalize_img,
              normalize_lyr=normalize_lyr,
              transform=train_ds_transf,
    )
    valid_ds = RetLSTMDataset(
              data_dir / "validation" / f"DME_{patch_width}",
              normalize_img=normalize_img,
              normalize_lyr=normalize_lyr,
              transform=valid_ds_transf,
              return_mask=True
    )
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


def functioning(model):
    x = torch.randn(64, 2, 496)
    out = model(x)
    return out.shape == (64, 2, 8)


def ls_models():
    ls = subprocess.run(["ls", "rec_models"], stdout=subprocess.PIPE)
    print(ls.stdout.decode("utf-8"))


def knormal(model):
    for m in model.modules():
        if (
                  isinstance(m, nn.Conv1d)
                  or isinstance(m, nn.Conv2d)
                  or isinstance(m, nn.Linear)
        ):
            nn.init.kaiming_normal_(
                      m.weight,
                      nonlinearity="linear" if isinstance(m, nn.Linear) else "relu"
            )
            nn.init.constant_(m.bias, 0)


def orig_init(model):
    for m in model.modules():
        if (
                  isinstance(m, nn.Linear)
                  or isinstance(m, nn.Conv1d)
                  or isinstance(m, nn.Conv2d)
        ):
            m.reset_parameters()


def kuni(model):
    for m in model.modules():
        if (
                  isinstance(m, nn.Conv1d)
                  or isinstance(m, nn.Conv2d)
                  or isinstance(m, nn.Linear)
        ):
            nn.init.kaiming_uniform_(
                      m.weight,
                      nonlinearity="linear" if isinstance(m, nn.Linear) else "relu"
            )
            nn.init.constant_(m.bias, 0)


def initialize(model, method="def"):
    if method == "def":
        orig_init(model)
    elif method == "knormal":
        knormal(model)
    elif method == "kuni":
        kuni(model)
    else:
        raise Exception("Error! Unknown method!")





def send_model(model, fname, init="uni"):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    if init == "uni":
        initialize(model)
    elif init == "normal":
        init_knormal(model)
    elif init == "def":
        orig_init(model)
    else:
        raise TypeError("ERROR! UNKNOWN init")
    assert functioning(model), "model not functioning!"
    comp(model)
    with open(f"{fname}.pkl", "wb") as file:
        pickle.dump(model, file)
    subprocess.run(["sendm", fname], stdout=subprocess.PIPE)
    subprocess.run(["rm", f"{fname}.pkl"])
    print("model sent")


def get_and_load_model(fname):
    subprocess.run(["getm", fname], stdout=subprocess.PIPE)
    with open(f"rec_models/{fname}.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def load_model(fname):
    with open(f"rec_models/{fname}.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def comp(model):
    print(sum(p.numel() for p in model.parameters()))


def get_and_load_ckp(model, path, epoch):
    subprocess.run(
              [
                  "to", f"{os.getenv('CKPS')}/{path}/epoch_{epoch}.pth", "."
              ]
    )
    model.load_state_dict(
              torch.load(
                        f"epoch_{epoch}.pth",
                        map_location=torch.device('cpu')
              )
    )
    print(">> checkpoint loaded")
    subprocess.run(["rm", f"epoch_{epoch}.pth"])
