#!/usr/bin/env python3

# Imports
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import transforms
from data import OCTDataset
from PIL import Image
from tqdm import tqdm
from models import RelayNet
from typing import Union

def get_loaders(
          data_dir,
          batch_size,
          train_transform,
          num_workers,
          train_val_ratio
):
    train_ds = OCTDataset(data_dir, transform=train_transform)
    valid_ds = OCTDataset(data_dir, transform=transforms.ToTensor())
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


def overfit():
    from data import OCTDataset
    from torch import optim
    dataset = OCTDataset("../dataset")
    data, mask = dataset[0]
    data = data.unsqueeze(0).unsqueeze(0).float()
    mask = mask.unsqueeze(0).long()
    model = RelayNet().float()


def get_layer_channels(data: torch.Tensor):
    """
    Creates the channels for each layer.
    :param data: torch.Tensor of shape N x H x W
    :return: torch.Tensor of shape N x 4 x H x W
    """

    data = data.unsqueeze(1)
    zeros = torch.zeros(data.shape[0], 2, *data.shape[2:])
    zeros.scatter_(1, data, 1)
    return zeros

def show_layers(data: torch.Tensor):
    """
    Shows the layers in different colors.
    :param data: torch.Tensor of shape [1 x C x] H x W
    :return: None
    """

    assert not (len(data.shape) == 4 and data.shape[0] != 1), "Only batch size of 1 can be shown!"
    data = data.cpu()
    data.squeeze_(0)
    if len(data.shape) == 3:
        data = data.max(dim=0).indices
    hue = torch.ones_like(data) * 10
    value = torch.ones_like(data) * 10
    data *= 50
    img_array = torch.stack((hue, value, data)).byte().numpy().transpose((1, 2, 0))
    img = Image.fromarray(img_array, mode="HSV")
    img.show()


@torch.no_grad()
def show_prediction(model, data, target):
    if len(data.shape) == 2:
        data.unsqueeze_(0).unsqueeze_(0)
    device = next(model.parameters()).device
    prediction = model(data.to(device).float())
    prediction = prediction.max(1).indices
    show_layers(target)
    show_layers(prediction)

def dice_coefficient(prediction, target):
    """
    Computes the dice coefficient.
    :param prediction: torch.Tensor of shape N x C x H x W
    :param target: torch.Tensor of shape N x H x W
    :return: torch.Tensor of shape N x C
    """

    target = get_layer_channels(target)
    intersection = 2 * (prediction * target).sum((-1, -2))
    denominator = (prediction + target).sum((-1, -2))
    dice_coeff = (intersection / denominator)
    return dice_coeff

def dice_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: Union[torch.Tensor, float] = 1.):
    """
    Computes the dice loss.
    :param predictions: torch.Tensor of shape N x C x H x W,
        the raw prediction of the model
    :param targets: torch.Tensor of shape N x H x W,
        the raw target from the dataset
    :param weights: torch.Tensor of shape C,
        per channel weights for loss weighting
    :return: the dice loss summed over the channels, averaged over the batch
    """
    predictions_max = predictions.max(dim=1).indices
    predictions_layered = get_layer_channels(predictions_max)
    dice_coeff = dice_coefficient(predictions_layered, targets).mean(0)
    loss = weights * (1 - dice_coeff).sum(0)
    return loss


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, loss_fn):
    """
    Validates the model.
    :param model: torch.nn.Module
    :param loader: torch.utils.data.DataLoader
    :param loss_fn: Function
    :return: torch.Tensor, torch.Tensor
    """
    model.eval()
    loader = tqdm(loader, desc="Validation", leave=False)
    acc = 0
    loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        prediction = model(data)
        curr_loss = loss_fn(prediction, target).item()
        loss += curr_loss
        curr_acc = dice_coefficient(prediction, target)
        acc += curr_acc
        loader.set_postfix(acc=curr_acc.item(), loss=curr_loss.item())
    model.train()
    acc /= (batch_idx + 1)
    loss /= (batch_idx + 1)
    return acc, loss



def train(model, train_dl, optimizer, loss_fn, epoch, num_epochs, device="cpu"):
    train_dl = tqdm(train_dl, desc=f"Epoch [{epoch} / {num_epochs}]", leave=False)
    for data, targets in train_dl:
        data = data.unsqueeze(1).to(device).float()
        targets = targets.unsqueeze(1).to(device).long()
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch.set_postfix(loss=loss.item())


def save_checkpoint(state, filename="checkpoint.pth"):
    print("=> Saving checkpoint...")
    torch.save(state, filename)


def load_checkpoint(model, checkpoint):
    print("=> Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])


if __name__ == "__main__":
    pass