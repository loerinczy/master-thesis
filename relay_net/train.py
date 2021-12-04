#!/usr/bin/env python3

# Imports
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from misc import get_layer_channels
from data import OCTDataset
from losses import dice_coefficient
from tqdm import tqdm
from models import RelayNet
from losses import CombinedLoss

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
    dice_avg = torch.zeros((num_classes,), dtype=float)
    for batch_idx, (data, target) in enumerate(loader):
        data = data.unsqueeze(1).float().to(device)
        prediction = model(data)
        target = target.long().to(device)
        cross_loss, dice_loss = loss_fn(prediction, target)
        prediction = get_layer_channels(prediction.max(dim=1).indices, num_classes)
        dice = dice_coefficient(prediction, target, num_classes=num_classes)
        dice_avg += dice.mean(0).cpu()
        loader.set_postfix(dice_coeff=dice.mean().item(), cross_loss=cross_loss.item(), dice_loss=dice_loss.item())
    model.train()
    return dice_avg / len(loader)



def train(model, train_dl, optimizer, loss_fn, dice_factor, epoch, num_epochs, writer, device="cpu"):
    train_dl = tqdm(train_dl, desc=f"Epoch [{epoch} / {num_epochs}]", leave=False)
    num_batches = len(train_dl)
    cross_losses = []
    dice_losses = []
    for batch_idx, (data, targets) in enumerate(train_dl):
        data = data.unsqueeze(1).to(device).float()
        targets = targets.to(device).long()
        predictions = model(data)
        cross_loss, dice_loss = loss_fn(predictions, targets)
        total_loss = cross_loss + dice_loss * dice_factor
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        writer.add_scalar("Train/cross_loss", cross_loss.item(), num_batches * epoch + batch_idx)
        writer.add_scalar("Train/dice_loss", dice_loss.item(), num_batches * epoch + batch_idx)
        train_dl.set_postfix(cross_loss=cross_loss.item(), dice_loss=dice_loss.item())
        cross_losses.append(cross_loss.item())
        dice_losses.append(dice_loss.item())
    cross_loss_avg = sum(cross_losses) / num_batches
    dice_loss_avg = sum(dice_losses) / num_batches
    return cross_loss_avg, dice_loss_avg


def save_checkpoint(state, filename="checkpoint.pth"):
    print("=> Saving checkpoint...")
    torch.save(state, filename)


def load_checkpoint(model, checkpoint):
    print("=> Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])


if __name__ == "__main__":
    from models import RelayNet
    from data import OCTDataset
    from torch import nn

    train_loader, valid_loader = get_loaders("../../generated/DME_64", 10, None, 2, .8)
    model = RelayNet().float()
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    loss_fn = CombinedLoss()
    train(model, train_loader, optimizer, loss_fn, 0, 1)
