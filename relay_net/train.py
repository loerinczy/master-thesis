#!/usr/bin/env python3

# Imports
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import transforms
from data import OCTDataset
from PIL import Image
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



def train(model, train_dl, optimizer, loss_fn, epoch, num_epochs, loss_list=None, device="cpu"):
    train_dl = tqdm(train_dl, desc=f"Epoch [{epoch} / {num_epochs}]", leave=False)
    for data, targets in train_dl:
        data = data.unsqueeze(1).to(device).float()
        targets = targets.to(device).long()
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss_list is not None:
            loss_list.append(loss.item())
        train_dl.set_postfix(loss=loss.item())


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
