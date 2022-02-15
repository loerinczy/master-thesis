import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc import get_layer_channels, get_layer_mask_from_boundaries
from utils.metric import (
    Metric, dice_acc, contour_error, sensitivity, intersection_over_union, mad_lt
)


@torch.no_grad()
def validate_relaynet(model: torch.nn.Module, loader: DataLoader, loss_fn, num_classes):
    """
    Validates the model.
    :param model: torch.nn.Module
    :param loader: torch.utils.data.DataLoader
    :param loss_fn: Function
    :param num_classes: int
    :return: torch.Tensor, torch.Tensor
    """
    model.eval()
    loader = tqdm(loader, desc="Validation", leave=False)
    device = next(model.parameters()).device
    cross_losses = []
    dice_losses = []
    ce = Metric(1, 8)
    mad = Metric(1, 7)
    dice = Metric(0, num_classes - 1)
    for batch_idx, (data, target) in enumerate(loader):
        data = data.unsqueeze(1).float().to(device)
        prediction = model(data)
        target = target.long().to(device)
        cross_loss, dice_loss = loss_fn(prediction, target)
        cross_losses.append(cross_loss.item())
        dice_losses.append(dice_loss.item())
        prediction_mask = prediction.max(dim=1).indices
        layer_mask = get_layer_channels(prediction_mask, num_classes)
        dice_avg = dice.update(dice_acc(layer_mask, target, num_classes), return_avg=True)
        ce_avg = ce.update(contour_error(prediction_mask, target), return_avg=True)
        mad_avg = mad.update(mad_lt(prediction_mask, target), return_avg=True)
        loader.set_postfix(dice=dice_avg, mad=mad_avg, ce=ce_avg)
    cross_loss = sum(cross_losses) / len(cross_losses)
    dice_loss = sum(dice_losses) / len(dice_losses)
    model.train()
    dice.normalize()
    ce.normalize()
    mad.normalize()
    return cross_loss, dice_loss, dice, ce, mad


@torch.no_grad()
def validate_deepretina(model: torch.nn.Module, loader: DataLoader, loss_fn, num_classes):
    """
    Validates the model.
    :param model: torch.nn.Module
    :param loader: torch.utils.data.DataLoader
    :param loss_fn: Function
    :param num_classes: int
    :return: torch.Tensor, torch.Tensor
    """
    model.eval()
    loader = tqdm(loader, desc="Validation", leave=False)
    device = next(model.parameters()).device
    losses = []
    iou = Metric(0, num_classes - 1)
    se = Metric(0, num_classes - 1)
    for batch_idx, (data, target) in enumerate(loader):
        data = data.unsqueeze(1).float().to(device)
        prediction = model(data)
        target = target.long().to(device)
        loss = loss_fn(prediction, target)
        losses.append(loss.item())
        prediction_mask = prediction.max(dim=1).indices
        layer_mask = get_layer_channels(prediction_mask, num_classes)
        iou_avg = iou.update(intersection_over_union(layer_mask, target, num_classes), return_avg=True)
        se_avg = se.update(sensitivity(layer_mask, target, num_classes), return_avg=True)
        loader.set_postfix(iou=iou_avg, se=se_avg)
    loss = sum(losses) / len(losses)
    model.train()
    iou.normalize()
    se.normalize()
    return loss, iou, se


@torch.no_grad()
def validate_retlstm(
          model: torch.nn.Module,
          loader: DataLoader,
          loss_fn,
          mean_std: list,
          num_classes: int,
          a_scan_length: int = 496,
):
    """
    Validates the model.
    :param model: torch.nn.Module
    :param loader: torch.utils.data.DataLoader
    :param loss_fn: Function
    :param mean_std: list
    :param num_classes: int
    :return: torch.Tensor, torch.Tensor
    """
    model.eval()
    loader = tqdm(loader, desc="Validation", leave=False)
    device = next(model.parameters()).device
    ce = Metric(0, num_classes - 2)
    mad = Metric(0, num_classes - 3)
    dice = Metric(0, num_classes - 1)
    losses = []
    for batch_idx, (data, target) in enumerate(loader):
        data, target = (
            data.swapaxes(0, 1).to(device),
            target.swapaxes(0, 1).to(device),
        )
        prediction = model(data)
        loss = loss_fn(prediction, target)
        losses.append(loss.item())
        prediction = (prediction * mean_std[3] + mean_std[2]) * a_scan_length
        target = (target * mean_std[3] + mean_std[2]) * a_scan_length
        mse = ((target - prediction)**2).mean((0, 1))
        ce_avg = ce.update({idx: round(value, 3) for idx, value in enumerate(mse.tolist())}, return_avg=True)
        mad_dict = {i: (prediction[:, :, i] - target[:, :, i]).abs().mean().item() for i in range(num_classes - 2)}
        mad_avg = mad.update(mad_dict, return_avg=True)
        target = get_layer_mask_from_boundaries(target.swapaxes(0, 1), data.shape[-1]).long()
        prediction_mask = get_layer_mask_from_boundaries(prediction.swapaxes(0, 1), data.shape[-1]).long()
        layer_mask = get_layer_channels(prediction_mask, num_classes)
        dice_avg = dice.update(dice_acc(layer_mask, target, num_classes), return_avg=True)
        loader.set_postfix(dice=dice_avg, mad=mad_avg, ce=ce_avg)
    loss = sum(losses) / len(losses)
    model.train()
    dice.normalize()
    ce.normalize()
    mad.normalize()
    return loss, dice, ce, mad