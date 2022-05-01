import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc import get_layer_channels, get_layer_mask_from_boundaries
from utils.metric import (
    Metric, dice_acc, contour_error, sensitivity, intersection_over_union, mad_lt
)
from torch.cuda.amp import autocast

@torch.no_grad()
def validate_relaynet(model: torch.nn.Module, loader: DataLoader, loss_fn, num_classes, use_lyr=True):
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
    dice = Metric(0, num_classes - 1)
    ce = Metric(1, 8)
    mad = Metric(1, 7)
    for batch_idx, (data, target) in enumerate(loader):
        data = data.unsqueeze(1).float().to(device)
        prediction = model(data)
        if num_classes == 10:
            target, fluid, lyr = target
            target[fluid.bool()] = 9
            target = target.long().to(device)
            fluid = fluid.to(device)
            lyr = lyr.to(device)
            target = (target, fluid)
        else:
            target = target.long().to(device)
        cross_loss, dice_loss = loss_fn(prediction, target)
        if num_classes == 10:
            target = target[0]
        cross_losses.append(cross_loss.item())
        dice_losses.append(dice_loss.item())
        prediction_mask = prediction.max(dim=1).indices
        layer_mask = get_layer_channels(prediction_mask, num_classes)
        dice_avg = dice.update(dice_acc(layer_mask, target, num_classes), return_avg=True)
        ce_avg = ce.update(contour_error(prediction_mask, lyr if use_lyr else target, use_lyr), return_avg=True)
        mad_avg = mad.update(mad_lt(prediction_mask, lyr if use_lyr else target, use_lyr), return_avg=True)
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
    ce = Metric(1, 8)
    for batch_idx, (data, target) in enumerate(loader):
        data = data.unsqueeze(1).float().to(device)
        prediction = model(data)
        if num_classes == 10:
            target, fluid, lyr = target
            target[fluid.bool()] = 9
            lyr = lyr.to(device)
        target = target.long().to(device)
        loss = loss_fn(prediction, target)
        losses.append(loss.item())
        prediction_mask = prediction.max(dim=1).indices
        layer_mask = get_layer_channels(prediction_mask, num_classes)
        iou_avg = iou.update(intersection_over_union(layer_mask, target, num_classes), return_avg=True)
        se_avg = se.update(sensitivity(layer_mask, target, num_classes), return_avg=True)
        ce_avg = ce.update(contour_error(prediction_mask, lyr, use_lyr=True), return_avg=True)
        loader.set_postfix(iou=iou_avg, se=se_avg, ce=ce_avg)
    loss = sum(losses) / len(losses)
    model.train()
    iou.normalize()
    se.normalize()
    ce.normalize()
    return loss, iou, se, ce


@torch.no_grad()
def validate_retlstm(
          model: torch.nn.Module,
          loader: DataLoader,
          loss_fn,
          normalize_lyr,
          standardize,
          loss_weight,
          mean_std: list,
          num_classes: int,
          a_scan_length: int=496,
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
    mad = Metric(1, num_classes - 2)
    dice = Metric(0, 9)
    losses = []
    for batch_idx, (data, target, corner, mask) in enumerate(loader):
        data, target, corner = (
            data.swapaxes(0, 1).to(device).float(),
            target.swapaxes(0, 1).to(device).float(),
            corner.swapaxes(0, 1).to(device)
        )
        mask = mask.to(device)
        with autocast():
            data[corner != 0] = 0
            prediction = model(data)
            lw = loss_weight.expand_as(target)[~target.isnan()]
            loss = (
                      loss_fn(prediction[~target.isnan()], target[~target.isnan()])
                      * lw
            ).mean()
        losses.append(loss.item())
        if standardize:
            prediction = (prediction * mean_std[3] + mean_std[2])
            target = (target * mean_std[3] + mean_std[2])
        if normalize_lyr:
            prediction *= a_scan_length
            target *= a_scan_length
        mae = (target - prediction).abs()
        ce_batch = {}
        for klass in range(0, num_classes - 1):
            curr_target = target[..., klass]
            ce_batch[klass] = mae[..., klass][~curr_target.isnan()].mean().item()
        ce_avg = ce.update(ce_batch, return_avg=True)
        mad_batch = {}
        for klass in range(1, num_classes - 1):
            curr_target = target[:, :, klass] - target[:, :, klass - 1]
            curr_pred = prediction[:, :, klass] - prediction[:, :, klass - 1]
            mad_batch[klass] = (curr_target[~curr_target.isnan()] - curr_pred[~curr_target.isnan()]).abs().mean().item()
        mad_avg = mad.update(mad_batch, return_avg=True)
        prediction_mask = get_layer_mask_from_boundaries(prediction.swapaxes(0, 1), data.shape[-1]).long()
        layer_mask = get_layer_channels(prediction_mask, num_classes)
        layer_mask = torch.cat([
                        layer_mask.swapaxes(-2, -1),
                        torch.zeros_like(mask, dtype=layer_mask.dtype, device=layer_mask.device).unsqueeze(1)
                    ], dim=1)
        dice_avg = dice.update(dice_acc(layer_mask, mask.long(), 10), return_avg=True)
        loader.set_postfix(dice=dice_avg, mad=mad_avg, ce=ce_avg)
    loss = sum(losses) / len(losses)
    model.train()
    dice.normalize()
    ce.normalize()
    mad.normalize()
    return loss, dice, ce, mad