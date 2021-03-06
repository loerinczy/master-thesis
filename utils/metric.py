import torch
import torch.nn as nn
from utils.misc import dice_coefficient, get_layer_channels


def contour_error(pred, target, use_lyr):
    classes = range(1, 9)
    ce_dict = {}
    for klass in classes:
        pred_mask = (pred == klass).int()
        pred_diff = (pred_mask[:, 1:] - pred_mask[:, :-1])
        pred_idx = pred_diff.max(1).indices
        if use_lyr:
            mad = (pred_idx.float() - target[:, klass - 1]).abs()
            mad = mad[~mad.isnan()]
        else:
            target_mask = (target == klass).int()
            target_diff = target_mask[:, 1:] - target_mask[:, :-1]
            target_idx = target_diff.max(1).indices
            mad = (pred_idx.float() - target_idx.float()).abs()
        ce_dict[klass] = mad.mean().item()
    return ce_dict


def mad_lt(pred, target, use_lyr):
    classes = range(1, 8)
    mad_dict = {}
    for klass in classes:
        pred_mask = (pred == klass).int()
        pred_diff = pred_mask[:, 1:] - pred_mask[:, :-1]
        pred_widths = pred_diff.shape[1] - 1 - pred_diff.flip(1).min(1).indices - pred_diff.max(1).indices
        if use_lyr:
            target_widths = target[:, klass] - target[:, klass-1]
        else:
            target_mask = (target == klass).int()
            target_diff = target_mask[:, 1:] - target_mask[:, :-1]
            target_widths = target_diff.shape[1] - 1 - target_diff.flip(1).min(1).indices - target_diff.max(1).indices
        mad = (pred_widths - target_widths).abs().float()
        mad = mad[~mad.isnan()].mean()
        mad_dict[klass] = mad.item()
    return mad_dict


def dice_acc(pred, target, num_classes):
    dice_coeff = dice_coefficient(pred, target, num_classes)
    if num_classes == 10:
        has_fluid = (target.view(pred.shape[0], -1) == 9).any(-1)
        if has_fluid.any():
            fluid_dice = dice_coeff[has_fluid, -1].mean()
        else:
            fluid_dice = torch.tensor(-1)
    dice = dice_coeff[:, :9].mean(0)
    dice_dict = {klass: dice[klass].item() for klass in range(9)}
    if num_classes == 10:
        dice_dict[9] = fluid_dice.item()
    return dice_dict


def intersection_over_union(prediction, target, num_classes):
    """
    Computes the intersection over union metric.
    :param prediction: torch.Tensor of shape N x C x H x W
    :param target: torch.Tensor of shape N x H x W
    :return: torch.Tensor of shape N x C
    """

    target = get_layer_channels(target, num_classes)
    intersection = prediction * target
    denominator = (prediction + target - intersection).sum((-1, -2))
    iou = intersection.sum((-1, -2)) / (denominator + 1e-12)
    if num_classes == 10:
        has_fluid = target[:, 9].view(prediction.shape[0], -1).any(-1)
        if has_fluid.any():
            fluid_iou = iou[has_fluid, -1].mean()
        else:
            fluid_iou = torch.tensor(-1)
    iou_avg = iou[:, :9].mean(0)
    iou_dict = {i: class_iou.item() for i, class_iou in enumerate(iou_avg)}
    if num_classes == 10:
        iou_dict[9] = fluid_iou.item()
    return iou_dict


def sensitivity(prediction, target, num_classes):
    """
    Computes the sensitivity metric.
    :param prediction: torch.Tensor of shape N x C x H x W
    :param target: torch.Tensor of shape N x H x W
    :return: torch.Tensor of shape N x C
    """

    target = get_layer_channels(target, num_classes)
    intersection = (prediction * target).sum((-1, -2))
    denominator = target.sum((-1, -2))
    se = intersection / (denominator + 1e-12)
    if num_classes == 10:
        has_fluid = target[:, 9].view(prediction.shape[0], -1).any(-1)
        if has_fluid.any():
            fluid_se = se[has_fluid, -1].mean()
        else:
            fluid_se = torch.tensor(-1)
    se_avg = se[:, :9].mean(0)
    se_dict = {i: class_se.item() for i, class_se in enumerate(se_avg)}
    if num_classes == 10:
        se_dict[9] = fluid_se.item()
    return se_dict


class Metric:
    def __init__(self, class_min, class_max):
        self.metric = {i: 0 for i in range(class_min, class_max + 1)}
        self.counter = {i: 0 for i in range(class_min, class_max + 1)}

    def update(self, curr_metric, return_avg=False):
        for key, value in curr_metric.items():
            if value != -1:
                self.metric[key] += value
                self.counter[key] += 1
        if return_avg:
            avg = sum(value for value in curr_metric.values()) / len(curr_metric.values())
            return avg

    def normalize(self):
        for key, value in self.metric.items():
            if self.counter[key]:
                self.metric[key] /= self.counter[key]
            else:
                self.metric[key] = -1

    def __str__(self):
        s = ""
        for key, value in self.metric.items():
            s += f"{key}: {round(value, 3)}; "
        return s

    def mean(self):
        s = sum(value for value in self.metric.values())
        mean = s / len(self.metric.values())
        return round(mean, 3)

