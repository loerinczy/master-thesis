import torch.nn as nn
from utils.misc import dice_coefficient, get_layer_channels


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
        ce_dict[klass - 1] = mse.sqrt().item()
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
    iou = intersection.sum((-1, -2)) / denominator
    iou_avg = iou.mean(0)
    iou_dict = {i: class_iou.item() for i, class_iou in enumerate(iou_avg)}
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
    se = intersection / denominator
    se_avg = se.mean(0)
    se_dict = {i: class_se.item() for i, class_se in enumerate(se_avg)}
    return se_dict


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

