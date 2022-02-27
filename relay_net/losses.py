import torch
from utils.misc import dice_coefficient, get_fluid_boundary
from torch.nn import functional as F
from torch import nn


class DiceLoss(nn.Module):

    def __init__(self, weight_channel=None, num_classes=9):
        super(DiceLoss, self).__init__()
        self.weight_channel = (
            weight_channel if weight_channel is not None
            else torch.zeros(num_classes)
        )
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
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
        predictions_exp = predictions.exp()
        dice_coeff = dice_coefficient(predictions_exp, targets, self.num_classes).mean(0)
        weight = 1 + self.weight_channel
        weight /= weight.sum()
        loss = (weight * (1 - dice_coeff)).sum()
        return loss


class CombinedLoss(nn.Module):

    def __init__(
              self,
              num_classes=9,
              weight_channel_cross=None,
              weight_channel_dice=None,
              weight_boundary_cross=1.
    ):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        self.dice_loss_fn = DiceLoss(weight_channel_dice, num_classes)
        self.weight_channel_cross = (
            weight_channel_cross if weight_channel_cross is not None
            else torch.ones(num_classes)
        )
        self.weight_boundary_cross = weight_boundary_cross
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        if self.num_classes == 10:
            targets, fluid = targets
            fluid_boundary = get_fluid_boundary(fluid)
        weight_boundary = F.pad(targets[:, 1:, :] - targets[:, :-1, :], (0, 0, 0, 1), "constant", 0)
        if self.num_classes == 10:
            weight_boundary += fluid_boundary
            targets[fluid.bool()] = 9
        weight_boundary = weight_boundary * self.weight_boundary_cross
        weight_channel = torch.zeros(*targets.shape, self.num_classes, device=targets.device).scatter_(
                  -1, targets.unsqueeze(-1), 1
        )
        weight_channel = weight_channel * self.weight_channel_cross
        weight_channel = weight_channel.max(-1).values
        weight = 1 + weight_boundary + weight_channel
        weight /= weight.sum()
        cross_entropy_loss = (self.cross_entropy_loss_fn(predictions, targets) * weight).sum()
        dice_loss = self.dice_loss_fn(predictions, targets)
        return cross_entropy_loss, dice_loss
