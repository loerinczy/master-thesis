import torch
from utils.misc import dice_coefficient
from torch.nn import functional as F
from torch import nn


class DiceLoss(nn.Module):

    def __init__(self, weight_channel, num_classes):
        super(DiceLoss, self).__init__()
        self.weight_channel = weight_channel / weight_channel.sum()
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
        loss = (self.weight_channel * (1 - dice_coeff)).mean(0)
        return loss


class CombinedLoss(nn.Module):

    def __init__(
              self,
              num_classes=9,
              weight_channel_cross=None,
              weight_channel_dice=1.,
              weight_boundary_cross=1.
    ):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        self.dice_loss_fn = DiceLoss(weight_channel_dice, num_classes)
        self.weight_channel_cross = weight_channel_cross
        self.weight_boundary_cross = weight_boundary_cross
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        weight_boundary = F.pad(targets[:, 1:, :] - targets[:, :-1, :], (0, 0, 0, 1), "constant", 0)
        weight_boundary = weight * self.weight_boundary_cross
        weight_channel = torch.zeros(*targets.shape, self.num_classes).scatter_(
                  -1, targets.unsqueeze(-1), 1
        )
        weight_channel = weight_channel * self.weight_channel_cross
        weight_channel = weight_channel.max(-1).values
        cross_entropy_loss = (self.cross_entropy_loss_fn(predictions, targets) * (
                  1 + weight_boundary + weight_channel
        )).mean()
        dice_loss = self.dice_loss_fn(predictions, targets)
        return cross_entropy_loss, dice_loss
