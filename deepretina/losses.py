from torch import nn
from torch.nn import functional  as F

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        prob = F.softmax(pred, 1)
        log_softmax = F.log_softmax(pred, 1)
        target = target.unsqueeze(1)
        loss = - self.alpha * (1 - prob.gather(1, target)) ** self.gamma * log_softmax.gather(1, target)
        loss_avg = loss.mean()
        return loss_avg
