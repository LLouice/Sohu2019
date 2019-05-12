import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.utils import to_onehot


class FocalLoss0(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss0, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        prob = prob.clamp(self.eps, 1. - self.eps)

        loss = -1 * target * torch.log(prob)
        loss = loss * (1 - logit) ** self.gamma

        return loss.sum()


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average=size_average

    def forward(self, input, target):
        y = to_onehot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit) ** self.gamma  # focal loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
