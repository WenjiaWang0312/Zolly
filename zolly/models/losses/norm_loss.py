import torch
import torch.nn as nn


class NormLoss(nn.Module):

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                loss_weight_override=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_weight = (loss_weight_override if loss_weight_override is not None
                       else self.loss_weight)
        loss = pred**2

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        loss *= loss_weight

        return loss


class MMDLoss(nn.Module):

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    @classmethod
    def mmd(cls, x, y):
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz

        XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
                      torch.zeros(xx.shape).to(x.device),
                      torch.zeros(xx.shape).to(x.device))

        for a in [0.05, 0.2, 0.9]:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

        return torch.mean(XX + YY - 2. * XY)

    def forward(self,
                pred,
                target,
                loss_weight_override=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_weight = (loss_weight_override if loss_weight_override is not None
                       else self.loss_weight)
        loss = self.mmd(pred, target)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        loss *= loss_weight

        return loss
