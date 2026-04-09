import torch
from torch import nn as nn
from torch.nn import functional as F



class TruncatedMSELoss(nn.Module):
    def __init__(self, clamp_max=4, reduction='mean', channels_last=True):
        super().__init__()

        assert (reduction in ['none', 'sum', 'mean'])
        self.reduction = reduction

        self.mse = nn.MSELoss(reduction='none')
        self.clamp_max = clamp_max
        self.channels_last = channels_last

    def forward(self, x, mask, logprobs=False):  # x is of shape N x S x C if channels_last is True else N x C x S;
                                                 # mask is N x S (1 --> keep, 0 --> mask)
        if self.channels_last is True:
            x = x.permute(0, 2, 1)  # N x C x S

        # ***** based on https://github.com/yabufarha/ms-tcn *****
        if logprobs is True:
            loss = self.mse(x[:, :, 1:], x.detach()[:, :, :-1])
        else:
            loss = self.mse(F.log_softmax(x[:, :, 1:], dim=1), F.log_softmax(x.detach()[:, :, :-1], dim=1))
        loss = torch.clamp(loss, min=0, max=self.clamp_max ** 2)  # N x C x (S - 1)
        loss = torch.mean(loss, dim=1)  # N x (S - 1)
        loss = loss * mask[:, 1:]

        if self.reduction == 'none':
            return loss
        else:
            sum = torch.sum(loss)
            if self.reduction == 'sum':
                return sum
            else:
                assert (self.reduction == 'mean')
                num_elems = mask[:, 1:].sum()
                return sum / num_elems


class MultiLabelLoss(nn.Module):
    def __init__(self, reduction='mean', pos_weight=None):
        super().__init__()

        assert (reduction in ['sum', 'mean'])
        self.reduction = reduction

        self.loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, x, target, valid_mask):  # x, target is of shape N x S x C;  valid_mask is N x S (1 --> keep, 0 --> mask)
        loss = self.loss(x[valid_mask], target[valid_mask])

        sum = torch.sum(loss)
        if self.reduction == 'sum':
            return sum
        else:
            assert (self.reduction == 'mean')
            num_elems = valid_mask.sum()
            return sum / num_elems




def create_multilabel_loss(pos_weight=None):
    loss = MultiLabelLoss(reduction='mean', pos_weight=pos_weight)

    return loss
