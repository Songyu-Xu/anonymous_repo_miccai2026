import torch
import torch.nn as nn
import numpy as np

class BCELoss(nn.Module):
    '''
    BCE Loss
    '''
    def __init__(self, eps=1e-8):
        super(BCELoss, self).__init__()

        self.eps = eps

    def forward(self, x, y, merge=True):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        los_pos =  y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1-y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if merge:
            batch_size = loss.shape[0]
            return -loss.sum() / batch_size
        else:
            return -loss


class RobustBCE(nn.Module):
    '''
    BCE with Robust Semi-supervised Learning
    '''
    def __init__(self, eps=1e-8):
        super(RobustBCE, self).__init__()
        self.eps = eps

    def forward(self, x, y, mask=None, clean_prob=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        mask: mask for clean set and relabeled set
        clean_prob: clean probabilities of all labels
        """

        if mask is None:
            mask = torch.ones_like(y).to(torch.bool)
        batch_size = mask.shape[0]
        x_pred = torch.sigmoid(x)

        # (1) BCE loss for Clean Set, Relabeled Set, and Unclean Set
        xs_pos = x_pred
        xs_neg = 1 - x_pred
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1-y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # (2) Reweighting Unclean Set with its 'clean_prob'
        if clean_prob is not None:
            clean_prob = mask * 1.0 + ~mask * clean_prob
            loss = loss * clean_prob

        # normalize w.r.t batch size
        loss = -torch.sum(loss) / batch_size

        return loss


class CSCCRobustBCE(nn.Module):
    '''
    BCE with Robust Semi-supervised Learning + CSCC weighting
    '''
    def __init__(self, eps=1e-8):
        super(CSCCRobustBCE, self).__init__()
        self.eps = eps

    def forward(self, x, y, cscc_weight=None):
        batch_size = x.shape[0]
        x_pred = torch.sigmoid(x)

        # BCE loss
        los_pos = y * torch.log(x_pred.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log((1 - x_pred).clamp(min=self.eps))
        loss = los_pos + los_neg  # [batch_size, num_classes]

        # Reweighting with CSCC weights (per-sample)
        if cscc_weight is not None:
            loss = loss * cscc_weight.unsqueeze(1)

        return -torch.sum(loss) / batch_size
    
