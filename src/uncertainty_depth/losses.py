import math

import torch
import torch.nn as nn


class EvidentialLoss(nn.Module):
    def __init__(self, lam=0.2, epsilon=1e-2, reduction='mean'):
        super().__init__()
        self.lam = lam
        self.epsilon = epsilon
        self.reduction = reduction

    def nig_nll(self, y, gamma, v, alpha, beta):
        twoBlambda = 2 * beta * (1 + v)
        nll = (
            0.5 * torch.log(math.pi / v)
            - alpha * torch.log(twoBlambda)
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        return nll

    def nig_reg(self, y, gamma, v, alpha, beta):
        error = torch.abs(y - gamma)
        evi = 2 * v + alpha
        reg = error * evi
        return reg

    def forward(self, y_pred, y_true):
        mu, v, alpha, beta = torch.split(y_pred, 1, dim=1)
        mu = mu.squeeze(1)
        v = v.squeeze(1)
        alpha = alpha.squeeze(1)
        beta = beta.squeeze(1)
        y_true = y_true.squeeze(1)

        nll_loss = self.nig_nll(y_true, mu, v, alpha, beta)
        reg_loss = self.nig_reg(y_true, mu, v, alpha, beta)
        total_loss = nll_loss + self.lam * (reg_loss - self.epsilon)

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss
