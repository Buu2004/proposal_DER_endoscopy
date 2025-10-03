import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DNormalGamma(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.filters = out_channels
        self.conv = nn.Conv2d(in_channels, 4 * out_channels, kernel_size, **kwargs)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        output = self.conv(x)
        mu, logv, logalpha, logbeta = torch.split(output, self.filters, dim=1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=1)


class _LoRA_qkv(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
