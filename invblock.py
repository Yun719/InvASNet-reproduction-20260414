from math import exp
import torch
import torch.nn as nn
import config as c
from rrdb_denselayer_1d import ResidualDenseBlock_out_1D

class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out_1D, clamp=c.clamp, harr=True, in_1=1, in_2=1):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 2
            self.split_len2 = in_2 * 2
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        v = self.clamp * 2 * (torch.sigmoid(s) - 0.5)
        v = torch.clamp(v, -10.0, 10.0)   # 防止 exp overflow
        return torch.exp(v)


    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)
