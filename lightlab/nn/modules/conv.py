from torch import nn
import torch


def autopad(k, p=None, d=1):
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    def __init__(
        self, c1, c2, k=1, s=1, p=None, use_bn=True, act=nn.SiLU, g=1, d=1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=not use_bn
        )
        self.bn = nn.BatchNorm2d(c2) if use_bn else nn.Identity()
        self.act = act() if issubclass(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


class SkipConn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.add(*x)
