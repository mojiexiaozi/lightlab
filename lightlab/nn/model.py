from torch import nn
import torch
from copy import deepcopy

from lightlab.nn.utils import load_yaml_model, parse_model
from lightlab.nn.utils import intersect_dicts
from lightlab.utils.logger import LOGGER
from lightlab.nn.modules import Detect, OBB, Pose


class Model(nn.Module):
    def __init__(self, cfg="yolov8-cls", scale="n", nc=None, verbose=True) -> None:
        super().__init__()
        self.cfg: dict = load_yaml_model(cfg)
        self.task = self.cfg["task"]
        self.inplace = self.cfg.get("inplace", True)
        if nc is not None:
            self.cfg["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.cfg), scale, verbose)

        m = self.model[-1]  # Detect() module
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: (
                self.forward(x)[0] if isinstance(m, (OBB, Pose)) else self.forward(x)
            )
            self.stride = torch.tensor(
                [s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))]
            )
            m.stride = self.stride
            m.bias_init()  # only run once

            # initialize weights
            for m in self.model:
                t = type(m)
                if t is nn.Conv2d:
                    pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif t is nn.BatchNorm2d:
                    m.eps = 1e-3
                    m.momentum = 0.03
                elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                    m.inplace = True
        else:
            self.stride = torch.Tensor([1])  # default stride

    def load(self, state_dict: dict, verbose=True):
        csd = intersect_dicts(state_dict, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(
                f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights"
            )

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                # from earlier layers
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)  # run the model
            if m.timm:
                y.extend(x)
                x = x[-1]  # last feature
            else:
                y.append(x if m.i in self.save else None)  # save output
            # if isinstance(x, (list, tuple)):
            #     try:
            #         print([a.shape for a in x], type(m))
            #     except:
            #         # pass
            #         print([a.shape for a in x[0]], type(m))
            #         print(x[0][0].shape, type(m))
            # else:
            #     print(x.shape, type(m))
        return x


if __name__ == "__main__":
    model = Model("yolov8-cls", nc=15)
    model.eval()
    ckpt = model.state_dict()
    res = model(torch.randn(1, 3, 640, 640))
    print(model)
