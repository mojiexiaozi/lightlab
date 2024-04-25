from torch import nn
import torch
import math

from .conv import Conv, autopad
from .block import DFL
from lightlab.utils.tal import dist2bbox, make_anchors, dist2rbox


class Classify(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class Detect(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()) -> None:
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers(P3, P4, P5)
        # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # B, C, H, W
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        if self.export and self.format in {
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "tfjs",
        }:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor(
                [grid_w, grid_h, grid_w, grid_h], device=box.device
            ).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(
                self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2]
            )
        else:
            dbox = (
                self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0))
                * self.strides
            )
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes, anchors):
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


class OBB(Detect):
    def __init__(self, nc=80, ne=1, ch=()):
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters
        self.detect = Detect.forward
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c4, 3),
                Conv(c4, c4, 3),
                nn.Conv2d(c4, self.ne, 1),
            )
            for x in ch
        )

    def forward(self, x):
        bs = x[0].shape[0]  # batch size
        # OBB theta logits
        angle = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
        )
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return (
            torch.cat([x, angle], 1)
            if self.export
            else (torch.cat([x[0], angle], 1), (x[1], angle))
        )

    def decode_bboxes(self, bboxes, anchors):
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()) -> None:
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c4, 3),
                Conv(c4, c4, 3),
                nn.Conv2d(c4, self.nk, 1),
            )
            for x in ch
        )

    def forward(self, x):
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1
        )  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return (
            torch.cat([x, pred_kpt], 1)
            if self.export
            else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))
        )

    def kpts_decode(self, bs, kpts: torch.Tensor):
        ndim = self.kpt_shape[1]
        if (
            self.export
        ):  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()
            y[:, 0::ndim] = (
                y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)
            ) * self.strides
            y[:, 1::ndim] = (
                y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)
            ) * self.strides
            return y


class Semantic(nn.Sequential):
    def __init__(self, c1, c2, k=3, up=1, mode="bicubic", act=None):
        return super().__init__(
            nn.Conv2d(c1, c2, k, padding=autopad(k)),
            nn.Upsample(scale_factor=up, mode=mode) if up > 1 else nn.Identity(),
            act() if isinstance(act, nn.Module) else nn.Identity(),
        )
