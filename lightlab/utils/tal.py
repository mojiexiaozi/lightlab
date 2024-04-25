import torch
from typing import List


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh=True, dim=1):
    """ltrb to xywh or xyxy"""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh box
    return torch.cat((x1y1, x2y2), dim)  # xyxy box


def dist2rbox(
    distance: torch.Tensor,
    pred_angle: torch.Tensor,
    anchor_points: torch.Tensor,
    dim=-1,
):
    lt, rb = distance.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # bs, h*w, 1
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def make_anchors(features: List[torch.Tensor], strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    dtype, device = features[0].dtype, features[1].device
    for i, stride in enumerate(strides):
        _, _, h, w = features[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)
