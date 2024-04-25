from typing import List
import torch
import numpy as np


def xyxy2xywh(boxes: np.ndarray | torch.Tensor) -> np.ndarray:
    assert boxes.shape[-1] == 4
    y = (
        torch.empty_like(boxes)
        if isinstance(boxes, torch.Tensor)
        else np.empty_like(boxes)
    )
    y[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2  # x center
    y[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2  # y center
    y[..., 2] = boxes[..., 2] - boxes[..., 0]  # width
    y[..., 3] = boxes[..., 3] - boxes[..., 1]  # height
    return y


def segments2boxes(segments: List[np.ndarray]) -> np.ndarray:
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # xyxy
    return xyxy2xywh(np.array(boxes))  # xywh
