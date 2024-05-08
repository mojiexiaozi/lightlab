from typing import List
import torch
import numpy as np
import cv2


def xywh2ltwh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def ltwh2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    return y


def xyxy2ltwh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def ltwh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # width
    y[..., 3] = x[..., 3] + x[..., 1]  # height
    return y


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


def xywh2xyxy(x):
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = (
        torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    )  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def xyxyxyxy2xywhr(corners):
    is_torch = isinstance(corners, torch.Tensor)
    points = corners.cpu().numpy() if is_torch else corners
    points = points.reshape(len(corners), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([x, y, w, h, angle / 180 * np.pi])
    return (
        torch.tensor(rboxes, device=corners.device, dtype=corners.dtype)
        if is_torch
        else np.asarray(rboxes, dtype=points.dtype)
    )  # rboxes


def segment2box(segment, width=640, height=640):
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy


def segments2boxes(segments: List[np.ndarray]) -> np.ndarray:
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # xyxy
    return xyxy2xywh(np.array(boxes))  # xywh


def resample_segments(segments, n=1000):
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate(
                [np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32
            )
            .reshape(2, -1)
            .T
        )  # segment xy
    return segments


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    return np.array(
        [
            polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio)
            for x in polygons
        ]
    )


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            imgsz,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index
