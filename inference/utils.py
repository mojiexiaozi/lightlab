import numpy as np
import cv2
from typing import Tuple, Union, List


def xywh2xyxy(x: np.ndarray, scale=None) -> np.ndarray:
    y = np.empty_like(x)
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    if scale is not None:
        if isinstance(scale, float):
            x_factory, y_factory = (scale, scale)
        else:
            x_factory, y_factory = scale
        y[..., 0] = (x[..., 0] - dw) * x_factory  # top left x
        y[..., 1] = (x[..., 1] - dh) * y_factory  # top left y
        y[..., 2] = (x[..., 0] + dw) * x_factory  # bottom right x
        y[..., 3] = (x[..., 1] + dh) * y_factory  # bottom right y
    else:
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = x[..., 0] + dw  # bottom right x
        y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def smallest_resize(
    img: np.ndarray, target_size: Union[int, Tuple[int, int]] = 256
) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    target_h, target_w = target_size

    # 计算调整尺寸
    ratio = min(target_h / h, target_w / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    return cv2.resize(img, (new_w, new_h)), ratio


def resize_by_shortest_edge(image, target_size):
    h, w = image.shape[:2]
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image


def letterbox_resize(
    image: np.ndarray, target_size: Union[int, Tuple[int, int]], center=True
):
    resized_img, _ = smallest_resize(image, target_size)
    if center:
        return center_padding(resized_img, target_size, (114, 114, 114))
    else:
        return padding(resized_img, target_size, (114, 114, 114))


def padding(
    img: np.ndarray,
    target_size: Union[int, Tuple[int, int]] = 256,
    pad_value: int = 0,
) -> np.ndarray:
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    target_h, target_w = target_size
    h, w = img.shape[:2]

    dh = max(0, target_h - h)
    dw = max(0, target_w - w)
    top, bottom = 0, int(round(dh + 0.1))
    left, right = 0, int(round(dw + 0.1))
    padded_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value
    )
    return padded_img


def center_padding(
    img: np.ndarray,
    target_size: Union[int, Tuple[int, int]] = 256,
    pad_value: int = 0,
) -> np.ndarray:
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    target_h, target_w = target_size
    h, w = img.shape[:2]

    dh = max(0, target_h - h) / 2.0
    dw = max(0, target_w - w) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value
    )
    return padded_img


def center_crop(img: np.ndarray, crop_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    h, w = img.shape[:2]

    # 如果 crop_size 是一个整数，则将其解析为 (crop_size, crop_size) 的元组
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    crop_h, crop_w = crop_size

    # 计算裁剪的起始坐标（考虑奇数尺寸）
    start_h = max(0, (h - crop_h) // 2)
    start_w = max(0, (w - crop_w) // 2)

    # 确保裁剪区域不超出图像边界
    end_h = min(h, start_h + crop_h)
    end_w = min(w, start_w + crop_w)

    # 进行中心裁剪
    cropped_img = img[start_h:end_h, start_w:end_w]

    return cropped_img


def get_preprocess_shape(
    old_w: int, old_h: int, long_side_length: int
) -> Tuple[int, int]:
    scale = long_side_length * 1.0 / max(old_w, old_h)
    new_h, new_w = old_h * scale, old_w * scale
    return (int(new_w + 0.5), int(new_h + 0.5))


def apply_coords(point_coords, original_size, target_length):
    old_h, old_w = original_size
    new_w, new_h = get_preprocess_shape(old_w, old_h, target_length)
    point_coords = np.array(point_coords, np.float32)
    point_coords[..., 0] = point_coords[..., 0] * (new_w / old_w)
    point_coords[..., 1] = point_coords[..., 1] * (new_h / old_h)
    return point_coords


def mask_to_shapes(mask: np.ndarray, mode="polygon") -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Refine contours
    approx_contours: List[np.ndarray] = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_contours.append(approx)

    # Remove too big contours ( >90% of image size)
    im_area = mask.shape[0] * mask.shape[1]
    if len(approx_contours) > 1:
        area_max = im_area * 0.9
        areas = [cv2.contourArea(contour) for contour in approx_contours]
        approx_contours = [
            contour for contour, area in zip(approx_contours, areas) if area < area_max
        ]

    # Remove small contours (area < 20% of average area)
    if len(approx_contours) > 1:
        areas = [cv2.contourArea(contour) for contour in approx_contours]
        avg_area = np.mean(areas)
        area_min = avg_area * 0.2
        approx_contours = [
            contour for contour, area in zip(approx_contours, areas) if area > area_min
        ]

    shapes = []
    if mode == "polygon":
        for contour in approx_contours:
            points = contour.reshape(-1, 2)

            points = points.tolist()
            if len(points) < 3:
                continue
            points.append(points[0])
            shapes.append({"shape_type": "polygon", "points": points})
    elif mode == "rectangle":
        for contour in approx_contours:
            points = contour.reshape(-1, 2)
            x_min = min(mask.shape[1], np.min(points[:, 0]))
            y_min = min(mask.shape[0], np.min(points[:, 1]))
            x_max = max(0, np.max(points[:, 0]))
            y_max = max(0, np.max(points[:, 1]))
            shapes.append(
                {
                    "shape_type": mode,
                    "points": [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                    ],
                }
            )
    elif mode == "circle":
        for contour in approx_contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            shapes.append({"shape_type": mode, "points": [center, radius]})
    elif mode == "rotation":
        for contour in approx_contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            shapes.append({"shape_type": mode, "points": box.tolist()})
    return shapes


from torchvision import transforms


if __name__ == "__main__":
    im = cv2.imread("assets/bus.jpg")
    # im_resized = resize_by_shortest_edge(im, 224)
    # cv2.imwrite("bus_resized.jpg", im_resized)

    # print(im_resized.shape, ratio)
    # im_padded = center_padding(im_resized, 640)
    # print(im_padded.shape)
    cv2.imwrite("im_crop.jpg", letterbox_resize(im, 640, False))
