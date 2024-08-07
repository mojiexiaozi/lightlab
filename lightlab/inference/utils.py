import numpy as np
import cv2
from typing import Tuple, Union, List, Dict


class Transform:
    def __init__(self, trans: List):
        self.trans = trans

    def __call__(self, inputs: np.ndarray):
        for tran in self.trans:
            inputs = tran(inputs)
        return inputs


class Normalize:
    def __init__(
        self, scale=None, std=(0.229, 0.224, 0.225), mean=(0.485, 0.456, 0.406)
    ) -> None:
        self.std = std
        self.mean = mean
        self.scale = scale

    def __call__(self, input: dict) -> Dict:
        img: np.ndarray = input["img"]
        scale = self.scale
        if scale is None:
            scale = 255.0 if img.dtype == np.uint8 else 65535.0

        img = img.astype(np.float32)
        if scale > 1.0:
            img /= scale

        if list(self.mean) != [0.0, 0.0, 0.0]:
            img -= self.mean
        if list(self.std) != [1.0, 1.0, 1.0]:
            img /= self.std
        input["img"] = img
        return input


class ToCHWImage:
    def __init__(self) -> None:
        pass

    def __call__(self, input: Dict) -> Dict:
        img: np.ndarray = input["img"]
        if len(img.shape) == 2:
            img = img[None]
        input["img"] = img.transpose((2, 0, 1))
        return input


class SmallestResize:
    def __init__(self, target_size: Union[int, Tuple[int, int]] = 256, stride=32):
        self.target_size = target_size
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        self.stride = stride

    def __call__(self, input: Dict) -> Dict:
        img: np.ndarray = input["img"]
        h, w = img.shape[:2]
        # 计算调整尺寸
        ratio = min(self.target_size[0] / h, self.target_size[1] / w)
        resize_h, resize_w = int(h * ratio), int(w * ratio)
        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)
        input["img"] = cv2.resize(img, (resize_w, resize_h))

        resize_h, resize_w = input["img"].shape[:2]
        input["scale"] = (h / resize_h, w / resize_w)
        return input


class ResizeLimitMax:
    def __init__(self, target_size: Union[int, Tuple[int, int]] = 256, stride=32):
        self.target_size = target_size
        if not isinstance(target_size, int):
            self.target_size = max(target_size)
        self.stride = stride

    def __call__(self, input: Dict) -> Dict:
        img: np.ndarray = input["img"]
        h, w = img.shape[:2]
        if max(h, w) > self.target_size:
            if h > w:
                ratio = float(self.target_size) / h
            else:
                ratio = float(self.target_size) / w
        else:
            ratio = 1.0
        # 计算调整尺寸
        resize_h, resize_w = int(h * ratio), int(w * ratio)
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        input["img"] = cv2.resize(img, (int(resize_w), int(resize_h)))

        resize_h, resize_w = input["img"].shape[:2]
        input["scale"] = (h / resize_h, w / resize_w)
        return input


class ResizeByShortEdge:
    def __init__(self, target_size: Union[int, Tuple[int, int]] = 256):
        self.target_size = target_size
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)

    def __call__(self, input: Dict) -> Dict:
        img: np.ndarray = input["img"]
        h, w = img.shape[:2]
        if h < w:
            new_h = self.target_size[0]
            ratio = self.target_size[0] / h
            new_w = int(w * ratio)
        else:
            new_w = self.target_size[1]
            ratio = self.target_size[1] / w
            new_h = int(h * ratio)
        input["img"] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        resize_h, resize_w = input["img"].shape[:2]
        input["scale"] = (h / resize_h, w / resize_w)
        return input


class Padding:
    def __init__(
        self, target_size: Union[int, Tuple[int, int]] = 256, pad_value: int = 0
    ):
        self.target_size = target_size
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        self.pad_value = pad_value

    def __call__(self, input: Dict) -> Dict:
        img: np.ndarray = input["img"]
        h, w = img.shape[:2]

        dh = max(0, self.target_size[0] - h)
        dw = max(0, self.target_size[1] - w)
        top, bottom = 0, int(round(dh + 0.1))
        left, right = 0, int(round(dw + 0.1))
        input["img"] = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.pad_value
        )
        return input


class CenterPadding:
    def __init__(
        self, target_size: Union[int, Tuple[int, int]] = 256, pad_value: int = 0
    ):
        self.target_size = target_size
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        self.pad_value = pad_value

    def __call__(self, input: Dict) -> Dict:
        img: np.ndarray = input["img"]
        h, w = img.shape[:2]

        dh = max(0, self.target_size[0] - h) / 2.0
        dw = max(0, self.target_size[1] - w) / 2.0
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        input["img"] = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.pad_value
        )
        return input


class CenterCrop:
    def __init__(self, target_size: Union[int, Tuple[int, int]] = 256):
        self.target_size = target_size
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)

    def __call__(self, input: Dict) -> Dict:
        img: np.ndarray = input["img"]
        h, w = img.shape[:2]

        # 计算裁剪的起始坐标（考虑奇数尺寸）
        start_h = max(0, (h - self.target_size[0]) // 2)
        start_w = max(0, (w - self.target_size[1]) // 2)

        # 确保裁剪区域不超出图像边界
        end_h = min(h, start_h + self.target_size[0])
        end_w = min(w, start_w + self.target_size[1])

        # 进行中心裁剪
        input["img"] = img[start_h:end_h, start_w:end_w]

        return input


class LetterBox:
    def __init__(
        self, target_size: Union[int, Tuple[int, int]] = 256, center=False, pad_val=114
    ):
        self.target_size = target_size
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        self.smallest_resize = SmallestResize(self.target_size)
        trans = [SmallestResize(self.target_size)]
        if center:
            trans.append(CenterPadding(self.target_size, pad_val))
        else:
            trans.append(Padding(self.target_size, pad_val))
        self.trans = Transform(trans)

    def __call__(self, input: Dict) -> Dict:
        return self.trans(input)


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


def xywh2xyxy(x: np.ndarray, scale=None) -> Dict:
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
