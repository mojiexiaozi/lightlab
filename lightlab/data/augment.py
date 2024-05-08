from torchvision import transforms as T
from PIL import Image
import torch
from typing import Union
import random
import numpy as np
import cv2
import math
from copy import deepcopy
import albumentations as A

from lightlab.utils.instance import Instances
from lightlab.utils.ops import (
    segment2box,
    polygons2masks_overlap,
    polygons2masks,
    xyxyxyxy2xywhr,
)
from lightlab.utils.metrics import bbox_ioa
from lightlab.cfg import AugmentParameters
from lightlab.utils import LOGGER


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        self.transforms.append(transform)

    def insert(self, index, transform):
        self.transforms.insert(index, transform)

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        assert isinstance(
            index, (int, list)
        ), f"The indices should be either list or int type but got {type(index)}"
        index = [index] if isinstance(index, int) else index
        return Compose([self.transforms[i] for i in index])

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        assert isinstance(
            index, (int, list)
        ), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(
                value, list
            ), f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(
                self.transforms
            ), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        return self.transforms

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"
        )


class BaseMixTransform:
    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        if random.uniform(0, 1) > self.p:
            return labels

        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        labels = self._update_label_text(labels)
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _mix_transform(self, labels):
        raise NotImplementedError

    def get_indexes(self):
        raise NotImplementedError

    def _update_label_text(self, labels):
        if "texts" not in labels:
            return labels

        mix_texts = sum(
            [labels["texts"]] + [x["texts"] for x in labels["mix_labels"]], []
        )
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {3, 4}, "grid must be equal to 3, 4 "
        super().__init__(dataset=dataset, p=p)
        self.dataset = dataset
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n

    def get_indexes(self, buffer=True):
        if buffer:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        assert (
            labels.get("rect_shape", None) is None
        ), "rect and mosaic are mutually exclusive."
        assert len(
            labels.get("mix_labels", [])
        ), "There are no other images for mosaic augment."
        # This code is modified for mosaic3 method.
        return self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels)

    def _mosaic3(self, labels):
        """Create a 1x3 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img3
            if i == 0:  # center
                img3 = np.full(
                    (s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8
                )  # base image with 3 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:  # left
                c = s - w, s + h0 - h, s, s + h0

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            img3[y1:y2, x1:x2] = img[
                y1 - padh :, x1 - padw :
            ]  # img3[ymin:ymax, xmin:xmax]
            # hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(
                labels_patch, padw + self.border[0], padh + self.border[1]
            )
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img3[
            -self.border[0] : self.border[0], -self.border[1] : self.border[1]
        ]
        return final_labels

    def _mosaic4(self, labels):
        """Create a 2x2 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (
            int(random.uniform(-x, 2 * s + x)) for x in self.border
        )  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full(
                    (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
                )  # base image with 4 tiles
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh):
        """Update labels."""
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class MixUp(BaseMixTransform):
    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate(
            [labels["instances"], labels2["instances"]], axis=0
        )
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


class RandomPerspective:
    def __init__(
        self,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        border=(0, 0),
        pre_transform=None,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # mosaic border
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        # Center
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(
            -self.perspective, self.perspective
        )  # x perspective (about y)
        P[2, 1] = random.uniform(
            -self.perspective, self.perspective
        )  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )  # x shear (deg)
        S[1, 0] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        )  # x translation (pixels)
        T[1, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]
        )  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (
            (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any()
        ):  # image changed
            if self.perspective:
                img = cv2.warpPerspective(
                    img, M, dsize=self.size, borderValue=(114, 114, 114)
                )
            else:  # affine
                img = cv2.warpAffine(
                    img, M[:2], dsize=self.size, borderValue=(114, 114, 114)
                )
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(
            n, 8
        )  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return (
            np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype)
            .reshape(4, n)
            .T
        )

    def apply_segments(self, segments, M):
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack(
            [segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0
        )
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints, M):
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (
            (xy[:, 0] < 0)
            | (xy[:, 1] < 0)
            | (xy[:, 0] > self.size[0])
            | (xy[:, 1] > self.size[1])
        )
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels):
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad

        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(
            bboxes, segments, keypoints, bbox_format="xyxy", normalized=False
        )
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T,
            box2=new_instances.bboxes.T,
            area_thr=0.01 if len(segments) else 0.10,
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + eps) > area_thr)
            & (ar < ar_thr)
        )  # candidates


class RandomHSV:
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        img = labels["img"]
        if self.hgain or self.sgain or self.vgain:
            r = (
                np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
            )  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge(
                (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
            )
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels


class RandomFlip:
    def __init__(self, p=0.5, direction="horizontal", flip_idx=None) -> None:
        assert direction in {
            "horizontal",
            "vertical",
        }, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # Flip up-down
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            # For keypoints
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(
                    instances.keypoints[:, self.flip_idx, :]
                )
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class CopyPaste:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, labels):
        im = labels["img"]
        cls = labels["cls"]
        h, w = im.shape[:2]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)
        if self.p and len(instances.segments):
            n = len(instances)
            _, w, _ = im.shape  # height, width, channels
            im_new = np.zeros(im.shape, np.uint8)

            # Calculate ioa first then select indexes randomly
            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)

            ioa = bbox_ioa(
                ins_flip.bboxes, instances.bboxes
            )  # intersection over area, (N, M)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                cv2.drawContours(
                    im_new,
                    instances.segments[[j]].astype(np.int32),
                    -1,
                    (1, 1, 1),
                    cv2.FILLED,
                )

            result = cv2.flip(im, 1)  # augment segments (flip left-right)
            i = cv2.flip(im_new, 1).astype(bool)
            im[i] = result[i]

        labels["img"] = im
        labels["cls"] = cls
        labels["instances"] = instances
        return labels


class LetterBox:
    def __init__(
        self,
        new_shape=(640, 640),
        auto=False,
        scaleFill=False,
        scaleup=True,
        center=True,
        stride=32,
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class Albumentations:
    def __init__(self, p=1.0):
        self.p = p
        self.transform = None
        try:

            # Transforms
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]
            self.transform = A.Compose(
                T,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )

            LOGGER.info(
                ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p)
            )
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{e}")

    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels["img"]
        cls = labels["cls"]
        if len(cls):
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*im.shape[:2][::-1])
            bboxes = labels["instances"].bboxes
            # TODO: add supports of segments and keypoints
            if self.transform and random.random() < self.p:
                new = self.transform(
                    image=im, bboxes=bboxes, class_labels=cls
                )  # transformed
                if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
            labels["instances"].update(bboxes=bboxes)
        return labels


class Format:
    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
        bgr=0.0,
    ):
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # set False when training detection only
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # keep the batch indexes
        self.bgr = bgr

    def __call__(self, labels):
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(
                    1 if self.mask_overlap else nl,
                    img.shape[0] // self.mask_ratio,
                    img.shape[1] // self.mask_ratio,
                )
            labels["masks"] = masks
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = (
            torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        )
        if self.return_keypoint:
            labels["keypoints"] = torch.from_numpy(instances.keypoints)
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments))
                if len(instances.segments)
                else torch.zeros((0, 5))
            )
        # NOTE: need to normalize obb in xywhr format for width-height consistency
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        # Then we can use collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(
            img[::-1] if random.uniform(0, 1) > self.bgr else img
        )
        img = torch.from_numpy(img)
        return img

    def _format_segments(self, instances, cls, w, h):
        """Convert polygon points to bitmap."""
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap(
                (h, w), segments, downsample_ratio=self.mask_ratio
            )
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks(
                (h, w), segments, color=1, downsample_ratio=self.mask_ratio
            )

        return masks, instances, cls


def v8_transforms(
    dataset,
    imgsz,
    hyp: AugmentParameters,
    stretch=False,
):
    pre_transform = Compose(
        [
            Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
            CopyPaste(p=hyp.copy_paste),
            RandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
            ),
        ]
    )
    flip_idx = dataset.flip_idx  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning(
                "WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'"
            )
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(
                f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}"
            )
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms


def classify_transforms(size=224, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        scale_size = size, size
    else:
        scale_size = size

    if scale_size[0] == scale_size[1]:
        transforms = [T.Resize(scale_size[0], interpolation=interpolation)]
    else:
        transforms = [T.Resize(scale_size)]

    transforms += [T.CenterCrop(size), T.ToTensor()]
    return T.Compose(transforms)


def classify_augmentations(
    size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    auto_augment=None,
    erasing=0.0,
    interpolation=Image.BILINEAR,
):
    if not isinstance(size, int):
        raise TypeError(
            f"classify_transforms() size {size} must be integer, not (list, tuple)"
        )

    scale = tuple(scale or (0.08, 1.0))
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))
    primary_tfl = [
        T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)
    ]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    if auto_augment:
        if auto_augment == "randaugment":
            secondary_tfl.append(T.RandAugment(interpolation=interpolation))
        elif auto_augment == "augmix":
            secondary_tfl.append(T.AugMix(interpolation=interpolation))
        elif auto_augment == "autoaugment":
            secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment" or None'
            )

    final_tfl = [
        T.ToTensor(),
        T.Normalize(
            mean=torch.tensor((0.0, 0.0, 0.0)), std=torch.tensor((1.0, 1.0, 1.0))
        ),
        T.RandomErasing(p=erasing, inplace=True),
    ]
    return T.Compose(primary_tfl + secondary_tfl + final_tfl)
