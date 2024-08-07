from torch.utils.data import Dataset
from pathlib import Path
from multiprocessing.pool import ThreadPool
from itertools import repeat
import numpy as np
import torch
import cv2
from copy import deepcopy
import math

from lightlab.cfg import AugmentParameters
from lightlab.utils import NUM_THREADS, TQDM, LOGGER, LOCAL_RANK
from lightlab.data.utils import (
    get_images,
    load_dataset_cache_file,
    save_dataset_cache_file,
    get_hash,
    verify_image_label,
)
from lightlab.utils.ops import resample_segments
from lightlab.utils.instance import Instances
from lightlab.data.augment import v8_transforms, Compose, LetterBox, Format


class YOLODataset(Dataset):
    def __init__(
        self,
        root,
        imgsz=640,
        nc=80,
        mode="train",
        task="detect",
        classes=None,
        batch_size=16,
        stride=32,
        rect=False,
        single_cls=False,
        pad=0.5,
        kpt_shape=None,
        flip_idx=[],
        params=AugmentParameters(),
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.imgsz = imgsz
        self.augment = mode == "train"
        self.single_cls = single_cls
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.kpt_shape = kpt_shape
        self.flip_idx = flip_idx
        self.pad = pad
        self.cfg = params
        self.nc = nc

        self.use_segments = task == "segment"
        self.use_obb = task == "obb"
        self.use_keypoints = task == "keypoints"

        self.im_files = get_images(self.root / "images")
        self.label_files = [
            Path(file)
            .parent.parent.joinpath(f"labels/{Path(file).stem}.txt")
            .as_posix()
            for file in self.im_files
        ]
        self.labels = self.get_labels()
        self.update_labels(classes)
        self.ni = len(self.labels)

        if self.rect:
            self.set_rectangle()

        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min(len(self.labels), self.batch_size)
        self.ims, self.im_hw0, self.im_hw = (
            [None] * self.ni,
            [None] * self.ni,
            [None] * self.ni,
        )
        self.transforms = self.build_transforms(params)

    def load_image(self, i, rect_mode=True):
        im, f = self.ims[i], self.im_files[i]
        if im is None:
            im = cv2.imread(f)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # origin hw
            # resize long side to imgsz while maintaining aspect ratio
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = min(math.ceil(w0 * r), self.imgsz), min(
                        math.ceil(h0 * r), self.imgsz
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(
                    im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                )
            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop()
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def close_mosaic(self, params: AugmentParameters):
        params.mosaic = 0.0
        params.copy_paste = 0.0
        params.mixup = 0.0
        self.transforms = self.build_transforms(params)

    def build_transforms(self, params: AugmentParameters):
        if self.augment:
            params.mosaic = 0.0 if self.rect else params.mosaic
            params.mixup = 0.0 if self.rect else params.mixup
            transforms = v8_transforms(self, self.imgsz, params)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=params.mask_ratio,
                mask_overlap=params.overlap_mask,
                bgr=params.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def set_rectangle(self):
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)
        nb = bi[-1] + 1  # number of batches
        shape = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = shape[:, 0] / shape[:, 1]  # aspect ratio
        irect = ar.argsort()  # aspect ratio order
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = (
            np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int)
            * self.stride
        )
        self.batch = bi  # batch index of image

    def update_labels(self, include_class=None):
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(axis=1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [
                        segments[si] for si, idx in enumerate(j) if idx
                    ]
                if keypoints:
                    self.labels[i]["keypoints"] = keypoints[j]

            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def get_labels(self):
        chace_file = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(chace_file), True
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(chace_file), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in (-1, 0):
            TQDM(
                None,
                desc=f"Scanning {chace_file}... {nf} images, {nm + ne} backgrounds, {nc} corrupt",
                total=n,
                initial=n,
            )
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        [cache.pop(k) for k in ("hash", "msgs")]
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(
                f"No labels found in {chace_file.parent}, training may not work correctly."
            )
        self.im_files = [lb["im_file"] for lb in labels]

        # Check if the dataset is all boxes or all segments
        lengths = (
            (len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels
        )
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(
                f"No labels found in {chace_file.parent}, training may not work correctly."
            )

        return labels

    def cache_labels(self, chace_file: Path):
        cache = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"Scanning {chace_file.parent / chace_file.stem}..."
        total = len(self.im_files)
        nkpt, ndim = (0, 0) if self.kpt_shape is None else self.kpt_shape
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.use_keypoints),
                    repeat(self.nc),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for (
                im_file,
                lb,
                shape,
                segments,
                keypoint,
                nm_f,
                nf_f,
                ne_f,
                nc_f,
                msg,
            ) in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    cache["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],
                            bboxes=lb[:, 1:],
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

            pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"No labels found in {chace_file.parent}")
        cache["hash"] = get_hash(self.label_files + self.im_files)
        cache["results"] = nf, nm, ne, nc, len(self.im_files)
        cache["msgs"] = msgs
        save_dataset_cache_file(chace_file, cache)
        return cache

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(
            index
        )
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def update_labels_info(self, label):
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            segments = np.stack(
                resample_segments(segments, n=segment_resamples), axis=0
            )
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(
            bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized
        )
        return label

    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            val = values[i]
            if k == "img":
                val = torch.stack(val, 0)
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                val = torch.cat(val, 0)
            new_batch[k] = val
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


if __name__ == "__main__":
    dataset = YOLODataset("assets/datasets/coco128/train")
    data = dataset[0]
    img = data["img"]
    img = img.permute(1, 2, 0).numpy()
    print(img.shape)
    from PIL import Image

    im = Image.fromarray(img)
    im.save("test.png")
