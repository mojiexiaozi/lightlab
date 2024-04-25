from torch.utils.data import Dataset
from pathlib import Path
from multiprocessing.pool import ThreadPool
from itertools import repeat
import numpy as np

from lightlab.cfg import AugmentParameters
from lightlab.utils import NUM_THREADS, TQDM, LOGGER, LOCAL_RANK
from lightlab.data.utils import (
    get_images,
    load_dataset_cache_file,
    save_dataset_cache_file,
    get_hash,
    verify_image_label,
)


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


if __name__ == "__main__":
    dataset = YOLODataset("assets/datasets/coco128/train")
