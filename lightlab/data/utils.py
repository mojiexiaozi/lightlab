from glob import glob
from PIL import Image, ImageOps
import contextlib
import gc, os
import numpy as np
from pathlib import Path
import hashlib

from lightlab.utils.types import IMG_SUFFIXS
from lightlab.utils import is_dir_writeable, LOGGER
from lightlab.utils.ops import segments2boxes


def get_hash(paths):
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()


def load_dataset_cache_file(path):
    gc.disable()
    cache = np.load(str(path), allow_pickle=True).item()
    gc.enable()
    return cache


def save_dataset_cache_file(path, cache):
    path = Path(path)
    if is_dir_writeable(path.parent):
        path.unlink(missing_ok=True)
        np.save(str(path), cache)
        path.with_suffix(".cache.npy").rename(path)
        LOGGER.info(f"New cache created: {path}")
    else:
        LOGGER.warning(
            f"Cache directory {path.parent} is not writeable, cache not saved."
        )


def get_images(images_dir, suffixs=IMG_SUFFIXS, recursive=False):
    images = [
        Path(file).as_posix()
        for file in glob(f"{images_dir}/**", recursive=recursive)
        if file.lower().endswith(suffixs)
    ]

    return images


def exif_size(im: Image.Image):
    s = im.size  # (width, height)
    if im.format == "JPEG":
        with contextlib.suppress(Exception):
            exif = im.getexif()
            if exif:
                rotation = exif.get(274, None)
                if rotation in [6, 8]:  # rotation 270 or 90
                    s = s[1], s[0]
    return s


def _verify_image(im_file):
    msg = ""
    im = Image.open(im_file)
    im.verify()
    w, h = exif_size(im)  # w, h
    assert (w > 9) and (h > 9), f"image size {w}x{h} < 10x10 pixels"
    assert f".{im.format.lower()}" in IMG_SUFFIXS, f"invalid image format {im.format}"
    if im.format.lower() in ("jpeg", "jpg"):
        with open(im_file, "rb") as f:
            f.seek(-2, 2)
            if f.read() != b"\xff\xd9":
                ImageOps.exif_transpose(Image.open(im_file)).save(
                    im_file, "JPEG", subsampling=0, quality=100
                )
                msg = f"WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
    return msg, (h, w)


def verify_image(args):
    # number (found, corrupt), message
    im_file = args[0]
    nf, nc, msg = 0, 0, ""
    try:
        msg, _ = _verify_image(im_file)
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"

    return args, nf, nc, msg


def verify_image_label(args):
    im_file, lb_file, keypoint, num_cls, nkpt, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify image
        msg, shape = _verify_image(im_file)

        # Verify label
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [
                        np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb
                    ]  # (cls, xy1...)
                    lb = np.concatenate(
                        (classes.reshape(-1, 1), segments2boxes(segments)), 1
                    )  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    # cls, x,y,w,h,(ndim*nkpt)
                    assert lb.shape[1] == (
                        5 + nkpt * ndim
                    ), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert (
                        lb.shape[1] == 5
                    ), f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                assert (
                    points.max() <= 1
                ), f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                # all labels
                max_cls = lb[:, 0].max()
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros(
                    (0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32
                )
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where(
                    (keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0
                ).astype(np.float32)
                # (nl, nkpt, 3)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1  # number corrupt
        msg = f"WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]
