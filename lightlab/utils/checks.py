import torch
import math
from copy import deepcopy
import numpy as np

from lightlab.nn.model import Model
from .torch_utils import profile
from .logger import LOGGER


def check_train_batch_size(model: Model, imgsz=640, amp=True):
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz, 0.4)


def autobatch(model, imgsz=640, fraction=0.60, batch_size=16):
    # Check device
    prefix = "AutoBatch: "
    LOGGER.info(f"{prefix}Computing optimal batch size for imgsz={imgsz}")
    device = next(model.parameters()).device  # get model device
    if device.type == "cpu":
        LOGGER.info(
            f"{prefix}CUDA not detected, using default CPU batch-size {batch_size}"
        )
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.warning(
            f"{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}"
        )
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    LOGGER.info(
        f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free"
    )

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)

        # Fit a solution
        y = [x[2] for x in results if x]  # memory [2]
        p = np.polyfit(batch_sizes[: len(y)], y, deg=1)  # first degree polynomial fit
        b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
        if None in results:  # some sizes failed
            i = results.index(None)  # first fail index
            if b >= batch_sizes[i]:  # y intercept above failure point
                b = batch_sizes[max(i - 1, 0)]  # select prior safe point
        if b < 1 or b > 1024:  # b outside of safe range
            b = batch_size
            LOGGER.warning(
                f"{prefix} CUDA anomaly detected, using default batch-size {batch_size}."
            )

        fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
        LOGGER.info(
            f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅"
        )
        return b
    except Exception as e:
        LOGGER.error(
            f"{prefix} error detected: {e},  using default batch-size {batch_size}."
        )
        return batch_size


def check_amp(model: Model):
    device = next(model.parameters()).device
    if device.type in ("cpu", "mps"):
        return False

    return False

    def amp_allclose(m, im):
        a = m(im, device=device, verbose=False)[0]


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TabError(f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. ")

    if len(imgsz) > max_dim:
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size.")
        LOGGER.warning(f"Updating to 'imgsz={max(imgsz)}'.")
        imgsz = [max(imgsz)]

    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]
    if sz != imgsz:
        LOGGER.warning(
            f"imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}"
        )
    sz = (
        [sz[0], sz[0]]
        if min_dim == 2 and len(sz) == 1
        else sz[0] if min_dim == 1 and len(sz) == 1 else sz
    )
    return sz
