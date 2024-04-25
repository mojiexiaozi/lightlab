import math
import torch
from torch import nn, distributed
import os
import cpuinfo
import random
import numpy as np
import thop
import time
from copy import deepcopy
from contextlib import contextmanager

import torch.distributed

from .logger import LOGGER


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """确保主进程执行完"""
    initialized = distributed.is_available() and distributed.is_initialized()
    if initialized and local_rank not in (-1, 0):
        distributed.barrier(device_ids=[local_rank])
    yield  # 使用with语句会在这里暂停，退出with语句会继续执行
    if initialized and local_rank == 0:
        distributed.barrier(device_ids=[0])


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10, device=None):
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # device
            m = (
                m.half()
                if hasattr(m, "half")
                and isinstance(x, torch.Tensor)
                and x.dtype is torch.float16
                else m
            )
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = (
                    thop.profile(m, inputs=[x], verbose=False)[0] / 1e9 * 2
                    if thop
                    else 0
                )  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        (
                            sum(yi.sum() for yi in y) if isinstance(y, list) else y
                        ).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float("nan")
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available()
                    else 0
                )  # (GB)
                s_in, s_out = (
                    tuple(x.shape) if isinstance(x, torch.Tensor) else "list"
                    for x in (x, y)
                )  # shapes
                p = (
                    sum(x.numel() for x in m.parameters())
                    if isinstance(m, nn.Module)
                    else 0
                )  # parameters
                LOGGER.info(
                    f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}"
                )
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                LOGGER.info(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def init_seeds(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def make_divisible(x, divisor=8):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def intersect_dicts(da: dict, db: dict, exclude=()):
    return {
        k: v
        for k, v in da.items()
        if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape
    }


def select_device(device="", batch=0):
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")

    cpu = device == "cpu"
    mps = device in ("mps", "mps:0")
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        if device == "cuda":
            device = "0"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        if not (
            torch.cuda.is_available()
            and torch.cuda.device_count() >= len(device.split(","))
        ):
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
            )

    info = ""
    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(",") if device else "0"
        n = len(device)
        if n > 1 and batch > 0 and batch % n != 0:
            raise ValueError(
                f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}."
            )

        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            info += f"{'' if i == 0 else ''}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and torch.backends.mps.is_available():
        info += f"MPS ({get_cpu_info()})\n"
        arg = "mps"
    else:
        info += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"

    LOGGER.info(info)
    return torch.device(arg)


def get_cpu_info():
    k = ("brand_raw", "hardware_raw", "arch_string_raw")
    info = cpuinfo.get_cpu_info()
    string = info.get(
        k[0] if k[0] in info else k[1] if k[1] in info else k[2], "unknown"
    )
    return string.replace("(R)", "").replace("CPU ", "").replace("@ ", "")


def is_parallel(model):
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def de_parallel(model):
    return model.modul if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / tau)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)
