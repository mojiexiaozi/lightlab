from dataclasses import dataclass


@dataclass
class AugmentParameters:
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    bgr: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    auto_augment: str = None  # (randaugment, autoaugment, augmix)
    erasing: float = 0.4
    overlap_mask: bool = True
    mask_ratio: int = 4


@dataclass
class HyperParameters:
    model: str = "yolov8-cls"
    scale: str = "n"
    data: str = ""
    nc: int = 3
    save: bool = True
    epochs: int = 100
    rect: bool = False
    patience: int = 100
    batch: int = -1
    imgsz: int = 224
    device: str = "cuda"
    workers: int = 8
    pretrain: bool = True
    optimizer: str = "auto"
    seed: int = 0
    amp: bool = True
    deterministic: bool = True
    save_period: int = 1
    verbose: bool = True
    cos_lr: bool = False
    close_mosaic: int = 10
    dropout = 0.0

    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    pose: float = 12.0
    kobj: float = 1.0
    label_smoothing: float = 0.0
    nbs: int = 64
