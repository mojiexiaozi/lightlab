from torchvision import transforms as T
from PIL import Image
import torch


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
