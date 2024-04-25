from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import cv2
import contextlib
from multiprocessing.pool import ThreadPool

from lightlab.data.utils import (
    get_images,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    get_hash,
)
from lightlab.data.augment import classify_augmentations, classify_transforms
from lightlab.cfg import AugmentParameters
from lightlab.utils import LOCAL_RANK, LOGGER, TQDM, NUM_THREADS


# 分类数据集
class ClassifyDataset(Dataset):
    def __init__(
        self,
        root,
        imgsz=224,
        class_to_idx=None,
        mode="train",
        params=AugmentParameters(),
    ) -> None:
        super().__init__()
        self.root = Path(root)
        classes = [path.name for path in Path(root).iterdir() if path.is_dir()]
        classes = sorted(classes)
        self.class_to_idx = class_to_idx
        if class_to_idx is None:
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        self.samples = [
            (file, self.class_to_idx[Path(file).parent.name])
            for file in get_images(root, recursive=True)
        ]
        self.samples = self.verify_images()

        if mode == "train":
            scale = (1.0 - params.scale, 1.0)
            self.transforms = classify_augmentations(
                size=imgsz,
                scale=scale,
                hflip=params.fliplr,
                vflip=params.flipud,
                erasing=params.erasing,
                auto_augment=params.auto_augment,
            )
        else:
            self.transforms = classify_transforms(size=imgsz)

    def verify_images(self):
        desc = f"Scanning {self.root}..."
        cache_file = self.root.with_suffix(".cache")
        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            cache = load_dataset_cache_file(cache_file)
            # identical hash
            assert cache["hash"] == get_hash([x[0] for x in self.samples])
            LOGGER.info(f"Loaded cache from {cache_file}")
            # found, corrupt, total, samples
            nf, nc, n, samples = cache.pop("results")
            if LOCAL_RANK in (-1, 0):
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        nf, nc, msgs, samples, cache = 0, 0, [], [], {}

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=self.samples)
            pbar = TQDM(results, desc=desc, total=len(self.samples))
            for sample, nf_f, nc_f, msg in pbar:
                if nf_f:
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f"{desc} {nf} images, {nc} corrupt"
            pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
        cache["hash"] = get_hash([x[0] for x in self.samples])
        cache["results"] = nf, nc, len(self.samples), samples
        cache["msgs"] = msgs
        save_dataset_cache_file(cache_file, cache)
        return samples

    def __getitem__(self, index):
        file, cls = self.samples[index]
        im = cv2.imread(file)  # BGR
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        return {"img": self.transforms(im), "cls": cls}

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    dataset = ClassifyDataset("assets/datasets/pole-cls/train", mode="val")
    print(len(dataset))
    print(dataset[0]["img"].max())
    import torchvision

    torchvision.utils.save_image(dataset[0]["img"], "cls.png")
    torchvision.utils.save_image(dataset[50]["img"], "cls1.png")
    torchvision.utils.save_image(dataset[100]["img"], "cls2.png")
    print(dataset[0]["cls"])
