from torch import nn
import torch

from lightlab.cfg import HyperParameters
from lightlab.core.trainer import Trainer
from lightlab.data.classify_dataset import ClassifyDataset
from lightlab.utils.metrics import ClassifyMetrics, Metrics


class ClassifyLoss:
    def __call__(self, preds, batch):
        loss = nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class ClassifyTrainer(Trainer):
    def __init__(self, cfg=HyperParameters()) -> None:
        self.loss_names = ["loss"]
        super().__init__(cfg)
        self.criterion = ClassifyLoss()

    def _setup_model(self):
        model = self.model
        for m in model.modules():
            if not self.cfg.pretrain and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.cfg.dropout:
                m.p = self.cfg.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training

    def _update_metrics(self, preds: torch.Tensor, batch):
        self.metrics.update(preds.softmax(1), batch["cls"])

    def _get_metrics(self) -> Metrics:
        return ClassifyMetrics()

    def _get_dataset(self, img_path, mode="train", batch=None):
        return ClassifyDataset(img_path, self.cfg.imgsz, mode=mode)

    def _preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def _progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )


if __name__ == "__main__":
    cfg = HyperParameters()
    cfg.model = "yolov8-cls"
    cfg.device = "cpu"
    cfg.pretrain = True
    cfg.imgsz = 224
    cfg.data = "assets/datasets/pole-cls"
    cfg.nc = 3
    cfg.epochs = 200
    cfg.batch = 8
    cfg.workers = 8
    trainer = ClassifyTrainer(cfg)
    trainer.train()
