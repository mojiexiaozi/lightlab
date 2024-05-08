import torch
import numpy as np


class ConfusionMetrix:
    def __init__(self, nc: int, conf=0.25, iou_thres=0.45, task="detect") -> None:
        self.task = task
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        self.matrix = (
            torch.zeros((nc + 1, nc + 1)) if task == "detect" else torch.zeros((nc, nc))
        )

    def process_cls_preds(self, preds: torch.Tensor, targets: torch.Tensor):
        preds, targets = torch.cat(preds), torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p, t] += 1

    def process_preds(
        self, preds: torch.Tensor, gt_bboxes: torch.Tensor, gt_cls: torch.Tensor
    ):
        if gt_cls.shape[0] == 0:
            if preds is not None:
                preds = preds[preds[:, 4] > self.conf]
                preds_classes = preds[:, 5].int()
                for dc in preds_classes:
                    self.matrix[dc, self.nc] += 1  # false positives
            return

        if preds is None:
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        preds = preds[preds[:, 4] > self.conf]
        gt_classes = gt_cls.int()
        preds_classes = preds[:, 5].int()
        is_obb = preds.shape[1] == 7 and gt_bboxes.shape[1] == 5

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false negatives
        # remove background class if task=detect
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)


class Metrics:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError

    def _process(self, preds, targets):
        raise NotImplementedError

    def __str__(self) -> str:
        results_dict = self.results_dict
        return f'{" | ".join([f"{k}: {v:.4f}" for k, v in results_dict.items()])}'

    @property
    def results_dict(self):
        raise NotImplementedError

    @property
    def keys(self):
        raise NotImplementedError

    @property
    def fitness(self):
        raise NotImplementedError


class ClassifyMetrics(Metrics):
    def reset(self):
        self.top1 = 0
        # self.top5 = 0
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds.argsort(1, descending=True)[:, :5])
        self.targets.append(targets)

    def _process(self, preds, targets):
        preds, targets = torch.cat(preds), torch.cat(targets)
        correct = (targets[:, None] == preds).float()
        acc = correct[:, 0]
        self.top1 = acc.mean(0).tolist()
        # acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)
        # self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def results_dict(self):
        self._process(self.preds, self.targets)
        return dict(zip(self.keys + ["fitness"], [self.top1, self.fitness]))

    @property
    def keys(self):
        return ["metrics/accuracy_top1"]

    @property
    def fitness(self):
        return self.top1


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (
        np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)
    ).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(
        0
    )

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)
