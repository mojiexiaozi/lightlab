from torch import nn, optim, distributed
import torch, os, math, time
import warnings
from datetime import timedelta
from pathlib import Path
import numpy as np

from lightlab.cfg import HyperParameters
from lightlab.utils.torch_utils import (
    select_device,
    init_seeds,
    one_cycle,
    torch_distributed_zero_first,
    ModelEMA,
)
from lightlab.utils import RANK, LOGGER, TQDM
from lightlab.utils.paths import PRETRAIN_PATH
from lightlab.utils.metrics import Metrics
from lightlab.utils.checks import check_amp, check_imgsz, check_train_batch_size
from lightlab.data.build import build_dataloader
from lightlab.nn.model import Model


class Trainer:
    def __init__(self, cfg=HyperParameters()) -> None:
        self.cfg = cfg
        self.device = select_device(self.cfg.device)
        init_seeds(self.cfg.seed + 1 + RANK, self.cfg.deterministic)

        if RANK in (-1, 0):
            pass

        self.save_period = self.cfg.save_period
        self.batch_size = self.cfg.batch
        self.epochs = self.cfg.epochs
        self.start_epoch = 0
        if RANK == -1:
            print(self.cfg)

        if self.device.type in ("cpu", "mps"):
            self.cfg.workers = 0

        self.model = Model(
            self.cfg.model, self.cfg.scale, self.cfg.nc, self.cfg.verbose
        )
        self.ema = None
        self.criterion = None
        data = Path(self.cfg.data)
        self.trainset, self.valset = data / "train", data / "val"

    def _check_resume(self):
        if self.cfg.pretrain:
            name = self.cfg.model
            if name.startswith("yolo"):
                name = f"{name[:6]}{self.cfg.scale}{name[6:]}"
            if name in PRETRAIN_PATH:
                ckpt = torch.load(PRETRAIN_PATH[name])
                self.model.load(ckpt, self.cfg.verbose)
        return False

    def _setup_amp(self, world_size):
        self.amp = torch.tensor(self.cfg.amp).to(self.device)
        if self.amp and RANK in (-1, 0):
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
        if RANK > -1 and world_size > 1:  # DDP
            distributed.broadcast(self.amp, src=0)

        self.amp = self.cfg.amp = bool(self.amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def _get_metrics(self) -> Metrics:
        raise NotImplementedError

    def _update_metrics(self, preds, batch):
        raise NotImplementedError

    def _get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        with torch_distributed_zero_first(rank):
            dataset = self._get_dataset(dataset_path, mode, batch_size)
        return build_dataloader(
            dataset, batch_size, self.cfg.workers, shuffle=mode == "train", rank=rank
        )

    def _get_dataset(self, img_path, mode="train", batch=None):
        raise NotImplementedError

    def _setup_ddp(self, world_size):
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # set to enforce timeout
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        distributed.init_process_group(
            backend="nccl" if distributed.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_optimizer(self):
        # accumulate loss before optimizing
        self.accumulate = max(round(self.cfg.nbs / self.batch_size), 1)
        decay = self.cfg.weight_decay * self.batch_size * self.accumulate / self.cfg.nbs
        iterations = (
            math.ceil(
                len(self.train_loader.dataset) / max(self.batch_size, self.cfg.nbs)
            )
            * self.epochs
        )

        g = [], [], []
        # normalization layers, i.e. BatchNorm2d()
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        if self.cfg.optimizer == "auto":
            LOGGER.info(
                "determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.cfg.nc
            # lr0 fit equation to 6 decimal places
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = (
                ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            )
            self.cfg.warmup_bias_lr = 0.0

        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
            )
        optimizer.add_param_group(
            {"params": g[0], "weight_decay": decay}
        )  # add g0 with weight_decay
        optimizer.add_param_group(
            {"params": g[1], "weight_decay": 0.0}
        )  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"optimizer: {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        self.optimizer = optimizer

    def _setup_scheduler(self):
        if self.cfg.cos_lr:
            self.lf = one_cycle(1, self.cfg.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = (
                lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.cfg.lrf)
                + self.cfg.lrf
            )  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_train(self, world_size):
        self.start_epoch = 0
        self._check_resume()
        self._setup_model()
        self.model.to(self.device)
        # freeze layers
        always_freeze_names = [".dfl"]  # always freeze these layers
        for k, v in self.model.named_parameters():
            if any(x in k for x in always_freeze_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif (
                not v.requires_grad and v.dtype.is_floating_point
            ):  # only floating point Tensor can require gradients
                # only floating point Tensor can require gradients
                LOGGER.warning(f"setting 'requires_grad=True' for frozen layer '{k}'. ")
                v.requires_grad = True

        # check amp
        self._setup_amp(world_size)

        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[RANK]
            )

        # check imgsz
        gs = max(
            int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32
        )  # grid size (max stride)
        self.cfg.imgsz = check_imgsz(self.cfg.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs

        # batch size
        if self.batch_size == -1 and RANK == -1:
            self.cfg.batch = self.batch_size = check_train_batch_size(
                self.model, self.cfg.imgsz, self.amp
            )

        # Dataloader
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self._get_dataloader(
            self.trainset, batch_size=batch_size, rank=RANK, mode="train"
        )
        if RANK in (-1, 0):
            # self.val_loader = self._get_dataloader(
            #     self.valset, batch_size=batch_size, rank=-1, mode="val"
            # )
            self.metrics = self._get_metrics()
            self.ema = ModelEMA(self.model)

        self._setup_optimizer()
        self._setup_scheduler()

        # resume_training
        self.scheduler.last_epoch = self.start_epoch - 1
        self.stop = False

    def _preprocess_batch(self, batch):
        return batch

    def _save_model(self):
        pass

    def _setup_model(self):
        raise NotImplementedError

    def _warmup(self, epoch, ni, nw):
        xi = [0, nw]
        self.accumulate = max(
            1, int(np.interp(ni, xi, [1, self.cfg.nbs / self.batch_size]).round())
        )
        for j, x in enumerate(self.optimizer.param_groups):
            xi = [0, nw]  # x interp
            self.accumulate = max(
                1, int(np.interp(ni, xi, [1, self.cfg.nbs / self.batch_size]).round())
            )
            for j, x in enumerate(self.optimizer.param_groups):
                # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x["lr"] = np.interp(
                    ni,
                    xi,
                    [
                        self.cfg.warmup_bias_lr if j == 0 else 0.0,
                        x["initial_lr"] * self.lf(epoch),
                    ],
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(
                        ni, xi, [self.cfg.warmup_momentum, self.cfg.momentum]
                    )

    def _train_step(self, pbar, epoch, nb, nw, world_size):
        self.model.train()
        self.metrics.reset()
        for i, batch in pbar:
            ni = i + nb * epoch
            if ni <= nw:
                self._warmup(epoch, ni, nw)

            # forward
            with torch.cuda.amp.autocast(self.amp):
                batch = self._preprocess_batch(batch)

                preds = self.model(batch["img"])
                self.loss, self.loss_items = self.criterion(preds, batch)
                if RANK != -1:
                    self.loss *= world_size
                if RANK in (0, -1):
                    self._update_metrics(preds, batch)

                self.tloss = (
                    (self.tloss * i + self.loss_items) / (i + 1)
                    if self.tloss is not None
                    else self.loss_items
                )

            # backward
            self.scaler.scale(self.loss).backward()

            # optimize
            if ni - self.last_opt_step >= self.accumulate:
                self._optimizer_step()
                self.last_opt_step = ni

            # log
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
            losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)

            if RANK in {-1, 0}:
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                    % (
                        f"{epoch + 1}/{self.epochs}",
                        mem,
                        *losses,
                        batch["cls"].shape[0],
                        batch["img"].shape[-1],
                    )
                )

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp()
        self._setup_train(world_size)

        nb = len(self.train_loader)
        nw = (
            max(round(self.cfg.warmup_epochs * nb), 100)
            if self.cfg.warmup_epochs > 0
            else -1
        )
        self.last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        LOGGER.info(
            f"Image sizes {self.cfg.imgsz} train, {self.cfg.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to 'bold', self.save_dir)\n"
            f"Starting training for " + (f"{self.epochs} epochs...")
        )

        # if self.cfg.close_mosaic:
        #     base_idx = (self.epochs - self.cfg.close_mosaic) * nb
        self.epoch = self.start_epoch
        while True:
            with warnings.catch_warnings():
                # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                warnings.simplefilter("ignore")
                self.scheduler.step()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(self.epoch)
            pbar = enumerate(self.train_loader)
            if self.epoch == (self.epochs - self.cfg.close_mosaic):
                self._close_dataloader_mosaic()

            if RANK in (-1, 0):
                LOGGER.info(self._progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)

            self.tloss = None
            self.optimizer.zero_grad()
            self._train_step(pbar, self.epoch, nb, nw, world_size)
            LOGGER.info(self.metrics)

            # for loggers
            self.lr = {
                f"lr/pg{ir}": x["lr"]
                for ir, x in enumerate(self.optimizer.param_groups)
            }  # for loggers

            if RANK in (-1, 0):
                final_epoch = self.epoch + 1 >= self.epochs
                self.ema.update_attr(
                    self.model,
                    include=["yaml", "nc", "args", "names", "stride", "class_weights"],
                )

                self.stop = final_epoch

                if self.cfg.save or final_epoch:
                    self._save_model()

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            torch.cuda.empty_cache()

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                distributed.broadcast_object_list(
                    broadcast_list, 0
                )  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]

            if self.stop:
                break

            self.epoch += 1
        torch.cuda.empty_cache()

    def _close_dataloader_mosaic(self):
        pass

    def _progress_string(self):
        """Returns a string describing training progress."""
        return ""

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=10.0
        )  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def train(self):
        if isinstance(self.cfg.device, str) and len(self.cfg.device):
            world_size = len(self.cfg.device.split(","))
        elif isinstance(self.cfg.device, (tuple, list)):
            world_size = len(self.cfg.device)
        elif torch.cuda.is_available():
            world_size = 1
        else:
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            if self.cfg.rect:
                LOGGER.warning(
                    "'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'"
                )
                self.cfg.rect = False
            if self.cfg.batch == -1:
                LOGGER.warning(
                    "'batch=-1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.cfg.batch = 16

            # cmd, file = generate_ddp_command(world_size, self)
            # try:
            #     LOGGER.info(f'"DDP:" debug command {" ".join(cmd)}')
            #     subprocess.run(cmd, check=True)
            # except Exception as e:
            #     raise e
            # finally:
            #     ddp_cleanup(self, str(file))
        else:
            self._do_train(world_size)
