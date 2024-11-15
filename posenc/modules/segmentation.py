from typing import Tuple

import lightning.pytorch as L
import monai.transforms as T
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet

from posenc.enums import (
    ModelType,
    OptimizerType,
    PosEncType,
    SchedulerType,
    ViTSettings,
)
from posenc.metrics import BraTSDice, get_segmentation_metrics
from posenc.nets.models import UNETRPOS
from posenc.nets.optim import WarmupWithCosineDecay, WarmupWithExponentialDecay


class SegmentationModule(L.LightningModule):
    def __init__(
        self,
        posenc: PosEncType,
        model_type: ModelType,
        optimizer: OptimizerType,
        lr: float,
        weight_decay: float,
        scheduler: SchedulerType,
        warmup_epochs: int,
        scale: float = 1.0,
        temperature: int = 10000,
        crop_size: Tuple[int, int, int] = (64, 64, 64),
    ):
        super().__init__()
        self.posenc = posenc
        self.model_type = model_type
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs

        if model_type == ModelType.CNN:
            self.model = SegResNet(
                spatial_dims=3,
                in_channels=4,
                out_channels=4,
            )
        else:
            vit_settings = ViTSettings(model_type.value)
            self.model = UNETRPOS(
                posenc,
                scale=scale,
                temperature=temperature,
                mlp_dim=vit_settings.mlp_dim,
                num_heads=vit_settings.num_heads,
                crop_size=crop_size,
            )

        self.loss = DiceCELoss(
            include_background=True,
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            softmax=True,
        )
        self.lr = lr

        # Metric trackers
        self.train_metric = get_segmentation_metrics("train")
        self.train_dice_metrics = BraTSDice("train")

        self.valid_metric = get_segmentation_metrics("valid")
        self.valid_dice_metrics = BraTSDice("valid")

        self.post_trans = T.Compose(
            [T.Activations(sigmoid=True), T.AsDiscrete(threshold=0.5)]
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["image"]
        y = batch["mask"]

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        return loss, y, y_hat

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)

        self.train_metric(y_hat[:, 1:, ...], y.type(torch.int)[:, 1:, ...])
        self.train_dice_metrics(self.post_trans(y_hat), y)

        self.log_dict(
            self.train_dice_metrics.aggregate_dict(), on_epoch=True, on_step=False
        )
        self.log_dict(self.train_metric, on_epoch=True, on_step=False)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)

        self.valid_metric(y_hat[:, 1:, ...], y.type(torch.int)[:, 1:, ...])
        self.valid_dice_metrics(self.post_trans(y_hat), y)

        self.log_dict(
            self.valid_dice_metrics.aggregate_dict(), on_epoch=True, on_step=False
        )
        self.log_dict(self.valid_metric, on_epoch=True, on_step=False)
        self.log("valid/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        if self.optimizer == OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == OptimizerType.SGD:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Invalid optimizer {self.optimizer}")

        if self.scheduler == SchedulerType.COSINE:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10, eta_min=10e-8
            )
        elif self.scheduler == SchedulerType.WARMUPCOSINE:
            scheduler = WarmupWithCosineDecay(
                optimizer, warmup_steps=self.warmup_epochs, eta_min=10e-8
            )
        elif self.scheduler == SchedulerType.WARMUPEXP:
            scheduler = WarmupWithExponentialDecay(
                optimizer, warmup_steps=self.warmup_epochs, eta_min=10e-8
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
