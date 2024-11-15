from collections.abc import Sequence

import lightning.pytorch as L
import torch
import torch.nn as nn

from posenc.enums import (
    ModelType,
    OptimizerType,
    PosEncType,
    SchedulerType,
    ViTSettings,
)
from posenc.metrics import get_regression_metrics
from posenc.nets.models import VideoViT
from posenc.nets.optim import WarmupWithCosineDecay, WarmupWithExponentialDecay


class VideoViTModule(L.LightningModule):
    def __init__(
        self,
        posenc: PosEncType,
        model_type: ModelType,
        optimizer: OptimizerType,
        lr: float,
        weight_decay: float,
        scheduler: SchedulerType,
        warmup_epochs: int,
        n_frames: int = 16,
        scale: float = 1.0,
        temperature: int = 10000,
    ):
        super().__init__()

        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.model_type = model_type
        self.optimizer = optimizer
        self.scale = scale
        self.temperature = temperature

        if model_type == ModelType.CNN:
            raise ValueError("CNN not supported for video generation!")

        vit_settings = ViTSettings(model_type.value)
        self.model = VideoViT(
            posenc=posenc,
            n_frames=n_frames,
            img_size=112,
            patch_size=16,
            mlp_dim=vit_settings.mlp_dim,
            num_layers=vit_settings.num_layers,
            num_heads=vit_settings.num_heads,
            scale=scale,
            temperature=temperature,
            cls_head=True,
        )

        self.loss = nn.MSELoss()
        self.lr = lr

        # Metric trackers
        self.train_metric = get_regression_metrics("train")
        self.valid_metric = get_regression_metrics("valid")

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x = batch["image"]
        y = batch["target"]

        y_hat, _ = self.forward(x)
        y_hat = y_hat.flatten().clip(0, 100)  # Ejection fraction is between 0 and 100
        loss = self.loss(y_hat, y)

        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)

        self.train_metric(y_hat, y)

        self.log_dict(self.train_metric, on_epoch=True, on_step=False)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)

        self.valid_metric(y_hat, y)

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
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
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
                optimizer, gamma=0.97, warmup_steps=self.warmup_epochs, eta_min=10e-8
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
