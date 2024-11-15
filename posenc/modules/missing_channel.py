import lightning.pytorch as L
import torch
from monai.networks.nets import SegResNet

from posenc.enums import (
    ModelType,
    OptimizerType,
    PosEncType,
    SchedulerType,
    ViTSettings,
)
from posenc.metrics import get_reconstruction_metrics
from posenc.nets.models import UNETRPOS
from posenc.nets.optim import WarmupWithCosineDecay, WarmupWithExponentialDecay


class MissingChannelModule(L.LightningModule):
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
    ):
        super().__init__()
        self.posenc = posenc
        self.model_type = model_type
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.scale = scale
        self.temperature = temperature

        if model_type == ModelType.CNN:
            self.model = SegResNet(in_channels=4, out_channels=1)
        elif "vit" in model_type.value:
            vit_settings = ViTSettings(model_type.value)
            self.model = UNETRPOS(
                posenc,
                scale=scale,
                temperature=temperature,
                mlp_dim=vit_settings.mlp_dim,
                num_heads=vit_settings.num_heads,
                out_channels=4,  # Always output all 4 channels. Dropped channel is selected later.
            )

        # Metric trackers
        self.train_metric = get_reconstruction_metrics("train")
        self.valid_metric = get_reconstruction_metrics("valid")

        self.loss_fn = torch.nn.MSELoss()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x: torch.Tensor = batch["image"]
        y: torch.Tensor = batch["channel_dropped"]
        dropped_idx: torch.Tensor = batch["channel_dropped_idx"]

        # Predict using masked image.
        y_hat: torch.Tensor = self.forward(x)

        # Select only the predictions for the dropped channel
        y_hat = y_hat[torch.arange(y_hat.shape[0]), dropped_idx.squeeze(), ...]

        # Remove the channel dimension
        y = y.squeeze()

        loss = self.loss_fn(y_hat, y)

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
                optimizer, warmup_steps=self.warmup_epochs, eta_min=10e-8
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
