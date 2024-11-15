from typing import Tuple

import lightning.pytorch as L
import torch
from einops import rearrange
from monai.networks.nets import BasicUNet

from posenc.enums import (
    ModelType,
    OptimizerType,
    PosEncType,
    SchedulerType,
    ViTSettings,
)
from posenc.metrics import get_reconstruction_metrics
from posenc.nets.loss_functions import VideoSSIMLoss
from posenc.nets.models import Vid2Vid
from posenc.nets.optim import WarmupWithCosineDecay, WarmupWithExponentialDecay


def get_frame_masking_indeces(n_frames, masking_percentage):
    """Get the indeces of the frames that will be masked."""
    n_masked = int(n_frames * masking_percentage)
    noise = torch.rand(n_frames)
    masked_indeces = torch.argsort(noise)[:n_masked]
    return masked_indeces


def drop_channel(video, n_frames=6, num_dropped_frames=4):
    vid = video.clone()
    noise = torch.rand(n_frames)
    dropped_idx = torch.argsort(noise)[:num_dropped_frames]
    vid[:, dropped_idx] = torch.zeros_like(vid[:, dropped_idx])
    return vid


def video_to_image_batch(video: torch.Tensor) -> torch.Tensor:
    """Convert a video tensor to a batch of images."""
    return rearrange(video, "b f c h w -> (b f) c h w")


class VideoGenViTModule(L.LightningModule):
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

        self.model_type = model_type
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.scale = scale
        self.temperature = temperature

        if model_type == ModelType.CNN:
            self.model = BasicUNet(spatial_dims=3, out_channels=1)
        else:
            vit_settings = ViTSettings(model_type.value)
            self.hidden_size = 252
            self.dropout_rate = 0.0
            self.model = Vid2Vid(
                posenc=posenc,
                mlp_dim=vit_settings.mlp_dim,
                num_heads=vit_settings.num_heads,
                num_layers=vit_settings.num_layers,
                hidden_size=self.hidden_size,
                patch_size=16,
                n_frames=16,
                dropout_rate=self.dropout_rate,
                scale=scale,
                temperature=temperature,
            )

        self.loss = VideoSSIMLoss(win_size=16)

        # Metric trackers
        self.train_metric = get_reconstruction_metrics("train")
        self.valid_metric = get_reconstruction_metrics("valid")

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # These are the input images with dropped frames.
        image: torch.Tensor = batch["image"]

        # Put in data range 0 and 1 for SSIM loss function
        image = (image - image.min()) / (image.max() - image.min())
        dropped = drop_channel(image, n_frames=16, num_dropped_frames=14)

        pred: torch.Tensor = self.forward(dropped)

        loss = self.loss(pred, image)

        return loss, pred, image

    def training_step(self, batch, batch_idx):
        loss, pred, image = self._step(batch)

        self.train_metric(video_to_image_batch(pred), video_to_image_batch(image))

        self.log_dict(self.train_metric, on_epoch=True, on_step=False)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, image = self._step(batch)

        self.valid_metric(video_to_image_batch(pred), video_to_image_batch(image))

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
