from functools import partial
from typing import Tuple

import lightning.pytorch as L
import torch
from monai.networks.nets import DenseNet121
from torch import nn
from torchvision.ops import complete_box_iou_loss

from posenc.enums import (
    ModelType,
    OptimizerType,
    PosEncType,
    SchedulerType,
    ViTSettings,
)
from posenc.metrics import IoU, get_classification_metrics
from posenc.nets.blocks import LinearHead
from posenc.nets.models import ViTDetection
from posenc.nets.optim import WarmupWithCosineDecay, WarmupWithExponentialDecay


class CNNDetection(nn.Module):
    def __init__(self):
        super().__init__()

        # Model
        self.model = DenseNet121(
            spatial_dims=2,
            in_channels=1,
            out_channels=1024,
            pretrained=False,
        )
        self.regression_head = LinearHead(in_features=1024, out_features=4, n_layers=3)
        self.classification_head = LinearHead(
            in_features=1024, out_features=8, n_layers=3
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.model(x)
        pred_labels = self.classification_head(x)
        pred_bbox = self.regression_head(x)
        return pred_labels, pred_bbox


class DetectionModule(L.LightningModule):
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
        reg_loss_weight: float = 50,
    ):
        super().__init__()

        self.posenc = posenc
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.reg_loss_weight = reg_loss_weight

        # Build model
        if model_type == ModelType.CNN:
            self.model = CNNDetection()
        elif "vit" in model_type.value:
            vit_settings = ViTSettings(model_type.value)
            self.model = ViTDetection(
                posenc,
                image_size=224,
                scale=scale,
                temperature=temperature,
                dim=vit_settings.mlp_dim,
                depth=vit_settings.num_layers,
                heads=vit_settings.num_heads,
            )

        # Loss functions
        self.cls_loss_function = nn.CrossEntropyLoss(
            torch.tensor(
                [
                    0.823,
                    0.852,
                    0.845,
                    0.882,
                    0.918,
                    0.912,
                    0.865,
                    0.902,
                ]
            )
        )
        self.reg_loss_function = partial(
            complete_box_iou_loss, reduction="mean"
        )  # nn.MSELoss()

        # Metrics
        self.train_cls_metrics = get_classification_metrics(
            "train", task="multiclass", num_classes=8
        )
        self.train_iou_metrics = IoU()

        self.valid_cls_metrics = get_classification_metrics(
            "valid", task="multiclass", num_classes=8
        )
        self.valid_iou_metrics = IoU()

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Return classification and regression outputs."""
        return self.model(x)

    def _step(self, batch) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Common forward step for training and validation with loss computation."""
        images, (labels, bboxes) = batch

        pred_labels, pred_bbox = self.forward(images)

        cls_loss = self.cls_loss_function(pred_labels, labels)
        reg_loss = self.reg_loss_function(pred_bbox.float(), bboxes.float())

        # Final loss is weighted average of classification and regression loss
        loss = self.reg_loss_weight * reg_loss + cls_loss

        return loss, cls_loss, reg_loss, pred_bbox, bboxes, pred_labels, labels

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, cls_loss, reg_loss, pred_bbox, bboxes, pred_labels, labels = self._step(
            batch
        )

        # Log metrics
        self.train_iou_metrics.update(
            pred_bbox.clone().detach().cpu().float(),
            bboxes.clone().detach().cpu().float(),
        )

        self.train_cls_metrics(pred_labels.softmax(1), labels)

        for i, s in enumerate(bboxes.std(dim=0)):
            self.log(f"train/bboxes/std_dim{i}", s.item())

        self.log("train/cls_loss", cls_loss)
        self.log("train/reg_loss", reg_loss)
        self.log("train/loss", loss, prog_bar=True)

        self.log("train/iou", self.train_iou_metrics, on_step=False, on_epoch=True)
        self.log_dict(self.train_cls_metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, cls_loss, reg_loss, pred_bbox, bboxes, pred_labels, labels = self._step(
            batch
        )

        # Log metrics
        self.valid_iou_metrics.update(
            pred_bbox.clone().detach().cpu().float(),
            bboxes.clone().detach().cpu().float(),
        )

        self.valid_cls_metrics(pred_labels.softmax(1), labels)

        for i, s in enumerate(bboxes.std(dim=0)):
            self.log(f"valid/bboxes/std_dim{i}", s.item())

        self.log("valid/cls_loss", cls_loss)
        self.log("valid/reg_loss", reg_loss)
        self.log("valid/loss", loss, prog_bar=True)

        self.log("valid/iou", self.valid_iou_metrics, on_step=False, on_epoch=True)
        self.log_dict(self.valid_cls_metrics, on_step=False, on_epoch=True)

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
