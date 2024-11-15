from typing import Tuple

import lightning.pytorch as L

from posenc.datasets.brats import BratsDatasetDataModule
from posenc.datasets.chestx import ChestObjDetDataModule, ChestXDataModule
from posenc.datasets.echonet import EchoNetDataModule
from posenc.enums import (
    DataTaskType,
    ModelType,
    OptimizerType,
    PatchEmbeddingType,
    PosEncType,
    SchedulerType,
)
from posenc.modules.missing_channel import MissingChannelModule
from posenc.modules.next_frame import VideoGenViTModule
from posenc.modules.object_detection import DetectionModule
from posenc.modules.segmentation import SegmentationModule
from posenc.modules.video_regression import VideoViTModule
from posenc.modules.vision_transformer import ViTBinaryClsModule, ViTMultiClsModule


def create_experiment(
    task: DataTaskType,
    model_type: ModelType = ModelType.VIT_B,
    posenc: PosEncType = PosEncType.SINCOS,
    batch_size: int = 32,
    num_workers: int = 4,
    optimizer: OptimizerType = OptimizerType.SGD,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    scheduler: SchedulerType = SchedulerType.WARMUPCOSINE,
    warmup_epochs: int = 10,
    scale: float = 1.0,
    temperature: int = 10000,
) -> Tuple[L.LightningDataModule, L.LightningModule]:
    if task == task.CHESTX_CLS:
        data_module = ChestXDataModule(
            "binary", batch_size, num_workers, do_cutmix=False, do_mixup=False
        )
        model = ViTBinaryClsModule(
            posenc,
            model_type,
            optimizer,
            lr,
            weight_decay,
            scheduler,
            warmup_epochs,
            img_size=224,
            patch_embedding=PatchEmbeddingType.CONV,
            scale=scale,
            temperature=temperature,
        )

    if task == task.CHESTX_MULTI:
        data_module = ChestXDataModule(
            "multilabel", batch_size, num_workers, do_cutmix=False, do_mixup=False
        )
        model = ViTMultiClsModule(
            posenc,
            model_type,
            optimizer,
            lr,
            weight_decay,
            scheduler,
            warmup_epochs,
            img_size=224,
            patch_embedding=PatchEmbeddingType.CONV,
            scale=scale,
            temperature=temperature,
        )

    elif task == task.CHESTX_OBJ:
        data_module = ChestObjDetDataModule(batch_size, num_workers)
        model = DetectionModule(
            posenc,
            model_type,
            optimizer,
            lr,
            weight_decay,
            scheduler,
            warmup_epochs,
            scale=scale,
            temperature=temperature,
        )

    elif task == task.BRATS_SEG:
        crop_size = (96, 112, 96)
        data_module = BratsDatasetDataModule(
            batch_size, num_workers, random_dropout_channel=False, crop_size=crop_size
        )
        model = SegmentationModule(
            posenc,
            model_type,
            optimizer,
            lr,
            weight_decay,
            scheduler,
            warmup_epochs,
            scale=scale,
            temperature=temperature,
            crop_size=crop_size,
        )

    elif task == task.BRATS_GEN:
        data_module = BratsDatasetDataModule(
            batch_size, num_workers, random_dropout_channel=True
        )
        model = MissingChannelModule(
            posenc,
            model_type,
            optimizer,
            lr,
            weight_decay,
            scheduler,
            warmup_epochs,
        )

    elif task == task.ECHONET_REG:
        n_frames = 16
        sampling_rate = 4
        data_module = EchoNetDataModule(
            batch_size, num_workers, clip_length=n_frames, sampling_rate=sampling_rate
        )
        model = VideoViTModule(
            posenc,
            model_type,
            optimizer,
            lr,
            weight_decay,
            scheduler,
            warmup_epochs,
            n_frames=n_frames,
            scale=scale,
            temperature=temperature,
        )

    elif task == task.ECHONET_GEN:
        clip_length = 16
        sampling_rate = 4
        data_module = EchoNetDataModule(
            batch_size,
            num_workers,
            clip_length=clip_length,
            sampling_rate=sampling_rate,
        )
        model = VideoGenViTModule(
            posenc,
            model_type,
            optimizer,
            lr,
            weight_decay,
            scheduler,
            warmup_epochs,
            scale=scale,
            temperature=temperature,
        )
    else:
        raise NotImplementedError(f"DataModule {task} not implemented.")

    return data_module, model
