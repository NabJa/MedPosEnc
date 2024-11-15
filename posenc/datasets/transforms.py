import pickle
from typing import List, Optional

import monai.transforms as T
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as F
from torchvision.transforms import v2


def read_summary(path):
    with open(path, mode="rb") as file:
        return pickle.load(file)


class NormalizeChannelWiseIntensity:
    def __init__(
        self,
        keys: str,
        mean: list,
        std: list,
        upper_thresholds: Optional[list] = None,
        lower_thresholds: Optional[list] = None,
        foreground_threshold: Optional[float] = None,
    ):
        self.keys = keys
        self.mean = mean
        self.std = std
        self.upper_thresholds = upper_thresholds
        self.lower_thresholds = lower_thresholds
        self.foreground_threshold = foreground_threshold

    def __call__(self, data: dict) -> dict:
        img = data[self.keys]
        nchannels = img.shape[0]

        assert (
            nchannels == len(self.mean) == len(self.std)
        ), f"Number of channels ({nchannels}) dont match the mean ({len(self.mean)}) and std ({len(self.std)})."

        for c in range(nchannels):
            channel = img[c]

            if self.lower_thresholds is not None:
                channel = torch.clip(channel, min=self.lower_thresholds[c])

            if self.upper_thresholds is not None:
                channel = torch.clip(channel, max=self.upper_thresholds[c])

            if self.foreground_threshold is not None:
                slices = channel > self.foreground_threshold
                masked = channel[slices]
            else:
                masked = channel

            channel[slices] = (masked - self.mean[c]) / self.std[c]
            img[c] = channel

        data[self.keys] = img

        return data


class NormalizeSpatial:
    def __init__(self, path_to_summary):
        self.summary = read_summary(path_to_summary)

    def __call__(self, img):
        img = (img - self.summary["mean_image"]) / self.summary["std_image"]
        return img


class Normalize:
    def __init__(self, path_to_summary):
        self.summary = read_summary(path_to_summary)

    def __call__(self, img):
        return F.normalize(img, self.summary["mean"], self.summary["std"])


class AddGaussianNoise:
    def __init__(self, p=0.0, mean=0.0, std=0.1):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, img):
        add_noise = np.random.binomial(1, self.p)
        if add_noise:
            return img + torch.randn(img.size()) * self.std + self.mean
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def get_cutmix_and_mixup_collate_function(
    do_cutmix: bool, do_mixup: bool, num_classes: int
):
    if do_cutmix and do_mixup:
        cutmix_or_mixup = v2.RandomChoice(
            [
                v2.CutMix(num_classes=num_classes),
                v2.MixUp(num_classes=num_classes),
            ]
        )

        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))

    elif do_cutmix:
        cutmix = v2.CutMix(num_classes=num_classes)

        def collate_fn(batch):
            return cutmix(*default_collate(batch))

    elif do_mixup:
        mixup = v2.MixUp(num_classes=num_classes)

        def collate_fn(batch):
            return mixup(*default_collate(batch))

    else:
        collate_fn = default_collate

    return collate_fn


class ProcessBratsMask:
    def __init__(self, mask_key="mask"):
        """Process the mask to have the following classes:
        0: Background
        1: GD-Enhancing Tumor
        2: Necrotic and Non-Enhancing Tumor
        3: Peritumoral Edema
        """
        self.mask_key = mask_key

    def __call__(self, data):
        """Fill masks to not have large holes."""
        mask = data[self.mask_key]

        mask[2] = mask[2] - mask[1]
        mask[1] = mask[1] + mask[2]

        # Ensure background is as expected. Sometimes there are artifacts because of e.g. padding.
        mask[0] = mask[1:].sum(dim=0) == 0

        data[self.mask_key] = mask

        return data


class DropoutRandomChannel:
    """Set a random channel to zero. This is for a image generation task."""

    def __init__(self, image_key: str, nchannels: int = 4):
        self.image_key = image_key
        self.nchannels = nchannels

    def __call__(self, data):

        droped_channel = torch.randint(0, self.nchannels, (1,))

        data["channel_dropped_idx"] = droped_channel
        data["channel_dropped"] = data[self.image_key][droped_channel]

        data[self.image_key][droped_channel] = torch.zeros_like(
            data[self.image_key][droped_channel]
        )

        return data


class ClearDroppedChannel:
    """Set all values to 0 in dropped channel. DropoutRandomChannel has to be applied first."""

    def __init__(self, image_key: str):
        self.image_key = image_key

    def __call__(self, data):

        channel_dropped_idx = data["channel_dropped_idx"]

        data[self.image_key][channel_dropped_idx] = torch.zeros_like(
            data[self.image_key][channel_dropped_idx]
        )

        return data


class ScaleChannelWisePercentile:
    def __init__(
        self,
        image_key: str,
        lower: List[float],
        upper: List[float],
        b_min: float = 0.0,
        b_max: float = 1.0,
        clip: bool = True,
    ) -> None:
        self.image_key = image_key
        self.lower = lower
        self.upper = upper
        self.b_min = b_min
        self.b_max = b_max
        self.scalers = [
            T.ScaleIntensityRange(amin, amax, self.b_min, self.b_max, clip=clip)
            for amin, amax in zip(self.lower, self.upper)
        ]

    def __call__(self, data):
        assert len(data[self.image_key]) == len(
            self.scalers
        ), "Number of channels mismatch."

        for i, scaler in enumerate(self.scalers):
            data[self.image_key][i] = scaler(data[self.image_key][i])

        return data
