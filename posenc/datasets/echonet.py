from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import imageio
import lightning.pytorch as L
import monai.transforms as T
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from monai.data import DataLoader
from torchvision.transforms.functional import rgb_to_grayscale
from tqdm import tqdm

ECHONET_STATS = {"mean": 57.6, "std": 54.1, "median": 40}


########################
#### Preprocessing #####
########################


def read_video(path):
    """Read video from path and return as numpy array with shape (frames, channels, height, width)."""
    vid = imageio.get_reader(path, "ffmpeg")
    video = [i for i in vid.iter_data()]
    return rearrange(
        video, "frames height width channels -> frames channels height width"
    )


def _save_transformed_video(path):
    video = torch.tensor(read_video(path))
    video = rgb_to_grayscale(video)
    torch.save(video, path.with_suffix(".pt"))


def transform_all_videos_to_grayscaled_tensors(input_dir, nprocesses=12):
    input_dir = Path(input_dir)
    videos = list(input_dir.glob("*.avi"))
    with Pool(nprocesses) as p:
        _ = list(p.imap(_save_transformed_video, tqdm(videos)))


########################
##### Transforms #######
########################


class ImageDropout:
    """Set a random channel and / or pixels to zero. This is for a image generation task."""

    def __init__(
        self,
        keys=None,
        channel_wise: bool = True,
        pixel_wise: bool = False,
        p: float = 0.5,
    ):
        """
        Args:
            channel_wise (bool, optional): Whether dropout channel-wise. Defaults to True.
            pixel_wise (bool, optional): Whether dropout pixel-wise. Defaults to False.
            p (float, optional): Probability of applying dropout. Must be in the range [0, 1]. Defaults to 0.5.

        Raises:
            AssertionError: If both channel_wise and pixel_wise are False.
            AssertionError: If p is not in the range [0, 1].
        """
        self.keys = keys
        self.channel_wise = channel_wise
        self.pixel_wise = pixel_wise
        self.p = p

        assert (
            channel_wise or pixel_wise
        ), "At least one of channel_wise or pixel_wise must be True."

        assert 0 < p < 1, "p must be in the range [0, 1]"

    def __call__(self, data: dict) -> torch.Tensor:
        """Set a random channel or values to zero. This is for a image generation task."""

        img = data[self.keys]

        if self.channel_wise:
            # Drop every channel with probability p
            mask = torch.rand(img.shape[0]) < self.p
            img[mask] = 0.0
            data["channel_dropped_idx"] = mask

        if self.pixel_wise:
            # Drop every foreground pixel with probability p
            fg_idx = torch.where(img > 0)
            fg_mask = torch.rand(fg_idx[0].shape) < self.p
            fg_idx = [i[fg_mask] for i in fg_idx]
            img[fg_idx] = 0

        data[self.keys] = img
        return data


class LoadVideoClip:
    """Read video and select random clip of length 'clip_length'."""

    def __init__(
        self, keys="image", clip_length: Optional[int] = 16, sampling_rate: int = 4
    ):
        """
        Args:
            keys (str): The keys to be loaded from the dataset. Defaults to "image".
            clip_length (int, optional): The length of each video clip. Defaults to 16. If None, the whole video is returned.
            sampling_rate (int, optional): The sampling rate for the video. Defaults to 1.
        """
        self.keys = keys
        self.clip_length = clip_length
        self.sampling_rate = sampling_rate

    def __call__(self, data: dict) -> torch.Tensor:
        """Read video and select random clip of length 'clip_length'."""

        path = data[self.keys]
        data["path"] = path

        video: torch.Tensor = torch.load(path)
        video = video.float()

        if self.clip_length is None:
            return video

        # Apply sampling rate
        if len(video) / self.sampling_rate > self.clip_length:
            video = video[:: self.sampling_rate]
        start = np.random.randint(0, len(video) - self.clip_length)
        data[self.keys] = video[start : start + self.clip_length]

        return data


########################
##### Dataclasses ######
########################


class EchoNetDataset:
    """
    The dataset is a collection of echocardiograms from the EchoNet challenge.
    It can be used to predict the ejection fraction (EF), end-diastolic volume (EDV), or end-systolic volume (ESV).
    We also use it for image generation tasks using the dropout transforms.
    """

    def __init__(
        self,
        mode: str = "train",
        output_variable: str = "ef",
        transforms=None,
    ) -> None:
        """
        Args:
            mode: The mode of the dataset. Can be "train", "val", or "test". Defaults to "train".
            output_variable: The output variable to predict. Can be "ef", "edv", or "esv". Defaults to "ef".
            transforms: A list of transforms to apply to the data. Defaults to None.
        """
        self.root = Path("/sc-scratch/sc-scratch-gbm-radiomics/posenc/echonet-dynamic")
        self.video_path = self.root / "processed"
        self.data = pd.read_csv(self.root / "FileList.csv")
        # self.volume_tracing = pd.read_csv(self.root / "VolumeTracings.csv")

        self.transforms = transforms

        assert output_variable.lower() in ["ef", "edv", "esv"]
        self.output_variable = output_variable.upper()

        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.data = self.data[self.data.Split.str.lower() == mode]

        # Remove videos with FrameWidth or FrameHeight unequal 112
        self.data = self.data[
            (self.data.FrameWidth == 112) & (self.data.FrameHeight == 112)
        ]

        self.data.reset_index(inplace=True, drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample from the dataset."""

        file_name, out_variable = self.data.loc[idx, ["FileName", self.output_variable]]
        video_path = str(self.video_path / f"{file_name}.pt")
        target = torch.tensor(out_variable, dtype=torch.float32)

        sample = {
            "image": video_path,
            "target": target,
            "target_name": self.output_variable,
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample


class EchoNetDataModule(L.LightningDataModule):
    """
    DataModule for the EchoNet dataset.
    This class is used to load the data, create the dataloaders and handle the specific taks transformations.
    Can be regression of output_variable or image generation with channel or pixel wise dropout.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 12,
        output_variable: str = "ef",
        clip_length: int = 16,
        sampling_rate: int = 1,
        dropout_channel: bool = False,
        dropout_pixel: bool = False,
        dropout_p: float = 0.5,
    ):
        """
        Initializes the Echonet dataset.

        Args:
            batch_size (int): The batch size for data loading. Default is 32.
            num_workers (int): The number of worker threads for data loading. Default is 12.
            output_variable (str): The output variable to predict. Default is "ef".
            clip_length (int): The length of video clips. Default is 16.
            dropout_channel (bool): Whether to apply channel dropout. Default is False.
            dropout_pixel (bool): Whether to apply pixel dropout. Default is False.
            dropout_p (float): The dropout probability. Default is 0.5.
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_variable = output_variable
        self.clip_length = clip_length
        self.drop_channel = dropout_channel
        self.drop_pixel = dropout_pixel
        self.dropout_p = dropout_p
        self.sampling_rate = sampling_rate

        dropout = dropout_channel or dropout_pixel

        self.train_transforms = self._get_transforms(augment=True, dropout=dropout)
        self.valid_transform = self._get_transforms(augment=False, dropout=dropout)

        self.save_hyperparameters()

    def _get_transforms(self, augment: bool = False, dropout=False):
        transforms = [
            LoadVideoClip(
                clip_length=self.clip_length, sampling_rate=self.sampling_rate
            ),
            T.NormalizeIntensityd(
                keys="image",
                subtrahend=ECHONET_STATS["mean"],
                divisor=ECHONET_STATS["std"],
                nonzero=True,
            ),
        ]

        if dropout:
            transforms += [
                T.CopyItemsd(keys="image"),
                ImageDropout(
                    keys="image",
                    channel_wise=self.drop_channel,
                    pixel_wise=self.drop_pixel,
                    p=self.dropout_p,
                ),
            ]

        if augment:
            transforms += [
                T.RandGaussianNoised(keys="image", prob=0.33, mean=0, std=0.1),
                T.RandShiftIntensityd(keys="image", prob=0.33, offsets=0.1),
                T.RandScaleIntensityd(keys="image", prob=0.33, factors=0.1),
            ]

        return T.Compose(transforms)

    def setup(self, stage: str = None):
        self.train = EchoNetDataset(
            mode="train",
            output_variable=self.output_variable,
            transforms=self.train_transforms,
        )
        self.val = EchoNetDataset(
            mode="val",
            output_variable=self.output_variable,
            transforms=self.valid_transform,
        )
        self.test = EchoNetDataset(
            mode="test",
            output_variable=self.output_variable,
            transforms=self.valid_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
