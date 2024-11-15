from pathlib import Path
from typing import Callable, Dict, Optional, Union

import lightning.pytorch as L
import monai.transforms as T
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader
from tqdm import tqdm

from posenc.datasets.transforms import ClearDroppedChannel, DropoutRandomChannel

BRATS_IMAGE_STATS = {
    "percentile_05": [169.0, 166.0, 85.0, 131.0],
    "percentile_95": [2853.0, 3798.0, 1467.0, 1750.0],
    "median_shape": [136, 176, 144],
    "mean": [840.3223, 1160.4885, 535.1234, 694.9304],
    "std": [918.5928, 1309.5450, 708.9212, 831.6685],
}


class BratsDataset:
    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
        image_folder_name="images",
    ):

        assert split in [
            "train",
            "val",
            "test",
        ], "split must be one of 'train', 'val', or 'test'"

        self.root = Path("/sc-scratch/sc-scratch-gbm-radiomics/posenc/brats")
        self.split_df = pd.read_csv(self.root / "split.csv")
        self.data = self.split_df[self.split_df["split"] == split]
        self.data.reset_index(drop=True, inplace=True)
        self.transform = transform
        self.image_folder_name = image_folder_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Union[str, torch.Tensor]]:
        patient_id = self.data.loc[idx, "patient"]

        patient = {
            "image": str(
                self.root / self.image_folder_name / f"{patient_id}-image.nii.gz"
            ),
            "mask": str(
                self.root / self.image_folder_name / f"{patient_id}-seg.nii.gz"
            ),
        }

        if self.transform:
            patient = self.transform(patient)

        return patient


class BratsDatasetDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        random_dropout_channel: bool = False,
        crop_size=[96, 112, 96],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_dropout_channel = random_dropout_channel
        self.crop_size = crop_size  # Should be divisable by 16.

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):

        if self.random_dropout_channel:
            train_transforms = self.train_transform_with_random_dropout()
            val_transforms = self.val_transform_with_random_dropout()
        else:
            train_transforms = self.train_transform_with_mask()
            val_transforms = self.val_transform_with_mask()

        self.train = BratsDataset(
            "train", transform=train_transforms, image_folder_name="cropped"
        )
        self.val = BratsDataset("val", transform=val_transforms)
        self.test = BratsDataset("test", transform=val_transforms)

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
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_transform_with_random_dropout(self):
        return T.Compose(
            [
                T.LoadImaged(keys="image", ensure_channel_first=True),
                T.Spacingd(
                    keys="image",
                    pixdim=1.0,
                    mode="bilinear",
                ),
                T.ResizeWithPadOrCropd(
                    keys="image",
                    spatial_size=BRATS_IMAGE_STATS["median_shape"],
                    mode="replicate",
                ),
                DropoutRandomChannel("image"),
                T.RandSpatialCropSamplesd(
                    keys=["image", "channel_dropped"],
                    roi_size=self.crop_size,
                    num_samples=1,
                ),
                T.RandFlipd(
                    keys=["image", "channel_dropped"],
                    prob=0.5,
                    spatial_axis=0,
                ),
                T.RandFlipd(
                    keys=["image", "channel_dropped"],
                    prob=0.5,
                    spatial_axis=1,
                ),
                T.RandFlipd(
                    keys=["image", "channel_dropped"],
                    prob=0.5,
                    spatial_axis=2,
                ),
                T.NormalizeIntensityd(
                    keys=["image", "channel_dropped"],
                    nonzero=True,
                    channel_wise=True,
                ),
                ClearDroppedChannel("image"),
            ]
        )

    def val_transform_with_random_dropout(self):
        return T.Compose(
            [
                T.LoadImaged(keys="image", ensure_channel_first=False),
                T.Spacingd(
                    keys="image",
                    pixdim=1.0,
                    mode="bilinear",
                ),
                T.CropForegroundd(
                    keys="image",
                    source_key="image",
                    select_fn=lambda x: x > 0,
                    allow_smaller=False,
                    k_divisible=8,
                ),
                T.ResizeWithPadOrCropd(
                    keys="image",
                    spatial_size=(136, 176, 144),
                    mode="replicate",
                ),
                DropoutRandomChannel("image"),
                T.RandSpatialCropSamplesd(
                    keys=["image", "channel_dropped"],
                    roi_size=self.crop_size,
                    num_samples=1,
                ),
                T.NormalizeIntensityd(
                    keys=["image", "channel_dropped"], nonzero=True, channel_wise=True
                ),
                ClearDroppedChannel("image"),
            ]
        )

    def train_transform_with_mask(self):
        return T.Compose(
            [
                T.LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
                T.Spacingd(
                    keys=["image", "mask"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                ),
                T.ResizeWithPadOrCropd(
                    keys=["image", "mask"],
                    spatial_size=BRATS_IMAGE_STATS["median_shape"],
                    mode="replicate",
                    allow_missing_keys=True,
                ),
                T.RandCropByPosNegLabeld(
                    keys=["image", "mask"],
                    label_key="mask",
                    spatial_size=self.crop_size,
                    num_samples=2,
                    pos=4,
                    neg=1,
                    fg_indices_key=3,
                    bg_indices_key=0,
                ),
                T.RandFlipd(
                    keys=["image", "mask"],
                    prob=0.5,
                    spatial_axis=0,
                ),
                T.RandFlipd(
                    keys=["image", "mask"],
                    prob=0.5,
                    spatial_axis=1,
                ),
                T.RandFlipd(
                    keys=["image", "mask"],
                    prob=0.5,
                    spatial_axis=2,
                ),
                T.NormalizeIntensityd(
                    keys="image",
                    nonzero=True,
                    channel_wise=True,
                ),
                T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )

    def val_transform_with_mask(self):
        return T.Compose(
            [
                T.LoadImaged(keys=["image", "mask"], ensure_channel_first=False),
                T.Spacingd(
                    keys=["image", "mask"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                T.CropForegroundd(
                    keys=["image", "mask"],
                    source_key="image",
                    select_fn=lambda x: x > 0,
                    allow_smaller=False,
                    k_divisible=8,
                ),
                T.ResizeWithPadOrCropd(
                    keys=["image", "mask"],
                    spatial_size=(136, 176, 144),
                    mode="replicate",
                ),
                T.RandCropByPosNegLabeld(
                    keys=["image", "mask"],
                    label_key="mask",
                    spatial_size=self.crop_size,
                    pos=1,
                    neg=1,
                    fg_indices_key=3,
                    bg_indices_key=0,
                ),
                T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )


def extract_patient_id(path: Path):
    """Extract patient ID from file name (e.g. BraTS-GLI-00444-000-seg.nii.gz -> BraTS-GLI-00444-000)."""
    return "-".join(Path(path).name.split("-")[:-1])


def generate_datasplit(dataset_path: Path, seed: int = 42) -> pd:
    """Generate a pandas DataFrame with patient IDs and split information."""

    all_patients = [extract_patient_id(p) for p in dataset_path.glob("*seg.nii.gz")]

    # Number of samples for each dataset
    num_train_samples = int(len(all_patients) * 0.6)  # 60% for training
    num_valid_samples = int(len(all_patients) * 0.2)  # 20% for validation

    # Randomly shuffle the IDs
    np.random.seed(seed)
    np.random.shuffle(all_patients)

    # Divide IDs into train, valid, and test sets
    train_ids = all_patients[:num_train_samples]
    valid_ids = all_patients[num_train_samples : num_train_samples + num_valid_samples]
    test_ids = all_patients[num_train_samples + num_valid_samples :]

    # Create pandas DataFrames for train, valid, and test datasets
    train_df = pd.DataFrame({"patient": train_ids, "split": ["train"] * len(train_ids)})
    valid_df = pd.DataFrame({"patient": valid_ids, "split": ["val"] * len(valid_ids)})
    test_df = pd.DataFrame({"patient": test_ids, "split": ["test"] * len(test_ids)})

    # Concatenate all DataFrames into one
    final_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    return final_df


def generate_cropped_dataset(
    cropped_dir_name: str = "cropped", num_workers: int = 0
) -> None:
    """Generate a cropped dataset from the original BRATS dataset."""
    crop = T.Compose(
        [
            T.LoadImaged(keys=["image", "mask"]),
            T.CropForegroundd(
                keys=["image", "mask"],
                source_key="image",
                select_fn=lambda x: x > 0,
                allow_smaller=False,
                k_divisible=8,
            ),
            T.SaveImaged(
                keys=["image", "mask"],
                output_dir=f"/sc-scratch/sc-scratch-gbm-radiomics/posenc/brats/{cropped_dir_name}",
                output_postfix="",
                squeeze_end_dims=False,
                separate_folder=False,
                print_log=False,
            ),
        ]
    )
    train_dataset = BratsDataset("train", crop)
    train_loader = DataLoader(train_dataset, num_workers=num_workers)

    for _ in tqdm(train_loader):
        continue


if __name__ == "__main__":
    generate_cropped_dataset(num_workers=0)
