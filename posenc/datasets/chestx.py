import argparse
import multiprocessing
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from urllib import request

import albumentations as A
import lightning.pytorch as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.ops import box_convert
from torchvision.transforms import v2
from tqdm import tqdm

from posenc.datasets.transforms import (
    AddGaussianNoise,
    get_cutmix_and_mixup_collate_function,
)

# URLs for the zip files
IMAGE_LINKS = {
    "01": "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    "02": "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    "03": "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "04": "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "05": "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "06": "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "07": "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    "08": "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "09": "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "10": "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "11": "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "12": "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
}


class BoxTransformer:
    """A class that transforms bounding box coordinates. And normalizes them using quantile transformer."""

    def __init__(self, format="xywh"):
        """
        Args:
            format (str, optional): The format of the bounding box coordinates. Defaults to "xywh".
        """
        self.format = format
        self.quant_transformer = QuantileTransformer(
            n_quantiles=250, output_distribution="normal", random_state=42
        )

        train_data = pd.read_csv(
            "/sc-scratch/sc-scratch-gbm-radiomics/posenc/chestx/ObjectDetection/train.csv"
        )
        self.quant_transformer.fit(train_data.loc[:, ["x", "y", "w", "h"]].to_numpy())

    def __call__(self, bboxes):
        """Transforms the given bounding box coordinates"""
        bboxes = self.quant_transformer.inverse_transform(bboxes)
        bboxes = box_convert(torch.tensor(bboxes), "xywh", self.format)
        return bboxes


class ChestXDataset:
    def __init__(
        self,
        mode="train",
        task="multilabel",
        image_path="resized256",
        transforms: Optional[Callable] = None,
    ):

        assert task in ["binary", "multilabel"], f"Task {task} not supported."

        self.root = Path("/sc-scratch/sc-scratch-gbm-radiomics/posenc/chestx")
        self.transforms = transforms
        self.task = task
        self.image_path = image_path

        assert mode in [
            "train",
            "val",
            "test",
        ], "mode should be one of 'train', 'val', or 'test'"
        self.mode = mode

        self.csv_path = (
            self.root / "PruneCXR" / f"miccai2023_nih-cxr-lt_labels_{mode}.csv"
        )

        self.data = pd.read_csv(self.csv_path)

        if "subj_id" in self.data:
            self.data.drop(["subj_id"], axis=1, inplace=True)

        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.data.loc[idx, "id"]
        img = str(self.root / "images" / self.image_path / image_name)
        img = Image.open(img).convert("L")

        if self.task == "binary":
            label = self.data.loc[idx, "No Finding"]
        else:
            label = self.data.loc[
                idx,
                ~self.data.columns.isin(
                    ["id", "sampling_weight", "binary_sampling_weight"]
                ),
            ].values.astype(int)

        if self.transforms:
            img = self.transforms(img)

        return img, torch.tensor(label).float()

    @property
    def labels(self):
        return list(self.data.columns.drop("id"))


class ChestXDataModule(L.LightningDataModule):
    def __init__(
        self,
        task: str,
        batch_size: int = 64,
        num_workers: int = 12,
        image_size: int = 224,
        image_dir_name: str = "resized256",
        do_cutmix: bool = True,
        do_mixup: bool = True,
    ):
        """
        Args:
            dataset (str): Dataset to use. Options: binary | multilabel | detection
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of CPU workers. Defaults to 12.
            train_transform (callable, optional): Transforms for train set. Defaults to None.
            valid_transform (callable, optional): Transforms for test and validation sets. Defaults to None.
            image_size (int, optional): Image size. Defaults to 224 as in original ViT. This is the size of the image after resizing.
            image_dir_name (str, optional): Image folder name to the images. Defaults to "resized256".
        """
        super().__init__()

        if task == "multilabel":
            self.num_classes = 20
            self.sampling_weights_key = (
                "binary_sampling_weight"  # TODO: Have to be computed.
            )
        elif task == "binary":
            self.num_classes = 2
            self.sampling_weights_key = "binary_sampling_weight"
        else:
            raise NotImplementedError(
                f"Only 'multilabel' and 'binary' are supported. Given {task}"
            )

        self.image_dir_name = image_dir_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.image_size = image_size
        self.do_cutmix = do_cutmix
        self.do_mixup = do_mixup

        # Define transforms
        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.526], std=[0.252]),
                v2.RandomCrop(image_size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=15),
                AddGaussianNoise(p=0.33, mean=0.0, std=0.1),
            ]
        )
        self.valid_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.526], std=[0.252]),
                v2.RandomCrop(image_size),
            ]
        )

        self.save_hyperparameters()

    def setup(self, stage: str = None):
        self.train = ChestXDataset(
            mode="train",
            task=self.task,
            image_path=self.image_dir_name,
            transforms=self.train_transform,
        )
        self.valid = ChestXDataset(
            mode="val",
            task=self.task,
            image_path=self.image_dir_name,
            transforms=self.valid_transform,
        )
        self.test = ChestXDataset(
            mode="test",
            task=self.task,
            image_path=self.image_dir_name,
            transforms=self.valid_transform,
        )

    def train_dataloader(self):
        # This sampler is used to balance the classes.
        # It samples with replacement to make ensure minority classes are oversampled.
        sampler = WeightedRandomSampler(
            self.train.data[self.sampling_weights_key],
            len(self.train.data),
            replacement=True,
        )

        # Add this transform to collate to apply cutmix or mixup with multi-processing.
        collate_fn = get_cutmix_and_mixup_collate_function(
            self.do_cutmix, self.do_mixup, self.num_classes
        )

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            drop_last=True,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )


class ChestObjDetDataset:
    root = Path("/sc-scratch/sc-scratch-gbm-radiomics/posenc/chestx/ObjectDetection")
    data_path = Path(
        "/sc-scratch/sc-scratch-gbm-radiomics/posenc/chestx/images/resized256"
    )

    def __init__(
        self,
        mode="train",
        transforms=None,
    ):
        """
        Args:
            mode (str, optional): Dataset to use. Options: train | val | test. Defaults to "train".
            transforms (callable, optional): Transforms for train set. Defaults to None.
        """

        self.label_dec_dict = {
            0: "Atelectasis",
            1: "Cardiomegaly",
            2: "Effusion",
            3: "Infiltrate",
            4: "Mass",
            5: "Nodule",
            6: "Pneumonia",
            7: "Pneumothorax",
        }
        self.transforms = transforms

        # Read dataset
        assert mode in [
            "train",
            "val",
            "test",
        ], "mode should be one of 'train', 'valid', or 'test'"
        self.data = pd.read_csv(self.root / f"{mode}.csv")

        if mode == "train":
            self.data = pd.concat([self.data, self.data]).reset_index(drop=True)

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.data["label_enc"] = self.label_encoder.fit_transform(self.data.label)

        self.data = self.data.loc[
            :, ["image", "label_enc", "x", "y", "w", "h"]
        ].to_records()

    def __len__(self):
        return len(self.data)

    def _process_bbox(self, x, y, w, h) -> List[List[int]]:
        """Get normalized bbox in xyxy format. Return as list for albumentations."""
        bbox = torch.tensor([[x, y, w, h]], dtype=torch.float32)

        bbox = box_convert(bbox, "xywh", "xyxy")
        bbox /= 1024

        return bbox.tolist()

    def __getitem__(self, idx):
        """Get image, label and bbox in specified format."""

        # Get image, label, and bounding box
        sample = self.data[idx]
        img_path, label = sample[1], sample[2]
        bbox = self._process_bbox(sample[3], sample[4], sample[5], sample[6])

        img = Image.open(self.data_path / img_path).convert("L")
        img = np.array(img)

        # Apply transforms
        if self.transforms:

            cropping_bbox = torch.zeros(4)
            cropping_bbox[:2] = torch.rand(2)
            cropping_bbox[2:] = (cropping_bbox[:2] + torch.rand(2)).clamp(0, 1)
            bbox.append(cropping_bbox.tolist())

            sample = self.transforms(
                image=img,
                bboxes=bbox,
                cropping_bbox=[0, 0, 1, 1],
                class_labels=[label, label],
            )
            img = sample["image"]
            bbox = torch.tensor(sample["bboxes"][0])
        #             bbox = torch.tensor(sample["cropping_bbox"])

        return img, (label, bbox)


def norm_pixels(x, *args, **kwargs):
    """Normalize the intensity to 0 and 1."""
    return x / 255


def norm_intensity(x, *args, **kwargs):
    """Normalize the intensity of the image with dataset mean and std."""
    return (x - 0.486) / 0.246


def add_channel_dim(x, *args, **kwargs):
    """Add channel dimension to the image."""
    return x[None, ...]


def to_tensor_image(x, *args, **kwargs):
    """Add channel dimension to the image."""
    return torch.tensor(x[None, ...], dtype=torch.float32)


class ChestObjDetDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 12,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transforms
        self.train_transform = A.Compose(
            [
                A.Lambda(image=norm_pixels),
                A.OneOf(
                    [
                        A.GaussNoise(p=1.0, var_limit=(0.002)),
                        A.MultiplicativeNoise(p=1.0),
                        A.PixelDropout(p=1.0, dropout_prob=0.01, drop_value=0.486),
                        A.MotionBlur(p=1.0),
                    ],
                    p=1.0,
                ),
                A.RandomSizedBBoxSafeCrop(
                    erosion_rate=0.0, p=1.0, width=256, height=256
                ),
                A.SafeRotate(limit=15, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Lambda(image=norm_intensity),
                A.Lambda(image=to_tensor_image),
            ],
            bbox_params=A.BboxParams(
                format="albumentations", label_fields=["class_labels"]
            ),
        )
        self.valid_transform = A.Compose(
            [
                A.Lambda(image=norm_pixels),
                A.Lambda(image=norm_intensity),
                A.CenterCrop(width=256, height=256),
                A.Lambda(image=to_tensor_image),
            ],
            bbox_params=A.BboxParams(
                format="albumentations", label_fields=["class_labels"]
            ),
        )

        self.save_hyperparameters()

    def setup(self, stage: str = None):
        self.train = ChestObjDetDataset(
            mode="train",
            transforms=self.train_transform,
        )
        self.valid = ChestObjDetDataset(
            mode="val",
            transforms=self.valid_transform,
        )
        self.test = ChestObjDetDataset(
            mode="test",
            transforms=self.valid_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )


def _download_image(output_path: Path, link_key: str) -> None:
    """
    Download a single image from the NIH website
    """
    filepath = output_path / Path(f"images_{link_key}.tar.gz")
    request.urlretrieve(IMAGE_LINKS[link_key], filepath)


def download_images(output_path: Path, processes: int = 12):
    """
    Download the images from the NIH website and save them to the output path
    """
    _download = partial(_download_image, Path(output_path))

    with Pool(processes) as p:
        p.map(_download, list(IMAGE_LINKS.keys()))


def train_valid_test_split_object_detection(path="BBox_List_2017.csv"):
    """Split the dataset into train, validation, and test sets for object detection. Ensure every patientid is only in one set."""
    boxes_df = pd.read_csv(path)
    boxes_df = boxes_df.dropna(axis=1)
    boxes_df.columns = ["image", "label", "x", "y", "w", "h"]

    boxes_df["patientid"], boxes_df["repetition"] = zip(
        *boxes_df["image"].str.split("_").apply(lambda x: (x[0], x[1].split(".")[0]))
    )

    # Get unique patientids
    unique_patientids = boxes_df["patientid"].unique()

    # Split the patientids into train and temporary sets
    train_patientids, temp_patientids = train_test_split(
        unique_patientids, test_size=0.3, random_state=42
    )

    # Split the remaining patientids into validation and test sets
    val_patientids, test_patientids = train_test_split(
        temp_patientids, test_size=0.5, random_state=42
    )

    # Filter the DataFrame based on the selected patientids for each split
    train_set = boxes_df[boxes_df["patientid"].isin(train_patientids)]
    val_set = boxes_df[boxes_df["patientid"].isin(val_patientids)]
    test_set = boxes_df[boxes_df["patientid"].isin(test_patientids)]

    return train_set, val_set, test_set


def resize_image(image_path, output_path, new_size):
    """
    Resize a single image and save it to the specified path.

    Args:
        image_path: Path to the input image.
        output_path: Path where resized image will be saved.
        new_size: Tuple specifying the new size (width, height) for the image.
    """
    # Load image using PIL
    image = Image.open(image_path)

    # Define transformation to resize image
    transform = v2.Resize(new_size)

    # Apply transformation to resize image
    resized_image = transform(image)

    # Get original file name
    _, filename = os.path.split(image_path)

    # Save resized image with the same original file name
    resized_image_path = os.path.join(output_path, filename)
    resized_image.save(resized_image_path)


def resize_images(input_path, output_path, new_size, num_processes=24):
    """
    Resize all images in a directory and save them to a specified path.

    Args:
        input_path: Path to the directory containing input images.
        output_path: Path where resized images will be saved.
        new_size: Tuple specifying the new size (width, height) for the images.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_path = Path(input_path)
    images = list(input_path.glob("*.png"))

    # Set up multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)

    _resize = partial(resize_image, output_path=output_path, new_size=new_size)

    # Resize images in parallel
    for _ in tqdm(
        pool.imap_unordered(_resize, images, chunksize=10), total=len(images)
    ):
        pass

    # Close the pool to free resources
    pool.close()
    pool.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path", type=str, help="Output path for preprocessing or downloading"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=12,
        help="Number of processes for downloading",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    download_images(Path(args.output_path), args.num_processes)
