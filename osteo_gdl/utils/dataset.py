# flake8: noqa

import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger as log
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


class OsteosarcomaDataset(Dataset):
    def __init__(self, imgs, feats, preprocessed: bool = True, train: bool = False):
        if imgs is not None:
            self.preprocessed = preprocessed
            self.train = train
            if preprocessed:
                self.imgs = imgs
                self.labels = feats
            else:
                self.img_names, self.imgs, self.labels = [], [], []
                for img_path in imgs:
                    img = Image.open(img_path)
                    case = os.path.splitext(os.path.basename(img_path))[0]
                    cls_vals = feats.loc[
                        feats["image.name"] == case, "classification"
                    ].values
                    if len(cls_vals) >= 1:
                        cls = cls_vals[0]
                    else:
                        continue
                    self.img_names.append(case)
                    self.imgs.append(img)
                    self.labels.append(cls)
            self.nimgs = len(self.imgs)
            self.maj_tfms = None
            self.min_tfms = None
            if preprocessed:
                self.make_tfms()

    def make_tfms(self):
        self.tfms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        if not self.preprocessed:
            return self.imgs[idx], self.labels[idx]

        y = self.labels[idx]
        img = self.imgs[idx]

        img = self.tfms(img)
        return img, torch.tensor(int(y), dtype=torch.long)

    def get_img_label_pairs(self):
        pairs = []
        for idx in range(self.nimgs):
            pairs.append(self[idx])
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


# Dataset
def split_data(imgs, labels, val_size: float, test_size: float):
    if val_size == 0.0:
        imgs_train, imgs_test, labels_train, labels_test = train_test_split(
            imgs, labels, test_size=test_size, stratify=labels, random_state=42
        )
        return (imgs_train, labels_train), (None, None), (imgs_test, labels_test)

    imgs_train_val, imgs_test, labels_train_val, labels_test = train_test_split(
        imgs, labels, test_size=test_size, stratify=labels, random_state=42
    )

    val_relative_size = val_size / (1 - test_size)
    imgs_train, imgs_val, labels_train, labels_val = train_test_split(
        imgs_train_val,
        labels_train_val,
        test_size=val_relative_size,
        stratify=labels_train_val,
        random_state=42,
    )
    return (
        (imgs_train, labels_train),
        (imgs_val, labels_val),
        (imgs_test, labels_test),
    )


def make_datasets(
    imgs: List[str],
    feats: pd.DataFrame,
    val_size: float,
    test_size: float,
    balance: bool = False,
) -> Tuple[Dataset, Union[None, Dataset], Dataset]:
    ds = OsteosarcomaDataset(imgs, feats, False)
    imgs, labels = ds.get_img_label_pairs()

    if balance:
        class_counts = np.bincount(labels)
        min_count = int(class_counts.min())

        # subsample at most min_count per class
        sub_imgs, sub_labels = [], []
        per_class_counter = {c: 0 for c in range(len(class_counts))}

        for img, lab in zip(imgs, labels):
            if per_class_counter[lab] < min_count:
                sub_imgs.append(img)
                sub_labels.append(lab)
                per_class_counter[lab] += 1

        imgs, labels = sub_imgs, sub_labels
        log.info(
            f"(Balancing) Using {min_count} samples per class → total {len(imgs)} images"
        )

    (
        (train_imgs, train_labels),
        (val_imgs, val_labels),
        (test_imgs, test_labels),
    ) = split_data(  # noqa: E501
        imgs,
        labels,
        val_size=val_size,
        test_size=test_size,
    )

    train_ds = OsteosarcomaDataset(train_imgs, train_labels, train=True)
    val_ds = (
        None
        if val_imgs is None
        else OsteosarcomaDataset(
            val_imgs,
            val_labels,
        )
    )
    test_ds = OsteosarcomaDataset(test_imgs, test_labels)
    return train_ds, val_ds, test_ds


def make_sample_weights(ds, device) -> torch.tensor:
    labels = ds.labels
    class_counts = np.bincount(labels)
    weights_per_class = 1.0 / class_counts
    weights = weights_per_class[labels]
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# DataLoader
def make_wrs(ds) -> WeightedRandomSampler:
    weights = make_sample_weights(ds)
    return WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)


def make_dataloader(ds: Dataset, batch_size: int, resample: bool = False) -> DataLoader:
    if resample:
        return DataLoader(
            ds, batch_size=batch_size, sampler=make_wrs(ds), num_workers=0
        )
    else:
        return DataLoader(ds, batch_size=batch_size, num_workers=0)
