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
import torch.nn as nn

class OsteosarcomaDataset(Dataset):
    def __init__(
        self,
        imgs,
        feats,
        preprocessed: bool = True,
        aug: bool = False,
        train: bool = False,
        model: nn.Module = None,
    ):
        if imgs is not None:
            self.preprocessed = preprocessed
            self.train = train
            if preprocessed:
                self.imgs = imgs
                self.labels = feats
            else:
                self.img_names, self.imgs, self.labels = [], [], []
                for img_path in imgs:
                    case = os.path.splitext(os.path.basename(img_path))[0]
                    cls_vals = feats.loc[
                        feats["image.name"] == case, "classification"
                    ].values
                    if len(cls_vals) >= 1:
                        cls = cls_vals[0]
                    else:
                        continue
                    self.img_names.append(case)
                    self.imgs.append(img_path)
                    self.labels.append(cls)
            self.nimgs = len(self.imgs)
            
            self.maj_tfms = None
            self.min_tfms = None
            self.aug = aug
            
            if preprocessed:
                self.make_tfms(model)

    def make_tfms(self, model: nn.Module):
        if self.train:  # val / test
            if self.aug:
                self.maj_tfms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                self.min_tfms = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                log.info(self.maj_tfms)
                log.info(self.min_tfms)
                
            else:
                self.tfms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                log.info(self.tfms)
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            log.info(self.tfms)
                    
    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        if not self.preprocessed:
            return self.imgs[idx], self.labels[idx]

        y = self.labels[idx]
        img = Image.open(self.imgs[idx]).convert("RGB")

        if self.aug and self.train:
            # This piece of code does not generalize well
            # TODO: Fix
            if y == 2: # NVT
                img = self.min_tfms(img)
            else:
                img = self.maj_tfms(img)
        else:
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
    aug: bool = False,
    balance: bool = False,
    model: nn.Module = None,
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

    train_ds = OsteosarcomaDataset(train_imgs, train_labels, aug=aug, train=True, model=model)
    val_ds = (
        None
        if val_imgs is None
        else OsteosarcomaDataset(
            val_imgs,
            val_labels,
            aug=aug, 
            model=model,
        )
    )
    test_ds = OsteosarcomaDataset(test_imgs, test_labels, aug=aug, model=model)
    return train_ds, val_ds, test_ds


def make_sample_weights(ds, device) -> torch.tensor:
    labels = np.array(ds.labels).astype(int)
    class_counts = np.bincount(labels)
    weights_per_class = 1.0 / class_counts
    weights = weights_per_class[labels]
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_class_weights(ds, device):
    labels = np.array(ds.labels).astype(int)
    class_counts = np.bincount(labels)
    weights_per_class = 1.0 / class_counts
    weights_per_class = weights_per_class / weights_per_class.sum() * len(class_counts)
    return torch.tensor(weights_per_class, dtype=torch.float32, device=device)


# DataLoader
def make_wrs(ds, device) -> WeightedRandomSampler:
    weights = make_sample_weights(ds, device)
    return WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)


def make_dataloader(ds: Dataset, batch_size: int, device: str = "cuda", resample: bool = False) -> DataLoader:
    if resample:
        return DataLoader(
            ds, batch_size=batch_size, sampler=make_wrs(ds, device), num_workers=0
        )
    else:
        return DataLoader(ds, batch_size=batch_size, num_workers=0)
