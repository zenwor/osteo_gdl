from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

import geoopt

from tqdm import tqdm
from loguru import logger as log

from osteo_gdl.utils.const import EUCL_OPTIM_MAP, RIEM_OPTIM_MAP, CRITERION_MAP
from osteo_gdl.utils.dataset import make_sample_weights

from osteo_gdl.models.riem.oehnet import OEHNet
from osteo_gdl.models.riem.ospnet import OSPNet
from osteo_gdl.models.riem.dospnet import DOSPNet

from torch.optim import Adam


def make_optim(
    model: nn.Module,
    eucl_lr: float = 1e-4,
    eucl_wd: Union[float, None] = 0.0,
    riem_lr: float = 1e-3,
):
    if isinstance(model, OEHNet):
        optim = geoopt.optim.RiemannianAdam(
            [
                {
                    "params": model.hyperbolic_head.parameters(),
                    "lr": riem_lr,
                    "manifold": model.hyperbolic_head.manifold,
                },
                {"params": model.backbone.parameters(), "lr": eucl_lr},
            ]
        )
    elif isinstance(model, OSPNet) or isinstance(model, DOSPNet):
        optim = Adam(model.parameters(), lr=eucl_lr, weight_decay=eucl_wd)
    else:
        raise ValueError("Cannot find optimizer for the given model.")
    return optim


def make_criterion(
    criterion: str, weighted: bool = False, weight_dataset: Dataset = None
) -> nn.Module:
    assert criterion in CRITERION_MAP, f"Cannot find given criterion: {criterion}."

    criterion_ = CRITERION_MAP[criterion]
    if weighted:
        assert (
            weight_dataset is not None
        ), f"For weighted loss, please provide the dataset."

        weights = make_sample_weights(weight_dataset)
        criterion = criterion_(weights=weights)
    else:
        criterion = criterion_()
    return criterion


@torch.no_grad()
def test(model: nn.Module, test_dl: DataLoader, epoch: int, device: str):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []

    loop = tqdm(test_dl, desc=f"test")
    for data in loop:
        x, y = data
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda"):
            out = model(x)
        pred = out.argmax(dim=1)

        correct += int((pred == y).sum())
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_acc = correct / len(test_dl.dataset)
    print(f"(test, {epoch + 1}) accuracy: {avg_acc * 100:.2f}%")
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:")
    print(cm)


def train(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    optim,
    criterion,
    epochs: int,
    device: str,
):
    model.train()

    scaler = torch.amp.GradScaler("cuda")
    training = True

    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        model.train()
        loop = tqdm(train_dl, desc=f"epoch {epoch+1}/{epochs} - train")

        for batch in loop:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)

            optim.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(xb.float())
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            running_corrects += (preds == yb).sum().item()
            total_samples += batch_size

        if not training:
            break

        epoch_avg_loss = running_loss / total_samples
        epoch_avg_acc = running_corrects / total_samples
        log.info(
            f"(train, {epoch + 1}) loss: {epoch_avg_loss:.2f}; acc: {epoch_avg_acc * 100:.2f}"
        )

        test(model, test_dl, epoch, device)
