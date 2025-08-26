import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, f1_score, recall_score, balanced_accuracy_score

import geoopt

from tqdm import tqdm
from loguru import logger as log

import copy
from typing import List, Union

from osteo_gdl.utils.const import EUCL_OPTIM_MAP, RIEM_OPTIM_MAP, CRITERION_MAP
from osteo_gdl.utils.dataset import make_sample_weights, make_class_weights
from osteo_gdl.utils.exp import log_exp, plot_metrics

from osteo_gdl.models import OEHNet, OSPNet, OsteoGNN


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
    elif isinstance(model, (OSPNet, OsteoGNN)):
        optim = Adam(model.parameters(), lr=eucl_lr, weight_decay=eucl_wd)
    else:
        raise ValueError("Cannot find optimizer for the given model.")
    return optim


def make_criterion(
    criterion: str, weighted: bool = False, weight_dataset: Dataset = None, device: str = "cuda",
) -> nn.Module:
    assert criterion in CRITERION_MAP, f"Cannot find given criterion: {criterion}."

    criterion_ = CRITERION_MAP[criterion]
    if weighted:
        assert (
            weight_dataset is not None
        ), f"For weighted loss, please provide the dataset."

        weights = make_class_weights(weight_dataset, device)
        criterion = criterion_(weight=weights)
    else:
        criterion = criterion_()
    return criterion


def ohem_loss(logits, targets, criterion=F.cross_entropy, keep_ratio=0.7):
    batch_size = logits.size(0)

    per_sample_loss = criterion(logits, targets, reduction="none")
    keep_num = max(1, int(batch_size * keep_ratio))

    _, idxs = per_sample_loss.topk(keep_num, largest=True)
    hard_loss = per_sample_loss[idxs]

    return hard_loss.mean()

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    use_ohem: bool,
    keep_ratio: float,
    device: str,
    epoch: int,
    split: str = "val",
):
    model.eval()
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    loop = tqdm(dataloader, desc=f"{split}")
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)

        assert xb.is_cuda, "Input is not on GPU!"
        assert next(model.parameters()).is_cuda, "Model is not on GPU!"
        
        with torch.no_grad():
            # logits = model(xb.float())
            logits = model(xb)
            if use_ohem:
                loss = ohem_loss(logits, yb, criterion, keep_ratio)
            else:
                loss = criterion(logits, yb)
            
        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)

        running_corrects += (preds == yb).sum().item()
        total_samples += batch_size

        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # metrics
    avg_loss = running_loss / total_samples
    avg_acc = running_corrects / total_samples
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"({split}, {epoch + 1}) loss: {avg_loss:.2f}; acc: {avg_acc * 100:.2f}%")
    print(f"{split.capitalize()} confusion matrix:")
    print(cm)

    return {
        "epoch": epoch + 1,
        "loss": avg_loss,
        "acc": avg_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "confusion_matrix": cm.tolist(),
    }


def train(
    # Train
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    optim,
    criterion,
    use_ohem: bool,
    epochs: int,
    device: str,
    # Log
    exp_name,
    model_name: str,
    backbone: str,
    backbone_out_dim: str,
    reduced_dims: List[int],
    oehnet_c: float,
    loss_name: str,
    batch_size: int,
    eucl_lr: float,
    eucl_wd: float,
    riem_lr: float,
    weighted_criterion: bool = False,
    classes: List[str] = [],
    num_epochs: int = 10,
    val_size: float = 0.10,
    test_size: float = 0.25,
):
    train_data, val_data, test_data = [], [], []

    scaler = torch.amp.GradScaler("cuda")

    val_iter = 0
    best_val_acc = 0.0
    patience = 15
    keep_ratio = 0.7
    epoch_iter = 0

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects, total_samples = 0.0, 0, 0

        loop = tqdm(train_dl, desc=f"epoch {epoch+1}/{epochs} - train")
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)

            optim.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                if use_ohem:
                    loss = ohem_loss(logits, yb, criterion, keep_ratio)
                else:
                    loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            running_corrects += (preds == yb).sum().item()
            total_samples += batch_size
        
        # Epoch metrics
        epoch_iter += 1
        epoch_avg_loss = running_loss / total_samples
        epoch_avg_acc = running_corrects / total_samples
        log.info(f"(train, {epoch+1}) loss: {epoch_avg_loss:.2f}; acc: {epoch_avg_acc*100:.2f}")

        # Validation
        if val_dl is not None:
            val_metrics = evaluate(model, val_dl, criterion, use_ohem, keep_ratio, device, epoch, split="val")
            val_acc = val_metrics["acc"]
            val_data.append(val_metrics)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                val_iter = 0
                best_model_wts = copy.deepcopy(model.state_dict())
                if use_ohem:
                    keep_ratio = max(0.5, keep_ratio - 0.05)
                    log.info(f"Val improved. Decrease keep_ratio to {keep_ratio:.2f}")
            else:
                val_iter += 1
                if use_ohem:
                    keep_ratio = min(0.95, keep_ratio + 0.05)
                    log.info(f"Val stagnated. Increase keep_ratio to {keep_ratio:.2f}")

        if val_iter >= patience:
            break
    
    # Load and validate the best model
    model.load_state_dict(best_model_wts)

    final_val_metrics = evaluate(model, val_dl, criterion, use_ohem, keep_ratio, device, epoch_iter, split="val") if val_dl else None
    final_test_metrics = evaluate(model, test_dl, criterion, False, False, device, epoch_iter, split="test") if test_dl else None

    if final_val_metrics:
        val_data.append(final_val_metrics)
    if final_test_metrics:
        test_data.append(final_test_metrics)

    log_exp(
        exp_name,
        model_name,
        backbone,
        backbone_out_dim,
        reduced_dims,
        oehnet_c,
        loss_name,
        batch_size,
        eucl_lr,
        eucl_wd,
        riem_lr,
        weighted_criterion,
        use_ohem,
        classes,
        num_epochs,
        val_size,
        test_size,
        train_data,
        val_data,
        test_data,
    )
    
    plot_metrics(
        train_data, val_data,
        save_dir=f"{os.getenv('EXP_PATH')}/{exp_name}/"
    )