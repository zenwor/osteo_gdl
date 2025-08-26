import os

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger as log
from typing import List


def collect_metrics(data_list, metric):
    if not data_list:
        return []
    return [epoch_data.get(metric, None) for epoch_data in data_list]

def log_exp(
    exp_name,
    model_name: str,
    backbone: str,
    backbone_out_dim: str,
    reduced_dims: list,
    oehnet_c: float,
    loss_name: str,
    batch_size: int,
    eucl_lr: float,
    eucl_wd: float,
    riem_lr: float,
    weighted_criterion: bool = False,
    use_ohem: bool = False,
    classes: list = [],
    num_epochs: int = 10,
    val_size: float = 0.10,
    test_size: float = 0.25,
    train_data: list = None,
    val_data: list = None,
    test_data: list = None,
    csv_path: str = f"{os.getenv('EXP_PATH')}/exp.csv",
):
    def collect_metrics(data_list, metric):
        """Safely extract metric from each epoch, defaulting to None if missing."""
        if not data_list:
            return []
        return [epoch_data.get(metric, None) for epoch_data in data_list]

    metrics = ["loss", "acc", "precision", "recall", "f1", "balanced_acc"]
    collected = {}
    for m in metrics:
        collected[f"train_{m}"] = collect_metrics(train_data, m)
        collected[f"val_{m}"] = collect_metrics(val_data, m)
        collected[f"test_{m}"] = collect_metrics(test_data, m)

    best_metrics = {}
    for split in ["train", "val", "test"]:
        for m in ["acc", "precision", "recall", "f1", "balanced_acc"]:
            values = [v for v in collected[f"{split}_{m}"] if v is not None]
            best_metrics[f"{split}_best_{m}"] = max(values) if values else None

    val_conf_mats = [epoch_data["confusion_matrix"] for epoch_data in val_data if "confusion_matrix" in epoch_data] if val_data else []
    test_conf_mats = [epoch_data["confusion_matrix"] for epoch_data in test_data if "confusion_matrix" in epoch_data] if test_data else []

    experiment_data = {
        "exp_name": exp_name,
        "model_name": model_name,
        "backbone": backbone,
        "backbone_out_dim": backbone_out_dim,
        "reduced_dims": str(reduced_dims),
        "oehnet_c": oehnet_c,
        "loss_fn": loss_name,
        "batch_size": batch_size,
        "eucl_lr": eucl_lr,
        "eucl_wd": eucl_wd,
        "riem_lr": riem_lr,
        "weighted_criterion": weighted_criterion,
        "use_ohem": use_ohem,
        "classes": str(classes),
        "num_epochs": num_epochs,
        "val_size": val_size,
        "test_size": test_size,
        # Per-epoch metrics
        **{f"train_{m}s": str(collected[f"train_{m}"]) for m in metrics},
        **{f"val_{m}s": str(collected[f"val_{m}"]) for m in metrics},
        **{f"test_{m}s": str(collected[f"test_{m}"]) for m in metrics},
        # Best metrics
        **best_metrics,
        # Confusion matrices
        "val_confusion_matrices": str(val_conf_mats),
        "test_confusion_matrices": str(test_conf_mats),
    }

    df = pd.DataFrame([experiment_data])
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as file:
        if not file_exists:
            df.to_csv(file, header=True, index=False)
        else:
            df.to_csv(file, header=False, index=False)

    log.info(f"Logged experiment data for {exp_name} to CSV.")
    
def plot_metrics(train_data = None, val_data = None, save_dir=f"{os.getenv('EXP_PATH')}"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    if train_data:
        ax[0].plot([d["epoch"] for d in train_data], [d["loss"] for d in train_data],
                   label="Train Loss", marker="o")
    if val_data:
        ax[0].plot([d["epoch"] for d in val_data], [d["loss"] for d in val_data],
                   label="Val Loss", marker="o")
    ax[0].set_title("Loss per Epoch")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.6)

    # --- Accuracy ---
    if train_data:
        ax[1].plot([d["epoch"] for d in train_data], [d["acc"] for d in train_data],
                   label="Train Acc", marker="s")
    if val_data:
        ax[1].plot([d["epoch"] for d in val_data], [d["acc"] for d in val_data],
                   label="Val Acc", marker="s")
    ax[1].set_title("Accuracy per Epoch")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    ax[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "train_val_loss_accuracy.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # --- Other metrics (precision, recall, f1, balanced_acc) ---
    metrics_to_plot = ["precision", "recall", "f1", "balanced_acc"]
    for metric in metrics_to_plot:
        plt.figure(figsize=(7, 5))
        if val_data:
            plt.plot([d["epoch"] for d in val_data],
                     [d[metric] for d in val_data],
                     label=f"Val {metric.capitalize()}", marker="o")
        plt.title(f"{metric.capitalize()} per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(save_dir, f"{metric}_per_epoch.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()