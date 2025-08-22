import argparse

from osteo_gdl.utils.const import (
    CLASSES,
    DEF_BATCH_SIZE,
    DEF_CLASSES,
    DEF_DEVICE,
    DEF_FEATS_PATH,
    DEF_IMGS_PATH,
    DEF_LR,
    DEF_NUM_EPOCHS,
    DEF_OPTIM,
    DEF_TEST_SIZE,
    DEF_VAL_SIZE,
    DEVICES,
    OPTIMS,
)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_name",
        type=str,
        default="default_experiment",
        help="Name of the experiment run with the given parameters.",
    )

    # Data
    parser.add_argument(
        "--imgs_path",
        type=str,
        default=DEF_IMGS_PATH,
        help="Path to root of all images directory.",
    )
    parser.add_argument(
        "--feats_path",
        type=str,
        default=DEF_FEATS_PATH,
        help="Path to features csv. Used for class extraction.",
    )

    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        choices=CLASSES,
        default=DEF_CLASSES,
        help="Classes to use for model training.",
    )

    # Training-related
    parser.add_argument(
        "--optim",
        type=str,
        choices=OPTIMS,
        default=DEF_OPTIM,
        help="Optimizer to use while training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEF_LR,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEF_NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEF_BATCH_SIZE, help="Batch size."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICES,
        default=DEF_DEVICE,
        help="Device to use for training.",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=DEF_VAL_SIZE,
        help="Validation set size.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=DEF_TEST_SIZE,
        help="Test set size.",
    )

    return parser.parse_args()
