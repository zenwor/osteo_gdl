import argparse

from osteo_gdl.utils.const import (
    CLASSES,
    DEF_BATCH_SIZE,
    DEF_CLASSES,
    DEF_DEVICE,
    DEF_FEATS_PATH,
    DEF_IMGS_PATH,
    DEF_EUCL_LR,
    DEF_EUCL_WD,
    DEF_RIEM_LR,
    DEF_NUM_EPOCHS,
    MODELS,
    DEF_MODEL,
    DEF_BACKBONE_OUT_DIM,
    DEF_OEHNET_C,
    EUCL_OPTIM_MAP,
    RIEM_OPTIM_MAP,
    DEF_EUCL_OPTIM,
    DEF_RIEM_OPTIM,
    DEF_CRITERION,
    CRITERIONS,
    DEF_TEST_SIZE,
    DEF_VAL_SIZE,
    DEVICES,
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

    # Model
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default=DEF_MODEL,
    )
    parser.add_argument("--backbone", type=str, help="Backbone for feature extraction.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained feature extraction backbone.",
    )
    parser.add_argument(
        "--backbone_out_dim",
        type=int,
        default=DEF_BACKBONE_OUT_DIM,
        help="Backbone output dimension.",
    )
    parser.add_argument(
        "--reduced_dims",
        type=int,
        nargs="+",
        default=[],
        help="Reduced dimensions for DOSPNet.",
    )
    parser.add_argument(
        "--oehnet_c",
        type=float,
        default=DEF_OEHNET_C,
        help="Parameter C for Hyperbolic head.",
    )

    # Training
    parser.add_argument(
        "--eucl_optim",
        type=str,
        choices=EUCL_OPTIM_MAP.keys(),
        default=DEF_EUCL_OPTIM,
        help="Euclidean optimizer.",
    )
    parser.add_argument(
        "--riem_optim",
        type=str,
        choices=RIEM_OPTIM_MAP.keys(),
        default=DEF_RIEM_OPTIM,
        help="Riemannian optimizer.",
    )

    parser.add_argument(
        "--eucl_lr",
        type=float,
        default=DEF_EUCL_LR,
        help="Euclidean learning rate.",
    )
    parser.add_argument(
        "--eucl_wd",
        type=float,
        default=DEF_EUCL_WD,
        help="Euclidean weight decay.",
    )
    parser.add_argument(
        "--riem_lr",
        type=float,
        default=DEF_RIEM_LR,
        help="Riemannian Learning rate.",
    )

    parser.add_argument(
        "--criterion",
        type=str,
        choices=CRITERIONS,
        default=DEF_CRITERION,
        help="Criterion / loss function.",
    )
    parser.add_argument(
        "--weighted_criterion",
        action="store_true",
        help="Use weights in loss function.",
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
