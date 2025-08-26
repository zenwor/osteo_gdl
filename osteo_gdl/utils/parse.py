import argparse

from osteo_gdl.utils.const import (
    # Data
    DEF_CLASSES,
    CLASSES,
    DEF_TEST_SIZE,
    DEF_VAL_SIZE,
    DEF_FEATS_PATH,
    DEF_IMGS_PATH,
    # General architecture
    DEF_MODEL,
    MODELS,
    # Backbone
    DEF_BACKBONE_OUT_DIM,
    # Riemmanian models
    DEF_OEHNET_C,
    DEF_REDUCED_DIMS,
    # GNN
    DEF_PATCH_SIZE,
    DEF_GNN_CONV,
    GNN_CONV_MAP,
    DEF_GNN_POOL,
    GNN_POOL_MAP,
    DEF_DROPOUT,
    ACT_MAP,
    DEF_K,
    DEF_GNN_ACT,
    DEF_GNN_DIMS,
    DEF_CLS_DIMS,
    DEF_CLS_ACT,
    # Training
    DEF_NUM_EPOCHS,
    DEF_EUCL_LR,
    DEF_EUCL_WD,
    DEF_RIEM_LR,
    DEF_BATCH_SIZE,
    DEF_CRITERION,
    CRITERIONS,
    DEF_DEVICE,
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
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Use data augmentations.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample throughout training.",
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
        default=DEF_REDUCED_DIMS,
        help="Reduced dimensions.",
    )
    parser.add_argument(
        "--oehnet_c",
        type=float,
        default=DEF_OEHNET_C,
        help="Parameter C for Hyperbolic head.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=DEF_PATCH_SIZE,
        help="Patch size for GNN models.",
    )
    parser.add_argument(
        "--gnn_conv",
        type=str,
        default=DEF_GNN_CONV,
        choices=GNN_CONV_MAP.keys(),
        help="Convolution to be used in GNN models.",
    )
    parser.add_argument(
        "--gnn_dims",
        type=int,
        default=DEF_GNN_DIMS,
        nargs="+",
        help="GNN convolution dimensions.",
    )
    parser.add_argument(
        "--pos_emb",
        action="store_true",
        help="Utilize patch positional embeddings.",
    )
    parser.add_argument(
        "--pool_type",
        type=str,
        default=DEF_GNN_POOL,
        choices=GNN_POOL_MAP.keys(),
        help="GNN pooling layer type.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEF_DROPOUT,
        help="Dropout layer percentage.",
    )
    parser.add_argument(
        "--gnn_act",
        type=str,
        default=DEF_GNN_ACT,
        choices=ACT_MAP.keys(),
        help="Activation function.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEF_K,
        help="K to be used in KNN for graph construction.",
    )
    # CLS
    parser.add_argument(
        "--cls_dims",
        type=int,
        default=DEF_CLS_DIMS,
        nargs="+",
        help="Classification layer dimensions.",
    )
    parser.add_argument(
        "--cls_act",
        type=str,
        default=DEF_CLS_ACT,
        choices=ACT_MAP.keys(),
        help="Activation function in classification layer.",
    )
    
    # Training
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
        "--ohem",
        action="store_true",
        help="Use OHEM loss.",
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
