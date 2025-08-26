import torch
import torchvision.models as backbone_models
from torch.optim import Adam, AdamW, SGD
from geoopt.optim import RiemannianAdam, RiemannianSGD
from torch.nn import CrossEntropyLoss
from torch.nn import ReLU, LeakyReLU, GELU

from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from torch_geometric.nn import (
    global_mean_pool, global_max_pool, global_add_pool
)

# Data
DEF_IMGS_PATH = "$DATA_PATH"
DEF_FEATS_PATH = "$DATA_PATH/feats.csv"
FEATS_COLS = ["image.name", "classification"]

# Model
DEF_MODEL = "oehnet"
MODELS = ["oehnet", "ospnet", "osteognn"]
DEF_PRETRAINED = True
BACKBONE_MAP = {
    "resnet18": backbone_models.resnet18,
    "resnet50": backbone_models.resnet50,
}
BACKBONES = BACKBONE_MAP.keys()
DEF_BACKBONE_OUT_DIM = 128

# Riemannian models
DEF_OEHNET_C = 1.0
DEF_REDUCED_DIMS = [64, 32, 16]

# GNN models
DEF_PATCH_SIZE = 16
GNN_CONV_MAP = {
    "gatconv": GATConv,
    "sageconv": SAGEConv,
    "gcnconv": GCNConv
}
GNN_POOL_MAP = {
    "mean": global_mean_pool,
    "max": global_max_pool,
    "add": global_add_pool,
    "global_mean_pool": global_mean_pool,
    "global_max_pool": global_max_pool,
    "global_add_pool": global_add_pool,
}
DEF_GNN_POOL = "mean"
ACT_MAP = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "gelu": GELU,
}
DEF_GNN_CONV = "sageconv"
DEF_GNN_DIMS = [256, 256, 128]
DEF_GNN_ACT = "relu"

DEF_DROPOUT = 0.3
DEF_K = 6

# CLS
DEF_CLS_DIMS = [128, 128]
DEF_CLS_ACT = "relu"

# Training
DEF_EUCL_OPTIM = "adamw"
DEF_RIEM_OPTIM = "riemannianadam"
EUCL_OPTIM_MAP = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
}
RIEM_OPTIM_MAP = {
    "riemannianadam": RiemannianAdam,
    "riemanniansgd": RiemannianSGD,
}

# Criterion
DEF_CRITERION = "crossentropyloss"
CRITERION_MAP = {"crossentropyloss": CrossEntropyLoss}
CRITERIONS = CRITERION_MAP.keys()

DEF_EUCL_LR = 1e-4
DEF_EUCL_WD = 0.01
DEF_RIEM_LR = 5e-4

DEF_NUM_EPOCHS = 10
DEF_BATCH_SIZE = 32

DEF_DEVICE = "cuda"
DEVICES = ["cuda", "cpu"]

DEF_CLASSES = ["Non-Tumor", "Non-Viable-Tumor", "Viable"]
CLASSES = DEF_CLASSES

DEF_VAL_SIZE = 0.10
DEF_TEST_SIZE = 0.25

def get_gnn_conv(conv: str):
    return GNN_CONV_MAP.get(conv, None)

def get_gnn_pool(pool: str):
    return GNN_POOL_MAP.get(pool, None)

def get_act(act: str):
    return ACT_MAP.get(act, None)