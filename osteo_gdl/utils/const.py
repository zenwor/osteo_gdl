import torch
from torch.optim import Adam, AdamW, SGD
from geoopt.optim import RiemannianAdam, RiemannianSGD
from torch.nn import CrossEntropyLoss
import torchvision.models as backbone_models

from osteo_gdl.models.riem.oehnet import OEHNet
from osteo_gdl.models.riem.ospnet import OSPNet
from osteo_gdl.models.riem.dospnet import DOSPNet

# Data
DEF_IMGS_PATH = "$DATA_PATH"
DEF_FEATS_PATH = "$DATA_PATH/feats.csv"
FEATS_COLS = ["image.name", "classification"]

# Model
DEF_PRETRAINED = True
MODEL_MAP = {"oehnet": OEHNet, "ospnet": OSPNet, "dospnet": DOSPNet}
MODELS = MODEL_MAP.keys()
DEF_MODEL = "oehnet"

BACKBONE_MAP = {
    "resnet18": backbone_models.resnet18,
    "resnet50": backbone_models.resnet50,
}
BACKBONES = BACKBONE_MAP.keys()
DEF_BACKBONE_OUT_DIM = 128
DEF_OEHNET_C = 1.0

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
