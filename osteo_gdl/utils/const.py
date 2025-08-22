# Data related
DEF_IMGS_PATH = "$DATA_PATH"
DEF_FEATS_PATH = "$DATA_PATH/feats.csv"
FEATS_COLS = ["image.name", "classification"]

# Model related
DEF_PRETRAINED = True

# Training related
OPTIMS = ["adam", "adamw", "sgd", "adadelta"]
DEF_OPTIM = "adamw"

DEF_LR = 1e-4

DEF_NUM_EPOCHS = 10
DEF_BATCH_SIZE = 32

DEF_DEVICE = "cuda"
DEVICES = ["cuda", "cpu"]

DEF_CLASSES = ["Non-Tumor", "Non-Viable-Tumor", "Viable"]
CLASSES = DEF_CLASSES

DEF_VAL_SIZE = 0.10
DEF_TEST_SIZE = 0.25
