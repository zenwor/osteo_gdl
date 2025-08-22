from loguru import logger as log

from osteo_gdl.utils.data import load_imgs_feats
from osteo_gdl.utils.dataset import make_dataloader, make_datasets
from osteo_gdl.utils.parse import parse_args

cls_map = dict()


def setup_env(args):
    global cls_map
    cls_map = {args.classes[idx]: idx for idx in range(len(args.classes))}


if __name__ == "__main__":
    # Setup general dynamic variables and parse the arguments
    args = parse_args()
    setup_env(args)

    # Load images and associated features (labels)
    imgs, feats = load_imgs_feats(args.imgs_path, args.feats_path, cls_map)
    log.info(f"Found {len(imgs)} samples.")

    # Create datasets
    train_ds, val_ds, test_ds = make_datasets(
        imgs, feats, args.val_size, args.test_size
    )
    log.info(f"Train dataset size: {len(train_ds)}")
    log.info(f"Validation dataset size: {len(val_ds)}")
    log.info(f"Test dataset size: {len(test_ds)}")

    # Create data loaders
    train_dl = make_dataloader(train_ds, args.batch_size, resample=False)
    val_dl = (
        None if val_ds is None else make_dataloader(val_ds, args.batch_size)
    )  # noqa: E501
    test_dl = make_dataloader(test_ds, args.batch_size)
    log.info("Successfully createad train, validation and test DataLoaders")
