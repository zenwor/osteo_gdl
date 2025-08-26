from loguru import logger as log

from osteo_gdl.utils.data import load_imgs_feats
from osteo_gdl.utils.dataset import make_dataloader, make_datasets
from osteo_gdl.utils.parse import parse_args

from osteo_gdl.train import make_optim, make_criterion, train
from osteo_gdl.models.model import make_model

cls_map = dict()


def setup_env(args):
    global cls_map
    cls_map = {args.classes[idx]: idx for idx in range(len(args.classes))}


if __name__ == "__main__":
    # Setup general dynamic variables and parse the arguments
    args = parse_args()
    setup_env(args)

    log.info(f"Running experiment: {args.exp_name}")

    # Create model -- useful for preparing transformations
    model = make_model(args).to(args.device)
    log.info(f"Model: {model}")

    # Load images and associated features (labels)
    imgs, feats = load_imgs_feats(args.imgs_path, args.feats_path, cls_map)
    log.info(f"Found {len(imgs)} samples.")

    # Create datasets
    train_ds, val_ds, test_ds = make_datasets(
        imgs, feats, args.val_size, args.test_size, args.aug,
        model=model
    )
    log.info(f"Train dataset size: {len(train_ds)}")
    log.info(f"Validation dataset size: {len(val_ds)}")
    log.info(f"Test dataset size: {len(test_ds)}")

    # Create data loaders
    train_dl = make_dataloader(train_ds, args.batch_size, device=args.device, resample=args.resample)
    val_dl = (
        None if val_ds is None else make_dataloader(val_ds, args.batch_size, device=args.device)
    )  # noqa: E501
    test_dl = make_dataloader(test_ds, args.batch_size, device=args.device)
    log.info("Successfully createad train, validation and test DataLoaders.")

    # Optimizer and loss function
    optim = make_optim(model, args.eucl_lr, args.eucl_wd, args.riem_lr)
    criterion = make_criterion(args.criterion, args.weighted_criterion, train_ds, device=args.device)
    log.info(f"Using optimizer: {optim}")
    log.info(f"Using criterion: {criterion}")

    # Launch training
    train(
        # Train
        model,
        train_dl,
        val_dl,
        test_dl,
        optim,
        criterion,
        args.ohem,
        args.num_epochs,
        args.device,
        # Log
        args.exp_name,
        args.model,
        args.backbone,
        args.backbone_out_dim,
        args.reduced_dims,
        args.oehnet_c,
        args.criterion,
        args.batch_size,
        args.eucl_lr,
        args.eucl_wd,
        args.riem_lr,
        args.weighted_criterion,
        args.classes,
        args.num_epochs,
        args.val_size,
        args.test_size,
    )
