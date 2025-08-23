import torch.nn as nn

from osteo_gdl.utils.const import MODEL_MAP
from osteo_gdl.models.riem.oehnet import OEHNet
from osteo_gdl.models.riem.ospnet import OSPNet
from osteo_gdl.models.riem.dospnet import DOSPNet


def make_model(args) -> nn.Module:
    model = args.model
    assert model in MODEL_MAP, f"Cannot find given model: {model}"

    backbone, pretrained = args.backbone, args.pretrained
    feature_dim = args.backbone_out_dim
    oehnet_c = args.oehnet_c
    reduced_dims = args.reduced_dims

    num_classes = len(args.classes)

    match model:
        case "oehnet":
            return OEHNet(
                backbone=backbone,
                pretrained=pretrained,
                feature_dim=feature_dim,
                oehnet_c=oehnet_c,
                num_classes=num_classes,
            )
        case "ospnet":
            return OSPNet(
                backbone=backbone,
                pretrained=pretrained,
                reduced_dim=feature_dim,
                num_classes=num_classes,
            )
        case "dospnet":
            return DOSPNet(
                backbone=backbone,
                reduced_dims=reduced_dims,
                num_classes=num_classes,
            )
