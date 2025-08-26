import torch.nn as nn
from osteo_gdl.models import OEHNet, OSPNet, OsteoGNN
from osteo_gdl.utils.const import MODELS

def make_model(args) -> nn.Module:
    model = args.model
    assert model in MODELS, f"Cannot find given model: {model}"

    num_classes = len(args.classes)

    match model:
        case "oehnet":
            return OEHNet(
                backbone=args.backbone,
                pretrained=args.pretrained,
                feature_dim=args.backbone_out_dim,
                oehnet_c=args.oehnet_c,
                num_classes=num_classes,
            )
        case "ospnet":
            return OSPNet(
                backbone=args.backbone,
                pretrained=args.pretrained,
                reduced_dim=args.backbone_out_dim,
                num_classes=num_classes,
            )
        
        case "osteognn":
            return OsteoGNN(
                # Backbone
                backbone=args.backbone,
                backbone_out_dim=args.backbone_out_dim,
                pretrained=args.pretrained,
                patch_size=args.patch_size,
                # GNN
                pos_emb=args.pos_emb,
                conv_dims=args.gnn_dims,
                conv_type=args.gnn_conv,
                pool_type=args.pool_type,
                dropout=args.dropout,
                act_type=args.gnn_act,
                k=args.k,
                # CLS
                cls_dims=args.cls_dims,
                cls_act=args.cls_act,
                num_classes=len(args.classes)
            )