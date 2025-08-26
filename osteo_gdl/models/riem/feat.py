import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ViTModel, ViTConfig

import warnings

warnings.filterwarnings(
    "ignore", message=".*pretrained.*deprecated.*", category=UserWarning
)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet50", out_dim=256, pretrained=True):
        super().__init__()
        if not hasattr(models, model_name):
            raise ValueError(
                f"Unknown model: {model_name}. Check torchvision.models for available options."
            )

        if pretrained:
            base = getattr(models, model_name)(pretrained=True)
        else:
            base = getattr(models, model_name)(weights=None)

        self.features = nn.Sequential(*list(base.children())[:-2])

        # Figure out feature dimension (last conv layer channels)
        if hasattr(base, "fc"):
            in_dim = base.fc.in_features
        else:
            raise ValueError(f"Cannot infer final feature dim for {model_name}")

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        f = self.features(x)
        f = self.proj(f)
        f = self.pool(f)
        f = f.squeeze(-1).squeeze(-1)
        return f


class ViTFeatureExtractor(nn.Module):
    def __init__(
        self, model_name="google/vit-base-patch16-224", out_dim=256, pretrained=True
    ):
        super().__init__()
        self.vit = (
            ViTModel.from_pretrained(model_name)
            if pretrained
            else ViTModel(ViTConfig())
        )
        hidden_dim = self.vit.config.hidden_size
        if out_dim == -1:
            out_dim = hidden_dim
        self.proj = nn.Linear(self.vit.config.hidden_size, out_dim)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        f = self.proj(cls_token)
        return f


def make_backbone(
    backbone: str,
    pretrained: bool = True,
    feature_dim: int = 128,
):
    # Backbone -- Try for ResNet, and if that is not the case, try for ViT
    if "resnet" in backbone:
        if feature_dim == -1:
            if not hasattr(models, backbone):
                raise ValueError(
                    f"Unknown model: {backbone}. Check torchvision.models for available options."
                )
            if pretrained:
                base = getattr(models, backbone)(pretrained=True)
            else:
                base = getattr(models, backbone)(pretrained=False)
            return base
        else:
            return ResNetFeatureExtractor(
                model_name=backbone, out_dim=feature_dim, pretrained=pretrained
            )
    else:
        return ViTFeatureExtractor(
            model_name=backbone, out_dim=feature_dim, pretrained=pretrained
        )
