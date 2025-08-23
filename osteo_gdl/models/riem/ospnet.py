import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from osteo_gdl.models.riem.spd import (
    batch_covariance,
    SPDRectifiedLinear,
    SPDTangentProjection,
    SPDFlatten,
)
from osteo_gdl.models.riem.feat import make_backbone


class OSPNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes=3,
        pretrained=True,
        reduced_dim=512,
        pool_size=5,
    ):
        super().__init__()
        base = make_backbone(backbone, pretrained, feature_dim=-1)

        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Reduce channels before SPD
        self.channel_reduce = nn.Conv2d(
            base.fc.in_features, reduced_dim, kernel_size=1, bias=False
        )

        # SPD pipeline
        self.spd_relu = SPDRectifiedLinear()
        self.spd_logmap = SPDTangentProjection()
        self.spd_flatten = SPDFlatten()

        # Classifier on tangent space
        feat_dim = (reduced_dim * (reduced_dim + 1)) // 2
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f = self.features(x)
        f = self.pool(f)
        f = self.channel_reduce(f)

        B, C, H, W = f.shape
        f = f.view(B, C, -1).transpose(1, 2)

        cov = batch_covariance(f)
        cov = self.spd_relu(cov)
        cov = self.spd_logmap(cov)
        flat = self.spd_flatten(cov)

        out = self.classifier(flat)
        return out
