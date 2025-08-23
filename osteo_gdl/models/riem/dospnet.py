import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from osteo_gdl.models.riem.spd import batch_covariance, SPDLayer, SPDFlatten
from osteo_gdl.models.riem.feat import make_backbone


class DOSPNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes=3,
        reduced_dims=[32, 16],
        pool_size=2,
        pretrained=True,
        fast_exp=True,
    ):
        super().__init__()
        # Backbone
        base = make_backbone(backbone, pretrained, feature_dim=-1)

        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.channel_reduce = nn.Conv2d(
            base.fc.in_features, reduced_dims[0], kernel_size=1, bias=False
        )

        # Stack SPD layers
        spd_layers = []
        for i in range(len(reduced_dims) - 1):
            spd_layers.append(
                SPDLayer(reduced_dims[i], reduced_dims[i + 1], fast_exp=fast_exp)
            )
        self.spd_stack = nn.Sequential(*spd_layers)

        # Classifier
        feat_dim = (reduced_dims[-1] * (reduced_dims[-1] + 1)) // 2
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        f = self.features(x)
        f = self.pool(f)
        f = self.channel_reduce(f)

        B, C, H, W = f.shape
        f = f.view(B, C, -1).transpose(1, 2)
        cov = batch_covariance(f, eps=1e-1, max_rank=16)

        for layer in self.spd_stack:
            cov = layer(cov)

        flat = SPDFlatten()(cov)
        out = self.classifier(flat)
        return out
