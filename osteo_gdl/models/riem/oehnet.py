import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from osteo_gdl.models.riem.feat import make_backbone


class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, c=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def mobius_matvec(self, x):
        x_tan = self.log_map(x)
        y_tan = self.linear(x_tan)
        y_hyp = self.exp_map(y_tan)
        return y_hyp

    def log_map(self, x):
        norm = torch.clamp(torch.norm(x, p=2, dim=-1, keepdim=True), min=1e-15)
        sqrt_c = self.c**0.5
        scale = (1.0 / sqrt_c) * torch.atanh(sqrt_c * norm) / norm
        return scale * x

    def exp_map(self, v):
        norm = torch.clamp(torch.norm(v, p=2, dim=-1, keepdim=True), min=1e-15)
        sqrt_c = self.c**0.5
        scale = (1.0 / sqrt_c) * torch.tanh(sqrt_c * norm) / norm
        return scale * v

    def forward(self, x):
        return self.mobius_matvec(x)


class HyperbolicHead(nn.Module):
    def __init__(self, in_dim, num_classes=3, c=1.0):
        super().__init__()
        self.manifold = geoopt.manifolds.PoincareBall(c=c)
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.c = c

        self.class_points = geoopt.ManifoldParameter(
            torch.empty(num_classes, in_dim), manifold=self.manifold
        )
        self.class_points.is_riemannian = True

        self.hyp_layers = nn.Sequential(
            HyperbolicLinear(in_dim, in_dim, c=c),
            nn.ReLU(),
            HyperbolicLinear(in_dim, in_dim, c=c),
            nn.ReLU(),
            HyperbolicLinear(in_dim, in_dim, c=c),
        )

        self.reset_parameters(radius=0.7)

    def reset_parameters(self, radius=0.7):
        with torch.no_grad():
            directions = torch.randn(self.num_classes, self.in_dim)
            directions = F.normalize(directions, dim=-1)
            perturb = 0.05 * torch.randn_like(directions)
            self.class_points.copy_((directions + perturb) * radius)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1) * 0.8
        z = self.manifold.expmap0(x)

        z = self.hyp_layers(z)

        dists = self.manifold.dist(z.unsqueeze(1), self.class_points)
        logits = -dists
        return logits


class OEHNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 128,
        num_classes: int = 3,
        oehnet_c: float = 1.0,
    ):
        super().__init__()

        self.backbone = make_backbone(backbone, pretrained, feature_dim)
        # Hyperbolic head for classification
        self.hyperbolic_head = HyperbolicHead(
            in_dim=feature_dim,
            num_classes=num_classes,
            c=oehnet_c,
        )

    def forward(self, x):
        f = self.backbone(x)  # (B, feature_dim)
        out = self.hyperbolic_head(f)  # (B, num_classes)
        return out
