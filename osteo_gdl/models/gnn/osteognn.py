import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ViTModel, ViTConfig
from torch_geometric.nn import knn_graph, BatchNorm
from typing import List
from osteo_gdl.utils.const import get_gnn_conv, get_gnn_pool, get_act
from osteo_gdl.models.riem.feat import make_backbone

def get_backbone_out_dim(backbone: nn.Module, input_size=(1, 3, 224, 224)) -> int:
    backbone.eval()
    with torch.no_grad():
        x = torch.zeros(input_size)
        out = backbone(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
    return out.shape[1]  


class PatchEmb(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        in_channels: int = 3,
        out_dim: int = 256,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.dropout = nn.Dropout(dropout)

        if "resnet" in self.backbone_name:
            backbone_model = getattr(models, backbone)(pretrained=pretrained)

            if in_channels != 3:
                backbone_model.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )

            self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
            feat_dim = get_backbone_out_dim(self.feature_extractor)
            self.fc = nn.Linear(feat_dim, out_dim)
            self.is_vit = False

        elif "vit" in self.backbone_name:
            self.vit = (
                ViTModel.from_pretrained(backbone) if pretrained else ViTModel(ViTConfig())
            )
            self.fc = nn.Linear(self.vit.config.hidden_size, out_dim)
            self.is_vit = True

        else:
            raise ValueError(f"Backbone {backbone} not supported. Use 'resnet' or 'vit'.")

    def forward(self, x):
        if self.is_vit:
            outputs = self.vit(pixel_values=x)
            cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
            x = self.fc(cls_token)
            x = self.dropout(x)
            return x
        else:
            x = self.feature_extractor(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            x = self.dropout(x)
            return x
    
class GNN(nn.Module):
    def __init__(
        self, 
        pos_emb: bool = False,
        conv_dims: List[int] = [],
        conv_type: str = "gatconv",
        pool_type: str = "mean",
        max_patches: int = 256, 
        dropout: float = 0.3,
        act_type: str = "relu",
        
    ):
        super().__init__()
        
        self.pos_emb = None
        if pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(max_patches, conv_dims[0]))
        
        self.convs, self.norms = self.make_layers(
            conv_dims,
            conv_type,
            act_type,
        )
        self.dropout = nn.Dropout(dropout)
        self.pool = get_gnn_pool(pool_type)
        
        
    def make_layers(self, conv_dims: List[int], conv_type: str, act_type: str):
        conv = get_gnn_conv(conv_type)
        convs, norms = nn.ModuleList(), nn.ModuleList()
        for in_dim, out_dim in zip(conv_dims[:-1], conv_dims[1:]):
            convs.append(conv(in_dim, out_dim))
            norms.append(BatchNorm(out_dim))
        
        self.activation = get_act(act_type)()
        
        return convs, norms

    def forward(self, x, edge_index, batch):
        # If applicable, add positional embeddings
        if self.pos_emb is not None:
            num_patches = x.size(0) 
            if num_patches <= self.pos_embed.size(0):
                pos_enc = self.pos_embed[:num_patches]
            else:
                raise ValueError(f"Number of patches ({num_patches}) exceeds max_patches.")
            x = x + pos_enc

        # conv -> norm -> activation -> dropout
        for conv, norm in zip(self.convs, self.norms):
            x = self.dropout(self.activation(norm(conv(x, edge_index))))
        
        x = self.pool(x, batch)
        return x

class MLP(nn.Module):
    def __init__(
        self, 
        dims: List[int], 
        act: str = "relu", 
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        act = get_act(act)
        
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))

        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class OsteoGNN(nn.Module):
    def __init__(
        self,
        # Backbone 
        backbone: str = "resnet18",
        backbone_out_dim: int = 256,
        pretrained: bool = True,
        patch_size: int = 16,
        # GNN
        pos_emb: bool = False,
        conv_dims: List[int] = [],
        conv_type: str = "gatconv",
        pool_type: str = "mean",
        dropout: float = 0.3,
        act_type: str = "relu",
        k: int = 6,
        # CLS
        cls_dims: List[int] = [], 
        cls_act: str = "relu",
        num_classes: int = 3,
    ):
        super().__init__()
        self.backbone = PatchEmb(backbone=backbone, out_dim=backbone_out_dim, pretrained=pretrained)

        self.gnn = GNN(
            pos_emb=pos_emb,
            conv_dims=conv_dims,
            conv_type=conv_type,
            max_patches=1024,
            pool_type=pool_type,
            dropout=dropout,
            act_type=act_type,
        )
        self.patch_size = patch_size
        self.k = k
        self.cls = MLP(cls_dims, cls_act, num_classes)
        
    def forward(self, imgs):
        B = imgs.size(0)
        
        if "vit" in self.backbone.backbone_name:
            # ViT handles patching internally
            outputs = self.backbone.vit(pixel_values=imgs)
            patch_embeds = outputs.last_hidden_state[:, 1:, :]
            B, P, D = patch_embeds.shape
            x = patch_embeds.reshape(B * P, D)
            batch_idx = torch.arange(B, device=imgs.device).repeat_interleave(P)
            
            # Optional: project embeddings if backbone.fc exists
            if hasattr(self.backbone, "fc"):
                x = self.backbone.fc(x)
                x = self.backbone.dropout(x)

        else:
            # ResNet backbone, patch manually
            patches = patch_images(imgs, self.patch_size)
            B, P, C, h, w = patches.shape
            patches = patches.view(B * P, C, h, w)
            batch_idx = torch.arange(B, device=imgs.device).repeat_interleave(P)
            x = self.backbone(patches)

        # Build KNN graph based on patch embeddings
        edge_index = knn_graph(x, k=self.k, batch=batch_idx, loop=False)

        # Get graph representation
        graph_repr = self.gnn(x, edge_index, batch_idx)

        # Classifier
        logits = self.cls(graph_repr)
        return logits
    
def patch_images(x, patch_size: int = 16):
    B, C, H, W = x.shape
    patch_h, patch_w = patch_size, patch_size
    patches = []
    for i in range(0, H, patch_h):
        for j in range(0, W, patch_w):
            patches.append(x[:, :, i:i+patch_h, j:j+patch_w])
    patches = torch.stack(patches, dim=1)
    return patches

