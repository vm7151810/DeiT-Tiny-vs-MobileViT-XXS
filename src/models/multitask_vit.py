import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict

class MultiTaskViT(nn.Module):
    """
    Multi-Task Vision Transformer backbone with multiple heads for:
    1. Primary Class Classification
    2. Attribute Prediction (Color, Material, Shape, etc.)
    """
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        attr_sizes: Dict[str, int],
        pretrained: bool = True,
        global_pool: str = "avg"
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0, 
            global_pool=global_pool
        )
        self.num_classes = num_classes
        self.attr_names = list(attr_sizes.keys())
        
        feats = self.backbone.num_features

        # classification head
        self.class_head = nn.Linear(feats, num_classes)

        # per-attribute heads
        self.attr_heads = nn.ModuleDict(
            {a: nn.Linear(feats, size) for a, size in attr_sizes.items()}
        )

        # initialize heads
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.class_head.weight, std=0.01)
        nn.init.constant_(self.class_head.bias, 0)
        for head in self.attr_heads.values():
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, 0)

    def forward(self, x):
        feat = self.backbone.forward_features(x)

        # Handle different backbone output shapes
        if feat.ndim == 3:
            feat = feat.mean(dim=1)
        elif feat.ndim == 4:
            feat = feat.mean(dim=[2, 3])

        cls_logits = self.class_head(feat)
        attr_logits = {a: head(feat) for a, head in self.attr_heads.items()}
        
        return cls_logits, attr_logits, feat
