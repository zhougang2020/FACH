"""
Six backbone image encoders used as victim-model backbones in FACH.

Each backbone wraps a torchvision pretrained model and exposes:
    forward(x) -> feature vector  (B, feat_dim)

Supported:
    'AlexNet'   -> 4096-dim
    'VGG11'     -> 4096-dim
    'RN50'      -> 2048-dim
    'RN152'     -> 2048-dim
    'IncV3'     -> 2048-dim  (Inception-v3, input must be ≥299×299)
    'DN161'     -> 2208-dim  (DenseNet-161)

Usage:
    from backbones import build_backbone
    backbone, feat_dim = build_backbone('VGG11', pretrained=True)
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ──────────────────────────────────────────────────────────────────────────────
# Output feature dimensions
# ──────────────────────────────────────────────────────────────────────────────
BACKBONE_FEAT_DIM = {
    'AlexNet': 4096,
    'VGG11':   4096,
    'RN50':    2048,
    'RN152':   2048,
    'IncV3':   2048,
    'DN161':   2208,
}

BACKBONE_NAMES = list(BACKBONE_FEAT_DIM.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Backbone wrappers
# ──────────────────────────────────────────────────────────────────────────────

class _AlexNetBackbone(nn.Module):
    """AlexNet — outputs 4096-dim ReLU feature from fc7."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.alexnet(
            weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features   = base.features
        self.avgpool    = base.avgpool
        # Keep only up to fc7 (index 4 in the original classifier):
        #   0: Dropout  1: Linear(9216→4096)  2: ReLU
        #   3: Dropout  4: Linear(4096→4096)  5: ReLU
        self.classifier = nn.Sequential(*list(base.classifier.children())[:6])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)   # (B, 4096)


class _VGG11Backbone(nn.Module):
    """VGG-11 — outputs 4096-dim feature from fc7."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.vgg11(
            weights=models.VGG11_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.avgpool  = base.avgpool
        # classifier[0..5]: Linear(25088→4096) ReLU Dropout Linear(4096→4096) ReLU Dropout
        self.classifier = nn.Sequential(*list(base.classifier.children())[:6])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)   # (B, 4096)


class _ResNetBackbone(nn.Module):
    """ResNet-50 or ResNet-152 — outputs 2048-dim average-pooled feature."""
    def __init__(self, depth: int, pretrained: bool = True):
        super().__init__()
        if depth == 50:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
        elif depth == 152:
            weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        # Remove the final FC layer
        self.encoder = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)   # (B, 2048)


class _InceptionV3Backbone(nn.Module):
    """Inception-v3 — outputs 2048-dim feature. Input should be ≥299×299."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.inception_v3(weights=weights, aux_logits=False)
        # Replace fc with identity to get 2048-dim feature
        base.fc = nn.Identity()
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # (B, 2048)


class _DenseNet161Backbone(nn.Module):
    """DenseNet-161 — outputs 2208-dim feature."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.densenet161(weights=weights)
        self.features = base.features
        self.relu     = nn.ReLU(inplace=True)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.features(x))
        x = self.pool(x)
        return x.flatten(1)   # (B, 2208)


# ──────────────────────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────────────────────

def build_backbone(name: str, pretrained: bool = True) -> tuple:
    """
    Build a backbone by name.

    Args:
        name      : one of BACKBONE_NAMES (case-insensitive).
        pretrained: load ImageNet weights.

    Returns:
        (backbone_module, feat_dim)
    """
    key = name.strip()
    # Normalise common aliases
    _ALIASES = {
        'alexnet': 'AlexNet',
        'vgg11':   'VGG11',
        'resnet50': 'RN50', 'rn50': 'RN50',
        'resnet152': 'RN152', 'rn152': 'RN152',
        'inceptionv3': 'IncV3', 'incv3': 'IncV3', 'inception': 'IncV3',
        'densenet161': 'DN161', 'dn161': 'DN161', 'densenet': 'DN161',
    }
    key = _ALIASES.get(key.lower(), key)

    if key == 'AlexNet':
        backbone = _AlexNetBackbone(pretrained)
    elif key == 'VGG11':
        backbone = _VGG11Backbone(pretrained)
    elif key == 'RN50':
        backbone = _ResNetBackbone(50, pretrained)
    elif key == 'RN152':
        backbone = _ResNetBackbone(152, pretrained)
    elif key == 'IncV3':
        backbone = _InceptionV3Backbone(pretrained)
    elif key == 'DN161':
        backbone = _DenseNet161Backbone(pretrained)
    else:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Choose from: {BACKBONE_NAMES}"
        )
    feat_dim = BACKBONE_FEAT_DIM[key]
    return backbone, feat_dim
