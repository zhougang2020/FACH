"""
Image substitute model for FACH.

Two modes:
  1. Raw-image mode  (use_backbone=True):
       VGG11 backbone → FC(4096→hidden) → FC(hidden→bit)
       Input: (B, 3, H, W) RGB images in [0,1].

  2. Feature mode (use_backbone=False):
       MLP on precomputed features (e.g. 4096-dim from VGG-F).
       Input: (B, image_dim) precomputed features.

In both cases the final real-valued output (before sign) is returned by
`forward()`.  `generate_hash()` returns sign-binarised codes.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ImgNet(nn.Module):
    def __init__(
        self,
        bit: int,
        image_dim: int = 4096,
        hidden_dim: int = 4096,
        use_backbone: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            bit        : hash code length K.
            image_dim  : input feature dimension (ignored when use_backbone=True).
            hidden_dim : size of the intermediate FC layer.
            use_backbone: if True, prepend VGG11 (processes raw RGB images).
            dropout    : dropout probability (0 = disabled).
        """
        super().__init__()
        self.bit = bit
        self.use_backbone = use_backbone

        if use_backbone:
            # VGG11 backbone — remove final classifier, keep features
            vgg = tv_models.vgg11(weights=tv_models.VGG11_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(vgg.features.children()),
                                          vgg.avgpool)
            backbone_out = 512 * 7 * 7  # default for 224×224 input

            self.fc_layers = nn.Sequential(
                nn.Linear(backbone_out, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, bit),
                nn.Tanh(),
            )
        else:
            # Feature-based MLP (matches DADH / other existing models)
            self.backbone = None
            layers = [nn.Linear(image_dim, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers += [nn.Linear(hidden_dim, bit), nn.Tanh()]
            self.fc_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns real-valued hash codes in (-1, 1)^K (no sign applied).
        x: (B, 3, H, W) for backbone mode, or (B, image_dim) for feature mode.
        """
        if self.use_backbone:
            feat = self.backbone(x).flatten(1)
        else:
            feat = x
        return self.fc_layers(feat)

    def generate_hash(self, x: torch.Tensor) -> torch.Tensor:
        """Returns binarised hash codes in {-1, +1}^K."""
        return self.forward(x).sign()
