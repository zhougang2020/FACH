"""
Base cross-modal hash model shared by all teacher methods.

Architecture:
    Image path : backbone(x) -> FC(feat_dim -> hidden) -> BN -> ReLU
                              -> FC(hidden -> bit) -> Tanh
    Text  path : FC(text_dim -> hidden) -> BN -> ReLU
                              -> FC(hidden -> bit) -> Tanh

All method files (DADH.py, DCMH.py, …) instantiate this class and only
differ in their loss functions and training logic.
"""

import os
import torch
import torch.nn as nn
from backbones import build_backbone


class TextNet(nn.Module):
    """3-layer MLP text encoder: text_dim → hidden → bit."""
    def __init__(self, text_dim: int, hidden_dim: int, bit: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, bit),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HashNet(nn.Module):
    """
    Generic cross-modal hash model.

    forward_img(x)  -> h_img  real-valued (B, bit)
    forward_txt(t)  -> h_txt  real-valued (B, bit)
    generate_img_code(x) -> sign(h_img)
    generate_txt_code(t) -> sign(h_txt)
    """

    def __init__(
        self,
        backbone_name: str,
        bit: int,
        text_dim: int,
        hidden_dim: int = 4096,
        pretrained: bool = True,
    ):
        super().__init__()
        self.bit = bit

        # ── Image branch ──────────────────────────────────────────────────────
        backbone, feat_dim = build_backbone(backbone_name, pretrained)
        self.backbone = backbone
        self.img_hash = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bit),
            nn.Tanh(),
        )

        # ── Text branch ───────────────────────────────────────────────────────
        self.txt_net = TextNet(text_dim, hidden_dim, bit)

        # ── Feature dim (exposed for loss functions) ──────────────────────────
        self.feat_dim   = feat_dim
        self.hidden_dim = hidden_dim

    # ──────────────────────────────────────────────────────────────────────────

    def forward_img(self, x: torch.Tensor) -> torch.Tensor:
        """Raw backbone features + hash head. Returns (B, bit) in (-1, 1)."""
        feat = self.backbone(x)
        return self.img_hash(feat)

    def forward_txt(self, t: torch.Tensor) -> torch.Tensor:
        """Text MLP. Returns (B, bit) in (-1, 1)."""
        return self.txt_net(t)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """Returns (h_img, h_txt) both real-valued."""
        return self.forward_img(x), self.forward_txt(t)

    def get_img_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Raw backbone feature (before hash head)."""
        return self.backbone(x)

    def generate_img_code(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_img(x).sign()

    def generate_txt_code(self, t: torch.Tensor) -> torch.Tensor:
        return self.forward_txt(t).sign()

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, device=None):
        state = torch.load(path, map_location=device or 'cpu')
        self.load_state_dict(state)
