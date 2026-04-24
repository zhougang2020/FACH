"""
Text substitute model for FACH.

Architecture: 3-layer MLP (as stated in the paper).
  Linear(text_dim → hidden) → ReLU → Linear(hidden → hidden) → ReLU
  → Linear(hidden → bit) → Tanh
"""

import torch
import torch.nn as nn


class TxtNet(nn.Module):
    def __init__(
        self,
        bit: int,
        text_dim: int,
        hidden_dim: int = 4096,
        dropout: float = 0.0,
    ):
        """
        Args:
            bit       : hash code length K.
            text_dim  : input text feature dimension.
            hidden_dim: size of hidden layers.
            dropout   : dropout probability (0 = disabled).
        """
        super().__init__()
        self.bit = bit

        layers = [nn.Linear(text_dim, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(hidden_dim, bit), nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns real-valued hash codes in (-1, 1)^K."""
        return self.net(x)

    def generate_hash(self, x: torch.Tensor) -> torch.Tensor:
        """Returns binarised hash codes in {-1, +1}^K."""
        return self.forward(x).sign()
