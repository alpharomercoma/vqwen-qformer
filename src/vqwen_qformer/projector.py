"""Linear projector: Q-Former hidden (768) -> Qwen3 hidden (2560).

BLIP-2 uses a single Linear here (not 2-layer MLP). We follow that.
"""
from __future__ import annotations

import torch
from torch import nn


class LinearProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
