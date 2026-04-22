"""
Position-wise Feed-Forward Network.
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
Paper Section 3.3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
