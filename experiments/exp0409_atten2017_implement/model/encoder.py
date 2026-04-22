"""
Transformer Encoder.
Each EncoderLayer: Multi-Head Self-Attention + Position-wise FFN,
both wrapped with residual connection and layer normalization.
Paper Section 3.1
"""
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import PositionwiseFFN


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,          # (B, T_src, d_model)
        src_mask: torch.Tensor | None = None,  # (B, 1, 1, T_src) padding mask
    ) -> torch.Tensor:
        # Sub-layer 1: self-attention with residual + layer norm
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Sub-layer 2: FFN with residual + layer norm
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
