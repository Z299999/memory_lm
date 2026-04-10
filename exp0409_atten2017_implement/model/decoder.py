"""
Transformer Decoder.
Each DecoderLayer has three sub-layers:
  1. Masked Multi-Head Self-Attention (causal mask)
  2. Multi-Head Cross-Attention over encoder output
  3. Position-wise FFN
All wrapped with residual connection and layer normalization.
Paper Section 3.1
"""
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import PositionwiseFFN


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,            # (B, T_tgt, d_model)
        enc_out: torch.Tensor,      # (B, T_src, d_model)
        src_mask: torch.Tensor | None = None,  # (B, 1, 1, T_src)
        tgt_mask: torch.Tensor | None = None,  # (B, 1, T_tgt, T_tgt) causal
    ) -> torch.Tensor:
        # Sub-layer 1: masked self-attention
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Sub-layer 2: cross-attention
        attn_out, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        # Sub-layer 3: FFN
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)
