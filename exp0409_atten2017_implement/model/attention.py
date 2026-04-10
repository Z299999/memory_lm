"""
Scaled Dot-Product Attention and Multi-Head Attention.
Paper Section 3.2
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,   # (B, H, T_q, d_k)
        k: torch.Tensor,   # (B, H, T_k, d_k)
        v: torch.Tensor,   # (B, H, T_k, d_v)
        mask: torch.Tensor | None = None,  # (B, 1, T_q, T_k) or (B, 1, 1, T_k)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        # (B, H, T_q, T_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v)   # (B, H, T_q, d_v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    Paper Section 3.2.2
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        q: torch.Tensor,   # (B, T_q, d_model)
        k: torch.Tensor,   # (B, T_k, d_model)
        v: torch.Tensor,   # (B, T_k, d_model)
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = q.size(0)

        # Project and split into heads: (B, T, d_model) -> (B, H, T, d_k)
        def project_and_split(x, w):
            return w(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        q = project_and_split(q, self.w_q)
        k = project_and_split(k, self.w_k)
        v = project_and_split(v, self.w_v)

        # Attention: (B, H, T_q, d_k)
        x, attn = self.attention(q, k, v, mask)

        # Concat heads: (B, T_q, d_model)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.w_o(x), attn
