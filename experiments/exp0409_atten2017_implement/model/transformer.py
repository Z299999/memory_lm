"""
Full Transformer model (encoder-decoder).
Shared source/target/pre-softmax embedding weights, scaled by sqrt(d_model).
Paper Section 3, Section 3.4
"""
import math
import torch
import torch.nn as nn
from config import Config
from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model

        # Shared embedding (paper Section 3.4: same weight matrix for src emb,
        # tgt emb, and pre-softmax linear transformation)
        self.embedding = nn.Embedding(cfg.vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, cfg.dropout, cfg.max_seq_len)

        self.encoder = Encoder(d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.dropout)
        self.decoder = Decoder(d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.dropout)

        # Pre-softmax linear shares weight with embedding (paper Section 3.4)
        self.out_proj = nn.Linear(d_model, cfg.vocab_size, bias=False)
        self.out_proj.weight = self.embedding.weight

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: torch.Tensor,         # (B, T_src)  token ids
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.embedding(src) * math.sqrt(self.cfg.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,         # (B, T_tgt)  token ids
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.embedding(tgt) * math.sqrt(self.cfg.d_model))
        return self.decoder(x, enc_out, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.out_proj(dec_out)   # (B, T_tgt, vocab_size) logits


def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Padding mask: (B, 1, 1, T_src), True where token is NOT padding."""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Combined padding + causal mask: (B, 1, T_tgt, T_tgt)."""
    T = tgt.size(1)
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)        # (B, 1, 1, T)
    causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # (T, T)
    return pad_mask & causal_mask
