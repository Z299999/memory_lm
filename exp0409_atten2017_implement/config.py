from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Model (paper Table 3 base model) ---
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    vocab_size: int = 37000       # shared src/tgt BPE vocabulary

    # --- Training ---
    warmup_steps: int = 4000
    batch_size: int = 32
    max_steps: int = 100_000
    label_smoothing: float = 0.1
    clip_grad_norm: float = 1.0

    # Adam (paper Section 5.3)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9

    # --- Data ---
    data_lang_pair: str = "zh-en"   # source-target, from Helsinki-NLP/opus-100
    max_samples: int = 100_000      # subset size for local experiments

    # --- Paths ---
    checkpoint_dir: str = "runs/checkpoints"
    log_dir: str = "runs/logs"

    # --- Derived (computed, do not set manually) ---
    @property
    def d_k(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def d_v(self) -> int:
        return self.d_k
