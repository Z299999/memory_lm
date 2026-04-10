"""
Training script for the Transformer.

Usage:
    # Toy overfit test (100 pairs, no data download needed)
    python train.py --toy --steps 500

    # Train on opus-100 subset
    python train.py --data zh-en --max_samples 100000

    # Resume from checkpoint
    python train.py --resume runs/checkpoints/step_10000.pt
"""
import argparse
import math
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from config import Config
from model.transformer import Transformer, make_src_mask, make_tgt_mask

PAD_ID = 0


# ---------------------------------------------------------------------------
# Learning rate schedule (paper eq. 3)
# ---------------------------------------------------------------------------

def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


# ---------------------------------------------------------------------------
# Label-smoothed cross-entropy loss (paper Section 5.4)
# ---------------------------------------------------------------------------

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B*T, V), target: (B*T,)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Smooth target distribution
        smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.pad_idx] = 0.0
        # Zero out padding positions
        mask = (target == self.pad_idx)
        smooth_dist[mask] = 0.0
        loss = -(smooth_dist * log_probs).sum(dim=-1)
        n_tokens = (~mask).sum().clamp(min=1)
        return loss.sum() / n_tokens


# ---------------------------------------------------------------------------
# Toy dataset for quick overfit test
# ---------------------------------------------------------------------------

def make_toy_batch(vocab_size: int = 100, seq_len: int = 10, batch_size: int = 8, device=None):
    """Random integer sequences; model should memorise src -> tgt mapping."""
    src = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
    tgt = src.clone()   # identity mapping: trivial but verifies loss drops
    return src, tgt


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config, toy: bool = False, resume: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = Adam(
        model.parameters(),
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        lr=1.0,   # actual lr set by scheduler
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, cfg.d_model, cfg.warmup_steps),
    )
    criterion = LabelSmoothingLoss(cfg.vocab_size, PAD_ID, cfg.label_smoothing).to(device)

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    if toy:
        _train_toy(model, optimizer, scheduler, criterion, cfg, device, start_step)
    else:
        from data.dataset import build_dataloaders
        train_loader, val_loader, tokenizer = build_dataloaders(cfg)
        _train_full(model, optimizer, scheduler, criterion, train_loader, cfg, device, start_step)


def _train_toy(model, optimizer, scheduler, criterion, cfg, device, start_step=0):
    max_steps = 500
    print(f"Toy overfit test: {max_steps} steps")
    model.train()
    for step in range(start_step + 1, max_steps + 1):
        src, tgt = make_toy_batch(cfg.vocab_size, seq_len=10, batch_size=32, device=device)
        src_mask = make_src_mask(src)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_mask = make_tgt_mask(tgt_in)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, cfg.vocab_size), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  step {step:4d} | loss {loss.item():.4f} | lr {lr:.2e}")

    print("Toy test done. If loss dropped below 1.0, implementation looks correct.")


def _train_full(model, optimizer, scheduler, criterion, train_loader, cfg, device, start_step=0):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    step = start_step
    model.train()
    pbar = tqdm(total=cfg.max_steps - start_step, desc="Training")
    epoch = 0

    while step < cfg.max_steps:
        epoch += 1
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = make_src_mask(src)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_mask = make_tgt_mask(tgt_in)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, cfg.vocab_size), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if step % 1000 == 0:
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"step_{step:06d}.pt")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, ckpt_path)
                print(f"\nSaved checkpoint: {ckpt_path}")

            if step >= cfg.max_steps:
                break

    pbar.close()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="Quick overfit test on random data")
    parser.add_argument("--data", type=str, default="zh-en", help="Language pair (zh-en or en-fr)")
    parser.add_argument("--max_samples", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = Config(
        data_lang_pair=args.data,
        max_samples=args.max_samples,
        max_steps=args.steps,
    )
    train(cfg, toy=args.toy, resume=args.resume)
