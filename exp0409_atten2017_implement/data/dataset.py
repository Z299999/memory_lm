"""
Data pipeline for opus-100 (zh-en or en-fr).
Uses HuggingFace datasets + tokenizers (BPE).

Usage:
    from data.dataset import build_dataloaders
    train_loader, val_loader, tokenizer = build_dataloaders(cfg)
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def _get_or_train_tokenizer(texts: list[str], vocab_size: int, save_path: str) -> Tokenizer:
    """Train a shared BPE tokenizer on all texts, or load from disk."""
    if os.path.exists(save_path):
        return Tokenizer.from_file(save_path)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", BOS_ID), ("[EOS]", EOS_ID)],
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


class TranslationDataset(Dataset):
    def __init__(self, pairs: list[tuple[list[int], list[int]]], max_len: int = 512):
        self.pairs = [
            (src, tgt)
            for src, tgt in pairs
            if 1 <= len(src) <= max_len and 1 <= len(tgt) <= max_len
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def _collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_padded = pad_sequence(srcs, batch_first=True, padding_value=PAD_ID)
    tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=PAD_ID)
    return src_padded, tgt_padded


def build_dataloaders(cfg, tokenizer_path: str = "runs/tokenizer.json"):
    """
    Returns (train_loader, val_loader, tokenizer).

    cfg.data_lang_pair: "zh-en" or "en-fr"
    cfg.max_samples: how many training pairs to use
    """
    lang_pair = cfg.data_lang_pair
    src_lang, tgt_lang = lang_pair.split("-")

    print(f"Loading opus-100 [{lang_pair}] ...")
    ds = load_dataset("Helsinki-NLP/opus-100", lang_pair, trust_remote_code=True)

    def extract_pairs(split, limit=None):
        rows = ds[split]["translation"]
        if limit:
            rows = rows[:limit]
        return [(r[src_lang], r[tgt_lang]) for r in rows]

    train_pairs_raw = extract_pairs("train", cfg.max_samples)
    val_pairs_raw = extract_pairs("validation")

    # Train a shared BPE tokenizer on all texts
    all_texts = [s for s, t in train_pairs_raw] + [t for s, t in train_pairs_raw]
    tokenizer = _get_or_train_tokenizer(all_texts, cfg.vocab_size, tokenizer_path)

    def encode_pairs(pairs):
        return [
            (
                tokenizer.encode(src).ids,
                tokenizer.encode(tgt).ids,
            )
            for src, tgt in pairs
        ]

    train_dataset = TranslationDataset(encode_pairs(train_pairs_raw), cfg.max_seq_len)
    val_dataset = TranslationDataset(encode_pairs(val_pairs_raw), cfg.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader, tokenizer
