"""
Evaluation: BLEU score + greedy/beam search decoding.

Usage:
    # Greedy decode a single sentence (interactive)
    python evaluate.py --checkpoint runs/checkpoints/step_10000.pt --interactive

    # Compute BLEU on validation set
    python evaluate.py --checkpoint runs/checkpoints/step_10000.pt --bleu

    # Visualise attention weights for one sample
    python evaluate.py --checkpoint runs/checkpoints/step_10000.pt --attn_vis
"""
import argparse
import math
import torch
import sacrebleu
from tokenizers import Tokenizer

from config import Config
from model.transformer import Transformer, make_src_mask, make_tgt_mask

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


# ---------------------------------------------------------------------------
# Greedy decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,          # (1, T_src)
    src_mask: torch.Tensor,
    max_len: int,
    device,
) -> list[int]:
    enc_out = model.encode(src, src_mask)
    tgt = torch.tensor([[BOS_ID]], device=device)
    for _ in range(max_len):
        tgt_mask = make_tgt_mask(tgt)
        logits = model.decode(tgt, enc_out, src_mask, tgt_mask)
        next_token = logits[:, -1, :].argmax(dim=-1)    # (1,)
        tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == EOS_ID:
            break
    return tgt[0, 1:].tolist()   # strip BOS


# ---------------------------------------------------------------------------
# Beam search decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_decode(
    model: Transformer,
    src: torch.Tensor,   # (1, T_src)
    src_mask: torch.Tensor,
    max_len: int,
    beam_size: int,
    device,
) -> list[int]:
    enc_out = model.encode(src, src_mask)   # (1, T_src, d_model)

    # Each beam: (score, token_ids)
    beams = [(0.0, [BOS_ID])]
    completed = []

    for _ in range(max_len):
        candidates = []
        for score, tokens in beams:
            if tokens[-1] == EOS_ID:
                completed.append((score, tokens))
                continue
            tgt = torch.tensor([tokens], device=device)
            tgt_mask = make_tgt_mask(tgt)
            logits = model.decode(tgt, enc_out, src_mask, tgt_mask)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)   # (V,)
            topk_probs, topk_ids = log_probs.topk(beam_size)
            for lp, tok in zip(topk_probs.tolist(), topk_ids.tolist()):
                candidates.append((score + lp, tokens + [tok]))

        if not candidates:
            break
        # Keep top beam_size beams
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

    completed += beams
    best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
    ids = best[1]
    # Strip BOS/EOS
    if ids and ids[0] == BOS_ID:
        ids = ids[1:]
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    return ids


# ---------------------------------------------------------------------------
# BLEU evaluation on validation set
# ---------------------------------------------------------------------------

def evaluate_bleu(model, val_loader, tokenizer: Tokenizer, device, beam_size: int = 4):
    model.eval()
    hypotheses, references = [], []
    for src_batch, tgt_batch in val_loader:
        for i in range(src_batch.size(0)):
            src = src_batch[i].unsqueeze(0).to(device)
            src_mask = make_src_mask(src)
            ids = beam_decode(model, src, src_mask, max_len=100, beam_size=beam_size, device=device)
            hyp = tokenizer.decode(ids)
            ref_ids = tgt_batch[i].tolist()
            ref_ids = [t for t in ref_ids if t not in (PAD_ID, BOS_ID, EOS_ID)]
            ref = tokenizer.decode(ref_ids)
            hypotheses.append(hyp)
            references.append([ref])

    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print(f"BLEU: {bleu.score:.2f}")
    return bleu.score


# ---------------------------------------------------------------------------
# Attention visualisation
# ---------------------------------------------------------------------------

def visualise_attention(model, src_ids, tokenizer: Tokenizer, device):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping visualisation.")
        return

    src = torch.tensor([src_ids], device=device)
    src_mask = make_src_mask(src)
    enc_out = model.encode(src, src_mask)

    # Patch first encoder self-attention to capture weights
    attn_weights = {}
    original_forward = model.encoder.layers[0].self_attn.attention.forward

    def hook(q, k, v, mask=None):
        out, attn = original_forward(q, k, v, mask)
        attn_weights["enc_layer0"] = attn.detach().cpu()
        return out, attn

    model.encoder.layers[0].self_attn.attention.forward = hook
    with torch.no_grad():
        model.encode(src, src_mask)
    model.encoder.layers[0].self_attn.attention.forward = original_forward

    if "enc_layer0" not in attn_weights:
        print("Could not capture attention weights.")
        return

    tokens = [tokenizer.id_to_token(i) for i in src_ids]
    weights = attn_weights["enc_layer0"][0, 0].numpy()   # head 0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, cmap="Blues")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_title("Encoder Self-Attention (Layer 0, Head 0)")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("runs/attn_vis.png")
    print("Saved: runs/attn_vis.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="runs/tokenizer.json")
    parser.add_argument("--bleu", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--attn_vis", action="store_true")
    parser.add_argument("--beam_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    model = Transformer(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = Tokenizer.from_file(args.tokenizer)

    if args.interactive:
        while True:
            text = input("Input (empty to quit): ").strip()
            if not text:
                break
            src_ids = tokenizer.encode(text).ids
            src = torch.tensor([src_ids], device=device)
            src_mask = make_src_mask(src)
            out_ids = beam_decode(model, src, src_mask, max_len=100,
                                  beam_size=args.beam_size, device=device)
            print("Output:", tokenizer.decode(out_ids))

    if args.bleu:
        from data.dataset import build_dataloaders
        _, val_loader, tokenizer = build_dataloaders(cfg)
        evaluate_bleu(model, val_loader, tokenizer, device, args.beam_size)

    if args.attn_vis:
        text = input("Input sentence for attention vis: ").strip()
        src_ids = tokenizer.encode(text).ids
        visualise_attention(model, src_ids, tokenizer, device)
