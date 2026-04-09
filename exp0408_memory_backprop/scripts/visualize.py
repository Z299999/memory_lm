from __future__ import annotations

"""Embedding visualization for the student model.

Usage:
    python3 scripts/visualize.py --model models/student --step 0 --out plots/step_0000
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# ── embedding extraction ───────────────────────────────────────────────────────

def load_embeddings(model_path: str) -> tuple[np.ndarray, list[str]]:
    """
    Load token embeddings from the model's input embedding matrix.
    Returns (embeddings, token_strings) for a curated set of tokens.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[viz] loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, device_map="cpu")
    model.eval()

    # Full embedding matrix: vocab_size × hidden_dim
    E = model.get_input_embeddings().weight.detach().cpu().numpy()
    print(f"[viz] embedding matrix shape: {E.shape}")

    # Curated token list: numbers, operators, math keywords
    probe_tokens = [
        # numbers
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "12", "20", "24", "48", "100",
        # operators / relations
        "+", "-", "×", "÷", "=", "%",
        "plus", "minus", "times", "divided", "equals", "total",
        # math keywords
        "dozen", "twice", "half", "third", "quarter",
        "more", "less", "each", "per", "sum", "product",
        "multiply", "subtract", "add", "divide",
        # generic
        "the", "a", "is", "of", "and", "in",
    ]

    ids, tokens = [], []
    for tok in probe_tokens:
        enc = tokenizer.encode(tok, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
            tokens.append(tok)

    probe_emb = E[ids]  # shape: (n_tokens, hidden_dim)
    return E, probe_emb, tokens


def sample_random_embeddings(E: np.ndarray, n: int = 2000, seed: int = 42) -> np.ndarray:
    # Filter out rows with NaN/Inf (can occur with float16 weights)
    valid = np.isfinite(E).all(axis=1)
    E_clean = E[valid].astype(np.float32)
    rng = np.random.default_rng(seed)
    n = min(n, len(E_clean))
    idx = rng.choice(len(E_clean), size=n, replace=False)
    return E_clean[idx]


# ── individual plots ───────────────────────────────────────────────────────────

def plot_cosine_sim_hist(E_sample: np.ndarray, out: Path, step: int) -> None:
    """Pairwise cosine similarity distribution of random embedding pairs."""
    from sklearn.preprocessing import normalize
    E_norm = normalize(_clean(E_sample))
    E_norm = E_norm[np.isfinite(E_norm).all(axis=1)]
    n = len(E_norm)
    rng = np.random.default_rng(0)
    idx_a = rng.choice(n, 1000)
    idx_b = rng.choice(n, 1000)
    sims = np.sum(E_norm[idx_a] * E_norm[idx_b], axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sims, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(sims.mean(), color="tomato", linestyle="--", label=f"mean={sims.mean():.3f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Pairwise Cosine Similarity Distribution (step {step})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"cosine_sim_hist_{step:04d}.png", dpi=150)
    plt.close(fig)
    print(f"[viz] cosine_sim_hist: mean={sims.mean():.4f}")


def _clean(E: np.ndarray) -> np.ndarray:
    E = E.astype(np.float32)
    return E[np.isfinite(E).all(axis=1)]


def plot_eigenvalue_spectrum(E_sample: np.ndarray, out: Path, step: int) -> None:
    """PCA eigenvalue spectrum + cumulative explained variance."""
    from sklearn.decomposition import PCA

    E = _clean(E_sample)
    E_centered = E - E.mean(axis=0)
    _, S, _ = np.linalg.svd(E_centered, full_matrices=False)
    eigenvalues = (S ** 2) / (len(E) - 1)
    explained_var_ratio = eigenvalues / eigenvalues.sum()
    k = min(100, len(eigenvalues))
    eigenvalues = eigenvalues[:k]
    cum_var = np.cumsum(explained_var_ratio[:k])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(range(1, k + 1), eigenvalues, color="steelblue", linewidth=1.5)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title(f"Eigenvalue Spectrum (step {step})")
    ax1.set_yscale("log")

    ax2.plot(range(1, k + 1), cum_var, color="seagreen", linewidth=1.5)
    ax2.axhline(0.9, color="tomato", linestyle="--", alpha=0.7, label="90%")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title(f"Cumulative Explained Variance (step {step})")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out / f"eigenvalue_spectrum_{step:04d}.png", dpi=150)
    plt.close(fig)

    n90 = int(np.searchsorted(cum_var, 0.9)) + 1
    print(f"[viz] eigenvalue_spectrum: {n90} components explain 90% variance")


def plot_covariance_heatmap(E_sample: np.ndarray, out: Path, step: int, n_dims: int = 64) -> None:
    """Correlation matrix of first n_dims raw embedding dimensions."""
    E = _clean(E_sample)
    k = min(n_dims, E.shape[1])
    corr = np.corrcoef(E[:, :k].T)  # k × k, raw dimensions

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Correlation Matrix of Top-{k} PCA Components (step {step})")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Component index")
    fig.tight_layout()
    fig.savefig(out / f"covariance_heatmap_{step:04d}.png", dpi=150)
    plt.close(fig)
    print(f"[viz] covariance_heatmap: done")


def plot_sphere_density(E_sample: np.ndarray, out: Path, step: int) -> None:
    """2D density of embeddings projected onto first 2 PCA directions (unit-normalized)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    E_norm = normalize(_clean(E_sample))
    E_norm = E_norm[np.isfinite(E_norm).all(axis=1)]
    # Use numpy SVD directly to avoid scipy finite check
    E_centered = E_norm - E_norm.mean(axis=0)
    _, _, Vt = np.linalg.svd(E_centered, full_matrices=False)
    coords = E_centered @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(6, 5))
    h = ax.hist2d(coords[:, 0], coords[:, 1], bins=60, cmap="YlOrRd")
    plt.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Unit-Sphere Density (PCA projection, step {step})")
    fig.tight_layout()
    fig.savefig(out / f"sphere_density_{step:04d}.png", dpi=150)
    plt.close(fig)
    print(f"[viz] sphere_density: done")


def plot_tsne(probe_emb: np.ndarray, tokens: list[str], out: Path, step: int) -> None:
    """t-SNE of curated probe tokens, colored by category."""
    from sklearn.manifold import TSNE

    categories = {
        "number": ["1","2","3","4","5","6","7","8","9","10","12","20","24","48","100"],
        "operator": ["+","-","×","÷","=","%","plus","minus","times","divided","equals","total"],
        "math_kw": ["dozen","twice","half","third","quarter","more","less","each","per",
                    "sum","product","multiply","subtract","add","divide"],
        "generic": ["the","a","is","of","and","in"],
    }
    color_map = {"number": "steelblue", "operator": "tomato", "math_kw": "seagreen", "generic": "gray"}

    token_cat = {}
    for cat, toks in categories.items():
        for t in toks:
            token_cat[t] = cat

    perp = min(30, len(probe_emb) - 1)
    coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(probe_emb)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cat, color in color_map.items():
        mask = [i for i, t in enumerate(tokens) if token_cat.get(t) == cat]
        if mask:
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=cat, s=60, alpha=0.85)
    for i, tok in enumerate(tokens):
        ax.annotate(tok, coords[i], fontsize=7, alpha=0.7)
    ax.legend()
    ax.set_title(f"t-SNE of Probe Tokens (step {step})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(out / f"embedding_tsne_{step:04d}.png", dpi=150)
    plt.close(fig)
    print(f"[viz] tsne: done ({len(tokens)} tokens)")


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/student")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--out", default="plots/step_0000")
    parser.add_argument("--n-sample", type=int, default=2000)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    E_full, probe_emb, tokens = load_embeddings(args.model)
    E_sample = sample_random_embeddings(E_full, n=args.n_sample)

    plot_cosine_sim_hist(E_sample, out, args.step)
    plot_eigenvalue_spectrum(E_sample, out, args.step)
    plot_covariance_heatmap(E_sample, out, args.step)
    plot_sphere_density(E_sample, out, args.step)
    plot_tsne(probe_emb, tokens, out, args.step)

    print(f"\n[viz] all plots saved to {out}/")


if __name__ == "__main__":
    main()
