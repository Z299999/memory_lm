"""Verify the theory hand-worked example for exp0513."""

from __future__ import annotations

import json
from pathlib import Path

import torch


EXPECTED_BP = torch.tensor([0.00625, 0.0125, -0.00625, -0.0125, 0.003125, 0.00625], dtype=torch.float32)
EXPECTED_INT = torch.tensor([0.00432, -0.00115, 0.00333, 0.00333, 0.00669, 0.00227], dtype=torch.float32)
EXPECTED_MIX = torch.tensor([0.00586, 0.00977, -0.00433, -0.00933, 0.00384, 0.00545], dtype=torch.float32)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "runs" / "hand_example"
    output_dir.mkdir(parents=True, exist_ok=True)

    B = torch.tensor([
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=torch.float32)

    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y_star = torch.tensor(1.0, dtype=torch.float32)
    w = torch.tensor([0.1, 0.2, -0.1, 0.1, 0.05, -0.05], dtype=torch.float32)
    v = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)
    q = torch.tensor([0.6, -0.2, 0.8], dtype=torch.float32)

    h1 = w[0] * x[0] + w[1] * x[1]
    h2 = w[2] * x[0] + w[3] * x[1]
    h3 = w[4] * x[0] + w[5] * x[1]
    y = h1 - h2 + 0.5 * h3
    delta = y - y_star

    g_bp = torch.tensor([
        delta * v[0] * x[0],
        delta * v[0] * x[1],
        delta * v[1] * x[0],
        delta * v[1] * x[1],
        delta * v[2] * x[0],
        delta * v[2] * x[1],
    ], dtype=torch.float32)
    eta_bp = 0.01
    delta_bp = -eta_bp * g_bp

    k = B.sum(dim=0)
    B_tilde = B / torch.sqrt(k).unsqueeze(0)
    s = B_tilde.matmul(q)
    eta_int = 0.01
    gamma = 1.0
    delta_int = eta_int * torch.tanh(gamma * s)

    lambda_value = 0.2
    delta_mix = (1.0 - lambda_value) * delta_bp + lambda_value * delta_int

    summary = {
        "y": float(y.item()),
        "delta": float(delta.item()),
        "delta_bp": [float(v) for v in delta_bp.tolist()],
        "delta_int": [float(v) for v in delta_int.tolist()],
        "delta_mix": [float(v) for v in delta_mix.tolist()],
        "bp_max_abs_err": float((delta_bp - EXPECTED_BP).abs().max().item()),
        "int_max_abs_err": float((delta_int - EXPECTED_INT).abs().max().item()),
        "mix_max_abs_err": float((delta_mix - EXPECTED_MIX).abs().max().item()),
    }
    (output_dir / "hand_example_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
