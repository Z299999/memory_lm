from __future__ import annotations

# This file loads a saved checkpoint, rebuilds the matching model, and writes a
# small evaluation report plus a prediction plot for the toy regression task.

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from config import Config
from data import build_dataset
from model import MLPBaseline, TMNNetwork
from utils import ensure_dir, save_prediction_plot


def build_model_from_config(config_dict: dict):
    filtered = {key: value for key, value in config_dict.items() if key in Config.__dataclass_fields__}
    config = Config(**filtered)
    if config.model_type == "tmn":
        return config, TMNNetwork(config)
    if config.model_type == "mlp":
        return config, MLPBaseline(config)
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved TMN or MLP checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config, model = build_model_from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = build_dataset(config)
    criterion = nn.MSELoss()
    with torch.no_grad():
        pred = model(dataset.x_val)
        val_loss = float(criterion(pred, dataset.y_val).item())
        plot_pred = model(dataset.x_plot).cpu().numpy()

    output_dir = checkpoint_path.parent / "evaluation"
    ensure_dir(output_dir)
    save_prediction_plot(
        dataset.x_plot.squeeze(-1).cpu().numpy(),
        dataset.y_plot.squeeze(-1).cpu().numpy(),
        plot_pred.squeeze(-1),
        output_dir / "prediction_eval.png",
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "val_loss": val_loss,
        "model_type": config.model_type,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
