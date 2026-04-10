from __future__ import annotations

# This file centralizes all experiment parameters and command-line options for
# the TMN and MLP toy regression runs.

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    # Experiment selection.
    model_type: str = "tmn"
    task_name: str = "sin"
    seed: int = 42

    # TMN structure. The default is the baby case described in the docs.
    L: int = 3
    n_in: int = 1
    n_out: int = 1
    d_model: int = 32
    node_activation: str = "relu"
    output_activation: str = "tanh"

    # Baseline MLP size. This is only used when model_type="mlp".
    mlp_hidden_dim: int = 64
    mlp_num_layers: int = 3

    # Training hyperparameters.
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 300

    # 1D regression dataset setup.
    num_train: int = 512
    num_val: int = 256
    num_plot: int = 512
    x_min: float = -6.283185307179586
    x_max: float = 6.283185307179586

    # Output controls.
    run_name: str = "default"
    save_plots: bool = True
    save_checkpoint: bool = True

    @property
    def run_dir(self) -> Path:
        # Every run writes into its own folder under runs/<model_type>/<run_name>.
        return Path(__file__).resolve().parent / "runs" / self.model_type / self.run_name

    def to_dict(self) -> dict:
        data = asdict(self)
        data["run_dir"] = str(self.run_dir)
        return data

    def save_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate a Triangular Memory Network.")

    # The same training script can run either the TMN or the comparison MLP.
    parser.add_argument("--model-type", choices=["tmn", "mlp"], default="tmn")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--seed", type=int, default=42)

    # TMN graph / hidden size parameters.
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--n-in", type=int, default=1)
    parser.add_argument("--n-out", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--node-activation", choices=["relu"], default="relu")
    parser.add_argument("--output-activation", choices=["tanh"], default="tanh")

    # Baseline-only parameters.
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-num-layers", type=int, default=3)

    # Optimizer / training loop parameters.
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)

    # Toy function fitting parameters.
    parser.add_argument("--task-name", choices=["sin"], default="sin")
    parser.add_argument("--num-train", type=int, default=512)
    parser.add_argument("--num-val", type=int, default=256)
    parser.add_argument("--num-plot", type=int, default=512)
    parser.add_argument("--x-min", type=float, default=-6.283185307179586)
    parser.add_argument("--x-max", type=float, default=6.283185307179586)

    # Artifact toggles.
    parser.add_argument("--no-save-plots", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        model_type=args.model_type,
        task_name=args.task_name,
        seed=args.seed,
        L=args.L,
        n_in=args.n_in,
        n_out=args.n_out,
        d_model=args.d_model,
        node_activation=args.node_activation,
        output_activation=args.output_activation,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_num_layers=args.mlp_num_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_train=args.num_train,
        num_val=args.num_val,
        num_plot=args.num_plot,
        x_min=args.x_min,
        x_max=args.x_max,
        run_name=args.run_name,
        save_plots=not args.no_save_plots,
        save_checkpoint=not args.no_save_checkpoint,
    )
