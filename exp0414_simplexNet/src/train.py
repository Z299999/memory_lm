"""Training loop for simplex memory networks.

This module provides the training loop with support for:
- Training and validation loss tracking
- Best model checkpointing
- Parameter tracing via trace_fn callback
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config


def build_dataset(config: Config):
    """Build dataset for the specified task.

    Args:
        config: Configuration object

    Returns:
        Dataset object with x_train, y_train, x_val, y_val, x_plot, y_plot
    """
    from data import build_dataset as _build
    return _build(config)


def build_model(config: Config):
    """Build model based on model_type.

    Args:
        config: Configuration object

    Returns:
        PyTorch nn.Module
    """
    if config.model_type == "smn":
        from model import SMNNetwork
        model = SMNNetwork(config)
        model.set_compiled(True)  # Enable compiled forward for faster training
        return model
    elif config.model_type == "mlp":
        from mlp import MLPBaseline
        return MLPBaseline(config)
    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")


def evaluate_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module
) -> float:
    """Evaluate model on given data.

    Args:
        model: PyTorch model
        x: Input tensor
        y: Target tensor
        criterion: Loss function

    Returns:
        Loss value
    """
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred, y)
    return float(loss.item())


def train_with_config(config: Config, trace_fn=None):
    """Train model with given configuration.

    Args:
        config: Configuration object
        trace_fn: Optional callback(trace_fn(model)) -> dict for tracking parameters

    Returns:
        Dictionary with training results:
        - metrics: final_train_loss, final_val_loss, model_type
        - model: trained model
        - train_losses: list of training losses
        - val_losses: list of validation losses
        - traced_params: dict of traced parameters (if trace_fn provided)
        - x_plot, y_plot, y_pred: for visualization
        - x1_grid, x2_grid: for 2D tasks
    """
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Build dataset and model
    dataset = build_dataset(config)
    train_loader = DataLoader(
        TensorDataset(dataset.x_train, dataset.y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = build_model(config)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Training loop
    train_losses = []
    val_losses = []
    traced_params = None
    best_val_loss = math.inf
    best_state_dict = None

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate_model(model, dataset.x_val, dataset.y_val, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Trace parameters if callback provided
        if trace_fn is not None:
            snapshot = trace_fn(model)
            if traced_params is None:
                traced_params = {k: [v] for k, v in snapshot.items()}
            else:
                for k, v in snapshot.items():
                    traced_params[k].append(v)

        # Keep best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"epoch={epoch + 1:04d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Final evaluation
    final_train_loss = evaluate_model(model, dataset.x_train, dataset.y_train, criterion)
    final_val_loss = evaluate_model(model, dataset.x_val, dataset.y_val, criterion)
    print(f"final_train_loss={final_train_loss:.6f}")
    print(f"final_val_loss={final_val_loss:.6f}")

    # Generate predictions for plotting
    model.eval()
    with torch.no_grad():
        y_pred = model(dataset.x_plot).cpu().numpy()

    metrics = {
        "model_type": config.model_type,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }

    # Handle 1D vs 2D input
    x_plot_np = dataset.x_plot.cpu().numpy()
    y_plot_np = dataset.y_plot.cpu().numpy()
    y_pred_np = y_pred

    if config.n_in == 2:
        # 2D input: return grid info
        n_side = int(math.sqrt(len(x_plot_np)))
        x1_grid = x_plot_np[:, 0].reshape(n_side, n_side)
        x2_grid = x_plot_np[:, 1].reshape(n_side, n_side)
        return {
            "metrics": metrics,
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "traced_params": traced_params,
            "x_plot": x_plot_np,
            "y_plot": y_plot_np,
            "y_pred": y_pred_np,
            "x1_grid": x1_grid,
            "x2_grid": x2_grid,
        }
    else:
        # 1D input: squeeze x but not y
        return {
            "metrics": metrics,
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "traced_params": traced_params,
            "x_plot": x_plot_np.squeeze(-1),
            "y_plot": y_plot_np,
            "y_pred": y_pred_np,
        }
