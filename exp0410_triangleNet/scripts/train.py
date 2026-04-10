from __future__ import annotations

# This file is the main training entry point. It builds the selected model,
# trains it on the toy regression task, and returns the result to run.py.

import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import build_dataset
from model import MLPBaseline, TMNNetwork


def build_model(config):
    if config.model_type == "tmn":
        return TMNNetwork(config)
    if config.model_type == "mlp":
        return MLPBaseline(config)
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred, y)
    return float(loss.item())


def train_with_config(config, trace_fn=None):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset = build_dataset(config)
    # The toy task is supervised regression on sampled (x, y) pairs.
    train_loader = DataLoader(
        TensorDataset(dataset.x_train, dataset.y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = build_model(config)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

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

        if trace_fn is not None:
            snapshot = trace_fn(model)
            if traced_params is None:
                traced_params = {k: [v] for k, v in snapshot.items()}
            else:
                for k, v in snapshot.items():
                    traced_params[k].append(v)

        if val_loss < best_val_loss:
            # Keep the best validation model rather than the last epoch.
            best_val_loss = val_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"epoch={epoch + 1:04d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_train_loss = evaluate_model(model, dataset.x_train, dataset.y_train, criterion)
    final_val_loss = evaluate_model(model, dataset.x_val, dataset.y_val, criterion)
    print(f"final_train_loss={final_train_loss:.6f}")
    print(f"final_val_loss={final_val_loss:.6f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(dataset.x_plot).cpu().numpy()

    metrics = {
        "model_type": config.model_type,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }

    return {
        "metrics": metrics,
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "traced_params": traced_params,
        "x_plot": dataset.x_plot.squeeze(-1).cpu().numpy(),
        "y_plot": dataset.y_plot.squeeze(-1).cpu().numpy(),
        "y_pred": y_pred.squeeze(-1),
    }
