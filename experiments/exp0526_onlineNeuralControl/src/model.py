"""Self-talk controller model for exp0526."""

from __future__ import annotations

import math

import torch
from torch import nn


def _activation_fn(name: str):
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu
    raise ValueError(f"Unsupported activation: {name!r}")


def _init_linear(linear: nn.Linear, activation: str) -> None:
    if activation == "tanh":
        nn.init.xavier_uniform_(linear.weight, gain=0.8)
    elif activation in {"relu", "leaky_relu"}:
        nn.init.kaiming_uniform_(linear.weight, a=0.0, nonlinearity="relu")
    else:
        raise ValueError(f"Unsupported activation: {activation!r}")
    nn.init.zeros_(linear.bias)


def build_fixed_sparse_signed_readout(
    hidden_dim: int,
    language_dim: int,
    *,
    coverage: int,
    seed: int,
) -> torch.Tensor:
    if language_dim <= 0:
        raise ValueError("language_dim must be positive.")
    if coverage <= 0 or coverage > language_dim:
        raise ValueError("coverage must be in [1, language_dim].")

    generator = torch.Generator().manual_seed(seed)
    readout = torch.zeros(hidden_dim, language_dim)
    for _ in range(coverage):
        hidden_order = torch.randperm(hidden_dim, generator=generator).tolist()
        sizes = [hidden_dim // language_dim] * language_dim
        for idx in range(hidden_dim % language_dim):
            sizes[idx] += 1
        head_tokens: list[int] = []
        for head_idx, block_size in enumerate(sizes):
            head_tokens.extend([head_idx] * block_size)

        accepted_pairs: list[tuple[int, int]] | None = None
        for _attempt in range(128):
            token_perm = torch.randperm(len(head_tokens), generator=generator).tolist()
            shuffled_heads = [head_tokens[idx] for idx in token_perm]
            pairs = list(zip(hidden_order, shuffled_heads))
            if all(readout[h_idx, head_idx] == 0.0 for h_idx, head_idx in pairs):
                accepted_pairs = pairs
                break
        if accepted_pairs is None:
            accepted_pairs = []
            for hidden_idx in hidden_order:
                available = torch.nonzero(readout[hidden_idx] == 0.0, as_tuple=False).flatten()
                if available.numel() == 0:
                    raise ValueError("No unused language head remained for a hidden unit.")
                choice = int(available[torch.randint(0, available.numel(), (1,), generator=generator)].item())
                accepted_pairs.append((hidden_idx, choice))

        for hidden_idx, head_idx in accepted_pairs:
            sign = 1.0 if torch.randint(0, 2, (1,), generator=generator).item() else -1.0
            readout[hidden_idx, head_idx] = sign

    counts = (readout != 0.0).sum(dim=0).to(torch.float32)
    for head_idx in range(language_dim):
        count = float(counts[head_idx].item())
        if count > 0.0:
            readout[:, head_idx] /= math.sqrt(count)
    return readout


class SelfTalkController(nn.Module):
    """Feed-forward controller with exp0522-style carried language state."""

    def __init__(
        self,
        *,
        trunk_dims: tuple[int, ...],
        activation: str,
        language_dim: int,
        language_readout_coverage: int,
        use_residual: bool,
        language_readout_all_layers: bool,
        message_carry_mode: str,
        seed: int,
        observation_dim: int = 2,
    ) -> None:
        super().__init__()
        self.activation_name = activation
        self.activation = _activation_fn(activation)
        self.observation_dim = int(observation_dim)
        self.language_dim = int(language_dim)
        self.use_language = self.language_dim > 0
        self.use_residual = bool(use_residual)
        self.language_readout_all_layers = bool(language_readout_all_layers)
        self.message_carry_mode = str(message_carry_mode)

        input_dim = self.observation_dim + self.language_dim
        dims = [input_dim, *trunk_dims]
        self.trunk = nn.ModuleList(nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1))
        for layer in self.trunk:
            _init_linear(layer, activation)
        self.output_head = nn.Linear(trunk_dims[-1], 1)
        _init_linear(self.output_head, activation)

        if self.use_language and self.message_carry_mode == "learnable_diagonal":
            self.message_carry_d = nn.Parameter(torch.zeros(self.language_dim))
        else:
            self.message_carry_d = None
        if self.use_language and self.message_carry_mode == "learnable_matrix":
            self.message_carry_D = nn.Parameter(torch.zeros(self.language_dim, self.language_dim))
        else:
            self.message_carry_D = None

        readout_dim = sum(trunk_dims) if self.language_readout_all_layers else trunk_dims[-1]
        if self.use_language:
            readout = build_fixed_sparse_signed_readout(
                readout_dim,
                self.language_dim,
                coverage=language_readout_coverage,
                seed=seed,
            )
        else:
            readout = torch.zeros(readout_dim, 0)
        self.register_buffer("language_readout", readout)

    def initial_message(self, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, self.language_dim, device=device)

    def _step_hidden(self, step_input: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = step_input
        all_hiddens: list[torch.Tensor] = []
        for layer in self.trunk:
            new_hidden = self.activation(layer(hidden))
            if self.use_residual and hidden.shape[-1] == new_hidden.shape[-1]:
                new_hidden = new_hidden + hidden
            hidden = new_hidden
            all_hiddens.append(hidden)
        return hidden, all_hiddens

    def forward_step(
        self,
        observation: torch.Tensor,
        *,
        message_prev: torch.Tensor | None,
        disable_language: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.shape != (1, self.observation_dim):
            raise ValueError(f"observation must have shape {(1, self.observation_dim)}, got {tuple(observation.shape)}.")
        device = observation.device
        parts = [observation]

        if self.use_language:
            if message_prev is None:
                message_prev = self.initial_message(device=device)
            if message_prev.shape != (1, self.language_dim):
                raise ValueError(f"message_prev must have shape {(1, self.language_dim)}, got {tuple(message_prev.shape)}.")
            if disable_language:
                language_input = torch.zeros_like(message_prev)
            elif self.message_carry_mode == "learnable_diagonal":
                language_input = (1.0 + self.message_carry_d) * message_prev
            elif self.message_carry_mode == "learnable_matrix":
                language_input = message_prev + message_prev @ self.message_carry_D
            else:
                language_input = message_prev
            parts.append(language_input)

        step_input = torch.cat(parts, dim=1)
        hidden, all_hiddens = self._step_hidden(step_input)
        raw_u = self.output_head(hidden)
        if self.use_language and not disable_language:
            readout_input = torch.cat(all_hiddens, dim=-1) if self.language_readout_all_layers else hidden
            message = readout_input @ self.language_readout
        elif self.use_language:
            message = torch.zeros(1, self.language_dim, device=device)
        else:
            message = torch.zeros(1, 0, device=device)
        return raw_u, message, hidden
