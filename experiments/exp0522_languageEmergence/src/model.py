"""Model definitions for exp0522."""

from __future__ import annotations

import math

import torch
from torch import nn


def _init_linear_tanh(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight, gain=0.8)
    nn.init.zeros_(linear.bias)


def build_fixed_sparse_signed_readout(
    hidden_dim: int,
    language_dim: int,
    *,
    coverage: int,
    seed: int,
) -> torch.Tensor:
    """Create a fixed sparse signed hidden->message readout matrix."""
    if language_dim <= 0:
        raise ValueError("language_dim must be positive.")
    if coverage <= 0:
        raise ValueError("coverage must be positive.")
    if coverage > language_dim:
        raise ValueError("coverage must be less than or equal to language_dim.")
    if hidden_dim < language_dim and coverage == 1:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be >= language_dim={language_dim} for disjoint readout blocks."
        )

    generator = torch.Generator().manual_seed(seed)
    readout = torch.zeros(hidden_dim, language_dim)

    # Each pass assigns every hidden unit to one language head. Repeating the
    # pass `coverage` times gives each hidden unit multiple distinct readers.
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
            candidate_pairs = list(zip(hidden_order, shuffled_heads))
            if all(readout[h_idx, head_idx] == 0.0 for h_idx, head_idx in candidate_pairs):
                accepted_pairs = candidate_pairs
                break

        if accepted_pairs is None:
            # Fallback: assign each hidden unit to a random unused language head.
            accepted_pairs = []
            for hidden_idx in hidden_order:
                available = torch.nonzero(readout[hidden_idx] == 0.0, as_tuple=False).flatten()
                if available.numel() == 0:
                    raise ValueError(
                        "No unused language head remained for a hidden unit; "
                        "coverage must be <= language_dim."
                    )
                choice = int(available[torch.randint(0, available.numel(), (1,), generator=generator)].item())
                accepted_pairs.append((hidden_idx, choice))

        for hidden_idx, head_idx in accepted_pairs:
            sign = 1.0 if torch.randint(0, 2, (1,), generator=generator).item() == 1 else -1.0
            readout[hidden_idx, head_idx] = sign

    counts = (readout != 0.0).sum(dim=0).to(torch.float32)
    for head_idx in range(language_dim):
        head_count = float(counts[head_idx].item())
        if head_count > 0.0:
            readout[:, head_idx] /= math.sqrt(head_count)
    return readout


class ExternalClockMLP(nn.Module):
    """Feed-forward agent with an optional fixed language channel."""

    def __init__(
        self,
        *,
        trunk_dims: tuple[int, ...],
        language_dim: int,
        language_readout_coverage: int = 1,
        pulse_dim: int = 1,
        use_language: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.pulse_dim = int(pulse_dim)
        self.language_dim = int(language_dim) if use_language else 0
        self.use_language = bool(use_language)
        input_dim = self.pulse_dim + self.language_dim
        dims = [input_dim, *trunk_dims]

        self.trunk = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1], bias=True)
            for idx in range(len(dims) - 1)
        )
        for layer in self.trunk:
            _init_linear_tanh(layer)

        self.output_head = nn.Linear(trunk_dims[-1], 1, bias=True)
        _init_linear_tanh(self.output_head)

        if self.use_language:
            readout = build_fixed_sparse_signed_readout(
                hidden_dim=trunk_dims[-1],
                language_dim=self.language_dim,
                coverage=language_readout_coverage,
                seed=seed,
            )
        else:
            readout = torch.zeros(trunk_dims[-1], 0)
        self.register_buffer("language_readout", readout)

    def _step_hidden(self, step_input: torch.Tensor) -> torch.Tensor:
        hidden = step_input
        for layer in self.trunk:
            hidden = torch.tanh(layer(hidden))
        return hidden

    def rollout(
        self,
        *,
        num_steps: int,
        pulse_value: float,
        disable_language: bool = False,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        device = self.output_head.weight.device
        batch_size = 1
        hidden_snapshots: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        messages: list[torch.Tensor] = []

        if self.use_language:
            message_prev = torch.zeros(batch_size, self.language_dim, device=device)
        else:
            message_prev = torch.zeros(batch_size, 0, device=device)
        pulse = torch.full((batch_size, self.pulse_dim), float(pulse_value), device=device)

        for _ in range(num_steps):
            if self.use_language:
                language_input = torch.zeros_like(message_prev) if disable_language else message_prev
                step_input = torch.cat([pulse, language_input], dim=1)
            else:
                step_input = pulse
            hidden = self._step_hidden(step_input)
            y_t = self.output_head(hidden)
            if self.use_language and not disable_language:
                message_t = hidden @ self.language_readout
            elif self.use_language:
                message_t = torch.zeros_like(message_prev)
            else:
                message_t = torch.zeros(batch_size, 0, device=device)

            outputs.append(y_t)
            messages.append(message_t)
            if return_hidden:
                hidden_snapshots.append(hidden)
            message_prev = message_t

        hidden_tensor = torch.cat(hidden_snapshots, dim=0) if return_hidden else None
        return torch.cat(outputs, dim=0), torch.cat(messages, dim=0), hidden_tensor
