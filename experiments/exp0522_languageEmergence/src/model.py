"""Model definitions for exp0522."""

from __future__ import annotations

import math

import torch
from torch import nn


def _init_linear_tanh(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight, gain=0.8)
    nn.init.zeros_(linear.bias)


def _init_linear_relu(linear: nn.Linear) -> None:
    nn.init.kaiming_uniform_(linear.weight, a=0.0, nonlinearity="relu")
    nn.init.zeros_(linear.bias)


def _activation_fn(name: str) -> callable:
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu
    raise ValueError(f"Unsupported activation: {name!r}")


def _init_linear(linear: nn.Linear, activation: str) -> None:
    if activation == "tanh":
        _init_linear_tanh(linear)
    elif activation in {"relu", "leaky_relu"}:
        _init_linear_relu(linear)
    else:
        raise ValueError(f"Unsupported activation: {activation!r}")


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


class AgentPool(nn.Module):
    """N-agent reservoir: independent trunks, shared readout head, inter-agent message matrix."""

    def __init__(
        self,
        *,
        num_agents: int,
        trunk_dims: tuple[int, ...],
        activation: str = "tanh",
        language_dim: int,
        language_readout_coverage: int = 1,
        pulse_dim: int = 1,
        use_error_input: bool = False,
        use_residual: bool = True,
        language_readout_all_layers: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.activation_name = str(activation)
        self.activation = _activation_fn(self.activation_name)
        self.pulse_dim = int(pulse_dim)
        self.error_dim = 1 if use_error_input else 0
        self.use_error_input = bool(use_error_input)
        self.language_dim = int(language_dim)
        self.use_language = self.language_dim > 0
        self.use_residual = bool(use_residual)
        self.language_readout_all_layers = bool(language_readout_all_layers)

        input_dim = self.pulse_dim + self.error_dim + self.language_dim
        dims = [input_dim, *trunk_dims]

        # Stacked weights: trunk_Ws[k] is (N, D_out, D_in) for layer k
        # Each agent's weights are initialized independently then stacked.
        trunk_Ws: list[nn.Parameter] = []
        trunk_bs: list[nn.Parameter] = []
        for k in range(len(dims) - 1):
            in_dim, out_dim = dims[k], dims[k + 1]
            Ws, bs = [], []
            for _ in range(self.num_agents):
                layer = nn.Linear(in_dim, out_dim, bias=True)
                _init_linear(layer, self.activation_name)
                Ws.append(layer.weight.data)   # (out_dim, in_dim)
                bs.append(layer.bias.data)     # (out_dim,)
            trunk_Ws.append(nn.Parameter(torch.stack(Ws)))  # (N, out_dim, in_dim)
            trunk_bs.append(nn.Parameter(torch.stack(bs)))  # (N, out_dim)
        self.trunk_Ws = nn.ParameterList(trunk_Ws)
        self.trunk_bs = nn.ParameterList(trunk_bs)

        # Shared reservoir readout: all agents' last hidden → scalar
        self.readout_head = nn.Linear(self.num_agents * trunk_dims[-1], 1, bias=True)
        _init_linear(self.readout_head, self.activation_name)

        # Error distribution: w_in[i] scales the shared error for agent i
        if self.use_error_input:
            self.w_in = nn.Parameter(torch.ones(self.num_agents))
        else:
            self.register_parameter("w_in", None)

        # Inter-agent communication D[i,j]: how agent i reads from agent j
        # D[i,i] = I at init (self-carry), D[i,j≠i] = 0 (no initial inter-agent signal)
        if self.use_language:
            # Residual parameterization: lang_input_i = activation(m_i + Σ_j DeltaD[i,j] @ m_j)
            # DeltaD initialized to 0 so training starts with pure identity self-carry
            self.DeltaD = nn.Parameter(
                torch.zeros(self.num_agents, self.num_agents, self.language_dim, self.language_dim)
            )
        else:
            self.register_parameter("DeltaD", None)

        # Fixed sparse readout per agent for language messages
        readout_input_dim = sum(trunk_dims) if self.language_readout_all_layers else trunk_dims[-1]
        if self.use_language:
            readouts = torch.stack([
                build_fixed_sparse_signed_readout(
                    hidden_dim=readout_input_dim,
                    language_dim=self.language_dim,
                    coverage=language_readout_coverage,
                    seed=seed + k,
                )
                for k in range(self.num_agents)
            ])
        else:
            readouts = torch.zeros(self.num_agents, readout_input_dim, 0)
        self.register_buffer("language_readouts", readouts)

    def _step_hidden_all(self, step_inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Batched trunk forward for all agents. step_inputs: (N, D_in) → last_hidden: (N, D_last)."""
        hidden = step_inputs
        all_hiddens: list[torch.Tensor] = []
        for W, b in zip(self.trunk_Ws, self.trunk_bs):
            # W: (N, D_out, D_in), b: (N, D_out), hidden: (N, D_in)
            new_hidden = self.activation(
                torch.bmm(W, hidden.unsqueeze(-1)).squeeze(-1) + b
            )
            if self.use_residual and hidden.shape[-1] == new_hidden.shape[-1]:
                new_hidden = new_hidden + hidden
            hidden = new_hidden
            all_hiddens.append(hidden)
        return hidden, all_hiddens

    def rollout(
        self,
        *,
        num_steps: int,
        pulse_value: float,
        target_sequence: torch.Tensor | None = None,
        y_target_sequence: torch.Tensor | None = None,
        initial_message: torch.Tensor | None = None,
        initial_error: torch.Tensor | None = None,
        detach_error_input: bool = True,
        force_zero_error_input: bool = False,
        disable_language: bool = False,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        device = self.readout_head.weight.device
        N = self.num_agents

        # Message state: (N, language_dim)
        if self.use_language:
            if initial_message is None:
                msg_prev = torch.zeros(N, self.language_dim, device=device)
            else:
                msg_prev = initial_message.to(device=device).reshape(N, self.language_dim)
        else:
            msg_prev = torch.zeros(N, 0, device=device)

        # Shared error state: (1, error_dim)
        if self.use_error_input:
            if initial_error is None:
                err_prev = torch.zeros(1, self.error_dim, device=device)
            else:
                err_prev = initial_error.to(device=device).reshape(1, self.error_dim)
        else:
            err_prev = torch.zeros(1, 0, device=device)

        if y_target_sequence is None:
            y_target_sequence = target_sequence
        if target_sequence is not None and tuple(target_sequence.shape) != (num_steps, 1):
            raise ValueError(f"target_sequence must have shape {(num_steps, 1)}, got {tuple(target_sequence.shape)}.")
        if self.use_error_input:
            if y_target_sequence is None:
                raise ValueError("y_target_sequence is required when use_error_input=True.")
            y_target_sequence = y_target_sequence.to(device=device)

        pulse = torch.full((1, self.pulse_dim), float(pulse_value), device=device)
        outputs: list[torch.Tensor] = []
        all_msg_steps: list[torch.Tensor] = []
        agent_hidden_steps: list[torch.Tensor] = []  # (N, D_last) per step

        for step in range(num_steps):
            # Language aggregation: language_input_i = activation(Σ_j D[i,j] @ m_j)
            # msg_prev: (N, d), D: (N_i, N_j, d, d)
            # lang_agg[i,e] = Σ_j Σ_d msg_prev[j,d] * D[i,j,e,d]
            if self.use_language and not disable_language:
                # (I + ΔD) @ m: identity self-carry + learned residual from all agents
                lang_agg = msg_prev + torch.einsum("jd,ijed->ie", msg_prev, self.DeltaD)
                lang_inputs = self.activation(lang_agg)  # (N, language_dim)
            else:
                lang_inputs = torch.zeros(N, self.language_dim, device=device)

            # Build all agents' inputs in one tensor (N, D_in) — no Python loop
            pulse_batch = pulse.expand(N, -1)  # (N, pulse_dim)
            parts = [pulse_batch]
            if self.use_error_input:
                if force_zero_error_input:
                    e_in = torch.zeros(N, self.error_dim, device=device)
                else:
                    e_in = self.w_in.unsqueeze(1) * err_prev  # (N, 1)
                parts.append(e_in)
            if self.use_language:
                parts.append(lang_inputs)  # (N, language_dim)
            step_inputs = torch.cat(parts, dim=1)  # (N, D_in)

            # Single batched trunk forward
            hidden_last, all_hiddens_per_layer = self._step_hidden_all(step_inputs)
            # hidden_last: (N, trunk_dims[-1])

            # Reservoir readout over all agents
            y_t = self.readout_head(hidden_last.reshape(1, -1))  # (1, 1)
            outputs.append(y_t)
            if return_hidden:
                agent_hidden_steps.append(hidden_last.detach())

            # Language messages via batched readout
            if self.use_language and not disable_language:
                readout_input = (
                    torch.cat(all_hiddens_per_layer, dim=-1)  # (N, sum(trunk_dims))
                    if self.language_readout_all_layers
                    else hidden_last  # (N, trunk_dims[-1])
                )
                # language_readouts: (N, readout_input_dim, d)
                msg_prev = torch.bmm(
                    readout_input.unsqueeze(1), self.language_readouts
                ).squeeze(1)  # (N, d)
            else:
                msg_prev = torch.zeros(N, self.language_dim, device=device)
            all_msg_steps.append(msg_prev.clone())

            # Error update (shared scalar)
            if self.use_error_input:
                e_t = y_target_sequence[step : step + 1] - y_t  # (1, 1)
                err_prev = e_t.detach() if detach_error_input else e_t

        outputs_tensor = torch.cat(outputs, dim=0)  # (num_steps, 1)
        all_msgs_tensor = torch.stack(all_msg_steps, dim=0).flatten(1, 2)  # (T, N*d)
        # hidden: (T, N, D_last) when return_hidden=True, else None
        hidden_tensor = torch.stack(agent_hidden_steps, dim=0) if agent_hidden_steps else None

        # final_message: (N, language_dim) — differs from single-agent (1, language_dim)
        # final_error: (1, error_dim) — same shape as single-agent
        return (
            outputs_tensor,
            outputs_tensor,  # raw == output (single readout head)
            all_msgs_tensor,
            hidden_tensor,   # (T, N, D_last) or None
            msg_prev,        # (N, language_dim)
            err_prev,   # (1, error_dim)
        )


class ExternalClockMLP(nn.Module):
    """Feed-forward agent with an optional fixed language channel."""

    def __init__(
        self,
        *,
        trunk_dims: tuple[int, ...],
        activation: str = "tanh",
        language_dim: int,
        language_readout_coverage: int = 1,
        pulse_dim: int = 1,
        use_error_input: bool = False,
        use_language: bool = True,
        use_residual: bool = True,
        language_readout_all_layers: bool = False,
        message_carry_mode: str = "identity",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.activation_name = str(activation)
        self.activation = _activation_fn(self.activation_name)
        self.pulse_dim = int(pulse_dim)
        self.error_dim = 1 if use_error_input else 0
        self.use_error_input = bool(use_error_input)
        self.language_dim = int(language_dim) if use_language else 0
        self.use_language = bool(use_language) and self.language_dim > 0
        self.use_residual = bool(use_residual)
        self.language_readout_all_layers = bool(language_readout_all_layers)
        self.message_carry_mode = str(message_carry_mode)
        input_dim = self.pulse_dim + self.error_dim + self.language_dim
        dims = [input_dim, *trunk_dims]

        self.trunk = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1], bias=True)
            for idx in range(len(dims) - 1)
        )
        for layer in self.trunk:
            _init_linear(layer, self.activation_name)

        self.output_head = nn.Linear(trunk_dims[-1], 1, bias=True)
        _init_linear(self.output_head, self.activation_name)

        if self.use_language and self.message_carry_mode == "learnable_diagonal":
            self.message_carry_d = nn.Parameter(torch.zeros(self.language_dim))
        else:
            self.message_carry_d = None
        if self.use_language and self.message_carry_mode == "learnable_matrix":
            self.message_carry_D = nn.Parameter(torch.zeros(self.language_dim, self.language_dim))
        else:
            self.message_carry_D = None

        readout_input_dim = sum(trunk_dims) if self.language_readout_all_layers else trunk_dims[-1]
        if self.use_language:
            readout = build_fixed_sparse_signed_readout(
                hidden_dim=readout_input_dim,
                language_dim=self.language_dim,
                coverage=language_readout_coverage,
                seed=seed,
            )
        else:
            readout = torch.zeros(readout_input_dim, 0)
        self.register_buffer("language_readout", readout)

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

    def rollout(
        self,
        *,
        num_steps: int,
        pulse_value: float,
        target_sequence: torch.Tensor | None = None,
        y_target_sequence: torch.Tensor | None = None,
        initial_message: torch.Tensor | None = None,
        initial_error: torch.Tensor | None = None,
        detach_error_input: bool = True,
        force_zero_error_input: bool = False,
        disable_language: bool = False,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        device = self.output_head.weight.device
        batch_size = 1
        hidden_snapshots: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        raw_outputs: list[torch.Tensor] = []
        messages: list[torch.Tensor] = []

        if self.use_language:
            if initial_message is None:
                message_prev = torch.zeros(batch_size, self.language_dim, device=device)
            else:
                if tuple(initial_message.shape) != (batch_size, self.language_dim):
                    raise ValueError(
                        f"initial_message must have shape {(batch_size, self.language_dim)}, "
                        f"got {tuple(initial_message.shape)}."
                    )
                message_prev = initial_message.to(device=device)
        else:
            message_prev = torch.zeros(batch_size, 0, device=device)
        pulse = torch.full((batch_size, self.pulse_dim), float(pulse_value), device=device)
        if target_sequence is not None and tuple(target_sequence.shape) != (num_steps, 1):
            raise ValueError(
                f"target_sequence must have shape {(num_steps, 1)}, got {tuple(target_sequence.shape)}."
            )
        if y_target_sequence is None:
            y_target_sequence = target_sequence
        if y_target_sequence is not None and tuple(y_target_sequence.shape) != (num_steps, 1):
            raise ValueError(
                f"y_target_sequence must have shape {(num_steps, 1)}, got {tuple(y_target_sequence.shape)}."
            )
        if self.use_error_input:
            if y_target_sequence is None:
                raise ValueError("y_target_sequence is required when use_error_input=True.")
            if tuple(y_target_sequence.shape) != (num_steps, 1):
                raise ValueError(
                    f"y_target_sequence must have shape {(num_steps, 1)}, got {tuple(y_target_sequence.shape)}."
                )
            if initial_error is None:
                error_prev = torch.zeros(batch_size, self.error_dim, device=device)
            else:
                if tuple(initial_error.shape) != (batch_size, self.error_dim):
                    raise ValueError(
                        f"initial_error must have shape {(batch_size, self.error_dim)}, "
                        f"got {tuple(initial_error.shape)}."
                    )
                error_prev = initial_error.to(device=device)
            y_target_sequence = y_target_sequence.to(device=device)
        else:
            error_prev = torch.zeros(batch_size, 0, device=device)

        for step_idx in range(num_steps):
            input_parts = [pulse]
            if self.use_error_input:
                input_parts.append(error_prev)
            if self.use_language:
                if disable_language:
                    language_input = torch.zeros_like(message_prev)
                elif self.message_carry_mode == "learnable_diagonal":
                    language_input = (1.0 + self.message_carry_d) * message_prev
                elif self.message_carry_mode == "learnable_matrix":
                    language_input = message_prev + message_prev @ self.message_carry_D
                else:
                    language_input = message_prev
                input_parts.append(language_input)
            step_input = torch.cat(input_parts, dim=1)
            hidden, all_hiddens = self._step_hidden(step_input)
            raw_t = self.output_head(hidden)
            y_t = raw_t
            if self.use_language and not disable_language:
                readout_input = torch.cat(all_hiddens, dim=-1) if self.language_readout_all_layers else hidden
                message_t = readout_input @ self.language_readout
            elif self.use_language:
                message_t = torch.zeros_like(message_prev)
            else:
                message_t = torch.zeros(batch_size, 0, device=device)

            outputs.append(y_t)
            raw_outputs.append(raw_t)
            messages.append(message_t)
            if return_hidden:
                hidden_snapshots.append(hidden)
            message_prev = message_t
            if self.use_error_input:
                if force_zero_error_input:
                    error_prev = torch.zeros_like(error_prev)
                else:
                    error_t = y_target_sequence[step_idx : step_idx + 1] - y_t
                    error_prev = error_t.detach() if detach_error_input else error_t

        hidden_tensor = torch.cat(hidden_snapshots, dim=0) if return_hidden else None
        final_message = message_prev
        final_error = error_prev
        return (
            torch.cat(outputs, dim=0),
            torch.cat(raw_outputs, dim=0),
            torch.cat(messages, dim=0),
            hidden_tensor,
            final_message,
            final_error,
        )
