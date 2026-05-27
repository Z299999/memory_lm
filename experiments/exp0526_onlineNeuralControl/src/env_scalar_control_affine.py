"""Scalar control-affine environment for exp0526."""

from __future__ import annotations

import ast
import math
from typing import Callable

import torch

try:
    from .config import EnvConfig
except ImportError:  # pragma: no cover
    from config import EnvConfig


_UNARY_OPS = {
    ast.UAdd: lambda x: x,
    ast.USub: lambda x: -x,
}

_BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}

_FUNCTIONS = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tanh": torch.tanh,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _scalar_tensor(value: float, *, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(float(value), dtype=like.dtype, device=like.device)


def _compile_scalar_expr(expr: str) -> Callable[[torch.Tensor], torch.Tensor]:
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST, x: torch.Tensor) -> torch.Tensor:
        if isinstance(node, ast.Expression):
            return _eval(node.body, x)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant in expression {expr!r}: {node.value!r}")
            return _scalar_tensor(float(node.value), like=x)
        if isinstance(node, ast.Name):
            if node.id == "x":
                return x
            if node.id in _CONSTANTS:
                return _scalar_tensor(_CONSTANTS[node.id], like=x)
            raise ValueError(f"Unsupported name in expression {expr!r}: {node.id!r}")
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator in expression {expr!r}.")
            return op(_eval(node.operand, x))
        if isinstance(node, ast.BinOp):
            op = _BINARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported binary operator in expression {expr!r}.")
            return op(_eval(node.left, x), _eval(node.right, x))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ValueError(f"Unsupported function in expression {expr!r}.")
            if node.keywords:
                raise ValueError(f"Keyword arguments are not allowed in expression {expr!r}.")
            if len(node.args) != 1:
                raise ValueError(f"Only single-argument functions are allowed in expression {expr!r}.")
            return _FUNCTIONS[node.func.id](_eval(node.args[0], x))
        raise ValueError(f"Unsupported syntax in expression {expr!r}: {ast.dump(node, include_attributes=False)}")

    def _compiled(x: torch.Tensor) -> torch.Tensor:
        result = _eval(tree, x)
        if result.shape != x.shape:
            result = torch.broadcast_to(result, x.shape)
        return result

    return _compiled


class ScalarControlAffineEnv:
    """Differentiable scalar control-affine system x_dot = f(x) + g(x) u."""

    def __init__(
        self,
        env_config: EnvConfig,
        *,
        device: torch.device,
        f_expr: str | None = None,
        g_expr: str | None = None,
    ) -> None:
        self.config = env_config
        self.device = device
        self.state_dim = 1
        self.f_expr = str(env_config.f_expr if f_expr is None else f_expr)
        self.g_expr = str(env_config.g_expr if g_expr is None else g_expr)
        self._f = _compile_scalar_expr(self.f_expr)
        self._g = _compile_scalar_expr(self.g_expr)

    @classmethod
    def from_cubic(cls, env_config: EnvConfig, *, device: torch.device) -> "ScalarControlAffineEnv":
        f_expr = f"{env_config.linear_coeff}*x + {env_config.cubic_coeff}*x**3"
        g_expr = f"{env_config.control_gain}"
        return cls(env_config, device=device, f_expr=f_expr, g_expr=g_expr)

    def initial_state(self) -> torch.Tensor:
        return torch.tensor([[float(self.config.x0)]], dtype=torch.float32, device=self.device)

    def eta(self, state: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]:
        x = state
        derived = {
            "x": float(x.detach().item()),
            "x_sq": float((x.detach() ** 2).item()),
            "abs_x": float(torch.abs(x.detach()).item()),
            "global_step": float(step),
        }
        return x, derived

    def step(self, state: torch.Tensor, u: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]:
        drift = self._f(state)
        control_field = self._g(state)
        raw_next_state = state + self.config.dt * (drift + control_field * u)
        next_state = self.config.state_limit * torch.tanh(raw_next_state / self.config.state_limit)
        next_derived = {
            "x": float(next_state.detach().item()),
            "x_sq": float((next_state.detach() ** 2).item()),
            "abs_x": float(torch.abs(next_state.detach()).item()),
            "global_step": float(step + 1),
        }
        return next_state, next_derived

    def diagnostics(self, state: torch.Tensor, derived: dict[str, float]) -> dict[str, float]:
        payload = dict(derived)
        payload["state_0"] = payload["x"]
        payload["state_norm_sq"] = payload["x_sq"]
        payload["abs_state_max"] = payload["abs_x"]
        return payload

    def equilibrium_points(self) -> list[dict[str, object]]:
        return []
