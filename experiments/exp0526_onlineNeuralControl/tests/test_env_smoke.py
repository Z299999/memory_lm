from pathlib import Path
import sys
import unittest

import torch


EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from src.config import load_config  # noqa: E402
from src.env import build_env  # noqa: E402
from src.model import SelfTalkController  # noqa: E402


def _config():
    return load_config(EXP_DIR / "config.yaml")


class Exp0526EnvSmokeTest(unittest.TestCase):
    def test_scalar_env_step_is_finite(self):
        cfg = _config()
        env = build_env(cfg, device=torch.device("cpu"))
        state = env.initial_state()
        next_state, _derived = env.step(state, torch.tensor([[0.0]]), 0)
        self.assertTrue(torch.isfinite(next_state).all())

    def test_uncontrolled_positive_state_grows_outward(self):
        cfg = _config()
        env = build_env(cfg, device=torch.device("cpu"))
        state = env.initial_state()
        next_state, _derived = env.step(state, torch.tensor([[0.0]]), 0)
        self.assertGreater(next_state.item(), state.item())

    def test_x_loss_reaches_model_parameters(self):
        cfg = _config()
        env = build_env(cfg, device=torch.device("cpu"))
        model = SelfTalkController(
            trunk_dims=cfg.model.trunk_dims,
            activation=cfg.model.activation,
            language_dim=cfg.model.language_dim,
            language_readout_coverage=cfg.model.language_readout_coverage,
            use_residual=cfg.model.use_residual,
            language_readout_all_layers=cfg.model.language_readout_all_layers,
            message_carry_mode=cfg.model.message_carry_mode,
            seed=cfg.run.seed,
            observation_dim=2,
        )
        state = env.initial_state()
        eta, _derived = env.eta(state, 0)
        observation = torch.cat([eta, torch.ones(1, 1)], dim=1)
        raw_u, _message, _hidden = model.forward_step(observation, message_prev=None)
        u = cfg.env.u_max * torch.tanh(raw_u)
        next_state, _ = env.step(state, u, 0)
        next_eta, _ = env.eta(next_state, 1)
        loss = torch.sum(next_eta ** 2)
        loss.backward()
        grad_sum = sum(
            p.grad.abs().sum().item()
            for p in model.parameters()
            if p.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)


if __name__ == "__main__":
    unittest.main()
