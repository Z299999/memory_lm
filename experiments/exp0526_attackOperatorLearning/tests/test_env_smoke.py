from pathlib import Path
import sys
import unittest

import torch


EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from src.config import load_config  # noqa: E402
from src.env import AgeStructuredPredatorPreyEnv  # noqa: E402
from src.model import SelfTalkController  # noqa: E402


def _config():
    return load_config(EXP_DIR / "config.yaml")


class Exp0526EnvSmokeTest(unittest.TestCase):
    def test_equilibrium_eta_is_zero_and_zeta_is_positive(self):
        cfg = _config()
        env = AgeStructuredPredatorPreyEnv(cfg.env, cfg.ecology, device=torch.device("cpu"))
        state = env.equilibrium_state(0)
        eta, derived = env.eta(state, 0)
        self.assertLess(torch.max(torch.abs(eta)).item(), 1e-5)
        self.assertGreater(derived.species[0].zeta.item(), 0.0)
        self.assertGreater(derived.species[1].zeta.item(), 0.0)

    def test_pde_step_preserves_positive_population(self):
        cfg = _config()
        env = AgeStructuredPredatorPreyEnv(cfg.env, cfg.ecology, device=torch.device("cpu"))
        state = env.initial_state()
        next_state, _derived = env.step(state, torch.tensor([[0.5]]), 0)
        self.assertTrue(torch.isfinite(next_state).all())
        self.assertGreater(torch.min(next_state).item(), 0.0)

    def test_eta_loss_reaches_model_parameters(self):
        cfg = _config()
        env = AgeStructuredPredatorPreyEnv(cfg.env, cfg.ecology, device=torch.device("cpu"))
        model = SelfTalkController(
            trunk_dims=cfg.model.trunk_dims,
            activation=cfg.model.activation,
            language_dim=cfg.model.language_dim,
            language_readout_coverage=cfg.model.language_readout_coverage,
            use_error_view=cfg.model.use_error_view,
            use_residual=cfg.model.use_residual,
            language_readout_all_layers=cfg.model.language_readout_all_layers,
            message_carry_mode=cfg.model.message_carry_mode,
            seed=cfg.run.seed,
        )
        state = env.initial_state()
        eta, _derived = env.eta(state, 0)
        observation = torch.cat([eta, torch.ones(1, 1)], dim=1)
        raw_u, _message, _hidden = model.forward_step(observation, error_view=eta, message_prev=None)
        next_state, _ = env.step(state, torch.nn.functional.softplus(raw_u), 0)
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
