from pathlib import Path
import sys
import unittest

import torch


EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from src.config import (  # noqa: E402
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    PlotConfig,
    RunConfig,
    TrainConfig,
)
from src.env import build_env  # noqa: E402
from src.model import SelfTalkController  # noqa: E402


def _make_config(*, env: EnvConfig, language_dim: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        run=RunConfig(run_name="test", seed=7, log_every=1, output_root="runs"),
        model=ModelConfig(
            trunk_dims=(16,),
            activation="tanh",
            language_dim=language_dim,
            language_readout_coverage=1 if language_dim > 0 else 0,
            use_residual=True,
            language_readout_all_layers=True,
            message_carry_mode="identity",
        ),
        env=env,
        train=TrainConfig(
            epochs=2,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            train_window_schedule="fixed(4)",
            control_loss_weight=1e-4,
        ),
        eval=EvalConfig(eval_steps=8, future_steps=16, eval_conditions=("full",)),
        plot=PlotConfig(plot_show_message_traces=False, plot_show_message_norm=False, plot_show_training_timeline=False),
    )


class Exp0526EnvSmokeTest(unittest.TestCase):
    def test_scalar_control_affine_step_is_finite(self):
        cfg = _make_config(
            env=EnvConfig(
                env_kind="scalar_control_affine",
                dt=0.01,
                x0=0.5,
                pulse_value=1.0,
                u_max=2.0,
                f_expr="x - x**3",
                g_expr="x + 2",
                state_limit=4.0,
            )
        )
        env = build_env(cfg, device=torch.device("cpu"))
        state = env.initial_state()
        next_state, _derived = env.step(state, torch.tensor([[0.0]]), 0)
        self.assertTrue(torch.isfinite(next_state).all())

    def test_scalar_cubic_compatibility_step_matches_expected_euler_step(self):
        cfg = _make_config(
            env=EnvConfig(
                env_kind="scalar_cubic",
                dt=0.01,
                x0=0.25,
                pulse_value=1.0,
                u_max=2.0,
                linear_coeff=1.0,
                cubic_coeff=-1.0,
                control_gain=0.5,
                state_limit=4.0,
            )
        )
        env = build_env(cfg, device=torch.device("cpu"))
        state = env.initial_state()
        u = torch.tensor([[0.2]])
        next_state, _derived = env.step(state, u, 0)
        raw_next = state + cfg.env.dt * (
            cfg.env.linear_coeff * state
            + cfg.env.cubic_coeff * (state ** 3)
            + cfg.env.control_gain * u
        )
        expected = cfg.env.state_limit * torch.tanh(raw_next / cfg.env.state_limit)
        self.assertAlmostEqual(next_state.item(), expected.item(), places=6)

    def test_scalar_x_loss_reaches_model_parameters(self):
        cfg = _make_config(
            env=EnvConfig(
                env_kind="scalar_control_affine",
                dt=0.01,
                x0=0.4,
                pulse_value=1.0,
                u_max=2.0,
                f_expr="x - x**3",
                g_expr="1.0",
                state_limit=4.0,
            )
        )
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
            observation_dim=env.state_dim + 1,
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
        grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_sum, 0.0)

    def test_planar_double_well_equilibria_match_expected_locations(self):
        cfg = _make_config(
            env=EnvConfig(
                env_kind="planar_double_well",
                dt=0.01,
                x0=(0.2, 0.0),
                pulse_value=1.0,
                u_max=1.5,
                alpha=1.0,
                beta=1.0,
                gamma=0.4,
                control_gain=1.0,
                state_limit=6.0,
            )
        )
        env = build_env(cfg, device=torch.device("cpu"))
        equilibria = env.equilibrium_points()
        self.assertEqual(len(equilibria), 3)
        self.assertEqual(equilibria[0]["point"], (-1.0, 0.0))
        self.assertEqual(equilibria[1]["point"], (0.0, 0.0))
        self.assertEqual(equilibria[2]["point"], (1.0, 0.0))
        self.assertTrue(equilibria[0]["stable"])
        self.assertFalse(equilibria[1]["stable"])
        self.assertTrue(equilibria[2]["stable"])

    def test_planar_double_well_step_is_finite(self):
        cfg = _make_config(
            env=EnvConfig(
                env_kind="planar_double_well",
                dt=0.01,
                x0=(0.15, -0.05),
                pulse_value=1.0,
                u_max=1.5,
                alpha=1.0,
                beta=1.0,
                gamma=0.4,
                control_gain=1.0,
                state_limit=6.0,
            ),
            language_dim=0,
        )
        env = build_env(cfg, device=torch.device("cpu"))
        state = env.initial_state()
        next_state, _derived = env.step(state, torch.tensor([[0.0]]), 0)
        self.assertEqual(tuple(next_state.shape), (1, 2))
        self.assertTrue(torch.isfinite(next_state).all())


if __name__ == "__main__":
    unittest.main()
