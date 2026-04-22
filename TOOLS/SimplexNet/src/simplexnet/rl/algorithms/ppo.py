"""PPO Algorithm for TOOLS/SimplexNet.

This module provides a continuous-control PPO-Clip implementation using
independent SMN actor and critic networks.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

try:
    from ...core.SMNmodule import SMNmodule
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from core.SMNmodule import SMNmodule

if TYPE_CHECKING:
    from ..mdp import RolloutBatch


class PPO:
    """Continuous-action PPO agent with independent actor/critic networks."""

    def __init__(
        self,
        actor_network: SMNmodule,
        critic_network: SMNmodule,
        act_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.0,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        action_low: np.ndarray | list[float] | float = -1.0,
        action_high: np.ndarray | list[float] | float = 1.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.act_dim = act_dim

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32).reshape(-1)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32).reshape(-1)
        if self.action_low.numel() == 1 and act_dim > 1:
            self.action_low = self.action_low.repeat(act_dim)
        if self.action_high.numel() == 1 and act_dim > 1:
            self.action_high = self.action_high.repeat(act_dim)
        if self.action_low.numel() != act_dim or self.action_high.numel() != act_dim:
            raise ValueError("action_low/action_high must match act_dim")

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=critic_lr)

        self._last_transition_info: dict[str, float] = {}

    def _split_actor_output(self, actor_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = actor_output[:, :self.act_dim]
        log_std = actor_output[:, self.act_dim:]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def _squash_action(self, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(pre_tanh_action)
        low = self.action_low.to(pre_tanh_action.device)
        high = self.action_high.to(pre_tanh_action.device)
        return low + 0.5 * (squashed + 1.0) * (high - low)

    def _unsquash_action(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_low.to(action.device)
        high = self.action_high.to(action.device)
        scaled = 2.0 * (action - low) / (high - low) - 1.0
        scaled = torch.clamp(scaled, -0.999999, 0.999999)
        return 0.5 * (torch.log1p(scaled) - torch.log1p(-scaled))

    def _log_prob_from_pre_tanh(
        self,
        dist: Normal,
        pre_tanh_action: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = dist.log_prob(pre_tanh_action).sum(dim=1)
        squash_correction = torch.log(1.0 - torch.tanh(pre_tanh_action).pow(2) + 1e-6).sum(dim=1)
        return log_prob - squash_correction

    def _policy_stats(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_output = self.actor_network(states)
        mean, log_std = self._split_actor_output(actor_output)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        if actions is None:
            pre_tanh_action = dist.rsample()
            squashed_action = self._squash_action(pre_tanh_action)
        else:
            squashed_action = actions
            pre_tanh_action = self._unsquash_action(actions)

        log_prob = self._log_prob_from_pre_tanh(dist, pre_tanh_action)
        entropy = dist.entropy().sum(dim=1)
        return squashed_action, log_prob, entropy, mean

    def get_last_transition_info(self) -> dict[str, float]:
        """Return old log-prob and critic value for the latest selected action."""
        return self._last_transition_info

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select a continuous action from the current policy."""
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            actor_output = self.actor_network(state_t)
            mean, log_std = self._split_actor_output(actor_output)
            value = self.critic_network(state_t).squeeze(1)
            std = torch.exp(log_std)
            dist = Normal(mean, std)

            if training:
                pre_tanh_action = dist.rsample()
            else:
                pre_tanh_action = mean

            action = self._squash_action(pre_tanh_action)
            log_prob = self._log_prob_from_pre_tanh(dist, pre_tanh_action)

        self._last_transition_info = {
            'old_log_prob': float(log_prob.item()),
            'value': float(value.item()),
        }
        return action.squeeze(0).cpu().numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """No-op for interface compatibility. PPO trains on rollout batches."""
        del state, action, reward, next_state, done

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def train(self, rollout_batch: 'RolloutBatch') -> Optional[dict[str, float]]:
        """Train actor and critic from an on-policy rollout batch."""
        if rollout_batch is None or len(rollout_batch) == 0:
            return None

        states = torch.as_tensor(np.array(rollout_batch.states), dtype=torch.float32)
        actions = torch.as_tensor(np.array(rollout_batch.actions), dtype=torch.float32)
        rewards = torch.as_tensor(rollout_batch.rewards, dtype=torch.float32)
        dones = torch.as_tensor(rollout_batch.dones, dtype=torch.float32)
        old_log_probs = torch.as_tensor(rollout_batch.old_log_probs, dtype=torch.float32)
        values = torch.as_tensor(rollout_batch.values, dtype=torch.float32)

        with torch.no_grad():
            next_states = torch.as_tensor(np.array(rollout_batch.next_states), dtype=torch.float32)
            next_values = self.critic_network(next_states).squeeze(1)

        advantages, returns = self._compute_gae(rewards, dones, values, next_values)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = states.shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)
        metrics = {
            'loss': 0.0,
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'value_mean': 0.0,
            'advantage_mean': float(advantages.mean().item()),
        }
        updates = 0

        for _ in range(self.update_epochs):
            perm = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                _, new_log_probs, entropy, _ = self._policy_stats(mb_states, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

                unclipped_obj = ratio * mb_advantages
                clipped_obj = clipped_ratio * mb_advantages
                actor_loss = -torch.min(unclipped_obj, clipped_obj).mean()
                entropy_bonus = entropy.mean()

                self.actor_optimizer.zero_grad()
                (actor_loss - self.entropy_coef * entropy_bonus).backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_pred = self.critic_network(mb_states).squeeze(1)
                critic_loss = F.mse_loss(value_pred, mb_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                approx_kl = (mb_old_log_probs - new_log_probs).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                total_metric_loss = actor_loss.item() + critic_loss.item()

                metrics['loss'] += total_metric_loss
                metrics['actor_loss'] += float(actor_loss.item())
                metrics['critic_loss'] += float(critic_loss.item())
                metrics['entropy'] += float(entropy_bonus.item())
                metrics['approx_kl'] += float(approx_kl.item())
                metrics['clip_fraction'] += float(clip_fraction.item())
                metrics['value_mean'] += float(value_pred.mean().item())
                updates += 1

        if updates > 0:
            for key in ('loss', 'actor_loss', 'critic_loss', 'entropy', 'approx_kl', 'clip_fraction', 'value_mean'):
                metrics[key] /= updates

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save PPO actor/critic checkpoint."""
        torch.save({
            'actor_state_dict': self.actor_network.state_dict(),
            'critic_state_dict': self.critic_network.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'act_dim': self.act_dim,
            'gamma': self.gamma,
            'clip_eps': self.clip_eps,
            'gae_lambda': self.gae_lambda,
            'entropy_coef': self.entropy_coef,
            'log_std_min': self.log_std_min,
            'log_std_max': self.log_std_max,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load PPO actor/critic checkpoint from disk."""
        checkpoint = torch.load(path, weights_only=False)
        self.load_checkpoint_from_dict(checkpoint)

    def load_checkpoint_from_dict(self, checkpoint: dict) -> None:
        """Load PPO actor/critic checkpoint from an in-memory dictionary."""
        self.actor_network.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_network.load_state_dict(checkpoint['critic_state_dict'])
        if checkpoint.get('actor_optimizer_state') is not None:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
        if checkpoint.get('critic_optimizer_state') is not None:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
