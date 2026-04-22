"""
SMNModule for Reinforcement Learning — Example: CartPole DQN

This example shows how to use SMNModule as a drop-in replacement
for a standard MLP in a Deep Q-Network (DQN) agent.

Key points:
1. SMNModule is a plain nn.Module — use it anywhere you'd use a PyTorch network
2. Set x_bounds to match your environment's observation space
3. Output layer is tanh — for Q-networks this is fine (we just need relative values)
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Import SMNModule
import sys
sys.path.insert(0, 'src')
from smn_fitter import SMNModule


# =============================================================================
# Option 1: SMNModule as Q-network (direct replacement for MLP)
# =============================================================================

class DQNAgent:
    """DQN agent using SMN as Q-network."""

    def __init__(self, state_dim: int, action_dim: int):
        # SMN as Q-network: state -> Q-values for each action
        # CartPole: state_dim=4, action_dim=2
        # x_bounds should match observation space
        self.q_network = SMNModule(
            n=2, m=4,                    # Architecture: triangle motif, 4 points/edge
            n_in=state_dim,              # 4-dim observation
            n_out=action_dim,            # 2 actions -> 2 Q-values
            activation='relu',
            x_bounds=[(-6, 6), (-6, 6), (-6, 6), (-6, 6)],  # CartPole obs bounds (approx)
        )

        self.target_network = SMNModule(
            n=2, m=4,
            n_in=state_dim,
            n_out=action_dim,
            activation='relu',
            x_bounds=[(-6, 6), (-6, 6), (-6, 6), (-6, 6)],
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 64):
        """One training step from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (double DQN style)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # MSE loss
        loss = nn.MSELoss()(current_q, target_q)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# =============================================================================
# Option 2: SMNModule as Policy Network (for Policy Gradient / Actor-Critic)
# =============================================================================

class PolicyAgent:
    """Policy gradient agent using SMN as policy network."""

    def __init__(self, state_dim: int, action_dim: int):
        # Policy network: state -> action probabilities
        self.policy_net = SMNModule(
            n=2, m=4,
            n_in=state_dim,
            n_out=action_dim,
            activation='relu',
            x_bounds=[(-6, 6), (-6, 6), (-6, 6), (-6, 6)],
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.gamma = 0.99

    def select_action(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        """Sample action from policy, return action and log-probability."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy_net(state_t)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def train_reinforce(self, log_probs: list, rewards: list):
        """REINFORCE-style policy gradient update."""
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)

        # Policy gradient loss: -sum(log_prob * return)
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# =============================================================================
# Training loop
# =============================================================================

def train_dqn(num_episodes: int = 500):
    """Train DQN agent with SMN Q-network."""
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_dim=4, action_dim=2)

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step(batch_size=64)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.update_target_network()
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}: reward={total_reward}, "
                  f"avg(10)={avg_reward:.1f}, eps={agent.epsilon:.2f}")

        # Early stopping: solve CartPole (avg >= 475)
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 475:
            print(f"Solved at episode {episode+1}!")
            break

    env.close()
    return episode_rewards


def train_reinforce(num_episodes: int = 1000):
    """Train REINFORCE agent with SMN policy network."""
    env = gym.make('CartPole-v1')
    agent = PolicyAgent(state_dim=4, action_dim=2)

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0

        for t in range(500):
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.train_reinforce(log_probs, rewards)
        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}: reward={total_reward}, "
                  f"avg(50)={avg_reward:.1f}")

    env.close()
    return episode_rewards


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Training DQN with SMN Q-network")
    print("=" * 60)
    rewards = train_dqn(num_episodes=500)
    print(f"\nBest 100-episode avg: {np.max([np.mean(rewards[max(0,i-100):i]) for i in range(100, len(rewards)+1)]):.1f}")
