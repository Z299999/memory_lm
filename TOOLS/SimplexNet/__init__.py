"""TOOLS/SimplexNet - Simplex Memory Network for Reinforcement Learning.

This package provides:
- SMNmodule: Core neural network architecture
- SMN_RL: High-level RL wrapper with training, testing, visualization
- rl.algorithms: RL algorithms (DQN, PPO, REINFORCE)
- tools: Utilities (checkpoint, logger, plot, gui)

Usage::

    from TOOLS.SimplexNet import SMN_RL, SMNmodule
    import gymnasium as gym

    # High-level API
    smn_rl = SMN_RL(env=gym.make('CartPole-v1'), n=2, m=4, n_in=4, n_out=2)
    rewards = smn_rl.train(num_episodes=500)

    # Low-level API
    from TOOLS.SimplexNet.rl.algorithms import DQN
    q_network = SMNmodule(n=2, m=4, n_in=4, n_out=2)
    dqn = DQN(q_network=q_network, obs_dim=4, act_dim=2)
"""

from .SMNmodule import SMNmodule
from .SMN_RL import SMN_RL

__all__ = ['SMNmodule', 'SMN_RL']
__version__ = '0.1.0'
