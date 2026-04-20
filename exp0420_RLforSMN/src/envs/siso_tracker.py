"""SISO Trajectory Tracking Environment.

A simple reinforcement learning environment for tracking a 1D target trajectory.

State: tracking error e(t) = target(t) - position(t)
Action: control force u(t) ∈ [-1, 1]
Reward: -e² - 0.1*u² (penalize error + control effort)

Dynamics:
  pos(t+1) = pos(t) + vel(t) * dt
  vel(t+1) = vel(t) + u(t) * dt
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SISOTrajectoryTracker(gym.Env):
    """SISO trajectory tracking environment.

    The agent learns to control a point mass to track a target trajectory.

    **Observation Space:**
    - `error`: tracking error = target - position
    - `velocity`: current velocity (optional, for better controllability)

    **Action Space:**
    - Continuous: control force u ∈ [-1, 1]

    **Reward:**
    - r = -error² - 0.1 * u²

    **Target Trajectories:**
    - 'sin': target(t) = A * sin(ω * t)
    - 'sin_mix': target(t) = 0.5*sin(t) + 0.3*sin(2*t) + 0.2*sin(3*t)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        target_freq: float = 0.1,
        target_amplitude: float = 1.0,
        episode_length: int = 200,
        dt: float = 0.02,
        use_velocity: bool = False,
        target_type: str = "sin",
        render_mode: str | None = None,
    ):
        """Initialize the environment.

        Args:
            target_freq: Frequency of sinusoidal target (Hz)
            target_amplitude: Amplitude of target trajectory
            episode_length: Number of time steps per episode
            dt: Time step size (seconds)
            use_velocity: If True, state = [error, velocity]; else state = [error]
            target_type: Target trajectory type ('sin' or 'sin_mix')
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()

        self.target_freq = target_freq
        self.target_amplitude = target_amplitude
        self.episode_length = episode_length
        self.dt = dt
        self.use_velocity = use_velocity
        self.target_type = target_type
        self.render_mode = render_mode

        # State space: [error] or [error, velocity]
        state_dim = 2 if use_velocity else 1
        self.observation_space = spaces.Box(
            low=np.array([-10.0] + ([-5.0] if use_velocity else [])),
            high=np.array([10.0] + ([5.0] if use_velocity else [])),
            dtype=np.float32
        )

        # Action space: continuous control force
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal state
        self.position = 0.0
        self.velocity = 0.0
        self.time_step = 0
        self._target_cache: np.ndarray | None = None

    def _target_function(self, t: float) -> float:
        """Compute target position at time t."""
        if self.target_type == "sin":
            return self.target_amplitude * np.sin(2 * np.pi * self.target_freq * t)
        elif self.target_type == "sin_mix":
            # Multi-frequency target for more challenging tracking
            return (0.5 * np.sin(t) + 0.3 * np.sin(2 * t) + 0.2 * np.sin(3 * t))
        else:
            return self.target_amplitude * np.sin(2 * np.pi * self.target_freq * t)

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        error = self._target_function(self.time_step * self.dt) - self.position
        if self.use_velocity:
            return np.array([error, self.velocity], dtype=np.float32)
        else:
            return np.array([error], dtype=np.float32)

    def _get_reward(self, error: float, action: np.ndarray) -> float:
        """Compute reward given error and action."""
        return -error**2 - 0.1 * float(action[0])**2

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.position = 0.0
        self.velocity = 0.0
        self.time_step = 0

        # Clear target cache for new episode
        self._target_cache = np.linspace(
            0, self.episode_length * self.dt, self.episode_length + 1
        )

        obs = self._get_obs()
        info = {
            "target": self._target_function(0),
            "position": self.position,
            "velocity": self.velocity,
        }

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step.

        Args:
            action: Control force u ∈ [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clamp action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Compute current error before update
        target = self._target_function(self.time_step * self.dt)
        error = target - self.position

        # Update dynamics (simple point mass)
        self.velocity += action[0] * self.dt
        self.position += self.velocity * self.dt

        # Compute reward
        reward = self._get_reward(error, action)

        # Advance time
        self.time_step += 1

        # Check termination
        terminated = False
        truncated = self.time_step >= self.episode_length

        # Info dict
        info = {
            "target": target,
            "position": self.position,
            "velocity": self.velocity,
            "error": error,
            "action": action[0],
        }

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment (not implemented for now)."""
        if self.render_mode == "human":
            print(f"t={self.time_step}: pos={self.position:.3f}, "
                  f"vel={self.velocity:.3f}, error={self._target_function(self.time_step * self.dt) - self.position:.3f}")

    def close(self):
        """Cleanup."""
        pass


# Test function
if __name__ == "__main__":
    print("Testing SISOTrajectoryTracker environment...")

    # Test 1: Basic environment (error only)
    print("\n=== Test 1: Error-only state ===")
    env = SISOTrajectoryTracker(target_freq=0.1, episode_length=100)
    obs, info = env.reset(seed=42)
    print(f"Initial obs: {obs}")
    print(f"Initial info: {info}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    for t in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if t % 5 == 0:
            print(f"  t={t}: obs={obs}, reward={reward:.4f}")
        if terminated or truncated:
            break
    print("Test 1: PASSED")

    # Test 2: Environment with velocity
    print("\n=== Test 2: Error + velocity state ===")
    env2 = SISOTrajectoryTracker(target_freq=0.1, use_velocity=True, episode_length=100)
    obs, info = env2.reset(seed=42)
    print(f"Initial obs: {obs}")
    assert len(obs) == 2, f"Expected 2D obs, got {len(obs)}"

    for t in range(20):
        action = env2.action_space.sample()
        obs, reward, terminated, truncated, info = env2.step(action)
        if terminated or truncated:
            break
    print("Test 2: PASSED")

    # Test 3: sin_mix target
    print("\n=== Test 3: sin_mix target ===")
    env3 = SISOTrajectoryTracker(target_type="sin_mix", episode_length=100)
    obs, info = env3.reset(seed=42)
    for t in range(50):
        action = env3.action_space.sample()
        obs, reward, terminated, truncated, info = env3.step(action)
        if terminated or truncated:
            break
    print("Test 3: PASSED")

    print("\n" + "="*50)
    print("All SISOTrajectoryTracker tests PASSED!")
    print("="*50)
