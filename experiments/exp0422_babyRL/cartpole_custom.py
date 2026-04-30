"""Custom CartPole environment implementing physics from scratch.

Inherits from gym.Env so it plugs directly into the existing DQN training loop.
Adds shaped reward: position and velocity penalties on top of the survival reward.

Physics: Lagrangian equations for an inverted pendulum on a cart.
Reference: Florian (2007) "Correct equations for the dynamics of the cart-pole system"
"""

from math import cos, sin, pi

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomCartPole(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # ── Physics constants (match gymnasium CartPole-v1 defaults) ──────────────
    gravity   = 9.8
    M         = 1.0    # cart mass  (kg)
    m         = 0.1    # pole mass  (kg)
    l         = 0.5    # half-pole length (m)
    dt        = 0.02   # timestep   (s)
    force_mag = 10.0   # force per action (N)

    # ── Termination thresholds ────────────────────────────────────────────────
    x_threshold     = 2.4
    theta_threshold = 12 * pi / 180   # ≈ 0.2094 rad

    def __init__(self, render_mode=None,
                 pos_weight=0.1, vel_weight=0.0,
                 angle_weight=0.0, angvel_weight=0.0):
        super().__init__()
        self.render_mode    = render_mode
        self.pos_weight     = pos_weight
        self.vel_weight     = vel_weight
        self.angle_weight   = angle_weight
        self.angvel_weight  = angvel_weight

        # Observation: [x, xdot, theta, thetadot]
        high = np.array([4.8, np.finfo(np.float32).max,
                         0.418, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space      = spaces.Discrete(2)   # 0=left, 1=right

        self.state  = None
        self._screen = None
        self._clock  = None

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def step(self, action):
        x, xdot, theta, thetadot = self.state

        F = self.force_mag if action == 1 else -self.force_mag

        # ── Lagrangian dynamics ───────────────────────────────────────────────
        costh = cos(theta)
        sinth = sin(theta)
        denom = self.M + self.m * (1.0 - costh ** 2)

        xddot     = (F + self.m * self.l * thetadot ** 2 * sinth
                     - self.m * self.gravity * sinth * costh) / denom
        thetaddot = (self.gravity * sinth - costh * xddot) / self.l

        # ── Euler integration ─────────────────────────────────────────────────
        x        += xdot     * self.dt
        xdot     += xddot    * self.dt
        theta    += thetadot * self.dt
        thetadot += thetaddot * self.dt

        self.state = np.array([x, xdot, theta, thetadot], dtype=np.float32)

        terminated = bool(
            abs(x)     > self.x_threshold or
            abs(theta) > self.theta_threshold
        )

        # ── Shaped reward ─────────────────────────────────────────────────────
        reward = (1.0
                  - self.pos_weight   * x        ** 2
                  - self.vel_weight   * xdot     ** 2
                  - self.angle_weight * theta     ** 2
                  - self.angvel_weight * thetadot ** 2)

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False, {}

    def render(self):
        try:
            import pygame
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                'pygame is not installed, run `pip install pygame`'
            ) from e

        screen_w, screen_h = 600, 400
        scale = screen_w / (2 * self.x_threshold * 2)   # pixels per meter
        cart_y = screen_h * 0.6                          # cart vertical position

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode((screen_w, screen_h))
                pygame.display.set_caption("CustomCartPole")
            else:
                self._screen = pygame.Surface((screen_w, screen_h))
            self._clock = pygame.time.Clock()

        # ── Draw ──────────────────────────────────────────────────────────────
        self._screen.fill((255, 255, 255))

        x, _, theta, _ = self.state
        cart_x = int(x * scale + screen_w / 2)

        # Cart
        cart_w, cart_h = 80, 30
        cart_rect = pygame.Rect(cart_x - cart_w // 2,
                                int(cart_y) - cart_h // 2,
                                cart_w, cart_h)
        pygame.draw.rect(self._screen, (100, 100, 100), cart_rect)

        # Pole
        pole_len = self.l * 2 * scale   # full pole in pixels
        pole_x2  = cart_x  + pole_len * sin(theta)
        pole_y2  = cart_y  - pole_len * cos(theta)
        pygame.draw.line(self._screen, (200, 80, 20),
                         (cart_x, int(cart_y)),
                         (int(pole_x2), int(pole_y2)), 6)

        # Axle dot
        pygame.draw.circle(self._screen, (50, 50, 50), (cart_x, int(cart_y)), 6)

        # Ground line
        pygame.draw.line(self._screen, (0, 0, 0),
                         (0, int(cart_y) + cart_h // 2 + 1),
                         (screen_w, int(cart_y) + cart_h // 2 + 1), 1)

        if self.render_mode == "human":
            pygame.event.pump()
            self._clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._screen = None
