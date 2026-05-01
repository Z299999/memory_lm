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
                 angle_weight=0.0, angvel_weight=0.0,
                 acc_weight=0.0, angacc_weight=0.0,
                 continuous: bool = False):
        super().__init__()
        self.render_mode    = render_mode
        self.pos_weight     = pos_weight
        self.vel_weight     = vel_weight
        self.angle_weight   = angle_weight
        self.angvel_weight  = angvel_weight
        self.acc_weight     = acc_weight
        self.angacc_weight  = angacc_weight
        self.continuous     = continuous

        # Observation: [x, xdot, theta, thetadot]
        high = np.array([4.8, np.finfo(np.float32).max,
                         0.418, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        if continuous:
            # continuous force F ∈ [-10, 10] N
            self.action_space = spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)
        else:
            # 21 discrete actions: 0→F=-10N, 10→F=0N, 20→F=+10N
            self.action_space = spaces.Discrete(21)

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

        if self.continuous:
            F = float(action[0])             # continuous: direct force value
        else:
            F = float(action - 10)           # discrete: 0..20 → -10..+10 N

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
                  - self.pos_weight    * x         ** 2
                  - self.vel_weight    * xdot      ** 2
                  - self.angle_weight  * theta      ** 2
                  - self.angvel_weight * thetadot   ** 2
                  - self.acc_weight    * xddot      ** 2
                  - self.angacc_weight * thetaddot  ** 2)

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False, {}

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        screen_w, screen_h = 600, 400
        scale       = screen_w / (self.x_threshold * 2)
        polewidth   = 10.0
        polelen     = scale * (2 * self.l)
        cartwidth   = 50.0
        cartheight  = 30.0
        carty       = 100                        # distance from top (pre-flip)
        axleoffset  = cartheight / 4.0

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode((screen_w, screen_h))
                pygame.display.set_caption("CustomCartPole")
            else:
                self._screen = pygame.Surface((screen_w, screen_h))
            self._clock = pygame.time.Clock()

        surf = pygame.Surface((screen_w, screen_h))
        surf.fill((255, 255, 255))

        x, _, theta, _ = self.state
        cartx = x * scale + screen_w / 2.0

        # ── Cart (anti-aliased filled polygon) ────────────────────────────────
        l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
        cart_coords = [(c[0] + cartx, c[1] + carty)
                       for c in [(l, b), (l, t), (r, t), (r, b)]]
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

        # ── Pole (rotated rectangle, wood colour) ─────────────────────────────
        l, r, t, b = (-polewidth/2, polewidth/2,
                      polelen - polewidth/2, -polewidth/2)
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-theta)
            pole_coords.append((coord[0] + cartx, coord[1] + carty + axleoffset))
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101))

        # ── Axle circle (blue-purple) ─────────────────────────────────────────
        gfxdraw.aacircle(surf, int(cartx), int(carty + axleoffset),
                         int(polewidth / 2), (129, 132, 203))
        gfxdraw.filled_circle(surf, int(cartx), int(carty + axleoffset),
                              int(polewidth / 2), (129, 132, 203))

        # ── Ground line ───────────────────────────────────────────────────────
        gfxdraw.hline(surf, 0, screen_w, carty, (0, 0, 0))

        # Flip y-axis (pygame y goes down, physics y goes up)
        surf = pygame.transform.flip(surf, False, True)
        self._screen.blit(surf, (0, 0))

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
