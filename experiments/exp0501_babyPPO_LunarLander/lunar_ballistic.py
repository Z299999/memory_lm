"""BallisticLunarLander — 弹道入射版 LunarLander。

改动：
1. reset() 注入弹道初速度（飞船高速飞入而非悬停）
2. step() 简化奖励函数：
     - 距离 shaping（差分，只保留距离，去掉速度和角度惩罚）
     - 腿触地奖励（包含在 shaping 内，差分形式）
     - 燃料惩罚（每步）
     - 终止 ±100（原版）
"""

import math
import numpy as np

# 3× 地图（必须在 from ... import 之前 patch）
import gymnasium.envs.box2d.lunar_lander as _ll
_ll.VIEWPORT_W = 1800   # 原始 600
_ll.VIEWPORT_H = 1200   # 原始 400

from gymnasium.envs.box2d.lunar_lander import (
    LunarLander, VIEWPORT_W, VIEWPORT_H, SCALE, FPS, LEG_DOWN
)


class BallisticLunarLander(LunarLander):
    def __init__(self,
                 render_mode=None,
                 entry_speed: float = 5.0,
                 entry_angle_deg: float = 45.0,
                 random_side: bool = True,
                 continuous: bool = True,
                 **kwargs):
        super().__init__(render_mode=render_mode, continuous=continuous, **kwargs)
        self.entry_speed     = entry_speed
        self.entry_angle_deg = entry_angle_deg
        self.random_side     = random_side
        self._prev_shaping   = None

    # ── reset：注入弹道初速度 ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_shaping = None

        angle_rad = math.radians(self.entry_angle_deg)
        if self.random_side:
            side = 1 if self.np_random.random() < 0.5 else -1
        else:
            side = -1

        vx =  side * self.entry_speed * math.cos(angle_rad)
        vy = -self.entry_speed * math.sin(angle_rad)

        import Box2D
        self.lander.linearVelocity = Box2D.b2Vec2(float(vx), float(vy))
        self.lander.angle = -side * angle_rad * 0.3

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = np.array([
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ], dtype=np.float32)

        return state, info

    # ── step：简化奖励 ────────────────────────────────────────────────────────

    def step(self, action):
        obs, orig_reward, terminated, truncated, info = super().step(action)

        # 终止时直接用原版 ±100
        if terminated:
            self._prev_shaping = None
            return obs, orig_reward, terminated, truncated, info

        # Shaping：只保留距离 + 腿触地（去掉速度和角度惩罚）
        shaping = (-100 * math.sqrt(obs[0] ** 2 + obs[1] ** 2)
                   + 10 * obs[6] + 10 * obs[7])

        reward = (shaping - self._prev_shaping) if self._prev_shaping is not None else 0.0
        self._prev_shaping = shaping

        # 燃料惩罚
        a0 = float(action[0])
        m_power = (a0 + 1.0) * 0.5 if a0 > 0 else 0.0
        reward -= m_power * 0.3 + abs(float(action[1])) * 0.03

        return obs, reward, terminated, truncated, info
