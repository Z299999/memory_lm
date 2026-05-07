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

# map_scale 由调用者在 import 前通过环境变量注入
import os
_map_scale = float(os.environ.get("MAP_SCALE", 1.0))

import gymnasium.envs.box2d.lunar_lander as _ll
_ll.VIEWPORT_W = int(600 * _map_scale)
_ll.VIEWPORT_H = int(400 * _map_scale)

from gymnasium.envs.box2d.lunar_lander import (
    LunarLander, VIEWPORT_W, VIEWPORT_H, SCALE, FPS, LEG_DOWN
)


class BallisticLunarLander(LunarLander):
    def __init__(self,
                 render_mode=None,
                 init_speed: float = 5.0,
                 init_flight_angle_deg: float = 45.0,
                 random_side: bool = True,
                 continuous: bool = True,
                 init_body_angle_deg: float = 13.5,
                 init_angular_velocity: float = 0.0,
                 init_altitude_m=None,
                 init_x_m: float = 0.0,
                 **kwargs):
        super().__init__(render_mode=render_mode, continuous=continuous, **kwargs)
        self.init_speed            = init_speed
        self.init_flight_angle_deg = init_flight_angle_deg
        self.random_side           = random_side
        self.init_body_angle_deg   = init_body_angle_deg
        self.init_angular_velocity = init_angular_velocity
        self.init_altitude_m       = init_altitude_m
        self.init_x_m              = init_x_m
        self._prev_shaping         = None

    # ── reset：注入弹道初速度 ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_shaping = None

        import Box2D

        # 1. 覆盖位置（x 和 y 独立控制，同时平移腿避免关节撕裂）
        old_x = self.lander.position.x
        old_y = self.lander.position.y
        x_phys = old_x
        y_phys = old_y
        if self.init_x_m != 0.0:
            x_phys = VIEWPORT_W / SCALE / 2 + self.init_x_m
        if self.init_altitude_m is not None:
            y_phys = self.helipad_y + self.init_altitude_m
        if self.init_x_m != 0.0 or self.init_altitude_m is not None:
            dx, dy = x_phys - old_x, y_phys - old_y
            self.lander.position = Box2D.b2Vec2(float(x_phys), float(y_phys))
            for leg in self.legs:   # 腿同步平移，避免关节约束失败
                leg.position = Box2D.b2Vec2(
                    leg.position.x + dx, leg.position.y + dy
                )

        # 2. 注入弹道速度
        flight_rad = math.radians(self.init_flight_angle_deg)
        if self.random_side:
            side = 1 if self.np_random.random() < 0.5 else -1
        else:
            side = -1

        vx =  side * self.init_speed * math.cos(flight_rad)
        vy = -self.init_speed * math.sin(flight_rad)
        self.lander.linearVelocity = Box2D.b2Vec2(float(vx), float(vy))

        # 3. 初始机身角度
        self.lander.angle = -side * math.radians(self.init_body_angle_deg)

        # 4. 初始角速度
        self.lander.angularVelocity = self.init_angular_velocity

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
