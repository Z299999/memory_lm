"""BallisticLunarLander — 弹道入射 + 物理驱动奖励函数。

在 gymnasium LunarLander 基础上做两件事：
1. reset() 时注入弹道初速度（飞船高速飞入而非悬停）
2. step() 用基于航天医学 + 弹道预测的奖励替换原版

奖励设计原则：
- 惩罚负 g（加速度朝飞船脚底，血涌向头）
- 惩罚角加速度（旋转力矩损伤结构）
- 惩罚预测落地偏差和落地速度（假设当前加速度不变向前积分）
- 不惩罚速度本身、角度本身（横飞减速是正当的弹道操纵）
"""

import math
import numpy as np
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
                 w_neg_g: float = 0.05,
                 w_pos_g: float = 0.01,
                 w_angacc: float = 0.005,
                 w_land_pos: float = 50.0,
                 w_land_vel: float = 2.0,
                 w_legs: float = 10.0,
                 w_fallback: float = 200.0,
                 w_main: float = 0.3,
                 w_side: float = 0.03,
                 **kwargs):
        super().__init__(render_mode=render_mode, continuous=continuous, **kwargs)
        self.entry_speed     = entry_speed
        self.entry_angle_deg = entry_angle_deg
        self.random_side     = random_side
        self.w_neg_g    = w_neg_g
        self.w_pos_g    = w_pos_g
        self.w_angacc   = w_angacc
        self.w_land_pos = w_land_pos
        self.w_land_vel = w_land_vel
        self.w_legs     = w_legs
        self.w_fallback = w_fallback
        self.w_main     = w_main
        self.w_side     = w_side

    # ── reset：注入弹道初速度 ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

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

    # ── step：物理驱动奖励 ────────────────────────────────────────────────────

    def step(self, action):
        # 保存上一步 Box2D 物理量（在 super().step() 之前）
        prev_lv = (self.lander.linearVelocity.x, self.lander.linearVelocity.y)
        prev_av = self.lander.angularVelocity

        obs, orig_reward, terminated, truncated, info = super().step(action)

        dt = 1.0 / FPS
        vel    = self.lander.linearVelocity
        theta  = self.lander.angle
        angvel = self.lander.angularVelocity

        # 加速度（Box2D m/s²）
        ax = (vel.x - prev_lv[0]) / dt
        ay = (vel.y - prev_lv[1]) / dt
        angacc = (angvel - prev_av) / dt

        # 1. g 力惩罚（负g重罚，正g轻罚）
        body_up = (-math.sin(theta), math.cos(theta))
        a_body  = ax * body_up[0] + ay * body_up[1]
        p_g     = (self.w_neg_g * max(0.0, -a_body) ** 2
                 + self.w_pos_g * max(0.0,  a_body) ** 2)

        # 2. 角加速度惩罚
        p_aa = self.w_angacc * angacc ** 2

        # 3. 弹道落地预测惩罚
        p_traj = self._traj_penalty(vel.x, vel.y, ax, ay, self.lander.position)

        # 4. 燃料惩罚
        p_fuel = (self.w_main * max(0.0, float(action[0]))
                + self.w_side * abs(float(action[1])))

        # 5. 触地奖励
        r_legs = self.w_legs * (obs[6] + obs[7])

        # 6. 终止奖励（直接复用原版 ±100，覆盖所有终止原因）
        t_bonus = orig_reward if terminated else 0.0

        reward = r_legs - p_g - p_aa - p_traj - p_fuel + t_bonus
        return obs, reward, terminated, truncated, info

    # ── 弹道落地预测 ──────────────────────────────────────────────────────────

    def _traj_penalty(self, vx, vy, ax, ay, pos):
        """
        假设当前 ax, ay 保持不变，解二次方程求落地时间：
            y_above + vy·t + 0.5·ay·t² = 0
        取最小正根，预测 x_land 和落地速度，返回加权惩罚。
        """
        y_above = pos.y - (self.helipad_y + LEG_DOWN / SCALE)
        x_pad   = VIEWPORT_W / SCALE / 2   # 着陆垫 x（Box2D m）
        x_norm  = VIEWPORT_W / SCALE / 2   # 归一化因子（10 m）

        t_land = None
        if abs(ay) < 0.1:
            # 近似匀速垂直运动
            if vy < -1e-3:
                t_land = -y_above / vy
        else:
            disc = vy * vy - 2.0 * ay * y_above
            if disc >= 0:
                sq = math.sqrt(disc)
                candidates = [t for t in [(-vy + sq) / ay, (-vy - sq) / ay] if t > 0.01]
                if candidates:
                    t_land = min(candidates)

        if t_land is None:
            return self.w_fallback

        t_land  = min(t_land, 10.0)   # 预测上限：10 s = 500 步
        x_land  = pos.x + vx * t_land + 0.5 * ax * t_land ** 2
        vx_land = vx + ax * t_land
        vy_land = vy + ay * t_land

        dx = (x_land - x_pad) / x_norm
        return self.w_land_pos * dx ** 2 + self.w_land_vel * (vx_land ** 2 + vy_land ** 2)
