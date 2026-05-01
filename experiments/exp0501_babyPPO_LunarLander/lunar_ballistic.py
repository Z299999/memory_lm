"""BallisticLunarLander — 弹道入射版 LunarLander。

继承 gymnasium 的 LunarLander，在 reset() 时注入初始速度，
模拟飞船以弹道轨迹飞入降落区域，而非静止悬浮于高空。

类比：SpaceX 猎鹰9号助推器高速飞回后减速着陆。

参数：
    entry_speed     初始速度大小（物理单位，默认 5.0）
    entry_angle_deg 速度方向与水平面的夹角（默认 45°，向下）
    random_side     True=随机从左/右飞入，False=固定从左飞入
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
                 **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        self.entry_speed     = entry_speed
        self.entry_angle_deg = entry_angle_deg
        self.random_side     = random_side

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # ── 注入弹道初速度 ────────────────────────────────────────────────────
        angle_rad = math.radians(self.entry_angle_deg)

        # 从哪一侧飞入（正 vx = 向右，负 vx = 向左）
        if self.random_side:
            side = 1 if self.np_random.random() < 0.5 else -1
        else:
            side = -1  # 固定从左侧飞入

        vx =  side * self.entry_speed * math.cos(angle_rad)
        vy = -self.entry_speed * math.sin(angle_rad)   # 始终向下

        # 设置 Box2D 物理体的初速度
        import Box2D
        self.lander.linearVelocity = Box2D.b2Vec2(float(vx), float(vy))

        # 飞船初始姿态略微倾向飞行方向（更真实）
        self.lander.angle = -side * angle_rad * 0.3

        # ── 重新计算 observation（与 step() 中相同的公式）────────────────────
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
