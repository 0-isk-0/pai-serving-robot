"""Hand-designed baseline controllers (no learning) and a wrapper for learned policies.

Every controller is called as act(env, obs) -> wheel command in [-1, 1]^2.
The classical controllers command (v, w) and pass through the same acceleration limiter,
so they are cup-aware by design: they never accelerate harder than A_MAX.
"""
import numpy as np

from serving_env import MAX_WHEEL, DT, N_RAYS

WHEEL_R, TRACK = 0.05, 0.22
V_MAX, A_MAX, W_MAX, AW_MAX = 0.6, 1.0, 2.0, 4.0


def vw_to_wheels(v, w):
    left = (v - w * TRACK / 2) / WHEEL_R
    right = (v + w * TRACK / 2) / WHEEL_R
    return np.clip(np.array([left, right]) / MAX_WHEEL, -1, 1)


class _Smooth:
    """Slew-rate limiter on (v, w): the classical way to keep a cup on a tray."""

    def __init__(self):
        self.v = self.w = 0.0

    def reset(self):
        self.v = self.w = 0.0

    def __call__(self, v, w):
        self.v += np.clip(v - self.v, -A_MAX * DT, A_MAX * DT)
        self.w += np.clip(w - self.w, -AW_MAX * DT, AW_MAX * DT)
        return vw_to_wheels(self.v, self.w)


def _goal_polar(env):
    p, yaw = env.pose()
    g = env.goal - p
    b = np.arctan2(g[1], g[0]) - yaw
    return np.linalg.norm(g), (b + np.pi) % (2 * np.pi) - np.pi


class GoToGoal:
    """Turn toward the goal and drive; ignores the LiDAR."""
    name = "直進制御（センサ不使用）"

    def __init__(self):
        self.s = _Smooth()

    def reset(self):
        self.s.reset()

    def __call__(self, env, obs):
        dist, b = _goal_polar(env)
        v = V_MAX * max(0.0, np.cos(b)) * min(1.0, dist / 0.8)
        return self.s(v if abs(b) < 0.8 else 0.0, np.clip(2.0 * b, -W_MAX, W_MAX))


class PotentialField:
    """Artificial potential field on the LiDAR scan + acceleration limiting (classical reactive planner)."""
    name = "ポテンシャル法（LiDAR使用・古典）"

    def __init__(self, d0=0.9, k_rep=0.05, goal_fade=2.0):  # tuned on seeds 50000+ (see results/pf_tuning.txt)
        self.s, self.d0, self.k_rep, self.goal_fade = _Smooth(), d0, k_rep, goal_fade
        self.ang = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)

    def reset(self):
        self.s.reset()

    def __call__(self, env, obs):
        dist, b = _goal_polar(env)
        scan = env.last_scan - 0.17                       # distance from robot surface
        f = np.array([np.cos(b), np.sin(b)])              # attraction (unit, robot frame)
        near = scan < self.d0
        if near.any():
            d = np.maximum(scan[near], 0.03)
            mag = self.k_rep * (1 / d - 1 / self.d0) / d
            mag *= min(1.0, dist / self.goal_fade)        # fade repulsion near the goal (goal-beside-table problem)
            f -= (mag[:, None] * np.stack([np.cos(self.ang[near]), np.sin(self.ang[near])], 1)).sum(0)
        head = np.arctan2(f[1], f[0])
        front = scan[(np.abs((self.ang + np.pi) % (2 * np.pi) - np.pi) < 0.6)].min()
        v = V_MAX * max(0.0, np.cos(head)) * min(1.0, dist / 0.8) * np.clip((front - 0.05) / 0.5, 0.15, 1)
        return self.s(v, np.clip(2.5 * head, -W_MAX, W_MAX))


class Learned:
    def __init__(self, path, name="強化学習（PPO）"):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")
        self.name = name

    def reset(self):
        pass

    def __call__(self, env, obs):
        return self.model.predict(obs, deterministic=True)[0]
