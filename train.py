"""Train the serving policy with PPO.

usage: python train.py --variant main --seed 0 --steps 10000000
variants:
  main     : cup on the plate, acceleration penalty, physics domain randomization, LiDAR safety layer
  nocup    : ablation - trained WITHOUT the cup and without the acceleration penalty (pure navigation)
  nodr     : ablation - trained without physics domain randomization (nominal masses/friction/motor)
  noshield : ablation - trained and run without the safety layer (pure end-to-end policy)
"""
import argparse
import os
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from serving_env import ServingEnv

VARIANTS = {
    "main": dict(),
    "nocup": dict(cup=False, accel_penalty=False),
    "nodr": dict(physics_dr=False),
    "noshield": dict(shield=False),
}
KEYS = ("success", "collision", "spill", "timeout")


class LogCallback(BaseCallback):
    def __init__(self, out, every=100_000):
        super().__init__()
        self.out, self.every, self.last, self.best, self.rows = out, every, 0, -1.0, []
        self.t0 = time.time()

    def _on_step(self):
        if self.num_timesteps - self.last < self.every:
            return True
        self.last = self.num_timesteps
        eps = list(self.model.ep_info_buffer)
        if not eps:
            return True
        rates = [np.mean([e[k] for e in eps]) for k in KEYS]
        rew = np.mean([e["r"] for e in eps])
        el = time.time() - self.t0
        self.rows.append((self.num_timesteps, el, rew, *rates))
        print(f"{self.num_timesteps:>10,} {el / 60:5.1f}min  R {rew:6.2f}  " +
              "  ".join(f"{k} {v:5.1%}" for k, v in zip(KEYS, rates)), flush=True)
        if rates[0] >= self.best:
            self.best = rates[0]
            self.model.save(f"{self.out}_best")
        np.savetxt(f"{self.out}_curve.csv", np.array(self.rows), delimiter=",", comments="",
                   header="steps,seconds,reward," + ",".join(KEYS))
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main", choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs("models", exist_ok=True)
    out = f"models/{a.variant}_s{a.seed}"
    torch.set_num_threads(1)
    env = make_vec_env(ServingEnv, n_envs=a.envs, seed=a.seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs=VARIANTS[a.variant], monitor_kwargs=dict(info_keywords=KEYS))
    model = PPO("MlpPolicy", env, seed=a.seed, n_steps=1024, batch_size=2048, n_epochs=10, learning_rate=3e-4,
                gamma=0.995, gae_lambda=0.95, ent_coef=0.0, clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0), device="cpu", verbose=0)
    model.learn(a.steps, callback=LogCallback(out))
    model.save(out)
    print("done", flush=True)
