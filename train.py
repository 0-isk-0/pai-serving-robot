"""Train a PPO navigation policy for the MuJoCo 2-wheel robot.

usage: python train.py [total_timesteps] [seed]
"""
import sys
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from robot_env import NavEnv

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = "models"


class LogCallback(BaseCallback):
    """Print rolling success / collision rate and save the best model."""

    def __init__(self, every=50_000):
        super().__init__()
        self.every, self.last, self.best = every, 0, -1.0
        self.hist = []

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            eps = list(self.model.ep_info_buffer)
            if eps:
                succ = np.mean([e["success"] for e in eps])
                coll = np.mean([e["collision"] for e in eps])
                rew = np.mean([e["r"] for e in eps])
                self.hist.append((self.num_timesteps, rew, succ, coll))
                print(f"steps {self.num_timesteps:>9,}  reward {rew:7.2f}  success {succ:5.1%}  collision {coll:5.1%}",
                      flush=True)
                if succ >= self.best:
                    self.best = succ
                    self.model.save(f"{OUT}/best")
                np.savetxt(f"{OUT}/curve.csv", np.array(self.hist), delimiter=",",
                           header="steps,reward,success,collision", comments="")
        return True


def make_env():
    return NavEnv()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(1)
    env = make_vec_env(make_env, n_envs=8, seed=SEED, vec_env_cls=SubprocVecEnv,
                       monitor_kwargs=dict(info_keywords=("success", "collision")))
    model = PPO("MlpPolicy", env, seed=SEED, n_steps=512, batch_size=1024, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
                policy_kwargs=dict(net_arch=[128, 128]), device="cpu", verbose=0)
    model.learn(TOTAL, callback=LogCallback())
    model.save(f"{OUT}/final")
    print("done", flush=True)
