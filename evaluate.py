"""Evaluate policies on unseen random layouts (seeds never used in training).

usage: python evaluate.py [model_path] [n_episodes]
Writes results/eval.json
"""
import json
import os
import sys
import numpy as np
from stable_baselines3 import PPO

from robot_env import NavEnv

SEED0 = 100_000  # training uses seeds 0..~10; evaluation uses a disjoint range


def naive_policy(obs):
    """Go straight to the goal; ignores the LiDAR (no AI, no obstacle avoidance)."""
    bearing = obs[-3] * np.pi
    fwd = max(0.0, 1 - abs(bearing))
    return np.array([fwd - 0.8 * bearing, fwd + 0.8 * bearing])


def run(env, act, seed):
    obs, _ = env.reset(seed=seed)
    path, last = 0.0, env._pose()[0]
    for t in range(env.max_steps):
        obs, r, term, trunc, info = env.step(act(obs))
        p = env._pose()[0]
        path += np.linalg.norm(p - last)
        last = p
        if term or trunc:
            break
    return dict(success=info["success"], collision=info["collision"], time=(t + 1) * 0.05, path=path)


def summarize(rows):
    s = [r for r in rows if r["success"]]
    return dict(
        episodes=len(rows),
        success_rate=float(np.mean([r["success"] for r in rows])),
        collision_rate=float(np.mean([r["collision"] for r in rows])),
        timeout_rate=float(np.mean([not r["success"] and not r["collision"] for r in rows])),
        mean_time_s=float(np.mean([r["time"] for r in s])) if s else None,
        mean_path_m=float(np.mean([r["path"] for r in s])) if s else None,
    )


if __name__ == "__main__":
    MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/final"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    env = NavEnv()
    model = PPO.load(MODEL, device="cpu")
    rng = np.random.default_rng(0)
    policies = {
        "PPO (learned)": lambda o: model.predict(o, deterministic=True)[0],
        "Naive go-to-goal (no AI)": naive_policy,
        "Random": lambda o: rng.uniform(-1, 1, 2),
    }
    res = {}
    for name, act in policies.items():
        res[name] = summarize([run(env, act, SEED0 + i) for i in range(N)])
        print(name, json.dumps(res[name]), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/eval.json", "w"), indent=2, ensure_ascii=False)
