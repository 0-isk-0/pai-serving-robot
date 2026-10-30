"""Extra measurements for the report -> results/analysis.json

- policy inference latency on CPU (is 20 Hz control feasible on a small computer?)
- how often the safety layer overrides the policy (is the policy doing the work, or the shield?)
- what happens right before each failure of the proposed method
"""
import json
import time
import numpy as np

from serving_env import ServingEnv
from controllers import Learned

N = 300
SEED0 = 100_000


def main():
    pol = Learned("models/main_s0")
    env = ServingEnv(final_stop=True)
    obs, _ = env.reset(seed=1)
    for _ in range(50):
        pol(env, obs)
    t = time.perf_counter()
    for _ in range(2000):
        pol(env, obs)
    latency_ms = (time.perf_counter() - t) / 2000 * 1000

    steps = active = 0
    failures = []
    by_people = {}
    for i in range(N):
        obs, _ = env.reset(seed=SEED0 + i)
        recent = []
        while True:
            obs, _, term, trunc, info = env.step(pol(env, obs))
            steps += 1
            active += env.shield_active
            recent = (recent + [env.shield_active])[-10:]
            if term or trunc:
                break
        b = by_people.setdefault(len(env.people), [0, 0])
        b[0] += 1
        b[1] += info["spill"]
        if not info["success"]:
            kind = ("hit_person" if info["hit_person"] else "collision" if info["collision"] else
                    "spill" if info["spill"] else "timeout")
            failures.append(dict(seed=SEED0 + i, kind=kind, people=len(env.people), peak_acc=round(info["peak_acc"], 2),
                                 shield_in_last_0_5s=int(sum(recent)),
                                 goal_dist=round(float(np.linalg.norm(env.goal - env.pose()[0])), 2)))
    out = dict(policy_latency_ms=latency_ms, shield_active_rate=active / steps, episodes=N,
               spill_by_people={k: dict(episodes=v[0], spills=v[1]) for k, v in sorted(by_people.items())},
               failures=failures)
    json.dump(out, open("results/analysis.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}), len(failures), "failures")
    for f in failures:
        print(f)


if __name__ == "__main__":
    main()
