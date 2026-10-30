"""Evaluate controllers on unseen restaurant layouts (seeds disjoint from training).

usage: python evaluate.py --out results/eval.json --n 300 [--models models/main_s0 ...]
Each method runs with its own safety-layer setting: classical baselines and the "noshield" model run
without it; "potential_field_shield" is the classical method with the same safety layer (fairness check).
Scenarios:
  standard : training distribution (0-4 people, randomized physics)
  stress   : always 4 people + physics randomized 1.6x wider than in training (extrapolation)
"""
import argparse
import json
import os
import numpy as np

from serving_env import ServingEnv
from controllers import GoToGoal, PotentialField, Learned

SEED0 = 100_000
SCENARIOS = {
    "standard": dict(),
    "stress": dict(min_people=4, max_people=4, dr_scale=1.6),
}


def run_episode(env, ctrl, seed, scenario):
    obs, _ = env.reset(seed=seed)
    ctrl.reset()
    path, last, min_clr, max_tilt, rev, shield = 0.0, env.pose()[0], np.inf, 0.0, 0, 0
    accs = []
    for t in range(env.max_steps):
        obs, _, term, trunc, info = env.step(ctrl(env, obs))
        accs.append(float(np.linalg.norm(env.acc)))
        rev += env.body_vel()[0][0] < -0.05
        shield += env.shield_active
        p = env.pose()[0]
        path += np.linalg.norm(p - last)
        last = p
        min_clr = min(min_clr, env.person_clearance())
        max_tilt = max(max_tilt, env.cup_state()[2])
        if term or trunc:
            break
    outcome = ("success" if info["success"] else "hit_person" if info["hit_person"] else
               "collision" if info["collision"] else "spill" if info["spill"] else "timeout")
    return dict(seed=seed, outcome=outcome, time=(t + 1) * 0.05, path=path, peak_acc=info["peak_acc"],
                min_person_clearance=None if np.isinf(min_clr) else float(min_clr), max_tilt=max_tilt,
                reverse_frac=rev / (t + 1), shield_frac=shield / (t + 1),
                arrival_acc=max(accs[-20:]))       # harshest acceleration in the last second before the end


def summarize(rows):
    out = {"episodes": len(rows)}
    for k in ("success", "collision", "hit_person", "spill", "timeout"):   # mutually exclusive outcomes
        out[k] = float(np.mean([r["outcome"] == k for r in rows]))      # ("collision" = furniture / walls only)
    ok = [r for r in rows if r["outcome"] == "success"]
    out["mean_time_s"] = float(np.mean([r["time"] for r in ok])) if ok else None
    out["mean_peak_acc"] = float(np.mean([r["peak_acc"] for r in rows]))
    out["arrival_acc"] = float(np.mean([r["arrival_acc"] for r in ok])) if ok else None
    out["reverse_frac"] = float(np.mean([r["reverse_frac"] for r in rows]))   # time spent driving backwards
    out["shield_frac"] = float(np.mean([r["shield_frac"] for r in rows]))     # steps where the safety layer acted
    clr = [r["min_person_clearance"] for r in rows if r["min_person_clearance"] is not None]
    out["near_miss_rate"] = float(np.mean([c < 0.2 for c in clr])) if clr else None  # passed within 20 cm of a person
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--no-baselines", action="store_true")
    a = ap.parse_args()
    ctrls = {} if a.no_baselines else {"go_to_goal": (GoToGoal(), False), "potential_field": (PotentialField(), False),
                                       "potential_field_shield": (PotentialField(), True)}
    for mp in a.models:
        name = os.path.basename(mp)
        ctrls[name] = (Learned(mp), not name.startswith("noshield"))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for sc in a.scenarios:
        for name, (c, shield) in ctrls.items():
            env = ServingEnv(shield=shield, final_stop=True, **SCENARIOS[sc])
            rows = [run_episode(env, c, SEED0 + i, sc) for i in range(a.n)]
            res.setdefault(sc, {})[name] = dict(summary=summarize(rows), episodes=rows)
            s = res[sc][name]["summary"]
            print(f"[{sc}] {name:>16}: success {s['success']:.1%}  collision {s['collision']:.1%} "
                  f"+ person {s['hit_person']:.1%}  spill {s['spill']:.1%}  timeout {s['timeout']:.1%}  "
                  f"peak_acc {s['mean_peak_acc']:.2f}  arrive_acc {s['arrival_acc'] or 0:.2f}  reverse {s['reverse_frac']:.0%}  shield {s['shield_frac']:.0%}", flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(res, open(a.out, "w"), ensure_ascii=False)
