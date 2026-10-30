# Design iterations of the proposed method (seed 0, 300 unseen layouts, standard / stress)

Numbers copied from the evaluation logs of each version. Only v4 is the final method; its
up-to-date numbers are in `results/eval.json` and `results/summary.json`.

| version | change | success | collision | spill | timeout | time driving backwards |
|---|---|---|---|---|---|---|
| v1 | PPO only, no safety layer (3M steps, stochastic training stats) | ~80% | ~18% | ~2% | ~0% | most collisions happened while reversing at 0.6-0.8 m/s |
| v2 | + LiDAR safety layer (stopping-distance speed limit) | 93.0% / 89.0% | 0.0% / 0.0% | 5.3% / 8.0% | 1.7% / 3.0% | all inspected spills happened while reversing |
| v3 | + reverse speed cap 0.25 m/s, 2x acceleration penalty | 90.7% / 87.0% | 1.0% / 1.7% | 1.7% / 1.7% | 6.7% / 9.7% | **95%** - exploit: commands full reverse and lets the cap act as a speed limiter (shield active 98% of steps) |
| v4 | + penalty on the amount the safety layer changes the command, stronger reverse penalty | 95.7% / 93.7% | 0.0% / 0.3% | 4.0% / 5.7% | 0.3% / 0.3% | 1% (shield active 6% / 9%) |
| v5 (final) | success now also requires braking to a full stop with the cup upright (training and evaluation) | 95.0% / 95.7% | 0.0% / 0.0% | 4.0% / 4.0% | 1.0% / 0.3% | 2% (shield active 6% / 9%) |

v2-v4 were evaluated with the old arrival rule (episode ends the moment the robot is inside the goal
slower than 0.25 m/s). Video inspection showed the v4 policy reaching the goal at full speed and
braking/spinning at the last moment; the cup was already falling when "success" was declared. v5
fixes the rule, so v2-v4 success rates are optimistic and not comparable to v5.
The v5 row is seed 0; the 3-seed means of the final method are in `results/summary.json`.

v1 is from training-time statistics; v2-v4 are deterministic evaluations on seeds 100000-100299.
