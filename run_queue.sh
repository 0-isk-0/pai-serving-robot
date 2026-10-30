#!/bin/bash
# Train every model used in the report, one after another (each job uses 8 worker processes).
cd "$(dirname "$0")"
for job in "main 0" "nocup 0" "noshield 0" "main 1" "main 2" "nodr 0"; do
  set -- $job
  PYTHONPATH=. ./.venv/bin/python train.py --variant $1 --seed $2 --steps 6000000 > logs_$1_s$2.txt 2>&1
done
echo queue-finished
