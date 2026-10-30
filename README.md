# 強化学習で障害物を避けて走る自律移動ロボット

MuJoCo 上の2輪ロボットが、LiDAR だけを頼りに、ランダムに置かれた障害物を避けてゴールへ向かう。
方策は強化学習（PPO）で獲得する。Physical AI 応用1 最終課題の提出物。

| | 内容 |
|---|---|
| シミュレータ | MuJoCo（摩擦・接触を含む物理モデル。グリッドワールドではない） |
| ロボット | 左右独立駆動の2輪 + 転倒防止キャスター |
| 観測（28次元） | LiDAR 24本の距離 / ゴールの距離・方位 / 前進速度・回転速度 |
| 行動（2次元） | 左右の車輪の速度指令 |
| 報酬 | ゴールへ近づいた距離 ×4 − 0.01 / 到達 +10 / 衝突 −5 |
| 学習 | Stable-Baselines3 PPO（MLP 128×128）、8並列、300万ステップ、CPUのみ |
| 評価 | 学習に使っていない配置 300 回（シード 100000〜） |

## 結果（`results/eval.json`）

| 方策 | 成功率 | 衝突率 |
|---|---|---|
| 学習済み PPO | 95.3 % | 4.0 % |
| LiDAR を使わず直進（AIなし） | 0.3 % | 99.7 % |
| ランダム行動 | 0.0 % | 7.7 % |

## 再現手順

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py 3000000 0      # 学習 -> models/final.zip, models/curve.csv
python evaluate.py models/final 300   # 評価 -> results/eval.json
python plot_curve.py           # 学習曲線 -> results/learning_curve.png
MUJOCO_GL=glfw python make_video.py models/final results/demo.mp4   # 動画（macOS）
```

学習済みモデル `models/final.zip` を同梱しているので、`evaluate.py` と `make_video.py` は学習なしで実行できる。

## ファイル

- `robot_env.py` — MuJoCo 環境（ロボット・障害物・LiDAR・報酬・衝突判定）
- `train.py` — PPO 学習
- `evaluate.py` — 学習済み方策とベースラインの比較評価
- `make_video.py` / `plot_curve.py` — 動画・図の生成
- `report/` — 提出レポートの元（HTML）

## 限界

2D 平面・静止障害物のみ。学習は1シードのみで、シード間のばらつきは測っていない。実機での検証はしていない。
