"""Plot the training curve from models/curve.csv -> results/learning_curve.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for p in ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
        break

d = np.genfromtxt("models/curve.csv", delimiter=",", names=True)
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
ax.plot(d["steps"] / 1e6, 100 * d["success"], color="#2e9e5b", lw=2.4, label="ゴール到達率")
ax.plot(d["steps"] / 1e6, 100 * d["collision"], color="#d9544d", lw=2.4, label="衝突率")
ax.set_xlabel("学習ステップ数（百万）")
ax.set_ylabel("直近エピソードの割合 [%]")
ax.set_title("PPO の学習曲線（毎回ランダムな障害物配置）")
ax.set_ylim(0, 102)
ax.grid(alpha=0.3)
ax.legend(loc="center right")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/learning_curve.png")
print("saved results/learning_curve.png")
