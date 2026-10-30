"""Figures for the report and video.

results/fig_results.png  : outcome breakdown per method (standard / stress scenarios)
results/fig_learning.png : learning curves of the proposed method over 3 seeds
results/summary.json     : the numbers used in the report (mean ± std over seeds for PPO)
"""
import glob
import json
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

METHODS = [("go_to_goal", "直進制御（センサなし）"),
           ("potential_field", "ポテンシャル法（古典・調整済み）"),
           ("potential_field_shield", "ポテンシャル法＋安全層"),
           ("noshield", "PPO単体（安全層なし）"),
           ("nocup", "PPO＋安全層：コップを考えず学習"),
           ("nodr", "PPO＋安全層：物理ランダム化なし"),
           ("main", "提案：PPO＋安全層")]
OUTCOMES = [("success", "成功", "#2e9e5b"), ("collision", "家具・壁に衝突", "#d9544d"),
            ("hit_person", "人と接触", "#a12a7a"), ("spill", "こぼした", "#3a7bd5"), ("timeout", "時間切れ", "#b0b0b0")]
SCEN = [("standard", "標準（学習時と同じ分布）"), ("stress", "厳しい条件（常に4人＋物理パラメータ1.6倍の幅）")]


def collect(ev, scen, key):
    """Return list of summaries (one per seed for learned methods)."""
    names = [k for k in ev[scen] if k == key or k.startswith(key + "_s")] if key not in ("go_to_goal", "potential_field", "potential_field_shield") \
        else [key]
    names = [n for n in names if not n.endswith("_best")]
    return [ev[scen][n]["summary"] for n in names if n in ev[scen]]


def video_figure(summary):
    """Large-font version for the video: standard scenario, the five methods that tell the story."""
    rows = [("go_to_goal", "直進制御（センサなし）"), ("potential_field", "ポテンシャル法（古典）"),
            ("noshield", "PPO単体（安全層なし）"), ("nocup", "コップを考えず学習したAI"), ("main", "提案：PPO＋安全層")]
    rows = [(k, l) for k, l in rows if k in summary["standard"]]
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=110)
    for yi, (key, label) in enumerate(rows):
        v = summary["standard"][key]
        left = 0
        for o, _, col in OUTCOMES:
            w = 100 * v[o][0]
            ax.barh(yi, w, left=left, color=col, height=0.66, edgecolor="white")
            if w >= 6 and o != "success":
                ax.text(left + w / 2, yi, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=14)
            left += w
        ax.text(2, yi, f"成功 {100 * v['success'][0]:.1f}%", va="center", color="white", fontsize=17, fontweight="bold")
    ax.set_yticks(range(len(rows)), [l for _, l in rows], fontsize=16)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.tick_params(axis="x", labelsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in OUTCOMES]
    fig.legend(handles, [l for _, l, _ in OUTCOMES], loc="lower center", ncol=5, frameon=False, fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("results/fig_video.png")


def main():
    ev = json.load(open("results/eval.json"))
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150, sharey=True)
    for ax, (scen, title) in zip(axes, SCEN):
        summary[scen] = {}
        ylabels = []
        for yi, (key, label) in enumerate(METHODS):
            ss = collect(ev, scen, key)
            if not ss:
                continue
            vals = {o: (float(np.mean([s[o] for s in ss])), float(np.std([s[o] for s in ss]))) for o, _, _ in OUTCOMES}
            vals["n_seeds"] = len(ss)
            vals["mean_time_s"] = float(np.mean([s["mean_time_s"] for s in ss if s["mean_time_s"]])) if any(
                s["mean_time_s"] for s in ss) else None
            vals["mean_peak_acc"] = float(np.mean([s["mean_peak_acc"] for s in ss]))
            arr = [s["arrival_acc"] for s in ss if s.get("arrival_acc") is not None]
            vals["arrival_acc"] = float(np.mean(arr)) if arr else None
            vals["shield_frac"] = float(np.mean([s.get("shield_frac", 0) for s in ss]))
            vals["reverse_frac"] = float(np.mean([s.get("reverse_frac", 0) for s in ss]))
            nm = [s["near_miss_rate"] for s in ss if s["near_miss_rate"] is not None]
            vals["near_miss_rate"] = float(np.mean(nm)) if nm else None
            summary[scen][key] = vals
            left = 0
            for o, _, col in OUTCOMES:
                w = 100 * vals[o][0]
                ax.barh(yi, w, left=left, color=col, height=0.62, edgecolor="white", linewidth=0.6)
                left += w
            sd = f" ±{100 * vals['success'][1]:.1f}" if len(ss) > 1 else ""
            ax.text(1.5, yi, f"{100 * vals['success'][0]:.1f}%{sd}", va="center", color="white", fontsize=10,
                    fontweight="bold")
            ylabels.append((yi, label + (f"（{len(ss)}シード）" if len(ss) > 1 else "")))
        ax.set_yticks([y for y, _ in ylabels], [l for _, l in ylabels])
        ax.set_xlim(0, 100)
        ax.set_xlabel("エピソードの割合 [%]（各300回・学習に使っていない配置）")
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in OUTCOMES]
    fig.legend(handles, [l for _, l, _ in OUTCOMES], loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("results/fig_results.png")

    # learning curves
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    for pattern, label, col in [("models/main_s*_curve.csv", "提案手法（成功率）", "#2e9e5b"),
                                ("models/nocup_s*_curve.csv", "コップなしで学習（参考・成功率）", "#999999")]:
        curves = [np.genfromtxt(f, delimiter=",", names=True) for f in sorted(glob.glob(pattern))]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        x = curves[0]["steps"][:n] / 1e6
        ys = np.array([100 * c["success"][:n] for c in curves])
        ax.plot(x, ys.mean(0), color=col, lw=2.3, label=f"{label}  n={len(curves)}")
        if len(curves) > 1:
            ax.fill_between(x, ys.min(0), ys.max(0), color=col, alpha=0.2)
        if pattern.startswith("models/main"):
            for k, lab, c2 in (("collision", "提案手法（衝突率）", "#d9544d"), ("spill", "提案手法（こぼし率）", "#3a7bd5")):
                yk = np.array([100 * c[k][:n] for c in curves])
                ax.plot(x, yk.mean(0), color=c2, lw=1.6, label=lab)
                if len(curves) > 1:
                    ax.fill_between(x, yk.min(0), yk.max(0), color=c2, alpha=0.15)
    ax.set_xlabel("学習ステップ数（百万）")
    ax.set_ylabel("直近100エピソードの割合 [%]")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="center right")
    ax.set_title("学習曲線（学習中・確率的方策。帯はシード間の最小〜最大）", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/fig_learning.png")
    video_figure(summary)
    json.dump(summary, open("results/summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({s: {k: round(v["success"][0], 3) for k, v in d.items()} for s, d in summary.items()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
