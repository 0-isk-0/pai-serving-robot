"""Render the submission video (MP4, 1280x720, 40 fps, real-time speed).

usage: MUJOCO_GL=glfw python make_video.py [model_path] [out.mp4]
Segments: title -> task -> (1) no-AI baseline -> (2) learned PPO -> results -> wrap-up.
"""
import json
import os
import sys
import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from robot_env import NavEnv, N_RAYS, GOAL_R
from evaluate import naive_policy

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/final"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/demo.mp4"
FPS = 40
W, H, SIM = 1280, 720, 720
SEEDS = [200_000 + i for i in range(10)]   # unseen layouts, shown in order (not hand-picked)

FONT_PATHS = ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]


def font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


BG = (18, 24, 38)
FG = (235, 240, 250)
DIM = (150, 165, 190)
GOOD, BAD, ACC = (90, 220, 130), (255, 110, 100), (110, 190, 255)


def card(lines, hold=4.0):
    """Static text card: lines = [(text, size, color)]."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    total = sum(sz * 1.6 for _, sz, _ in lines)
    y = (H - total) / 2
    for text, sz, col in lines:
        f = font(sz)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=col)
        y += sz * 1.6
    return [np.array(img)] * int(hold * FPS)


def image_card(path, caption, hold=8.0):
    img = Image.new("RGB", (W, H), BG)
    pic = Image.open(path).convert("RGB")
    pic.thumbnail((1180, 590))
    img.paste(pic, ((W - pic.width) // 2, 24))
    d = ImageDraw.Draw(img)
    f = font(24)
    w = d.textlength(caption, font=f)
    d.text(((W - w) / 2, 646), caption, font=f, fill=DIM)
    return [np.array(img)] * int(hold * FPS)


def sim_frame(env, renderer, trail, show_lidar):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0]
    cam.distance, cam.azimuth, cam.elevation = 9.2, 90, -88
    renderer.update_scene(env.d, cam)
    scn = renderer.scene
    p, yaw = env._pose()

    def line(a, b, w, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w, a, b)
        scn.ngeom += 1

    for a, b in zip(trail[:-1], trail[1:]):
        line([a[0], a[1], 0.02], [b[0], b[1], 0.02], 0.025, [1.0, 0.85, 0.2, 0.9])
    if show_lidar:
        org = np.array([p[0], p[1], 0.12])
        for k in range(N_RAYS):
            ang = yaw + env.ray_angles[k]
            dist = env._last_lidar[k]
            hit = org + dist * np.array([np.cos(ang), np.sin(ang), 0])
            near = dist < 1.0
            line(org, hit, 0.008, [1.0, 0.35, 0.3, 0.8] if near else [0.3, 0.8, 1.0, 0.45])
    return renderer.render()


def compose(sim, title, sub, status, scol, timer, tally, note):
    img = Image.new("RGB", (W, H), BG)
    img.paste(Image.fromarray(sim), (0, 0))
    d = ImageDraw.Draw(img)
    x = SIM + 30
    d.text((x, 34), title, font=font(34), fill=FG)
    d.text((x, 86), sub, font=font(22), fill=DIM)
    d.text((x, 170), status, font=font(46), fill=scol)
    d.text((x, 240), f"経過時間  {timer:4.1f} s", font=font(28), fill=FG)
    y = 320
    for color, txt in [((51, 115, 230), "ロボット（2輪・物理モデル）"), ((51, 204, 77), "ゴール"),
                       ((217, 89, 64), "障害物"), ((255, 217, 51), "走行軌跡"), ((77, 204, 255), "LiDAR 24本（赤=近い）")]:
        d.rectangle([x, y + 6, x + 22, y + 26], fill=color)
        d.text((x + 36, y), txt, font=font(22), fill=FG)
        y += 38
    d.text((x, 540), tally, font=font(26), fill=ACC)
    d.text((x, 640), note, font=font(20), fill=DIM)
    return np.array(img)


def run_segment(env, act, seeds, title, sub, label, writer):
    renderer = mujoco.Renderer(env.m, SIM, SIM)
    wins = 0
    for n, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        env.max_steps = 1000  # 40 Hz stepping
        trail, t, info = [env._pose()[0].copy()], 0, {}
        env._last_lidar = obs[:N_RAYS] * 3.0
        done, hold = False, 0
        while not done or hold < int(1.3 * FPS):
            if not done:
                action = act(obs)
                for _ in range(2):  # same action for 2 sub-steps => 40 fps video, 20 Hz policy
                    obs, _, term, trunc, info = env.step(action)
                    env._last_lidar = obs[:N_RAYS] * 3.0
                    t += 1
                    trail.append(env._pose()[0].copy())
                    if term or trunc:
                        done = True
                        break
                status, scol = "走行中", FG
            else:
                hold += 1
            if done:
                ok = info.get("success")
                status, scol = ("ゴール到達！", GOOD) if ok else ("衝突…" if info.get("collision") else "時間切れ", BAD)
            frame = sim_frame(env, renderer, trail[::3] + [trail[-1]], show_lidar=True)
            writer.append_data(compose(frame, label, sub, status, scol, t * 0.025, f"レイアウト {n + 1} / {len(seeds)}",
                                       "等倍速（再生速度の加工なし）"))
        wins += 1 if info.get("success") else 0
    return wins


class Writer:
    def __init__(self, path):
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
                                    pixelformat="yuv420p")

    def append_data(self, f):
        self.w.append_data(f)

    def extend(self, frames):
        for f in frames:
            self.w.append_data(f)

    def close(self):
        self.w.close()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    ev = json.load(open("results/eval.json")) if os.path.exists("results/eval.json") else None
    env = NavEnv()
    model = PPO.load(MODEL, device="cpu")
    w = Writer(OUT)

    w.extend(card([("強化学習で障害物を避けて走る", 56, FG), ("自律移動ロボット", 56, FG),
                   ("MuJoCo 物理シミュレーション  ×  PPO  ×  LiDAR", 28, ACC),
                   ("Physical AI 応用1 最終課題", 24, DIM)], 4))
    w.extend(card([("タスク", 44, ACC),
                   ("ランダムに配置された障害物を、LiDAR だけを頼りに避け", 30, FG),
                   ("ゴールまで自律走行する", 30, FG),
                   ("・ロボット：左右独立駆動の2輪（摩擦・接触を含む物理モデル）", 24, DIM),
                   ("・入力：LiDAR 24本 + ゴールの方向と距離 + 速度", 24, DIM),
                   ("・出力：左右の車輪の速度指令", 24, DIM),
                   ("・毎回レイアウトが変わる（学習時に見ていない配置で評価）", 24, DIM)], 8))

    w.extend(card([("AIの仕組み（強化学習 PPO）", 44, ACC),
                   ("観測：LiDAR 24本の距離 + ゴールの方向・距離 + 前進速度・回転速度（28次元）", 25, FG),
                   ("行動：左右の車輪の速度指令（連続値2つ）", 25, FG),
                   ("報酬：ゴールへ近づくほどプラス / ゴール到達 +10 / 衝突 -5", 25, FG),
                   ("学習：配置をランダムに変えながら 300万ステップ試行錯誤（CPUのみ・数分）", 25, FG),
                   ("使う技術：MuJoCo（物理）・Stable-Baselines3 PPO・MLP 128×128", 22, DIM)], 9))

    w.extend(card([("① AIなし：ゴールへ一直線に進む制御", 44, BAD),
                   ("障害物を見ない単純な制御。経路上に障害物があると…", 28, FG)], 3))
    run_segment(env, naive_policy, SEEDS[:3], "① AIなし", "ゴール方向へ直進するだけ", "① AIなし", w)

    w.extend(card([("② 学習済みAI（PPO 強化学習）", 44, GOOD),
                   ("同じ3つのレイアウトで、LiDAR を使って避けながら進む", 28, FG)], 3))
    wins = run_segment(env, lambda o: model.predict(o, deterministic=True)[0], SEEDS, "② 学習済みAI（PPO）",
                       "LiDAR で障害物を見て避ける", "② 学習済みAI", w)

    w.extend(image_card("results/learning_curve.png",
                        "試行錯誤を重ねるほどゴール到達率が上がり、衝突率が下がる（学習中・確率的方策での値）", 8))
    if ev:
        ppo, nv, rd = ev["PPO (learned)"], ev["Naive go-to-goal (no AI)"], ev["Random"]
        w.extend(card([("評価結果（学習に使っていない配置 %d 回）" % ppo["episodes"], 40, ACC),
                       ("学習済みAI（PPO）   成功 %.1f %%   衝突 %.1f %%" % (100 * ppo["success_rate"], 100 * ppo["collision_rate"]), 32, GOOD),
                       ("AIなし（直進）          成功 %.1f %%   衝突 %.1f %%" % (100 * nv["success_rate"], 100 * nv["collision_rate"]), 32, BAD),
                       ("ランダム行動             成功 %.1f %%" % (100 * rd["success_rate"]), 32, DIM)], 7))
    w.extend(card([("まとめ", 44, ACC),
                   ("・物理モデル上で、センサ入力から車輪指令を出す方策を強化学習で獲得", 26, FG),
                   ("・障害物の配置が毎回変わっても、見ていない配置で高い成功率", 26, FG),
                   ("・限界：2D 平面・静止障害物のみ（実機・動く障害物は今後の課題）", 24, DIM)], 6))
    w.close()
    print("saved", OUT)
