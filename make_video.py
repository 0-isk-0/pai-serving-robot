"""Render the submission video (MP4, 1280x720, 40 fps).

usage: MUJOCO_GL=glfw python make_video.py --out results/demo.mp4
Needs: models/main_s0.zip, models/nocup_s0.zip, results/eval.json, results/fig_*.png (from plot_results.py)
"""
import argparse
import json
import os
import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

from serving_env import ServingEnv, AX, AY, TABLE_R, CHAIR_R, PERSON_R, ROBOT_R, KITCHEN, N_RAYS
from controllers import GoToGoal, PotentialField, Learned, vw_to_wheels

W, H, MAIN_W = 1280, 720, 880
FPS = 40
BG, PANEL = (16, 22, 34), (24, 31, 46)
FG, DIM, ACC = (236, 240, 248), (150, 162, 185), (110, 190, 255)
GOOD, BAD, WARN = (90, 220, 130), (255, 105, 95), (255, 200, 80)
FONT_PATHS = ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]
_fonts = {}


def font(size):
    if size not in _fonts:
        p = next((p for p in FONT_PATHS if os.path.exists(p)), None)
        _fonts[size] = ImageFont.truetype(p, size) if p else ImageFont.load_default()
    return _fonts[size]


def centered(d, y, text, size, color):
    f = font(size)
    d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=color)


def card(lines, seconds):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = (H - sum(s * 1.55 for _, s, _ in lines)) / 2
    for text, s, c in lines:
        centered(d, y, text, s, c)
        y += s * 1.55
    return np.array(img), seconds


def image_card(path, caption, seconds, title=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    top = 20
    if title:
        centered(d, 18, title, 34, ACC)
        top = 75
    pic = Image.open(path).convert("RGB")
    pic.thumbnail((1200, 640 - top))
    img.paste(pic, ((W - pic.width) // 2, top))
    centered(d, 668, caption, 22, DIM)
    return np.array(img), seconds


def diagram_card(seconds):
    """Sensors -> neural network -> wheels, plus what was rewarded and randomized."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    centered(d, 40, "しくみ：センサから車輪まで、1つのニューラルネットが判断する", 34, ACC)

    def box(x, y, w, h, title, lines, col):
        d.rounded_rectangle([x, y, x + w, y + h], 16, outline=col, width=3, fill=PANEL)
        d.text((x + 18, y + 14), title, font=font(26), fill=col)
        for i, t in enumerate(lines):
            d.text((x + 18, y + 58 + i * 32), t, font=font(20), fill=FG)

    box(60, 140, 330, 250, "入力（センサ）", ["LiDAR 36本 × 2時刻", "（人の動きが分かる）", "ゴールの方向・距離",
                                          "速度・IMU加速度", "直前の指令"], ACC)
    box(475, 170, 330, 190, "方策ネットワーク", ["多層パーセプトロン", "256 × 256", "強化学習（PPO）で獲得"], GOOD)
    box(890, 170, 330, 190, "出力（行動）", ["左右の車輪の", "速度指令", "20回 / 秒"], WARN)
    for x0, x1 in ((390, 475), (805, 890)):
        d.line([x0 + 8, 265, x1 - 12, 265], fill=FG, width=5)
        d.polygon([(x1 - 4, 265), (x1 - 22, 253), (x1 - 22, 277)], fill=FG)
    d.text((80, 440), "報酬（何を良しとするか）", font=font(26), fill=ACC)
    for i, t in enumerate(["ゴールに近づく ＋　／　配膳完了（止まって渡す） ＋10",
                           "衝突・こぼす −5　／　急加速 −　／　人に近づきすぎ −"]):
        d.text((100, 488 + i * 36), t, font=font(23), fill=FG)
    d.text((80, 580), "学習の工夫", font=font(26), fill=ACC)
    d.text((100, 626), "店内配置・人の数と動き・質量・摩擦・モーター特性・センサノイズを毎回ランダムに変えて学習",
           font=font(21), fill=FG)
    return np.array(img), seconds


class Harsh:
    """Smooth speed-up (cup stays), then an emergency stop with no acceleration limit (cup tips)."""
    name = "急加速・急停止する制御"

    def reset(self):
        self.k = 0

    def __call__(self, env, obs):
        self.k += 1
        if self.k < 50:
            return vw_to_wheels(min(0.7, 0.7 * self.k / 30), 0.0)
        return vw_to_wheels(-0.8, 0.0)


def empty_stage(env):
    """Clear the room and put the robot in the open, facing +x (for the cup-physics demo)."""
    env.reset(seed=0)
    for mids in (env.tb, env.ch, env.pp):
        for i, mid in enumerate(mids):
            env.d.mocap_pos[mid] = [50 + i, 50, 0]
    env.tables, env.chairs, env.people = [], [], []
    env.set_goal([3.2, 0.0])
    q = env.cup_q
    env.d.qpos[:7] = [-2.8, 0, 0.08, 1, 0, 0, 0]
    env.d.qpos[q:q + 7] = [-2.8, 0, 0.1716, 1, 0, 0, 0]
    env.d.qvel[:] = 0
    mujoco.mj_forward(env.m, env.d)
    for _ in range(20):
        mujoco.mj_step(env.m, env.d)
    env.prev_scan = None
    obs, env.prev_dist = env._obs()
    return obs


class Scene:
    def __init__(self, env):
        self.env = env
        self.r = mujoco.Renderer(env.m, H, MAIN_W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.az = None

    def main_view(self):
        p, yaw = self.env.pose()
        target = np.degrees(yaw) + 30          # MuJoCo azimuth = viewing direction -> camera sits behind
        if self.az is None:
            self.az = target
        self.az += 0.06 * ((target - self.az + 180) % 360 - 180)   # smooth chase camera
        c = self.cam
        c.lookat[:] = [p[0], p[1], 0.25]
        c.distance, c.azimuth, c.elevation = 3.2, self.az, -52
        self.r.update_scene(self.env.d, c)
        return self.r.render()

    def minimap(self, trail):
        e, s = self.env, 50                                     # 50 px per metre -> 400 x 300
        img = Image.new("RGB", (int(2 * AX * s), int(2 * AY * s)), (226, 214, 196))
        d = ImageDraw.Draw(img)
        P = lambda q: ((q[0] + AX) * s, (AY - q[1]) * s)

        def circ(q, r, col):
            x, y = P(q)
            d.ellipse([x - r * s, y - r * s, x + r * s, y + r * s], fill=col)

        d.rectangle([0, (AY - 1.1) * s, 0.55 * s, (AY + 1.1) * s], fill=(70, 70, 76))
        p, yaw = e.pose()
        for k in range(N_RAYS):
            a = yaw + e.ray_ang[k]
            dist = e.last_scan[k]
            q = p + dist * np.array([np.cos(a), np.sin(a)])
            d.line([P(p), P(q)], fill=(235, 90, 80) if dist < 0.8 else (90, 170, 230), width=1)
        if len(trail) > 1:
            d.line([P(q) for q in trail], fill=(240, 170, 20), width=3)
        circ(e.goal, 0.22, (60, 190, 100))
        for c in e.tables:
            circ(c, TABLE_R, (140, 95, 60))
        for c in e.chairs:
            circ(c, CHAIR_R, (85, 100, 120))
        for q in e.people:
            circ(q["pos"], PERSON_R, (225, 120, 70))
        circ(p, ROBOT_R, (40, 90, 220))
        d.line([P(p), P(p + 0.22 * np.array([np.cos(yaw), np.sin(yaw)]))], fill=(255, 255, 255), width=3)
        d.rectangle([0, 0, img.width - 1, img.height - 1], outline=(90, 90, 90), width=2)
        return img


STATUS = {"success": ("配膳完了！", GOOD), "collision": ("衝突", BAD), "hit_person": ("人と接触", BAD),
          "spill": ("こぼれた！", BAD), "timeout": ("時間切れ", WARN)}


def compose(main, mini, title, tcol, sub, status, scol, t, tilt, acc, served, speed, shield=False):
    img = Image.new("RGB", (W, H), PANEL)
    img.paste(Image.fromarray(main), (0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, MAIN_W, 54], fill=BG)
    d.text((18, 10), title, font=font(28), fill=tcol)
    d.text((MAIN_W - 18 - d.textlength(speed, font=font(18)), 18), speed, font=font(18), fill=DIM)
    x = MAIN_W + 16
    d.text((x, 12), sub, font=font(19), fill=DIM)
    d.text((x, 40), status, font=font(36), fill=scol)
    d.text((x + 240, 54), f"{t:5.1f} 秒", font=font(22), fill=FG)
    img.paste(mini.resize((368, 276)), (x, 100))

    def gauge(y, label, val, vmax, warn, unit):
        d.text((x, y), label, font=font(19), fill=FG)
        txt = f"{val:4.1f} {unit}"
        d.text((x + 368 - d.textlength(txt, font=font(19)), y), txt, font=font(19), fill=FG)
        y0, w = y + 30, 368
        d.rectangle([x, y0, x + w, y0 + 16], fill=(45, 54, 72))
        col = BAD if val >= warn else GOOD if val < warn * 0.6 else WARN
        d.rectangle([x, y0, x + int(w * min(val / vmax, 1.0)), y0 + 16], fill=col)
        wx = x + int(w * warn / vmax)
        d.line([wx, y0 - 3, wx, y0 + 19], fill=FG, width=2)

    gauge(392, "コップの傾き（線=転倒）", tilt, 40, 35, "°")
    gauge(452, "加速度（線=危険）", acc, 8, 4.0, "m/s²")
    d.text((x, 514), f"配膳  {served}", font=font(24), fill=ACC)
    if shield:
        d.rounded_rectangle([x + 190, 512, x + 368, 544], 8, fill=(120, 80, 20))
        d.text((x + 202, 515), "安全層：減速中", font=font(20), fill=WARN)
    legend = [((40, 90, 220), "ロボット"), ((60, 190, 100), "ゴール"), ((225, 120, 70), "人"),
              ((140, 95, 60), "テーブル"), ((240, 170, 20), "軌跡"), ((90, 170, 230), "LiDAR")]
    for i, (col, txt) in enumerate(legend):
        lx, ly = x + (i % 3) * 124, 562 + (i // 3) * 32
        d.rectangle([lx, ly + 5, lx + 16, ly + 21], fill=col)
        d.text((lx + 24, ly), txt, font=font(18), fill=FG)
    return np.array(img)


class Video:
    def __init__(self, path):
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
                                    pixelformat="yuv420p")
        self.n = 0

    def still(self, item):
        frame, sec = item
        for _ in range(int(sec * FPS)):
            self.frame(frame)

    def frame(self, f):
        self.w.append_data(f)
        self.n += 1

    def close(self):
        self.w.close()
        print(f"{self.n / FPS:.1f} s of video", flush=True)


def crossing_stage(env):
    """Open floor; one person walks across the robot's path (shows the safety layer + learned policy)."""
    obs = empty_stage(env)
    env.people = [dict(pos=np.array([0.2, -2.4]), wp=np.array([0.2, 2.4]), speed=0.7)]
    env.d.mocap_pos[env.pp[0]] = [0.2, -2.4, 0]
    env._new_waypoint = lambda pos: np.array([pos[0], -np.sign(pos[1]) * 2.4])   # keep pacing back and forth
    return obs


def start_tour(env, seed, n_tables):
    """Kitchen -> n tables -> kitchen."""
    env.reset(seed=seed, options={"start_kitchen": True})
    rng = np.random.default_rng(seed)
    order = sorted(range(len(env.tables)), key=lambda k: env.tables[k][0])   # serve near to far
    goals = []
    for k in order[:n_tables]:
        spots = env.table_spots(k)
        goals.append(spots[rng.integers(len(spots))])
    goals.append(KITCHEN.copy())
    env.set_goal(goals[0])
    env.tour = goals[1:]
    env.max_steps = 3000
    obs, env.prev_dist = env._obs()
    return obs


def play(video, env, ctrl, title, tcol, sub, seed=None, tour=None, speed=1, max_s=None, n_goals=1, setup=None):
    """Run one episode and stream it. speed=2 shows every other frame (labelled 2x)."""
    if setup:
        obs = setup(env)
    else:
        obs = start_tour(env, seed, tour) if tour else env.reset(seed=seed)[0]
    ctrl.reset()
    sc = Scene(env)
    trail = [env.pose()[0].copy()]
    st = dict(status="走行中", col=FG, tilt=0.0, served=0, t=0.0)
    every = max(1, int(round(5 * speed)))   # physics sub-steps per video frame (5 => 40 fps at 1x)
    label = "等倍速" if every == 5 else f"{every / 5:g}倍速" + ("（スロー）" if every < 5 else "")
    count = [0]

    def emit():
        video.frame(compose(sc.main_view(), sc.minimap(trail), title, tcol, sub, st["status"], st["col"],
                            st["t"], st["tilt"], float(np.linalg.norm(env.acc)), f"{st['served']} / {n_goals}", label,
                            getattr(env, "shield_active", False) and st["status"] == "走行中"))

    def on_sub(i):
        count[0] += 1
        if count[0] % every == 0:
            st["tilt"] = env.cup_state()[2]
            emit()

    while True:
        obs, _, term, trunc, info = env.step(ctrl(env, obs), on_substep=on_sub)
        trail.append(env.pose()[0].copy())
        st["t"] += 0.05
        st["served"] = info["delivered"]
        if term or trunc or (max_s and st["t"] >= max_s):
            break
    key = ("success" if info["success"] else "hit_person" if info["hit_person"] else "collision"
           if info["collision"] else "spill" if info["spill"] else "timeout")
    st["status"], st["col"] = STATUS[key]
    env.d.ctrl[:] = 0
    for _ in range(int(1.6 * FPS)):          # hold the end; physics keeps running so a falling cup keeps falling
        for _ in range(5):
            mujoco.mj_step(env.m, env.d)
        st["tilt"] = env.cup_state()[2]
        emit()
    return key


def pick_comparison_seed(ev):
    """A layout where the four controllers show their typical failure modes (statistics are shown separately)."""
    by = {k: {r["seed"]: r["outcome"] for r in v["episodes"]} for k, v in ev["standard"].items()}
    want = [lambda s: by["main_s0"][s] == "success", lambda s: by["nocup_s0"][s] == "spill",
            lambda s: by["go_to_goal"][s] in ("collision", "hit_person"), lambda s: by["potential_field"][s] == "timeout"]
    for k in (4, 3, 2):
        for s in sorted(by["main_s0"]):
            if all(f(s) for f in want[:k]):
                return s
    return min(by["main_s0"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/demo.mp4")
    ap.add_argument("--tour-seed", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=None, help="override the comparison layout")
    a = ap.parse_args()
    ev = json.load(open("results/eval.json"))
    main = Learned("models/main_s0", "提案手法：強化学習（PPO）")
    nocup = Learned("models/nocup_s0", "比較：コップを考えずに学習したAI")
    v = Video(a.out)

    v.still(card([("こぼさない配膳ロボット", 64, FG),
                  ("人が行き交うレストランを、強化学習で走る", 32, ACC),
                  ("MuJoCo 物理シミュレーション × PPO × LiDAR", 24, DIM),
                  ("Physical AI 応用1 最終課題", 22, DIM)], 4))
    v.still(card([("課題", 42, ACC),
                  ("ロボットの天板に、固定されていないコップが載っている", 30, FG),
                  ("テーブルや椅子の配置も、歩く人も、毎回ちがう店内で", 30, FG),
                  ("① ぶつからない　② こぼさない　③ テーブルの横で止まって渡す", 30, WARN),
                  ("コップは物理的な剛体。急加速・急停止・衝突で本当に倒れる", 24, DIM)], 7))
    play(v, ServingEnv(randomize=False, max_people=0, shield=False, final_stop=True), Harsh(), "コップは本物の物理演算", WARN,
         "ゆっくり加速 → 急ブレーキすると…", setup=empty_stage, max_s=4)
    v.still(diagram_card(10))

    seed = a.seed if a.seed is not None else pick_comparison_seed(ev)
    print("comparison layout seed", seed, flush=True)
    v.still(card([("同じ店内・同じ人の動きで、4つの方法を比べる", 38, ACC),
                  ("（ここに映るのは代表例。統計は後で示す）", 24, DIM)], 3))
    for ctrl, col, sub, shield in [(GoToGoal(), BAD, "センサを使わずゴールへ向かう", False),
                                   (PotentialField(), WARN, "LiDARで障害物から離れる古典手法", False),
                                   (nocup, WARN, "ぶつからないが、コップを考えていない", True),
                                   (main, GOOD, "コップと人を考えて学習した方策＋安全層", True)]:
        play(v, ServingEnv(shield=shield, final_stop=True), ctrl, ctrl.name, col, sub, seed=seed,
             speed=2 if isinstance(ctrl, PotentialField) else 0.4 if ctrl is nocup else 1,
             max_s=20 if isinstance(ctrl, PotentialField) else 30)

    v.still(card([("人が目の前を横切ったら", 40, ACC),
                  ("安全層が「コップを倒さない減速度で止まれる速度」に抑え、", 28, FG),
                  ("学習した方策が回り込み方を決める", 28, FG)], 4))
    play(v, ServingEnv(randomize=False, max_people=0, final_stop=True), main, "提案手法：人の横断", GOOD, "人の後ろを回り込んで進む",
         setup=crossing_stage, max_s=15)

    v.still(card([("連続配膳デモ", 42, GOOD),
                  ("キッチン → 3つのテーブル → キッチンへ戻る（4人が歩行中）", 28, FG),
                  ("学習に使っていない店内配置", 24, DIM)], 3))
    play(v, ServingEnv(min_people=4, max_people=4, final_stop=True), main, "連続配膳（提案手法）", GOOD, "3卓に配膳してキッチンへ戻る",
         seed=a.tour_seed, tour=3, n_goals=4)

    v.still(image_card("results/fig_video.png", "学習に使っていない店内配置 各300回（提案手法は3シードの平均）", 10,
                       "評価結果：配膳の成功率と失敗の内訳"))
    v.still(image_card("results/fig_learning.png", "3つの乱数シードで学習（線は平均、帯は最小〜最大）", 6, "学習曲線"))
    sm = json.load(open("results/summary.json"))["standard"]["main"]
    v.still(card([("まとめ", 42, ACC),
                  (f"物理的に載ったコップを、人の間を抜けてテーブルまで届ける（配膳成功 {100 * sm['success'][0]:.0f}%）", 26, FG),
                  ("古典手法・安全層なし・「コップを考えないAI」と同条件で比べ、各工夫の効果を数値で示した", 26, FG),
                  (f"残る課題：人が多い場面でのこぼし（{100 * sm['spill'][0]:.1f}%）、到着直前の強めの減速", 23, WARN),
                  ("限界：シミュレーションのみ。実機ではセンサ遅延や床の段差、液体の揺れへの対応が必要", 23, DIM)], 8))
    v.close()
