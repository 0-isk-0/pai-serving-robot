"""Restaurant serving robot in MuJoCo.

A 2-wheel robot carries a free-standing cup on its top plate and must deliver it to a
table through a randomized restaurant with walking people, without collisions and
without spilling (the cup is a real rigid body; harsh acceleration tips it over).

Everything that moves at runtime (tables, chairs, people, goal) is a mocap body:
world-body geoms are static in MuJoCo and their collision shape would not follow
runtime position changes.
"""
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

AX, AY = 4.0, 3.0                 # arena half sizes [m]
N_TABLES, N_CHAIRS, N_PEOPLE = 6, 12, 4
TABLE_R, CHAIR_R, PERSON_R = 0.40, 0.15, 0.20
ROBOT_R = 0.17
N_RAYS, MAX_RANGE = 36, 4.0
MAX_WHEEL = 16.0                  # rad/s  (~0.8 m/s)
DT = 0.05                         # control period [s]
GOAL_R, STOP_V = 0.25, 0.25       # arrive = within GOAL_R and slower than STOP_V
TOP_Z = 0.11                      # top plate height
KITCHEN = np.array([-3.3, 0.0])
WHEEL_R, TRACK = 0.05, 0.22
A_SAFE, SAFE_MARGIN = 1.5, 0.05   # safety layer: brake at a cup-safe deceleration, keep 5 cm
V_REVERSE = 0.25                  # safety layer: reversing (into the robot's blind side for people) is slow only
FAR = 50.0


def build_xml():
    parts = []
    for i in range(N_TABLES):
        parts.append(f"""
    <body name="tb{i}" mocap="true" pos="{FAR} {FAR + i} 0">
      <geom name="table{i}" type="cylinder" size="{TABLE_R} 0.36" pos="0 0 0.36" rgba="0.55 0.38 0.25 1"/>
      <geom type="cylinder" size="{TABLE_R + 0.06} 0.015" pos="0 0 0.735" rgba="0.95 0.95 0.92 1"
            contype="0" conaffinity="0" group="2"/>
    </body>""")
    for i in range(N_CHAIRS):
        parts.append(f"""
    <body name="ch{i}" mocap="true" pos="{FAR + 2} {FAR + i} 0">
      <geom name="chair{i}" type="cylinder" size="{CHAIR_R} 0.23" pos="0 0 0.23" rgba="0.35 0.42 0.5 1"/>
    </body>""")
    colors = ["0.9 0.5 0.3", "0.4 0.6 0.9", "0.6 0.8 0.4", "0.85 0.7 0.3"]
    for i in range(N_PEOPLE):
        parts.append(f"""
    <body name="pp{i}" mocap="true" pos="{FAR + 4} {FAR + i} 0">
      <geom name="person{i}" type="capsule" fromto="0 0 0.25 0 0 1.3" size="{PERSON_R}" rgba="{colors[i]} 1"/>
      <geom type="sphere" size="0.12" pos="0 0 1.62" rgba="0.95 0.8 0.68 1" contype="0" conaffinity="0" group="2"/>
    </body>""")
    walls = ""
    for n, (px, py, sx, sy) in {"wN": (0, AY, AX, .05), "wS": (0, -AY, AX, .05),
                                "wE": (AX, 0, .05, AY), "wW": (-AX, 0, .05, AY)}.items():
        walls += f'<geom name="{n}" type="box" size="{sx} {sy} 0.3" pos="{px} {py} 0.3" rgba="0.82 0.8 0.76 1"/>\n'
    return f"""
<mujoco>
  <option timestep="0.005" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.9 0.92 0.95" rgb2="0.6 0.65 0.72" width="64" height="64"/>
    <texture name="floor" type="2d" builtin="checker" width="256" height="256" rgb1="0.80 0.72 0.60" rgb2="0.74 0.65 0.53"/>
    <material name="floor" texture="floor" texrepeat="8 6" reflectance="0.02"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light pos="-3 -2 4" dir="0.5 0.3 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="{AX} {AY} .1" material="floor" group="2"/>
    {walls}
    <geom name="counter" type="box" size="0.25 1.1 0.45" pos="{-AX + 0.3} 0 0.45" rgba="0.3 0.3 0.33 1"/>
    {''.join(parts)}
    <body name="goal" mocap="true" pos="{FAR} {-FAR} 0">
      <geom type="cylinder" size="{GOAL_R} 0.004" pos="0 0 0.004" rgba="0.2 0.85 0.4 0.7" contype="0" conaffinity="0" group="2"/>
      <geom type="cylinder" size="0.015 0.5" pos="0 0 0.5" rgba="0.2 0.85 0.4 0.35" contype="0" conaffinity="0" group="2"/>
    </body>
    <body name="chassis" pos="0 0 0.08">
      <freejoint name="root"/>
      <geom name="body" type="box" size="0.13 0.09 0.03" mass="1.5" rgba="0.95 0.95 0.97 1" group="1"/>
      <geom type="cylinder" size="0.085 0.003" pos="0 0 0.031" rgba="0.25 0.5 0.9 1" contype="0" conaffinity="0" group="1"/>
      <geom name="nose" type="box" size="0.02 0.05 0.025" pos="0.14 0 0" mass="0.01" rgba="0.25 0.5 0.9 1" group="1"
            contype="0" conaffinity="0"/>
      <geom name="caster_f" type="sphere" size="0.0235" pos="0.10 0 -0.055" mass="0.05" friction="0 0 0" priority="1" group="1"/>
      <geom name="caster_b" type="sphere" size="0.0235" pos="-0.10 0 -0.055" mass="0.05" friction="0 0 0" priority="1" group="1"/>
      <body name="wl" pos="0 0.11 -0.03">
        <joint name="jl" type="hinge" axis="0 1 0" damping="0.01" armature="0.004"/>
        <geom name="wheel_l" type="cylinder" size="0.05 0.015" euler="90 0 0" mass="0.1" friction="1.2 .005 .0001" rgba=".1 .1 .1 1" group="1"/>
      </body>
      <body name="wr" pos="0 -0.11 -0.03">
        <joint name="jr" type="hinge" axis="0 1 0" damping="0.01" armature="0.004"/>
        <geom name="wheel_r" type="cylinder" size="0.05 0.015" euler="90 0 0" mass="0.1" friction="1.2 .005 .0001" rgba=".1 .1 .1 1" group="1"/>
      </body>
    </body>
    <body name="cup" pos="0 0 0.2">
      <freejoint name="cupj"/>
      <geom name="cup" type="cylinder" size="0.035 0.06" mass="0.3" friction="0.6 0.005 0.0001" priority="2" condim="4"
            rgba="0.75 0.9 1.0 0.85" group="1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="al" joint="jl" kv="0.5" ctrlrange="-{MAX_WHEEL} {MAX_WHEEL}"/>
    <velocity name="ar" joint="jr" kv="0.5" ctrlrange="-{MAX_WHEEL} {MAX_WHEEL}"/>
  </actuator>
</mujoco>"""


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


class ServingEnv(gym.Env):
    """Single delivery per episode (training). Pass options={"tour": [...]} to chain goals."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, randomize=True, max_people=N_PEOPLE, min_people=0, cup=True, accel_penalty=True, physics_dr=True, shield=True, final_stop=True, max_steps=600,
                 dr_scale=1.0):
        self.m = mujoco.MjModel.from_xml_string(build_xml())
        self.d = mujoco.MjData(self.m)
        self.randomize, self.max_people, self.min_people, self.use_cup = randomize, max_people, min_people, cup
        self.accel_penalty, self.max_steps, self.dr_scale = accel_penalty, max_steps, dr_scale
        self.physics_dr, self.shield = physics_dr, shield
        self.final_stop = final_stop   # after arriving, brake to a full stop; success only if the cup is still upright
        self.substeps = int(round(DT / self.m.opt.timestep))
        m = self.m
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        mid = lambda n: m.body_mocapid[bid(n)]
        self.tb = [mid(f"tb{i}") for i in range(N_TABLES)]
        self.ch = [mid(f"ch{i}") for i in range(N_CHAIRS)]
        self.pp = [mid(f"pp{i}") for i in range(N_PEOPLE)]
        self.goal_mid = mid("goal")
        self.furniture_ids = ({gid(f"table{i}") for i in range(N_TABLES)} | {gid(f"chair{i}") for i in range(N_CHAIRS)}
                              | {gid(n) for n in ("wN", "wS", "wE", "wW", "counter")})
        self.person_ids = {gid(f"person{i}") for i in range(N_PEOPLE)}
        self.robot_ids = {gid(n) for n in ("body", "caster_f", "caster_b", "wheel_l", "wheel_r")}
        self.chassis = bid("chassis")
        self.cup_body = bid("cup")
        self.cup_geom = gid("cup")
        self.cup_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cupj")]
        self.cup_v = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cupj")]
        self.geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # LiDAR sees group 0 only
        self.ray_ang = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
        # nominal physical parameters for domain randomization
        self.nom = dict(mass=m.body_mass[self.chassis], inertia=m.body_inertia[self.chassis].copy(),
                        wfric=m.geom_friction[gid("wheel_l")].copy(), cfric=m.geom_friction[self.cup_geom].copy(),
                        cmass=m.body_mass[self.cup_body], cinertia=m.body_inertia[self.cup_body].copy(),
                        kv=m.actuator_gainprm[0, 0])
        self.wheel_geoms = [gid("wheel_l"), gid("wheel_r")]
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * N_RAYS + 9,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng()
        self.tour = []

    # ------------------------------------------------------------------ layout
    def _free(self, p, r, people=True):
        if abs(p[0]) > AX - 0.1 - r or abs(p[1]) > AY - 0.1 - r:
            return False
        if p[0] < -AX + 0.55 + r and abs(p[1]) < 1.1 + r:     # kitchen counter
            return False
        if any(np.linalg.norm(p - c) < TABLE_R + r for c in self.tables):
            return False
        if any(np.linalg.norm(p - c) < CHAIR_R + r for c in self.chairs):
            return False
        if people and any(np.linalg.norm(p - q["pos"]) < PERSON_R + r for q in self.people):
            return False
        return True

    def _sample_free(self, r, tries=500):
        for _ in range(tries):
            p = self.rng.uniform([-AX + 0.3, -AY + 0.3], [AX - 0.3, AY - 0.3])
            if self._free(p, r):
                return p
        return None

    def table_spots(self, k):
        """Approach points beside table k (where the robot should stop to serve)."""
        c = self.tables[k]
        out = []
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            p = c + (TABLE_R + ROBOT_R + 0.2) * np.array([np.cos(a), np.sin(a)])
            if self._free(p, ROBOT_R + 0.15, people=False):
                out.append(p)
        return out

    def _layout(self):
        r = self.rng
        n_tab = r.integers(4, N_TABLES + 1) if self.randomize else N_TABLES
        grid = [(x, y) for x in (-1.6, 0.6, 2.8) for y in (-1.6, 1.6)]
        sel = r.permutation(len(grid))[:n_tab]
        self.tables = [np.array(grid[i]) + r.uniform(-0.35, 0.35, 2) for i in sel]
        self.chairs = []
        for c in self.tables:
            for a in r.uniform(0, 2 * np.pi, 2):
                p = c + (TABLE_R + CHAIR_R + 0.07) * np.array([np.cos(a), np.sin(a)])
                if abs(p[0]) < AX - 0.3 and abs(p[1]) < AY - 0.3 and all(np.linalg.norm(p - q) > 2 * CHAIR_R + 0.05 for q in self.chairs):
                    self.chairs.append(p)
        self.people = []

    def _new_waypoint(self, pos):
        for _ in range(60):
            w = self._sample_free(PERSON_R + 0.15, tries=20)
            if w is None:
                continue
            if all(seg_dist(c, pos, w) > TABLE_R + PERSON_R + 0.1 for c in self.tables) and \
               all(seg_dist(c, pos, w) > CHAIR_R + PERSON_R + 0.05 for c in self.chairs):
                return w
        return pos.copy()

    def _place_mocap(self):
        d = self.d
        for i, mid in enumerate(self.tb):
            d.mocap_pos[mid] = [*self.tables[i], 0] if i < len(self.tables) else [FAR, FAR + i, 0]
        for i, mid in enumerate(self.ch):
            d.mocap_pos[mid] = [*self.chairs[i], 0] if i < len(self.chairs) else [FAR + 2, FAR + i, 0]
        for i, mid in enumerate(self.pp):
            d.mocap_pos[mid] = [*self.people[i]["pos"], 0] if i < len(self.people) else [FAR + 4, FAR + i, 0]
        d.mocap_pos[self.goal_mid] = [*self.goal, 0]

    # ------------------------------------------------------------------ randomization
    def _randomize_physics(self):
        m, n, r, s = self.m, self.nom, self.rng, self.dr_scale
        on = self.randomize and self.physics_dr
        u = lambda lo, hi: 1 + s * r.uniform(lo, hi) if on else 1.0
        k = u(-0.25, 0.25)
        m.body_mass[self.chassis], m.body_inertia[self.chassis] = n["mass"] * k, n["inertia"] * k
        k = u(-0.3, 0.3)
        m.body_mass[self.cup_body], m.body_inertia[self.cup_body] = n["cmass"] * k, n["cinertia"] * k
        k = u(-0.3, 0.2)
        for g in self.wheel_geoms:
            m.geom_friction[g] = n["wfric"] * np.array([k, 1, 1])
        m.geom_friction[self.cup_geom] = n["cfric"] * np.array([u(-0.3, 0.3), 1, 1])
        kv = n["kv"] * u(-0.3, 0.3)
        m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 2] = kv, -kv
        self.lidar_noise = 0.02 * s if on else 0.0
        mujoco.mj_setConst(m, self.d)   # recompute derived constants (inv. weights) after changing masses

    # ------------------------------------------------------------------ sensing
    def pose(self):
        q = self.d.qpos[3:7]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return self.d.qpos[:2].copy(), yaw

    def body_vel(self):
        _, yaw = self.pose()
        c, s = np.cos(yaw), np.sin(yaw)
        v = self.d.qvel[:2]
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]]), self.d.qvel[5]

    def lidar(self):
        p, yaw = self.pose()
        ang = yaw + self.ray_ang
        vec = np.zeros((N_RAYS, 3))
        vec[:, 0], vec[:, 1] = np.cos(ang), np.sin(ang)
        dist = np.zeros(N_RAYS)
        gid = np.zeros(N_RAYS, dtype=np.int32)
        mujoco.mj_multiRay(self.m, self.d, np.array([p[0], p[1], 0.12]), vec.ravel(), self.geomgroup, 1,
                           self.chassis, gid, dist, None, N_RAYS, MAX_RANGE)
        dist[dist < 0] = MAX_RANGE
        dist = np.minimum(dist, MAX_RANGE)
        self.true_scan = dist.copy()                       # noiseless, used only for reward shaping
        if self.lidar_noise > 0:
            dist = dist + self.rng.normal(0, self.lidar_noise, N_RAYS)
            dist[self.rng.random(N_RAYS) < 0.02] = MAX_RANGE  # dropouts
        return np.clip(dist, 0, MAX_RANGE)

    def _obs(self):
        p, yaw = self.pose()
        scan = self.lidar()
        self.last_scan = scan
        if self.prev_scan is None:
            self.prev_scan = scan
        to_g = self.goal - p
        dist = np.linalg.norm(to_g)
        b = np.arctan2(to_g[1], to_g[0]) - yaw
        v, w = self.body_vel()
        o = np.concatenate([scan / MAX_RANGE, np.clip((scan - self.prev_scan) * 2.0, -1, 1),
                            [min(dist, 8) / 8, np.sin(b), np.cos(b), v[0] / 0.8, w / 4,
                             np.clip(self.acc[0] / 5, -2, 2), np.clip(self.acc[1] / 5, -2, 2)],
                            self.prev_action])
        self.prev_scan = scan
        return o.astype(np.float32), dist

    # ------------------------------------------------------------------ status
    def cup_state(self):
        """Returns (spilled, horizontal offset, tilt angle)."""
        if not self.use_cup:
            return False, 0.0, 0.0
        cq = self.d.qpos[self.cup_q:self.cup_q + 7]
        off = np.linalg.norm(cq[:2] - self.d.qpos[:2])
        w, x, y, z = cq[3:7]
        upz = 1 - 2 * (x * x + y * y)                         # cup axis z-component
        tilt = np.degrees(np.arccos(np.clip(upz, -1, 1)))
        return bool(off > 0.10 or cq[2] < TOP_Z or tilt > 35), off, tilt

    def contacts(self):
        hit_f = hit_p = False
        for c in self.d.contact[:self.d.ncon]:
            a, b = c.geom1, c.geom2
            if b in self.robot_ids:
                a, b = b, a
            if a in self.robot_ids:
                hit_f |= b in self.furniture_ids
                hit_p |= b in self.person_ids
        return hit_f, hit_p

    def person_clearance(self):
        p, _ = self.pose()
        if not self.people:
            return np.inf
        return min(np.linalg.norm(p - q["pos"]) for q in self.people) - PERSON_R - ROBOT_R

    # ------------------------------------------------------------------ people
    def _move_people(self):
        p, _ = self.pose()
        for q in self.people:
            to = q["wp"] - q["pos"]
            dist = np.linalg.norm(to)
            if dist < 0.1:
                q["wp"] = self._new_waypoint(q["pos"])
                continue
            dirv = to / dist
            rel = p - q["pos"]
            # people politely stop if the robot is right in front of them
            if np.linalg.norm(rel) < 0.6 and np.dot(rel, dirv) > 0:
                continue
            q["pos"] = q["pos"] + dirv * min(q["speed"] * DT, dist)
        for i, q in enumerate(self.people):
            self.d.mocap_pos[self.pp[i]] = [*q["pos"], 0]

    # ------------------------------------------------------------------ gym API
    def set_goal(self, g):
        self.goal = np.asarray(g, dtype=float)
        self.d.mocap_pos[self.goal_mid] = [*self.goal, 0]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        options = options or {}
        mujoco.mj_resetData(self.m, self.d)
        self._randomize_physics()
        for _ in range(100):
            self._layout()
            if options.get("start_kitchen") or r.random() < 0.3:
                start = KITCHEN + r.uniform(-0.3, 0.3, 2)
            else:
                start = self._sample_free(ROBOT_R + 0.25)
            if start is None or not self._free(start, ROBOT_R + 0.2):
                continue
            if r.random() < 0.85:
                k = r.integers(len(self.tables))
                spots = self.table_spots(k)
                if not spots:
                    continue
                goal = spots[r.integers(len(spots))]
            else:
                goal = KITCHEN.copy()
            if np.linalg.norm(goal - start) > 1.5:
                break
        self.start = start
        self.goal = goal
        n_people = r.integers(self.min_people, self.max_people + 1) if self.randomize else self.max_people
        for _ in range(n_people):
            for _ in range(100):
                pp = self._sample_free(PERSON_R + 0.1)
                if pp is not None and np.linalg.norm(pp - start) > 1.2 and np.linalg.norm(pp - goal) > 0.8:
                    self.people.append(dict(pos=pp, wp=self._new_waypoint(pp), speed=r.uniform(0.3, 0.8)))
                    break
        self._place_mocap()
        yaw = r.uniform(-np.pi, np.pi)
        self.d.qpos[:3] = [start[0], start[1], 0.08]
        self.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        cq = self.cup_q
        if self.use_cup:
            self.d.qpos[cq:cq + 3] = [start[0], start[1], TOP_Z + 0.0605]
        else:
            self.d.qpos[cq:cq + 3] = [FAR, -FAR, 0.06]
        self.d.qpos[cq + 3:cq + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)
        for _ in range(20):                      # let the cup settle
            mujoco.mj_step(self.m, self.d)
        self.tour = list(options.get("tour", []))
        self.t, self.acc, self.peak_acc = 0, np.zeros(2), 0.0
        self.prev_action = np.zeros(2)
        self.prev_scan = None
        self.prev_v = self.body_vel()[0]
        self.delivered = 0
        o, self.prev_dist = self._obs()
        return o, {}

    def safety_layer(self, action):
        """LiDAR speed limit: never drive faster than what lets us stop at A_SAFE before the nearest
        obstacle in the direction of travel, v <= sqrt(2 * A_SAFE * (d - margin)). Uses the noisy scan
        (what a real robot would have). Turning rate is left untouched."""
        wl, wr = action * MAX_WHEEL * WHEEL_R
        v, w = (wl + wr) / 2, (wr - wl) / TRACK
        if abs(v) < 1e-6:
            return action, False
        limited = v < -V_REVERSE
        if limited:
            v = -V_REVERSE
        rel = (self.ray_ang + (0.0 if v > 0 else np.pi) + np.pi) % (2 * np.pi) - np.pi
        d = self.last_scan[np.abs(rel) < np.radians(50)].min() - ROBOT_R
        vmax = np.sqrt(2 * A_SAFE * max(0.0, d - SAFE_MARGIN))
        if abs(v) <= vmax and not limited:
            return action, False
        v = np.sign(v) * min(abs(v), vmax)
        out = np.array([v - w * TRACK / 2, v + w * TRACK / 2]) / WHEEL_R / MAX_WHEEL
        return np.clip(out, -1, 1), True

    def brake_to_stop(self, on_substep=None):
        """Bring the robot to rest at the safety layer's cup-safe deceleration, then let the cup settle.
        Returns True if the cup spilled while stopping."""
        v0, w0 = self.body_vel()
        v0 = v0[0]
        dt, t = self.m.opt.timestep, 0.0
        t_stop = max(abs(v0) / A_SAFE, abs(w0) / 4.0)
        while t < t_stop + 0.4:
            v = np.sign(v0) * max(0.0, abs(v0) - A_SAFE * t)
            w = np.sign(w0) * max(0.0, abs(w0) - 4.0 * t)
            self.d.ctrl[:] = np.clip(np.array([v - w * TRACK / 2, v + w * TRACK / 2]) / WHEEL_R, -MAX_WHEEL, MAX_WHEEL)
            mujoco.mj_step(self.m, self.d)
            if on_substep is not None:
                on_substep(0)
            t += dt
        return self.cup_state()[0]

    def step(self, action, on_substep=None):
        """on_substep(i) is called after every physics sub-step (used only for smooth video)."""
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        executed, self.shield_active = self.safety_layer(action) if self.shield else (action, False)
        self.intervention = float(np.abs(action - executed).sum())
        self.d.ctrl[:] = executed * MAX_WHEEL
        for i in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
            if on_substep is not None:
                on_substep(i)
        self._move_people()
        mujoco.mj_forward(self.m, self.d)
        v, _ = self.body_vel()
        self.acc = (v - self.prev_v) / DT
        self.prev_v = v
        amag = float(np.linalg.norm(self.acc))
        self.peak_acc = max(self.peak_acc, amag)
        self.t += 1
        o, dist = self._obs()
        self.prev_action = executed

        reward = 3.0 * (self.prev_dist - dist) - 0.01
        if self.accel_penalty:
            reward -= 0.06 * max(0.0, amag - 2.0)
        clr = self.person_clearance()
        if clr < 0.4:
            reward -= 0.05 * (0.4 - clr) / 0.4
        gap = self.true_scan.min() - ROBOT_R                # clearance to the nearest obstacle surface
        if gap < 0.2:
            reward -= 0.1 * (0.2 - gap) / 0.2
        reward -= 0.04 * max(0.0, -v[0]) / V_REVERSE        # prefer driving forward (predictable for people)
        reward -= 0.03 * self.intervention                  # relying on the safety layer is not free (no exploiting it)
        self.prev_dist = dist

        hit_f, hit_p = self.contacts()
        spilled, _, _ = self.cup_state()
        info = dict(success=False, collision=False, hit_person=False, spill=False, timeout=False,
                    delivered=self.delivered, peak_acc=self.peak_acc)
        terminated = truncated = False
        if hit_f or hit_p:
            reward -= 10.0
            terminated = info["collision"] = True
            info["hit_person"] = hit_p
        elif spilled:
            reward -= 5.0
            terminated = info["spill"] = True
        elif dist < GOAL_R and abs(v[0]) < STOP_V and self.final_stop and self.brake_to_stop(on_substep):
            reward -= 5.0                                   # arrived, but the cup fell while stopping
            terminated = info["spill"] = True
        elif dist < GOAL_R and abs(v[0]) < STOP_V:
            reward += 10.0
            self.delivered += 1
            info["delivered"] = self.delivered
            if self.tour:
                self.set_goal(self.tour.pop(0))
                o, self.prev_dist = self._obs()
            else:
                terminated = info["success"] = True
        elif self.t >= self.max_steps:
            truncated = info["timeout"] = True
        return o, float(reward), terminated, truncated, info
