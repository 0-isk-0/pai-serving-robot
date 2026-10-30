"""MuJoCo 2-wheel robot navigating a randomized obstacle field with LiDAR.

The robot is a real physics model (wheels, friction, contacts), not a grid world.
Observation: 24 LiDAR rays + goal (distance, bearing) + forward/yaw speed.
Action: left / right wheel velocity command in [-1, 1].
"""
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ARENA = 3.0          # half size of the square arena [m]
N_RAYS = 24
MAX_RANGE = 3.0
N_OBS = 6
N_BLOCK = 3          # obstacles forced onto the start->goal line
OBS_R = 0.25
ROBOT_R = 0.17
WHEEL_R = 0.05
MAX_WHEEL = 16.0     # rad/s  (~0.8 m/s)
GOAL_R = 0.25


def build_xml():
    obs = ""
    for i in range(N_OBS):
        # mocap bodies: world-body geoms are static, so moving them at runtime would not move their collision shape
        obs += (f'<body name="ob{i}" mocap="true" pos="{10 + i} 10 0.2"><geom name="obs{i}" type="cylinder" '
                f'size="{OBS_R} 0.2" rgba="0.85 0.35 0.25 1" group="0"/></body>\n')
    walls = ""
    for n, (px, py, sx, sy) in {"wN": (0, ARENA, ARENA, .05), "wS": (0, -ARENA, ARENA, .05),
                                "wE": (ARENA, 0, .05, ARENA), "wW": (-ARENA, 0, .05, ARENA)}.items():
        walls += f'<geom name="{n}" type="box" size="{sx} {sy} 0.2" pos="{px} {py} 0.2" rgba="0.3 0.3 0.35 1" group="0"/>\n'
    return f"""
<mujoco>
  <option timestep="0.005" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1=".88 .9 .92" rgb2=".78 .82 .85"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse=".9 .9 .9"/>
    <geom name="floor" type="plane" size="{ARENA} {ARENA} .1" material="grid" group="2"/>
    {walls}
    {obs}
    <geom name="goal" type="cylinder" size="{GOAL_R} 0.005" pos="2 2 0.005" rgba="0.2 0.8 0.3 0.8"
          contype="0" conaffinity="0" group="2"/>
    <body name="chassis" pos="0 0 0.08">
      <freejoint name="root"/>
      <geom name="body" type="box" size="0.13 0.09 0.03" mass="1.5" rgba="0.2 0.45 0.9 1" group="1"/>
      <geom name="nose" type="box" size="0.02 0.05 0.03" pos="0.14 0 0" mass="0.01" rgba="1 1 1 1" group="1"
            contype="0" conaffinity="0"/>
      <geom name="caster_f" type="sphere" size="0.0235" pos="0.10 0 -0.055" mass="0.05" friction="0 0 0" priority="1" group="1"/>
      <geom name="caster_b" type="sphere" size="0.0235" pos="-0.10 0 -0.055" mass="0.05" friction="0 0 0" priority="1" group="1"/>
      <body name="wl" pos="0 0.11 -0.03">
        <joint name="jl" type="hinge" axis="0 1 0" damping="0.01" armature="0.004"/>
        <geom type="cylinder" size="{WHEEL_R} 0.015" euler="90 0 0" mass="0.1" friction="1.2 .005 .0001" rgba=".1 .1 .1 1" group="1"/>
      </body>
      <body name="wr" pos="0 -0.11 -0.03">
        <joint name="jr" type="hinge" axis="0 1 0" damping="0.01" armature="0.004"/>
        <geom type="cylinder" size="{WHEEL_R} 0.015" euler="90 0 0" mass="0.1" friction="1.2 .005 .0001" rgba=".1 .1 .1 1" group="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="al" joint="jl" kv="0.5" ctrlrange="-{MAX_WHEEL} {MAX_WHEEL}"/>
    <velocity name="ar" joint="jr" kv="0.5" ctrlrange="-{MAX_WHEEL} {MAX_WHEEL}"/>
  </actuator>
</mujoco>"""


class NavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, n_obstacles=N_OBS, max_steps=500, seed=None):
        self.m = mujoco.MjModel.from_xml_string(build_xml())
        self.d = mujoco.MjData(self.m)
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.substeps = 10  # control dt = 0.05 s
        g = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.obs_ids = [g(f"obs{i}") for i in range(N_OBS)]
        self.static_ids = set(self.obs_ids + [g(n) for n in ("wN", "wS", "wE", "wW")])
        self.robot_ids = {g(n) for n in ("body", "caster_f", "caster_b")}
        self.goal_id = g("goal")
        self.mocap = [self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"ob{i}")]
                      for i in range(N_OBS)]
        self.chassis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # rays see group 0 only
        self.ray_angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
        self.observation_space = spaces.Box(-np.inf, np.inf, (N_RAYS + 4,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(seed)
        self._renderer = None

    # ---- helpers -------------------------------------------------------
    def _pose(self):
        p = self.d.qpos[:2].copy()
        q = self.d.qpos[3:7]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return p, yaw

    def _lidar(self, p, yaw):
        out = np.empty(N_RAYS)
        origin = np.array([p[0], p[1], 0.12])
        gid = np.zeros(1, dtype=np.int32)
        for i, a in enumerate(self.ray_angles):
            ang = yaw + a
            vec = np.array([np.cos(ang), np.sin(ang), 0.0])
            dist = mujoco.mj_ray(self.m, self.d, origin, vec, self.geomgroup, 1, self.chassis, gid)
            out[i] = MAX_RANGE if dist < 0 else min(dist, MAX_RANGE)
        return out

    def _obs(self):
        p, yaw = self._pose()
        to_goal = self.goal - p
        dist = np.linalg.norm(to_goal)
        bearing = np.arctan2(to_goal[1], to_goal[0]) - yaw
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        v = self.d.qvel[:2]
        fwd = float(v @ np.array([np.cos(yaw), np.sin(yaw)]))
        o = np.concatenate([self._lidar(p, yaw) / MAX_RANGE,
                            [dist / (2 * ARENA), bearing / np.pi, fwd, self.d.qvel[5] / 3.0]])
        return o.astype(np.float32), dist

    def _collided(self):
        for c in self.d.contact[:self.d.ncon]:
            a, b = c.geom1, c.geom2
            if (a in self.robot_ids and b in self.static_ids) or (b in self.robot_ids and a in self.static_ids):
                return True
        return False

    # ---- gym API -------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        r = self.rng
        while True:  # sample start / goal / obstacles with clearances
            start = r.uniform(-2.4, 2.4, 2)
            goal = r.uniform(-2.4, 2.4, 2)
            if np.linalg.norm(start - goal) > 3.0:
                break
        pts = []
        for i in range(N_OBS):
            if i >= self.n_obstacles:
                self.d.mocap_pos[self.mocap[i]] = [10 + i, 10, 0.2]
                continue
            for _ in range(200):
                if i < N_BLOCK:  # force obstacles onto the straight start->goal line so the path is blocked
                    c = start + (goal - start) * r.uniform(0.25, 0.75) + r.normal(0, 0.2, 2)
                else:
                    c = r.uniform(-2.5, 2.5, 2)
                c = np.clip(c, -2.5, 2.5)
                if (np.linalg.norm(c - start) > 0.75 and np.linalg.norm(c - goal) > 0.7
                        and all(np.linalg.norm(c - q) > 0.9 for q in pts)):
                    break
            pts.append(c)
            self.d.mocap_pos[self.mocap[i]] = [c[0], c[1], 0.2]
        self.goal = goal
        self.m.geom_pos[self.goal_id] = [goal[0], goal[1], 0.005]
        yaw = r.uniform(-np.pi, np.pi)
        self.d.qpos[:3] = [start[0], start[1], 0.08]
        self.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, self.d)
        self.t = 0
        o, self.prev_dist = self._obs()
        return o, {}

    def step(self, action):
        a = np.clip(action, -1, 1) * MAX_WHEEL
        self.d.ctrl[:] = a
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        o, dist = self._obs()
        reward = 4.0 * (self.prev_dist - dist) - 0.01
        self.prev_dist = dist
        terminated = truncated = False
        info = {"success": False, "collision": False}
        if dist < GOAL_R:
            reward += 10.0
            terminated = True
            info["success"] = True
        elif self._collided():
            reward -= 5.0
            terminated = True
            info["collision"] = True
        elif self.t >= self.max_steps:
            truncated = True
        return o, float(reward), terminated, truncated, info

    def render(self, width=640, height=480, camera="top"):
        if self._renderer is None or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self.m, height, width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 0, 0]
        cam.distance, cam.azimuth, cam.elevation = 9.0, 90, -75
        self._renderer.update_scene(self.d, cam)
        return self._renderer.render()
