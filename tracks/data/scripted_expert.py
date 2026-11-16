"""Track C2: scripted expert for the ALOHA two-arm cube transfer.

The expert drives the end-effector version of the task (`gym_aloha/tasks/sim_end_effector.py`):
each arm's gripper link is welded to a mocap body, so the expert only has to say where each
gripper should be (position + orientation) and how open it is. MuJoCo solves the arm joints.

That weld is soft and the arm joints have large friction, so an open-loop mocap command
leaves the gripper several centimetres off target. The expert therefore works closed loop:
it plans where the finger "pad point" should be, measures where it is each step, and shifts
the mocap command by the accumulated error (see `rollout_ee`). Tracking error is a few mm.

Plan, per episode (400 steps at 50 Hz, the same length as the human demonstrations):
  1. right gripper, tilted down, moves above the cube and opens;
  2. descends, closes on the cube, lifts it to a meeting point above the middle of the table;
  3. left gripper, rolled 90 degrees so its fingers close top/bottom (the right gripper's
     fingers hold the cube's sides), comes in horizontally and closes on the cube;
  4. right gripper opens and retreats; left keeps holding the cube off the table.

Variation ("noise injected so demonstrations vary"): approach/retreat waypoints get Gaussian
jitter, the meeting point is randomised, and key frame times shift by a few steps.
All randomness comes from one seed, so every demonstration can be regenerated exactly.

The joint angles the arm went through are then replayed in the ordinary joint-space task
(`replay_joint`), which is the simulator the policy is trained and evaluated in.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gym_aloha.constants import ASSETS_DIR, DT
from gym_aloha.tasks.sim_end_effector import TransferCubeEndEffectorTask

EPISODE_LEN = 400  # steps, matches lerobot/aloha_sim_transfer_cube_human
# m, gripper_link -> "pad point" along the gripper axis. Fingertips are 0.153 m out (measured
# from the finger mesh); the pad point sits ~2 cm short of the tips, where the fingers close on
# the cube. The mocap weld's own anchor is 0.1347 m out, so this is also roughly what it tracks.
PAD_REACH = 0.13
CUBE_REST_Z = 0.02  # cube centre once it settles on the table (spawned at 0.05, half-size 0.02)
CUBE_X_RANGE = (0.0, 0.2)  # gym_aloha/utils.py sample_box_pose
CUBE_Y_RANGE = (0.4, 0.6)


# ----------------------------------------------------------------------------- geometry helpers
def quat_about_y(deg: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for a rotation of `deg` degrees about the world y axis."""
    h = np.deg2rad(deg) / 2
    return np.array([np.cos(h), 0.0, np.sin(h), 0.0])


def slerp(q0: np.ndarray, q1: np.ndarray, f: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / np.sin(th)


def gripper_axis(side: str, tilt_deg: float) -> np.ndarray:
    """World direction the fingers point in after tilting the gripper down by `tilt_deg`.

    Mocap quaternion identity = start pose: left gripper points +x, right gripper points -x.
    The mocap weld applies the mocap rotation in the world frame, so a rotation about world y
    tilts the gripper down (left: +deg, right: -deg).
    """
    t = np.deg2rad(tilt_deg)
    return np.array([np.cos(t), 0, -np.sin(t)]) if side == "left" else np.array([-np.cos(t), 0, -np.sin(t)])


def side_quat(side: str, tilt_deg: float) -> np.ndarray:
    return quat_about_y(tilt_deg if side == "left" else -tilt_deg)


def pad_point(physics, side: str) -> np.ndarray:
    """Actual pad point of one gripper: link position + PAD_REACH along the link's x axis
    (the link's local x axis is the direction the fingers point)."""
    name = f"vx300s_{side}/gripper_link"
    return physics.named.data.xpos[name] + PAD_REACH * physics.named.data.xmat[name].reshape(3, 3)[:, 0]


# ----------------------------------------------------------------------------- environment
class _FixedCubeEETask(TransferCubeEndEffectorTask):
    """End-effector task with a caller-chosen cube pose and no camera rendering (fast)."""

    def __init__(self, cube_pose: np.ndarray):
        super().__init__()
        self.cube_pose = np.asarray(cube_pose, dtype=float)

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        i = physics.model.name2id("red_box_joint", "joint")
        np.copyto(physics.data.qpos[i : i + 7], self.cube_pose)
        super(TransferCubeEndEffectorTask, self).initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["gripper_ctrl"] = physics.data.ctrl.copy()
        obs["env_state"] = self.get_env_state(physics)
        return obs


def make_ee_env(cube_pose: np.ndarray) -> control.Environment:
    physics = mujoco.Physics.from_xml_path(str(ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"))
    return control.Environment(
        physics, _FixedCubeEETask(cube_pose), float("inf"), control_timestep=DT, n_sub_steps=None, flat_observation=False
    )


def sample_cube_pose(rng: np.random.Generator, x_range=CUBE_X_RANGE, y_range=CUBE_Y_RANGE) -> np.ndarray:
    """Same distribution as gym_aloha's sample_box_pose, but driven by our own generator."""
    xyz = [rng.uniform(*x_range), rng.uniform(*y_range), 0.05]
    return np.array([*xyz, 1.0, 0.0, 0.0, 0.0])


# ----------------------------------------------------------------------------- the expert
def quat_about_x(deg: float) -> np.ndarray:
    h = np.deg2rad(deg) / 2
    return np.array([np.cos(h), np.sin(h), 0.0, 0.0])


@dataclass
class Waypoint:
    t: int
    pad: np.ndarray  # desired pad-point position (world, m)
    quat: np.ndarray  # mocap orientation command
    grip: float  # normalised gripper command: 0 closed, 1 open


@dataclass
class ExpertParams:
    pos_noise: float = 0.004  # m, std of per-waypoint position jitter (approach / retreat points)
    time_noise: int = 4  # steps, max shift of each key frame time
    meet_noise: float = 0.02  # m, std of meeting-point jitter (x, y; half that in z)
    right_tilt: float = 60.0  # deg, commanded downward tilt of the picking (right) gripper
    left_roll: float = 90.0  # deg, receiving (left) gripper rolled so its fingers close top/bottom
    meet_xyz: tuple = (0.0, 0.5, 0.25)  # nominal handover point (cube centre)
    grasp_dz: float = 0.01  # m, pad point above cube centre at the grasp
    left_dx: float = -0.005  # m, left pad point relative to the cube centre at the handover
    gain: float = 0.15  # per-step gain of the pad-position correction
    extra: dict = field(default_factory=dict)


class ScriptedTransferExpert:
    """Waypoint plan in pad-point space. `__call__(t)` gives desired (left, right) waypoints."""

    def __init__(self, cube_xyz, start_pad_left, start_pad_right, start_q_left, start_q_right,
                 rng: np.random.Generator | None = None, params: ExpertParams | None = None):
        self.p = p = params or ExpertParams()
        rng = rng or np.random.default_rng(0)

        def jit(v, s=p.pos_noise):
            return np.asarray(v, float) + (rng.normal(0, s, 3) if s > 0 else 0)

        def dt():
            return int(rng.integers(-p.time_noise, p.time_noise + 1)) if p.time_noise else 0

        cube = np.array([cube_xyz[0], cube_xyz[1], CUBE_REST_Z])
        meet = np.asarray(p.meet_xyz, float)
        if p.meet_noise > 0:
            meet = meet + rng.normal(0, p.meet_noise, 3) * [1, 1, 0.5]

        rq = side_quat("right", p.right_tilt)
        lq = quat_about_x(p.left_roll)
        ax_r = gripper_axis("right", p.right_tilt)
        ax_l = gripper_axis("left", 0.0)

        T = {k: v + dt() for k, v in dict(above=80, down=120, close=150, lift=175, meet=260,
                                          l_near=250, l_in=295, l_close=315, r_open=335, r_away=370).items()}
        T["down"] = max(T["down"], T["above"] + 25)
        T["close"] = max(T["close"], T["down"] + 20)
        T["lift"] = max(T["lift"], T["close"] + 20)
        T["meet"] = max(T["meet"], T["lift"] + 60)
        T["l_in"] = max(T["l_in"], T["meet"] + 15, T["l_near"] + 30)
        T["l_close"] = max(T["l_close"], T["l_in"] + 15)
        T["r_open"] = max(T["r_open"], T["l_close"] + 15)
        T["r_away"] = max(T["r_away"], T["r_open"] + 20)
        self.T = T

        grasp = cube + [0, 0, p.grasp_dz]
        left_pad = meet + [p.left_dx, 0, 0]
        r_away = meet - 0.10 * ax_r + [0, 0, 0.04]
        self.right = [
            Waypoint(0, np.asarray(start_pad_right), np.asarray(start_q_right), 0.0),
            Waypoint(T["above"], jit(grasp - 0.08 * ax_r), rq, 1.0),
            Waypoint(T["down"], grasp, rq, 1.0),
            Waypoint(T["close"], grasp, rq, 0.0),
            Waypoint(T["lift"], grasp + [0, 0, 0.05], rq, 0.0),
            Waypoint(T["meet"], meet, rq, 0.0),
            Waypoint(T["r_open"], meet, rq, 0.0),
            Waypoint(T["r_open"] + 12, meet, rq, 1.0),
            Waypoint(T["r_away"], jit(r_away), rq, 1.0),
            Waypoint(EPISODE_LEN, r_away, rq, 1.0),
        ]
        self.left = [
            Waypoint(0, np.asarray(start_pad_left), np.asarray(start_q_left), 0.0),
            Waypoint(T["l_near"], jit(left_pad - 0.09 * ax_l), lq, 1.0),
            Waypoint(T["l_in"], left_pad, lq, 1.0),
            Waypoint(T["l_close"], left_pad, lq, 0.0),
            Waypoint(EPISODE_LEN, left_pad + [-0.03, 0, 0.02], lq, 0.0),
        ]

    @staticmethod
    def _interp(wps: list[Waypoint], t: int):
        for a, b in zip(wps[:-1], wps[1:]):
            if a.t <= t <= b.t:
                f = (t - a.t) / max(b.t - a.t, 1)
                return a.pad + f * (b.pad - a.pad), slerp(a.quat, b.quat, f), a.grip + f * (b.grip - a.grip)
        w = wps[-1]
        return w.pad, w.quat, w.grip

    def __call__(self, t: int):
        return self._interp(self.left, t), self._interp(self.right, t)


# ----------------------------------------------------------------------------- rollout
def rollout_ee(cube_pose: np.ndarray, rng: np.random.Generator | None, params: ExpertParams | None = None,
               frame_cb=None):
    """Run the expert in the end-effector sim, closed loop on the measured pad points.

    The weld between mocap and gripper is soft and the joints have large friction, so the
    gripper settles centimetres away from a commanded mocap pose. Each step the expert
    measures where its pad point really is and shifts the mocap command by an accumulated
    correction (an integral term), so the fingers end up where the plan says.

    Returns the joint trajectory to replay in the joint-space sim (`joint_traj`, 400x14,
    gripper entries = the expert's normalised gripper commands), per-step reward, and success.
    """
    params = params or ExpertParams()
    env = make_ee_env(cube_pose)
    ts = env.reset()
    ph = env.physics
    sides = ("left", "right")
    start_pad = {s: pad_point(ph, s) for s in sides}
    start_q = {s: ph.data.mocap_quat[i].copy() for i, s in enumerate(sides)}
    expert = ScriptedTransferExpert(cube_pose[:3], start_pad["left"], start_pad["right"],
                                    start_q["left"], start_q["right"], rng, params)
    # initial correction = offset between mocap and pad point at rest
    bias = {s: ph.data.mocap_pos[i] - start_pad[s] for i, s in enumerate(sides)}

    qpos, rewards, grip_cmd = [ts.observation["qpos"].copy()], [], []
    for t in range(EPISODE_LEN):
        plan = dict(zip(sides, expert(t)))
        action = []
        for s in sides:
            pad, quat, grip = plan[s]
            bias[s] = bias[s] + params.gain * (pad - pad_point(ph, s))
            action += [pad + bias[s], quat, [grip]]
        action = np.concatenate(action)
        ts = env.step(action)
        qpos.append(ts.observation["qpos"].copy())
        rewards.append(float(ts.reward))
        grip_cmd.append([plan["left"][2], plan["right"][2]])
        if frame_cb is not None:
            frame_cb(t, ph, plan)

    qpos = np.array(qpos)
    grip_cmd = np.array(grip_cmd)
    joint_traj = qpos[1:].copy()
    joint_traj[:, 6] = grip_cmd[:, 0]
    joint_traj[:, 13] = grip_cmd[:, 1]
    rewards = np.array(rewards)
    return {"joint_traj": joint_traj, "rewards": rewards, "max_stage": int(rewards.max()),
            "success": bool(rewards.max() == 4), "cube_pose": np.asarray(cube_pose), "T": expert.T}


# ----------------------------------------------------------------------------- joint-space replay
TASK_TEXT = "Pick up the cube with the right arm and transfer it to the left arm."  # same as the human set


def make_joint_env(render_top: bool):
    """The normal joint-position-controlled task the policy is trained and evaluated in."""
    from gym_aloha.tasks import sim as joint_sim

    class _Task(joint_sim.TransferCubeTask):
        def get_observation(self, physics):
            obs = collections.OrderedDict()
            obs["qpos"] = self.get_qpos(physics)
            if render_top:
                obs["top"] = physics.render(height=480, width=640, camera_id="top")
            return obs

    physics = mujoco.Physics.from_xml_path(str(ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"))
    return control.Environment(physics, _Task(), float("inf"), control_timestep=DT, n_sub_steps=None,
                               flat_observation=False)


def replay_joint(joint_traj: np.ndarray, cube_pose: np.ndarray, render_top: bool = False, on_frame=None):
    """Replay a joint trajectory in the joint-space sim with the same cube pose.

    `on_frame(t, obs, action)` is called with the observation the policy would see *before*
    action t is applied, so (obs_t, action_t) pairs are exactly what a dataset frame holds.
    Returns per-step rewards.
    """
    from gym_aloha.tasks import sim as joint_sim

    joint_sim.BOX_POSE[0] = np.asarray(cube_pose)  # read by TransferCubeTask.initialize_episode
    env = make_joint_env(render_top)
    ts = env.reset()
    rewards = []
    for t, action in enumerate(joint_traj):
        if on_frame is not None:
            on_frame(t, ts.observation, action)
        ts = env.step(action)
        rewards.append(float(ts.reward))
    return np.array(rewards)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
