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


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def gripper_quat(side: str, tilt_deg: float, roll_deg: float) -> np.ndarray:
    """Mocap orientation: first roll the gripper about its own pointing axis (world x at the
    start pose), then tilt it down about world y."""
    return quat_mul(side_quat(side, tilt_deg), quat_about_x(roll_deg))


@dataclass
class Waypoint:
    t: int
    pad: np.ndarray  # desired pad-point position (world, m)
    quat: np.ndarray  # mocap orientation command
    grip: float  # gripper command, same scale as the dataset (0 = fully closed, 1 = fully open)


@dataclass
class ExpertParams:
    """Defaults are set from the human demonstrations (lerobot/aloha_sim_transfer_cube_human),
    measured with forward kinematics on their joint states: see tracks/data/compare_datasets.py.
    Orientation commands are a few degrees beyond the measured human angles because the soft
    weld lets the gripper sag back by about that much."""

    # pick: right gripper tilted down, fingers closing sideways (human: 51 deg tilt, 4 deg roll)
    pick_tilt: float = 58.0
    grasp_dz: float = 0.017  # pad point above cube centre (human pad height at grasp ~0.030 m)
    # handover: right gripper levelled out and rolled so its fingers close top/bottom
    # (human: pointing along -x, ~75 deg roll); left horizontal, fingers sideways (human: -4 deg, ~14 deg roll)
    meet_tilt_right: float = -13.0  # commanded above level: the soft weld sags it ~12 deg (measured)
    meet_roll_right: float = 80.0
    meet_tilt_left: float = -16.0  # same sag compensation
    meet_roll_left: float = 14.0
    meet_right: tuple = (-0.004, 0.497, 0.288)  # human mean right pad point at the handover
    meet_left_dx: float = 0.006  # human pads are 1.6 cm apart, but here the cube sits deeper in the right gripper, so the left reaches further in (tested: -0.016 -> 11/20 hold, +0.006 -> 18/20)
    meet_noise: tuple = (0.014, 0.011, 0.011)  # human std of the meeting point
    # gripper command levels (human means: left 0.20 closed / 1.03 open, right 0.19 / 1.10)
    grip_closed: tuple = (0.20, 0.19)  # (left, right)
    grip_open: tuple = (1.03, 1.10)
    grip_noise: float = 0.03
    # timing (human means: right opens by ~72, closes 156, left closes 298, right releases 322)
    t_right_open: int = 72
    t_above: int = 118
    t_down: int = 142
    t_close: int = 156
    t_lift: int = 185
    t_meet: int = 262
    t_left_near: int = 245
    t_left_in: int = 286
    t_left_close: int = 298
    t_release: int = 322
    t_away: int = 360
    # timing spread: one shift for the whole pick and one more for the handover (human std ~30 steps)
    pick_shift_std: float = 22.0
    handover_shift_std: float = 15.0
    event_jitter: int = 4
    pos_noise: float = 0.004  # m, jitter of approach / retreat points
    away_dz: float = 0.0  # tested 0.03 -> 35/40 hold, 0.0 -> 36/40; m, rise of the right gripper while it backs away after releasing
    gain: float = 0.15  # per-step gain of the pad-position correction


def noiseless_params() -> ExpertParams:
    return ExpertParams(meet_noise=(0.0, 0.0, 0.0), grip_noise=0.0, pick_shift_std=0.0, handover_shift_std=0.0,
                        event_jitter=0, pos_noise=0.0)


class ScriptedTransferExpert:
    """Waypoint plan in pad-point space. `__call__(t)` gives desired (left, right) waypoints."""

    def __init__(self, cube_xyz, start_pad_left, start_pad_right, start_q_left, start_q_right,
                 rng: np.random.Generator | None = None, params: ExpertParams | None = None):
        self.p = p = params or ExpertParams()
        rng = rng or np.random.default_rng(0)

        def jit(v, s=p.pos_noise):
            return np.asarray(v, float) + (rng.normal(0, s, 3) if s > 0 else 0)

        def ev():
            return int(rng.integers(-p.event_jitter, p.event_jitter + 1)) if p.event_jitter else 0

        # --- timing: shared shifts + per-event jitter, then enforce order and minimum gaps
        pick = int(np.clip(rng.normal(0, p.pick_shift_std), -45, 45)) if p.pick_shift_std else 0
        hand = int(np.clip(rng.normal(0, p.handover_shift_std), -30, 30)) if p.handover_shift_std else 0
        T = dict(r_open=p.t_right_open + pick // 2 + ev(), above=p.t_above + pick + ev(), down=p.t_down + pick + ev(),
                 close=p.t_close + pick + ev(), lift=p.t_lift + pick + ev(), meet=p.t_meet + pick + hand + ev(),
                 l_near=p.t_left_near + pick + hand + ev(), l_in=p.t_left_in + pick + hand + ev(),
                 l_close=p.t_left_close + pick + hand + ev(), release=p.t_release + pick + hand + ev(),
                 away=p.t_away + pick + hand + ev())
        T["r_open"] = max(T["r_open"], 30)
        T["above"] = max(T["above"], T["r_open"] + 25)
        T["down"] = max(T["down"], T["above"] + 18)
        T["close"] = max(T["close"], T["down"] + 10)
        T["lift"] = max(T["lift"], T["close"] + 22)
        T["meet"] = max(T["meet"], T["lift"] + 60)
        T["l_near"] = max(T["l_near"], T["meet"] - 30)
        T["l_in"] = max(T["l_in"], T["meet"] + 15, T["l_near"] + 30)
        T["l_close"] = max(T["l_close"], T["l_in"] + 10)
        T["release"] = max(T["release"], T["l_close"] + 18)
        T["away"] = max(T["away"], T["release"] + 25)
        over = T["away"] - (EPISODE_LEN - 20)  # keep at least 20 steps of holding at the end
        if over > 0:
            for k in ("meet", "l_near", "l_in", "l_close", "release", "away"):
                T[k] -= over
        self.T = T

        # --- geometry
        cube = np.array([cube_xyz[0], cube_xyz[1], CUBE_REST_Z])
        meet_r = np.asarray(p.meet_right, float) + (rng.normal(0, 1, 3) * p.meet_noise)
        meet_l = meet_r + [p.meet_left_dx, 0, 0]
        gl_c, gr_c = (np.asarray(p.grip_closed) + (rng.normal(0, p.grip_noise, 2) if p.grip_noise else 0))
        gl_o, gr_o = (np.asarray(p.grip_open) + (rng.normal(0, p.grip_noise, 2) if p.grip_noise else 0))

        q_pick = gripper_quat("right", p.pick_tilt, 0.0)
        q_meet_r = gripper_quat("right", p.meet_tilt_right, p.meet_roll_right)
        q_meet_l = gripper_quat("left", p.meet_tilt_left, p.meet_roll_left)
        ax_pick = gripper_axis("right", p.pick_tilt)
        ax_meet_r = gripper_axis("right", p.meet_tilt_right)
        ax_l = gripper_axis("left", p.meet_tilt_left)

        grasp = cube + [0, 0, p.grasp_dz]
        r_start, l_start = np.asarray(start_pad_right), np.asarray(start_pad_left)
        r_away = meet_r - 0.10 * ax_meet_r + [0, 0, p.away_dz]
        self.right = [
            Waypoint(0, r_start, np.asarray(start_q_right), gr_c),
            Waypoint(T["r_open"], jit(r_start + 0.4 * (grasp - 0.08 * ax_pick - r_start)), q_pick, gr_o),
            Waypoint(T["above"], jit(grasp - 0.08 * ax_pick), q_pick, gr_o),
            Waypoint(T["down"], grasp, q_pick, gr_o),
            Waypoint(T["close"], grasp, q_pick, gr_c),
            Waypoint(T["lift"], grasp + [0, 0, 0.05], q_pick, gr_c),
            Waypoint(T["meet"], meet_r, q_meet_r, gr_c),
            Waypoint(T["release"], meet_r, q_meet_r, gr_c),
            Waypoint(T["release"] + 10, meet_r, q_meet_r, gr_o),
            Waypoint(T["away"], jit(r_away), q_meet_r, gr_o),
            Waypoint(EPISODE_LEN, r_away, q_meet_r, gr_o),
        ]
        self.left = [
            Waypoint(0, l_start, np.asarray(start_q_left), gl_c),
            Waypoint(max(T["l_near"] - 60, 1), jit(l_start + 0.3 * (meet_l - 0.10 * ax_l - l_start)), q_meet_l, gl_o),
            Waypoint(T["l_near"], jit(meet_l - 0.09 * ax_l), q_meet_l, gl_o),
            Waypoint(T["l_in"], meet_l, q_meet_l, gl_o),
            Waypoint(T["l_close"], meet_l, q_meet_l, gl_c),
            Waypoint(EPISODE_LEN, meet_l + [-0.03, 0, 0.0], q_meet_l, gl_c),
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
