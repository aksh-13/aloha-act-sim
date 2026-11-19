"""Perturbed variants of the gym-aloha transfer-cube environment (issue B2).

A `Perturbation` is a plain record of what differs from the nominal scene. `VARIANTS` maps a
name to one of them, so the harness can select a variant by name. Every field left at its
default leaves the simulator untouched, and `nominal` changes nothing at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.rl import control
from gym_aloha.constants import ASSETS_DIR, DT
from gym_aloha.env import AlohaEnv
from gym_aloha.tasks import sim as aloha_sim

# Training range of the cube start position, from gym_aloha.utils.sample_box_pose.
CUBE_X_RANGE = (0.0, 0.2)
CUBE_Y_RANGE = (0.4, 0.6)
CUBE_START_Z = 0.05
CUBE_REST_Z = 0.02  # centre height of the 4 cm cube lying on the table

RIGHT_FINGERS = ("vx300s_right/10_left_gripper_finger", "vx300s_right/10_right_gripper_finger")
LEFT_FINGERS = ("vx300s_left/10_left_gripper_finger", "vx300s_left/10_right_gripper_finger")

STRIPES_ASSET = (
    '<texture name="table_stripes" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" '
    'rgb2="0.6 0.6 0.6" width="256" height="256"/>'
    '<material name="table_stripes" texture="table_stripes" texrepeat="6 6" texuniform="true"/>'
)


@dataclass(frozen=True)
class Perturbation:
    family: str = "nominal"
    severity: int = 0  # 0 = nominal, 1..3 = graded steps within a family
    description: str = "Unmodified gym-aloha transfer-cube scene."

    # Cube start: sampled from the band lying between `lo` and `hi` metres outside the
    # training rectangle. None keeps the training range.
    cube_band: tuple[float, float] | None = None

    # Camera `top`, expressed in the camera's own frame (x right, y up in the image, z towards
    # the viewer). Any non-zero value freezes the camera orientation first, so a shift is a pure
    # translation rather than a re-aim at the table.
    cam_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)  # metres
    cam_roll_deg: float = 0.0  # about the optical axis
    cam_tilt_deg: float = 0.0  # about the image x axis

    # Appearance.
    cube_rgba: tuple[float, float, float, float] | None = None
    table_rgba: tuple[float, float, float, float] | None = None
    table_stripes: bool = False
    light_scale: float = 1.0  # multiplies every light and the headlight

    # Physics.
    cube_mass_scale: float = 1.0
    cube_friction_scale: float = 1.0  # sliding friction of every cube contact

    # External push on the cube: `push_force` newtons, horizontal, random direction per episode,
    # held for `push_steps` control steps. Starts at control step `push_at_step`, or, when
    # `push_on_lift` is set, `push_at_step` steps after the cube is first lifted.
    push_force: float = 0.0
    push_steps: int = 5
    push_at_step: int = 30
    push_on_lift: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _p(family: str, severity: int, description: str, **kwargs) -> Perturbation:
    return Perturbation(family=family, severity=severity, description=description, **kwargs)


VARIANTS: dict[str, Perturbation] = {
    "nominal": Perturbation(),
    # Cube start outside the training range.
    "cube_band_1": _p("cube_band", 1, "Cube starts 0-1.5 cm outside the training range.", cube_band=(0.0, 0.015)),
    "cube_band_2": _p("cube_band", 2, "Cube starts 1.5-3 cm outside the training range.", cube_band=(0.015, 0.03)),
    "cube_band_3": _p("cube_band", 3, "Cube starts 3-6 cm outside the training range.", cube_band=(0.03, 0.06)),
    # Camera translated sideways.
    "cam_shift_1": _p("cam_shift", 1, "Top camera moved 1 cm sideways.", cam_shift=(0.01, 0.0, 0.0)),
    "cam_shift_2": _p("cam_shift", 2, "Top camera moved 3 cm sideways.", cam_shift=(0.03, 0.0, 0.0)),
    "cam_shift_3": _p("cam_shift", 3, "Top camera moved 6 cm sideways.", cam_shift=(0.06, 0.0, 0.0)),
    # Camera rolled about its optical axis.
    "cam_roll_1": _p("cam_roll", 1, "Top camera rolled 1 degree.", cam_roll_deg=1.0),
    "cam_roll_2": _p("cam_roll", 2, "Top camera rolled 3 degrees.", cam_roll_deg=3.0),
    "cam_roll_3": _p("cam_roll", 3, "Top camera rolled 6 degrees.", cam_roll_deg=6.0),
    # Camera tilted off vertical.
    "cam_tilt_1": _p("cam_tilt", 1, "Top camera tilted 1 degree.", cam_tilt_deg=1.0),
    "cam_tilt_2": _p("cam_tilt", 2, "Top camera tilted 3 degrees.", cam_tilt_deg=3.0),
    "cam_tilt_3": _p("cam_tilt", 3, "Top camera tilted 6 degrees.", cam_tilt_deg=6.0),
    # Cube colour, stepping away from red around the hue circle.
    "cube_colour_1": _p("cube_colour", 1, "Orange cube.", cube_rgba=(1.0, 0.5, 0.0, 1.0)),
    "cube_colour_2": _p("cube_colour", 2, "Green cube.", cube_rgba=(0.0, 0.8, 0.0, 1.0)),
    "cube_colour_3": _p("cube_colour", 3, "Blue cube.", cube_rgba=(0.0, 0.2, 1.0, 1.0)),
    # Table appearance.
    "table_1": _p("table", 1, "Table lightened to mid grey.", table_rgba=(0.4, 0.4, 0.4, 1.0)),
    "table_2": _p("table", 2, "Table changed to wood brown.", table_rgba=(0.55, 0.38, 0.22, 1.0)),
    "table_3": _p("table", 3, "Table given light and dark grey stripes.", table_stripes=True),
    # Lighting.
    "light_1": _p("light", 1, "Lighting at 70%.", light_scale=0.7),
    "light_2": _p("light", 2, "Lighting at 40%.", light_scale=0.4),
    "light_3": _p("light", 3, "Lighting at 160%.", light_scale=1.6),
    # Cube mass (nominal 50 g).
    "mass_1": _p("mass", 1, "Cube mass x2 (100 g).", cube_mass_scale=2.0),
    "mass_2": _p("mass", 2, "Cube mass x4 (200 g).", cube_mass_scale=4.0),
    "mass_3": _p("mass", 3, "Cube mass x8 (400 g).", cube_mass_scale=8.0),
    # Cube friction (nominal sliding coefficient 1.0).
    "friction_1": _p("friction", 1, "Cube friction x0.5.", cube_friction_scale=0.5),
    "friction_2": _p("friction", 2, "Cube friction x0.25.", cube_friction_scale=0.25),
    "friction_3": _p("friction", 3, "Cube friction x0.1.", cube_friction_scale=0.1),
    # Push while the cube is still on the table, before the gripper arrives.
    "push_table_1": _p("push_table", 1, "0.5 N push slides the resting cube about 1 cm at step 30.", push_force=0.5),
    "push_table_2": _p("push_table", 2, "0.65 N push slides the resting cube about 3 cm at step 30.", push_force=0.65),
    "push_table_3": _p("push_table", 3, "0.75 N push slides the resting cube about 6 cm at step 30.", push_force=0.75),
    # Push while the right gripper is carrying the cube.
    "push_carry_1": _p("push_carry", 1, "2 N push 15 steps after lift.", push_force=2.0, push_on_lift=True, push_at_step=15),
    "push_carry_2": _p("push_carry", 2, "5 N push 15 steps after lift.", push_force=5.0, push_on_lift=True, push_at_step=15),
    "push_carry_3": _p("push_carry", 3, "10 N push 15 steps after lift.", push_force=10.0, push_on_lift=True, push_at_step=15),
}


def sample_cube_xy(rng: np.random.RandomState, band: tuple[float, float] | None) -> tuple[float, float]:
    """Cube start position. With no band this is the training distribution."""
    if band is None:
        # Same draw order as gym_aloha.utils.sample_box_pose, so equal seeds give equal poses.
        x, y, _ = rng.uniform([CUBE_X_RANGE[0], CUBE_Y_RANGE[0], 0.0], [CUBE_X_RANGE[1], CUBE_Y_RANGE[1], 0.0])
        return float(x), float(y)
    lo, hi = band
    while True:
        x = rng.uniform(CUBE_X_RANGE[0] - hi, CUBE_X_RANGE[1] + hi)
        y = rng.uniform(CUBE_Y_RANGE[0] - hi, CUBE_Y_RANGE[1] + hi)
        if distance_outside_training_range(x, y) > lo:
            return float(x), float(y)


def distance_outside_training_range(x: float, y: float) -> float:
    """Largest per-axis distance (metres) by which (x, y) lies outside the training rectangle."""
    dx = max(CUBE_X_RANGE[0] - x, x - CUBE_X_RANGE[1], 0.0)
    dy = max(CUBE_Y_RANGE[0] - y, y - CUBE_Y_RANGE[1], 0.0)
    return max(dx, dy)


class _Task(aloha_sim.TransferCubeTask):
    """TransferCubeTask that renders only the camera the policy sees and can push the cube."""

    def __init__(self, perturbation: Perturbation):
        super().__init__()
        self.perturbation = perturbation
        self.push_direction = np.zeros(2)
        self.step_index = 0
        self.lift_step: int | None = None
        self.push_start: int | None = None

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        physics.named.data.xfrc_applied["box"] = 0.0
        self.step_index = 0
        self.lift_step = None
        self.push_start = None if self.perturbation.push_on_lift else self.perturbation.push_at_step

    def before_step(self, action, physics):
        super().before_step(action, physics)
        p = self.perturbation
        if p.push_force <= 0.0:
            return
        if self.push_start is None and self.lift_step is not None:
            self.push_start = self.lift_step + p.push_at_step
        pushing = self.push_start is not None and self.push_start <= self.step_index < self.push_start + p.push_steps
        force = p.push_force * self.push_direction if pushing else np.zeros(2)
        physics.named.data.xfrc_applied["box", :2] = force

    def after_step(self, physics):
        super().after_step(physics)
        if self.lift_step is None and self.get_reward(physics) == 2:
            self.lift_step = self.step_index
        self.step_index += 1

    def get_observation(self, physics):
        return {
            "qpos": self.get_qpos(physics),
            "qvel": self.get_qvel(physics),
            "env_state": self.get_env_state(physics),
            "images": {"top": physics.render(height=480, width=640, camera_id="top")},
        }


def _load_physics(perturbation: Perturbation) -> dm_mujoco.Physics:
    xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
    if not perturbation.table_stripes:
        return dm_mujoco.Physics.from_xml_path(str(xml_path))
    # The table mesh has no texture coordinates, so the checker pattern lands as stripes.
    # A texture has to exist at compile time, so patch the scene file in memory.
    assets = {path.name: path.read_bytes() for path in ASSETS_DIR.iterdir() if path.is_file()}
    scene = assets["scene.xml"].decode()
    scene = scene.replace("</asset>", STRIPES_ASSET + "</asset>", 1)
    scene = scene.replace('name="table" rgba="0.2 0.2 0.2 1"', 'name="table" material="table_stripes"', 1)
    if "table_stripes" not in scene.split("<worldbody>")[1]:
        raise RuntimeError("scene.xml changed upstream; the table geom was not found")
    assets["scene.xml"] = scene.encode()
    return dm_mujoco.Physics.from_xml_string(xml_path.read_text(), assets=assets)


def apply_to_model(physics: dm_mujoco.Physics, p: Perturbation) -> None:
    """Edit the compiled model in place. Does nothing for fields left at their defaults."""
    model = physics.model
    cube_geom = model.name2id("red_box", "geom")
    cube_body = model.name2id("box", "body")
    table_geom = model.name2id("table", "geom")
    cam = model.name2id("top", "camera")

    if p.cam_shift != (0.0, 0.0, 0.0) or p.cam_roll_deg or p.cam_tilt_deg:
        # The camera tracks the table body. Freeze its current orientation, then move it.
        physics.forward()
        base = np.zeros(4)
        mujoco.mju_mat2Quat(base, physics.data.cam_xmat[cam].copy())
        roll, tilt = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(roll, np.array([0.0, 0.0, 1.0]), np.deg2rad(p.cam_roll_deg))
        mujoco.mju_axisAngle2Quat(tilt, np.array([1.0, 0.0, 0.0]), np.deg2rad(p.cam_tilt_deg))
        quat, local = np.zeros(4), np.zeros(4)
        mujoco.mju_mulQuat(local, tilt, roll)
        mujoco.mju_mulQuat(quat, base, local)
        shift_world = physics.data.cam_xmat[cam].reshape(3, 3) @ np.array(p.cam_shift)
        model.cam_mode[cam] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        model.cam_quat[cam] = quat
        model.cam_pos[cam] = model.cam_pos[cam] + shift_world

    if p.cube_rgba is not None:
        model.geom_rgba[cube_geom] = p.cube_rgba
    if p.table_rgba is not None:
        model.geom_rgba[table_geom] = p.table_rgba
    if p.light_scale != 1.0:
        model.light_diffuse[:] *= p.light_scale
        model.light_specular[:] *= p.light_scale
        model.vis.headlight.ambient[:] *= p.light_scale
        model.vis.headlight.diffuse[:] *= p.light_scale
    if p.cube_mass_scale != 1.0:
        model.body_mass[cube_body] *= p.cube_mass_scale
        model.body_inertia[cube_body] *= p.cube_mass_scale
    if p.cube_friction_scale != 1.0:
        # MuJoCo uses the larger friction of the two touching geoms unless one has priority,
        # so give the cube priority to make its own coefficient decide.
        model.geom_friction[cube_geom, 0] *= p.cube_friction_scale
        model.geom_priority[cube_geom] = 1
    physics.forward()


def cube_signals(physics: dm_mujoco.Physics) -> dict:
    """What the failure classifier needs from one simulator step."""
    right = left = table = False
    model = physics.model
    for i in range(physics.data.ncon):
        contact = physics.data.contact[i]
        if contact.dist > 0:
            continue
        names = {model.id2name(contact.geom1, "geom"), model.id2name(contact.geom2, "geom")}
        if "red_box" not in names:
            continue
        right = right or bool(names & set(RIGHT_FINGERS))
        left = left or bool(names & set(LEFT_FINGERS))
        table = table or "table" in names
    return {
        "right_touch": right,
        "left_touch": left,
        "table_touch": table,
        "cube_z": float(physics.named.data.qpos["red_box_joint"][2]),
    }


class PerturbedTransferCubeEnv(AlohaEnv):
    """gym-aloha transfer cube with one `Perturbation` applied.

    Observations, actions, reward and success are those of `AlohaEnv`. `info` additionally
    carries the reward stage and the contact and height signals of the step.
    """

    def __init__(self, perturbation: Perturbation | str = "nominal", episode_length: int = 400):
        self.perturbation = VARIANTS[perturbation] if isinstance(perturbation, str) else perturbation
        self.episode_length = episode_length
        self.steps = 0
        self.cube_start = np.zeros(3)
        super().__init__(task="transfer_cube", obs_type="pixels_agent_pos")

    def _make_env_task(self, task_name):
        physics = _load_physics(self.perturbation)
        apply_to_model(physics, self.perturbation)
        return control.Environment(
            physics, _Task(self.perturbation), float("inf"), control_timestep=DT, flat_observation=False
        )

    @property
    def physics(self) -> dm_mujoco.Physics:
        return self._env.physics

    def reset(self, seed=None, options=None):
        rng = np.random.RandomState(seed)
        x, y = sample_cube_xy(rng, self.perturbation.cube_band)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        self.cube_start = np.array([x, y, CUBE_START_Z])
        self._env.task.push_direction = np.array([np.cos(angle), np.sin(angle)])
        self._env.task._random = np.random.RandomState(seed)
        aloha_sim.BOX_POSE[0] = np.concatenate([self.cube_start, [1.0, 0.0, 0.0, 0.0]])
        self.steps = 0
        raw = self._env.reset()
        return self._format_raw_obs(raw.observation), {"is_success": False, **cube_signals(self.physics)}

    def step(self, action):
        _, reward, _, raw_obs = self._env.step(action)
        self.steps += 1
        success = reward == 4
        info = {"is_success": success, "stage": int(reward), **cube_signals(self.physics)}
        return self._format_raw_obs(raw_obs), reward, success, self.steps >= self.episode_length, info
