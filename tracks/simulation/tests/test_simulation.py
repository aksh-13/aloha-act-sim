"""Run with: python -m unittest discover -s tracks/simulation/tests -t ."""

import unittest
from pathlib import Path

import numpy as np
from gym_aloha.constants import START_ARM_POSE
from gym_aloha.utils import sample_box_pose

from tracks.simulation import failures
from tracks.simulation.evaluate import REPO_ROOT, checkpoint_name
from tracks.simulation.perturbations import (
    VARIANTS,
    PerturbedTransferCubeEnv,
    distance_outside_training_range,
    sample_cube_xy,
)

HOLD = np.array(START_ARM_POSE[:6] + [1.0] + START_ARM_POSE[8:14] + [1.0], dtype=np.float32)


def trace(steps):
    """steps: list of (stage, right, left, table, z)."""
    keys = ("stage", "right_touch", "left_touch", "table_touch", "cube_z")
    return {key: [s[i] for s in steps] for i, key in enumerate(keys)}


REST = (0, False, False, True, 0.02)
TOUCH = (1, True, False, True, 0.02)
HELD = (2, True, False, False, 0.10)
BOTH = (3, True, True, False, 0.10)


class ClassifierTest(unittest.TestCase):
    def label(self, steps):
        return failures.classify(**trace(steps))["label"]

    def test_success(self):
        self.assertEqual(self.label([REST, TOUCH, HELD, HELD, HELD, (4, False, True, False, 0.1)]), "success")

    def test_never_grasped(self):
        self.assertEqual(self.label([REST] * 5 + [TOUCH] * 5), "never_grasped")

    def test_left_bumping_a_resting_cube_is_not_a_handover(self):
        self.assertEqual(self.label([REST] * 5 + [(3, False, True, True, 0.02)] * 5), "never_grasped")

    def test_one_step_bounce_is_not_a_lift(self):
        self.assertEqual(self.label([TOUCH, HELD, TOUCH, TOUCH]), "never_grasped")

    def test_grasped_then_dropped(self):
        self.assertEqual(self.label([TOUCH] + [HELD] * 5 + [REST] * 5), "grasped_then_dropped")

    def test_lifted_no_handover(self):
        self.assertEqual(self.label([TOUCH] + [HELD] * 9), "lifted_no_handover")

    def test_handover_failed(self):
        self.assertEqual(self.label([TOUCH] + [HELD] * 5 + [BOTH] * 3 + [REST] * 3), "handover_failed")


class SamplingTest(unittest.TestCase):
    def test_nominal_matches_gym_aloha(self):
        for seed in (0, 7, 1000):
            x, y = sample_cube_xy(np.random.RandomState(seed), None)
            np.testing.assert_allclose([x, y], sample_box_pose(seed)[:2])

    def test_band_stays_in_band(self):
        rng = np.random.RandomState(0)
        for _ in range(500):
            d = distance_outside_training_range(*sample_cube_xy(rng, (0.03, 0.06)))
            self.assertTrue(0.03 < d <= 0.06)


class EnvTest(unittest.TestCase):
    def test_every_variant_builds_and_steps(self):
        for name in VARIANTS:
            env = PerturbedTransferCubeEnv(name)
            env.reset(seed=1)
            _, _, _, _, info = env.step(HOLD)
            self.assertIn("cube_z", info, name)

    def test_nominal_image_matches_gym_aloha(self):
        import gymnasium as gym
        import gym_aloha  # noqa: F401

        reference = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
        expected, _ = reference.reset(seed=3)
        actual, _ = PerturbedTransferCubeEnv("nominal").reset(seed=3)
        np.testing.assert_array_equal(actual["pixels"]["top"], expected["pixels"]["top"])
        np.testing.assert_allclose(actual["agent_pos"], expected["agent_pos"])

    def test_table_push_moves_the_cube_further_with_more_force(self):
        moved = []
        for name in ("nominal", "push_table_1", "push_table_3"):
            env = PerturbedTransferCubeEnv(name)
            env.reset(seed=5)
            for _ in range(80):
                env.step(HOLD)
            moved.append(np.linalg.norm(env.physics.named.data.qpos["red_box_joint"][:2] - env.cube_start[:2]))
        self.assertLess(moved[0], 1e-3)
        self.assertLess(moved[1], moved[2])
        self.assertGreater(moved[1], 5e-3)

    def test_physical_edits_reach_the_model(self):
        nominal = PerturbedTransferCubeEnv("nominal").physics.model
        heavy = PerturbedTransferCubeEnv("mass_2").physics.model
        slick = PerturbedTransferCubeEnv("friction_3").physics.model
        box, geom = nominal.name2id("box", "body"), nominal.name2id("red_box", "geom")
        self.assertAlmostEqual(heavy.body_mass[box], 4 * nominal.body_mass[box])
        self.assertAlmostEqual(slick.geom_friction[geom, 0], 0.1 * nominal.geom_friction[geom, 0])


class NamingTest(unittest.TestCase):
    def test_checkpoint_names(self):
        self.assertEqual(checkpoint_name(REPO_ROOT / "checkpoints/a2-chunk-50/020000"), "a2-chunk-50-020000")
        self.assertEqual(checkpoint_name(REPO_ROOT / "checkpoints/act_transfer_cube_published"), "act_transfer_cube_published")
        run = REPO_ROOT / "outputs/train/act_transfer_cube/checkpoints/080000/pretrained_model"
        self.assertEqual(checkpoint_name(run), "act_transfer_cube-080000")
        self.assertEqual(checkpoint_name(Path("/tmp/some_model")), "some_model")


if __name__ == "__main__":
    unittest.main()
