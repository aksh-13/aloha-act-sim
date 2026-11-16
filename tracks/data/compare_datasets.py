"""Track C: how close is a scripted dataset to the human one?

Compares lerobot/aloha_sim_transfer_cube_human with one or more scripted datasets on:
  * format: fps, episode length, features, camera, image size;
  * handoff style, from forward kinematics on the recorded joint states: gripper tilt at the
    pick, gripper roll at the handover, meeting point, grasp height;
  * timing of the key events (right closes, left closes, right releases);
  * gripper command levels;
  * joint-angle and command distributions (per-joint 1-D Wasserstein distance, averaged);
  * motion smoothness (mean step-to-step change of the commands).

Yardstick: every number is also computed between the two halves of the human data
(episodes 0-24 vs 25-49). A scripted set is "as close as human is to human" on a metric
when its gap to the full human set is about that size.

    python tracks/data/compare_datasets.py --scripted aloha_sim_transfer_cube_scripted \
        --scripted aloha_sim_transfer_cube_scripted_v1
Writes tracks/data/results/dataset_comparison.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).parent))
import scripted_expert as se  # noqa: E402

HUMAN = "lerobot/aloha_sim_transfer_cube_human"
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # joint columns without the grippers


# ----------------------------------------------------------------------------- loading
def load(repo_id: str, root: str | None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    hf = ds.hf_dataset
    ep = np.asarray(hf["episode_index"]).ravel()
    return {
        "name": repo_id, "ds": ds, "ep": ep,
        "state": np.stack([np.asarray(x) for x in hf["observation.state"]]).astype(float),
        "action": np.stack([np.asarray(x) for x in hf["action"]]).astype(float),
    }


def subset(d, episodes):
    m = np.isin(d["ep"], episodes)
    return {**d, "ep": d["ep"][m], "state": d["state"][m], "action": d["action"][m]}


# ----------------------------------------------------------------------------- kinematics
class FK:
    def __init__(self):
        from dm_control import mujoco
        from gym_aloha.constants import ASSETS_DIR

        self.ph = mujoco.Physics.from_xml_path(str(ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"))

    def __call__(self, state):
        from gym_aloha.constants import unnormalize_puppet_gripper_position

        q = self.ph.data.qpos
        for o, j in ((0, 0), (8, 7)):
            q[o:o + 6] = state[j:j + 6]
            g = unnormalize_puppet_gripper_position(state[j + 6])
            q[o + 6], q[o + 7] = g, -g
        self.ph.forward()
        out = {}
        for s in ("left", "right"):
            R = self.ph.named.data.xmat[f"vx300s_{s}/gripper_link"].reshape(3, 3)
            out[s] = {"pad": se.pad_point(self.ph, s), "axis": R[:, 0].copy(), "close": R[:, 1].copy()}
        return out


def first_drop(x, after):
    """First step after `after` where gripper command x falls below halfway between open and closed."""
    thr = 0.5 * (x[after:].max() + x.min())
    idx = np.where((np.arange(len(x)) > after) & (x < thr))[0]
    return int(idx[0]) if len(idx) else len(x) - 1


def per_episode(d, fk: FK):
    rows = []
    for e in np.unique(d["ep"]):
        S, A = d["state"][d["ep"] == e], d["action"][d["ep"] == e]
        rg, lg = A[:, 13], A[:, 6]
        r_close = first_drop(rg, int(np.argmax(rg[:250])))
        l_close = first_drop(lg, int(np.argmax(lg)))
        thr = 0.5 * (rg.max() + rg.min())
        rel = np.where((np.arange(len(rg)) > l_close) & (rg > thr))[0]
        r_release = int(rel[0]) if len(rel) else len(rg) - 1
        g, h = fk(S[r_close]), fk(S[min(l_close + 5, len(S) - 1)])
        tilt = lambda v: float(np.degrees(np.arcsin(np.clip(-v[2], -1, 1))))
        roll = lambda v: float(np.degrees(np.arcsin(np.clip(abs(v[2]), 0, 1))))
        rows.append({
            "right_close_step": r_close, "left_close_step": l_close, "right_release_step": r_release,
            "pick_tilt_right_deg": tilt(g["right"]["axis"]), "pick_roll_right_deg": roll(g["right"]["close"]),
            "grasp_height_m": float(g["right"]["pad"][2]),
            "handover_tilt_right_deg": tilt(h["right"]["axis"]), "handover_roll_right_deg": roll(h["right"]["close"]),
            "handover_tilt_left_deg": tilt(h["left"]["axis"]), "handover_roll_left_deg": roll(h["left"]["close"]),
            "meet_x_m": float(h["right"]["pad"][0]), "meet_y_m": float(h["right"]["pad"][1]),
            "meet_z_m": float(h["right"]["pad"][2]),
            "grip_closed_left": float(lg.min()), "grip_open_left": float(lg.max()),
            "grip_closed_right": float(rg.min()), "grip_open_right": float(rg.max()),
            "command_speed_rad_per_step": float(np.abs(np.diff(A[:, ARM], axis=0)).mean()),
            "command_jerk": float(np.abs(np.diff(A[:, ARM], n=3, axis=0)).mean()),
        })
    return rows


def summarise(rows):
    keys = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))} for k in keys}


def distribution_gap(a, b):
    """Mean over joints of the 1-D Wasserstein distance (rad for arm joints)."""
    return {
        "state_arm_joints": float(np.mean([wasserstein_distance(a["state"][:, j], b["state"][:, j]) for j in ARM])),
        "action_arm_joints": float(np.mean([wasserstein_distance(a["action"][:, j], b["action"][:, j]) for j in ARM])),
        "action_grippers": float(np.mean([wasserstein_distance(a["action"][:, j], b["action"][:, j]) for j in (6, 13)])),
    }


def fmt_feature_summary(d):
    ds = d["ds"]
    lens = np.bincount(d["ep"])
    lens = lens[lens > 0]
    shape = list(ds.features["observation.images.top"]["shape"])
    feats = sorted(k for k in ds.features if k not in ("index", "frame_index", "episode_index", "task_index", "timestamp"))
    return {"fps": ds.fps, "episodes": int(ds.num_episodes), "frames": int(ds.num_frames),
            "episode_length": sorted(set(lens.tolist())), "cameras": list(ds.meta.camera_keys),
            "image_shape": shape, "features": feats}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripted", action="append", required=True,
                    help="local dataset name under --datasets-dir (repeatable)")
    ap.add_argument("--datasets-dir", type=Path, default=Path("datasets"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "dataset_comparison.json")
    args = ap.parse_args()

    fk = FK()
    human = load(HUMAN, None)
    half_a, half_b = subset(human, range(0, 25)), subset(human, range(25, 50))
    h_rows = per_episode(human, fk)
    h_sum = summarise(h_rows)
    ha_sum, hb_sum = summarise(per_episode(half_a, fk)), summarise(per_episode(half_b, fk))

    report = {"human": {"format": fmt_feature_summary(human), "style": h_sum},
              "yardstick_human_half_vs_half": {
                  "style_mean_gap": {k: abs(ha_sum[k]["mean"] - hb_sum[k]["mean"]) for k in h_sum},
                  "distribution_gap": distribution_gap(half_a, half_b)},
              "scripted": {}}

    for name in args.scripted:
        s = load(f"local/{name}", str(args.datasets_dir / name))
        s_sum = summarise(per_episode(s, fk))
        report["scripted"][name] = {
            "format": fmt_feature_summary(s),
            "format_matches_human": fmt_feature_summary(s) | {"episodes": 0, "frames": 0}
                                    == fmt_feature_summary(human) | {"episodes": 0, "frames": 0},
            "style": s_sum,
            "style_mean_gap": {k: abs(s_sum[k]["mean"] - h_sum[k]["mean"]) for k in h_sum},
            "distribution_gap": distribution_gap(s, human),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    # ---- table
    names = list(report["scripted"])
    yard = report["yardstick_human_half_vs_half"]
    head = f"{'metric':32s} {'human mean (std)':>18s} {'human-vs-human gap':>19s}" + "".join(f" {n[-24:]:>26s}" for n in names)
    print(head)
    print("-" * len(head))
    for k in h_sum:
        line = f"{k:32s} {h_sum[k]['mean']:9.3f} ({h_sum[k]['std']:6.3f}) {yard['style_mean_gap'][k]:19.3f}"
        for n in names:
            st = report["scripted"][n]["style"][k]
            line += f" {st['mean']:9.3f} ({st['std']:6.3f}) gap {report['scripted'][n]['style_mean_gap'][k]:6.3f}"
        print(line)
    print("\nDistribution gap (mean 1-D Wasserstein over joints; 0 = identical)")
    for k in yard["distribution_gap"]:
        line = f"{k:32s} {'':18s} {yard['distribution_gap'][k]:19.3f}"
        for n in names:
            line += f" {report['scripted'][n]['distribution_gap'][k]:26.3f}"
        print(line)
    print("\nFormat matches human (fps, length, features, camera, image size): "
          + ", ".join(f"{n}: {report['scripted'][n]['format_matches_human']}" for n in names))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
