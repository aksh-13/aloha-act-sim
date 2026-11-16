"""Track C2: record scripted-expert demonstrations as a LeRobot dataset.

For each episode:
  1. sample a cube position (default: the task's own range; widen it for C4 repair data);
  2. run the scripted expert in the end-effector sim -> joint trajectory;
  3. replay that trajectory in the joint-space sim (the one the policy sees), rendering the
     `top` camera, and record (observation, action) per step;
  4. keep the episode only if the replay reaches stage 4 (cube held by the left gripper
     off the table) and is still at stage 4 on the last step. Others are logged and skipped.

The dataset has the same features, fps, episode length and task string as
lerobot/aloha_sim_transfer_cube_human, so `lerobot-train` loads it the same way:

    lerobot-train --policy.type=act --dataset.repo_id=local/aloha_sim_transfer_cube_scripted \
        --dataset.root=datasets/aloha_sim_transfer_cube_scripted ...

Usage:
    python tracks/data/generate_dataset.py --episodes 50 --seed 100000 \
        --root datasets/aloha_sim_transfer_cube_scripted

A manifest (seed, cube position, stages per kept/skipped episode) is written next to the
script under results/ so the dataset can be regenerated exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import scripted_expert as se  # noqa: E402

MOTORS = ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle",
          "left_wrist_rotate", "left_gripper", "right_waist", "right_shoulder", "right_elbow",
          "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper"]

FEATURES = {
    "observation.images.top": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": [14], "names": {"motors": MOTORS}},
    "action": {"dtype": "float32", "shape": [14], "names": {"motors": MOTORS}},
    "next.done": {"dtype": "bool", "shape": [1], "names": None},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=50, help="number of successful episodes to keep")
    ap.add_argument("--max-attempts", type=int, default=None, help="default: 2 x episodes")
    ap.add_argument("--seed", type=int, default=100_000,
                    help="attempt i uses seed + i (default keeps clear of measure_expert.py's seeds)")
    ap.add_argument("--repo-id", default="local/aloha_sim_transfer_cube_scripted")
    ap.add_argument("--root", type=Path, default=Path("datasets/aloha_sim_transfer_cube_scripted"))
    ap.add_argument("--cube-x", type=float, nargs=2, default=se.CUBE_X_RANGE)
    ap.add_argument("--cube-y", type=float, nargs=2, default=se.CUBE_Y_RANGE)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="delete --root first if it exists")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default: tracks/data/results/<root name>_manifest.json")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.root.exists():
        if not args.overwrite:
            sys.exit(f"{args.root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.root)
    max_attempts = args.max_attempts or 2 * args.episodes
    params = se.noiseless_params() if args.no_noise else se.ExpertParams()

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=50, features=FEATURES, root=args.root,
                               robot_type="aloha", use_videos=True)
    kept, skipped, t0 = [], [], time.time()
    for attempt in range(max_attempts):
        if len(kept) == args.episodes:
            break
        seed = args.seed + attempt
        rng = np.random.default_rng(seed)
        cube = se.sample_cube_pose(rng, args.cube_x, args.cube_y)
        ee = se.rollout_ee(cube, rng, params)
        info = {"seed": seed, "cube_xy": cube[:2].round(4).tolist(), "ee_max_stage": ee["max_stage"]}
        if not ee["success"]:
            skipped.append({**info, "reason": "expert failed in end-effector sim"})
            print(f"attempt {attempt}: skip (ee stage {ee['max_stage']})", flush=True)
            continue

        n_frames = len(ee["joint_traj"])

        def on_frame(t, obs, action):
            ds.add_frame({
                "observation.images.top": obs["top"],
                "observation.state": obs["qpos"].astype(np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "next.done": np.array([t == n_frames - 1]),
                "task": se.TASK_TEXT,
            })

        rw = se.replay_joint(ee["joint_traj"], cube, render_top=True, on_frame=on_frame)
        info.update(replay_max_stage=int(rw.max()), replay_final_stage=int(rw[-1]))
        # Keep only clean demonstrations: the handover must succeed AND the left gripper must
        # still hold the cube off the table at the last step (a few replays succeed, then drop it).
        if rw.max() < 4 or rw[-1] < 4:
            ds.clear_episode_buffer()
            reason = "joint replay failed" if rw.max() < 4 else "cube dropped after the handover"
            skipped.append({**info, "reason": reason})
            print(f"attempt {attempt}: skip ({reason}; max stage {int(rw.max())}, final {int(rw[-1])})", flush=True)
            continue
        ds.save_episode()
        kept.append({"episode_index": len(kept), **info})
        print(f"attempt {attempt}: kept as episode {len(kept) - 1}  "
              f"({len(kept)}/{args.episodes}, {time.time() - t0:.0f}s)", flush=True)
    ds.finalize()

    manifest = {
        "repo_id": args.repo_id, "root": str(args.root), "task": se.TASK_TEXT,
        "kept": len(kept), "attempts": len(kept) + len(skipped), "seed_start": args.seed,
        "cube_x_range": list(args.cube_x), "cube_y_range": list(args.cube_y), "noise": not args.no_noise,
        "expert_params": {k: v for k, v in vars(params).items() if k != "extra"},
        "episodes": kept, "skipped": skipped, "seconds": round(time.time() - t0, 1),
    }
    out = args.manifest or Path(__file__).parent / "results" / f"{args.root.name}_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    print(f"\nKept {len(kept)} of {len(kept) + len(skipped)} attempts -> {args.root}. Manifest: {out}")
    if len(kept) < args.episodes:
        sys.exit(f"only {len(kept)} episodes kept; raise --max-attempts")


if __name__ == "__main__":
    main()
