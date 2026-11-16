"""Track C2: measure the scripted expert's own success rate.

Runs the expert on N random cube positions (same distribution as the task) and reports:
  * success in the end-effector sim, where the expert acts directly;
  * success when its joint trajectory is replayed in the joint-space sim, which is what ends
    up in the dataset (only episodes that succeed here are kept as demonstrations).

No rendering, so it is fast (about 1 s per episode on a laptop CPU).

    python tracks/data/measure_expert.py --episodes 100 --seed 0 \
        --out tracks/data/results/expert_success.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import scripted_expert as se  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="episode i uses seed + i")
    ap.add_argument("--no-noise", action="store_true", help="turn off waypoint/time/meeting-point noise")
    ap.add_argument("--cube-x", type=float, nargs=2, default=se.CUBE_X_RANGE)
    ap.add_argument("--cube-y", type=float, nargs=2, default=se.CUBE_Y_RANGE)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "expert_success.json")
    args = ap.parse_args()

    params = se.ExpertParams(pos_noise=0, time_noise=0, meet_noise=0) if args.no_noise else se.ExpertParams()
    rows, t0 = [], time.time()
    for i in range(args.episodes):
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        cube = se.sample_cube_pose(rng, args.cube_x, args.cube_y)
        ee = se.rollout_ee(cube, rng, params)
        rw = se.replay_joint(ee["joint_traj"], cube)
        rows.append({"seed": seed, "cube_xy": cube[:2].round(4).tolist(), "ee_max_stage": ee["max_stage"],
                     "replay_max_stage": int(rw.max()), "replay_final_stage": int(rw[-1])})
        print(f"[{i + 1}/{args.episodes}] seed {seed} cube {cube[:2].round(3)} "
              f"ee stage {ee['max_stage']} replay stage {int(rw.max())}", flush=True)

    n = len(rows)
    k_ee = sum(r["ee_max_stage"] == 4 for r in rows)
    k_rp = sum(r["replay_max_stage"] == 4 for r in rows)
    k_hold = sum(r["replay_final_stage"] == 4 for r in rows)
    summary = {
        "what": "scripted expert success (Track C2); success = reward stage 4 reached at any step, as in lerobot-eval",
        "episodes": n, "seed_start": args.seed, "noise": not args.no_noise,
        "cube_x_range": list(args.cube_x), "cube_y_range": list(args.cube_y),
        "ee_sim": {"successes": k_ee, "rate": k_ee / n, "wilson95": se.wilson_interval(k_ee, n)},
        "joint_replay": {"successes": k_rp, "rate": k_rp / n, "wilson95": se.wilson_interval(k_rp, n)},
        "joint_replay_still_holding_at_end": {"count": k_hold, "rate": k_hold / n},
        "expert_params": {k: v for k, v in vars(params).items() if k != "extra"},
        "machine": platform.platform(), "seconds": round(time.time() - t0, 1),
        "episodes_detail": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    lo, hi = summary["joint_replay"]["wilson95"]
    print(f"\nEE sim: {k_ee}/{n}.  Joint replay: {k_rp}/{n} (95% CI {lo:.3f}-{hi:.3f}).  "
          f"Holding at end: {k_hold}/{n}.  Wrote {args.out}")


if __name__ == "__main__":
    main()
