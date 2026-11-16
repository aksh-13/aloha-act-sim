# Track C — Data

Data-scaling curve, scripted expert, generated datasets, targeted repair. See PLAN.md.

## Status

| Item | State |
|---|---|
| C1 subsets (10 / 25 / 50) | Done: `subsets.json` (fixed seed, nested). Training runs not started (need the GPU budget, issue #2). |
| C2 scripted expert | Done, built to hand off the way the human demos do. Over 100 episodes: 100/100 handovers in the joint-space sim (95% Wilson 0.963–1.000), 81/100 still holding the cube on the last step. `results/expert_success.json`. |
| C2 dataset | Done: 50 episodes kept of 70 attempts, 20,000 frames, same format as the human set; loads in `LeRobotDataset`, and `lerobot-train` trained 2 steps on an earlier version. Lives under `datasets/` (not in git); `results/aloha_sim_transfer_cube_scripted_manifest.json` lists every seed. |
| Human vs scripted comparison | Done: `compare_datasets.py`, `results/dataset_comparison.json`, and a side-by-side video (`make_comparison_video.py`). |
| C3, C4 | Not started: need training (C3) and Track B's robustness matrix (C4). |

Checked on Windows 11, Python 3.12, LeRobot 0.7.0, MuJoCo 3.8.1, CPU only. Not checked on the Macs or the 4090.

## Files

| File | What it does |
|---|---|
| `scripted_expert.py` | The expert (waypoint plan + closed-loop gripper tracking in the end-effector sim) and the joint-space replay. |
| `measure_expert.py` | Expert success over N random cube positions, both sims, with interval. No rendering, about 2 s per episode. |
| `generate_dataset.py` | Records expert demonstrations as a LeRobot dataset with the same features as the human set. About 30–60 s per episode on CPU. |
| `compare_datasets.py` | How close a scripted dataset is to the human one (handoff geometry, timing, gripper levels, joint distributions), against a human-vs-human yardstick. |
| `make_comparison_video.py` | Side-by-side video of one human demo and the expert from the same cube start. |
| `define_subsets.py` / `subsets.json` | The fixed C1 subsets and the `--dataset.episodes=[...]` flag for each. |
| `results/` | Expert success, dataset manifest, comparison JSON. |

## How the expert works

**It plans in end-effector space.** `gym_aloha`'s end-effector task welds each gripper to a
"mocap" target, so the expert only says where each gripper goes and how open it is.

**It copies the human handoff style, measured from the human data.** Forward kinematics on
the human joint states shows the right gripper picks at ~51° tilt with fingers closing
sideways, then levels out and rolls ~75° while carrying; the left gripper stays upright and
grabs from the sides; they meet ~0.29 m high; gripper commands run ~0.2 (closed) to ~1.1
(open); key events vary by ~30 steps between demos. The expert's defaults are set from
those numbers (`ExpertParams`).

**It is closed loop, because the weld is soft.** Commanded open-loop, the gripper ended up
about 14 cm off target. The expert tracks a "pad point" 13 cm out along each gripper
(fingertips are at 15.3 cm), measures it each step and shifts the command by the
accumulated error. Orientation is commanded ~12° above the target because the weld lets
the gripper sag by about that much.

**The dataset is recorded from a joint-space replay.** The joint angles the expert's arms
went through are replayed in the normal joint-space task (the one the policy sees),
rendering the `top` camera. An episode is kept only if the replay reaches stage 4 and still
holds the cube on the last step.

## How close it is to the human data

Yardstick: the same numbers computed between human episodes 0–24 and 25–49. Full table in
`results/dataset_comparison.json`; `v1` is the first expert (left gripper rolled instead of
the right).

| | Human | Scripted (now) | Scripted v1 | Human vs human gap |
|---|---|---|---|---|
| Format (fps, length, features, camera, image size) | — | same | same | — |
| Right closes / left closes / right releases (step) | 156 / 298 / 322 | 150 / 291 / 326 | 136 / 306 / 342 | 3 / 12 / 9 |
| Spread of those times (std, steps) | 30–35 | ~23 | ~2 | — |
| Pick tilt, right (°) | 51 | 50 | 53 | 1 |
| Handover roll right / left (°) | 75 / 14 | 78 / 12 | 4 / 78 | 2 / 1 |
| Handover tilt right / left (°) | −2 / −4 | 6 / 5 | 45 / 12 | 2 / 0.2 |
| Meeting height (m) | 0.288 | 0.285 | 0.243 | 0.001 |
| Gripper closed / open (left) | 0.20 / 1.03 | 0.20 / 1.03 | 0.00 / 1.00 | 0.02 / 0.02 |
| Joint-angle distribution gap (rad) | — | 0.159 | 0.286 | 0.026 |
| Command jerk | 0.002 | ~0.000 | ~0.000 | ~0.000 |

**What still differs:**
- Handover tilts are 8–9° off. Aiming further up made the left grip less reliable.
- Overall joint-angle distributions are ~6× further from human than human is from itself (v1: ~11×). The handoff poses now match, but the paths between them don't: the expert moves in straight lines between waypoints, and humans don't.
- Human commands are jittery (teleoperation), the expert's are smooth. I didn't add artificial jitter.
- Timing varies less than in the human demos (std ~23 vs ~30–35 steps).

**The cost of matching:** the human-style handoff is harder for the expert to finish cleanly. The left arm holds the cube to the end in 81/100 episodes, against 97/100 for v1; the filter throws out the rest. Episodes are discarded more often for some cube positions than others, which can slightly skew where cubes start in the kept set. Not measured yet.

## Commands

```bash
python tracks/data/measure_expert.py --episodes 100 --seed 0
python tracks/data/generate_dataset.py --episodes 50 --max-attempts 120 --seed 100000 --root datasets/aloha_sim_transfer_cube_scripted
python tracks/data/compare_datasets.py --scripted aloha_sim_transfer_cube_scripted
python tracks/data/make_comparison_video.py --human-episode 0 --out videos/human_vs_scripted_ep0.mp4

# train on the scripted set
lerobot-train --policy.type=act --dataset.repo_id=local/aloha_sim_transfer_cube_scripted \
  --dataset.root=datasets/aloha_sim_transfer_cube_scripted ...

# C1: train on a fixed human subset (flag from subsets.json)
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes=[17,21,25,27,28,30,37,43,44,47] ...

# C4 later: demonstrations outside the training range, e.g. a wider cube x range
python tracks/data/generate_dataset.py --episodes 50 --cube-x -0.05 0.25 --root datasets/repair_wide_x
```

## Known limits

- The expert uses privileged simulator state (exact cube position, finger positions). That's normal for a scripted expert, but worth stating when comparing with human data (C3).
- The expert hasn't been tested outside the default cube range yet. C4 will need that.
- On Windows, `lerobot-train` crashes at its first checkpoint save, while creating the `checkpoints/last` symlink (needs admin or Developer Mode). This doesn't affect the Macs or the Linux 4090.
