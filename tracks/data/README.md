# Track C — Data

Data-scaling curve, scripted expert, generated datasets, targeted repair. See PLAN.md.

## Status

| Item | State |
|---|---|
| C1 subsets (10 / 25 / 50) | Done: `subsets.json` (fixed seed, nested). Training runs not started (need the GPU budget, issue #2). |
| C2 scripted expert | Done: 100/100 in the end-effector sim and 100/100 when replayed in the joint-space sim (95% Wilson interval 0.963–1.000); 97/100 still holding the cube on the last step. `results/expert_success.json`. |
| C2 dataset | Script done and checked: a 2-episode dataset loads in `LeRobotDataset` and `lerobot-train` trains on it. Full 50-episode set: generated locally under `datasets/` (not in git). |
| C3, C4 | Not started: need training (C3) and Track B's robustness matrix (C4). |

Checked on Windows 11, Python 3.12, LeRobot 0.7.0, MuJoCo 3.8.1, CPU only. Not checked on the Macs or the 4090.

## Files

| File | What it does |
|---|---|
| `scripted_expert.py` | The expert (waypoint plan + closed-loop gripper tracking in the end-effector sim) and the joint-space replay. |
| `measure_expert.py` | Expert success over N random cube positions, both sims, with interval. No rendering, about 2 s per episode. |
| `generate_dataset.py` | Records expert demonstrations as a LeRobot dataset with the same features as the human set. About 1 min per episode on CPU (rendering + AV1 encoding). |
| `define_subsets.py` / `subsets.json` | The fixed C1 subsets and the `--dataset.episodes=[...]` flag for each. |
| `results/` | Expert success JSON and dataset manifests (seed and cube position per episode, plus skipped attempts). |

## How the expert works (one paragraph each)

**It plans in end-effector space.** `gym_aloha`'s end-effector task welds each gripper to a
"mocap" target, so the expert only says where each gripper goes and how open it is. Right
gripper tilts down, picks the cube, carries it to a meeting point above the table centre;
left gripper, rolled 90° so its fingers close top-and-bottom (the right one holds the sides),
comes in horizontally, closes; right opens and backs away.

**It is closed loop, because the weld is soft.** Commanded open-loop, the gripper ended up
about 14 cm from its target (soft weld, high joint friction). The expert instead tracks a
"pad point" 13 cm out along each gripper (fingertips are at 15.3 cm), measures it every
step and shifts the command by the accumulated error. Tracking error is a few millimetres.

**Variation comes from a seed.** Approach and retreat points get ±4 mm noise, the meeting
point ±2 cm, key frame times ±4 steps. Episode *i* uses seed *start + i*, so any
demonstration can be regenerated exactly.

**The dataset is recorded from a joint-space replay.** The joint angles the expert's arms
went through are replayed in the normal joint-space task (the one the policy sees) with the
same cube pose, rendering the `top` camera. An episode is kept only if the replay reaches
stage 4 and still holds the cube on the last step.

## Commands

```bash
# expert success (C2 report number)
python tracks/data/measure_expert.py --episodes 100 --seed 0

# 50 scripted demonstrations (C2 dataset, used by C3)
python tracks/data/generate_dataset.py --episodes 50 --seed 100000 --root datasets/aloha_sim_transfer_cube_scripted

# train on it
lerobot-train --policy.type=act --dataset.repo_id=local/aloha_sim_transfer_cube_scripted \
  --dataset.root=datasets/aloha_sim_transfer_cube_scripted ...

# C1: train on a fixed human subset (flag from subsets.json)
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes=[17,21,25,27,28,30,37,43,44,47] ...

# C4 later: demonstrations outside the training range, e.g. a wider cube x range
python tracks/data/generate_dataset.py --episodes 50 --cube-x -0.05 0.25 --root datasets/repair_wide_x
```

## Known limits

- The expert uses privileged simulator state (exact cube position, finger positions). That is normal for a scripted expert but worth saying when comparing with human data (C3).
- Scripted demonstrations are smoother and more uniform than human ones; C3 is what measures whether that matters.
- The expert has not been tested outside the default cube range yet. C4 will need that.
- On Windows, `lerobot-train` crashes at its first checkpoint save, while creating the `checkpoints/last` symlink (needs admin or Developer Mode). This does not affect the Macs or the Linux 4090.
