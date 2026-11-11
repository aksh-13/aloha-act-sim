# ACT on the ALOHA two-arm cube transfer (simulation)

Train an Action Chunking Transformer (ACT, Zhao et al. 2023, arXiv 2304.13705) from 50 human
demonstrations, then test it in closed loop in `gym-aloha`, a MuJoCo simulation of the two-arm
ALOHA robot. Built on Hugging Face LeRobot 0.7.0.

Published target: 83.0% success over 500 episodes after 80,000 training steps
(https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human).

## Plan and tracks

The full plan is in [PLAN.md](PLAN.md). Three tracks, one owner each:

| Track | Question | Folder |
|---|---|---|
| A — Method | How much does action chunking matter, and how does the policy cope with slow inference? | `tracks/method/` |
| B — Simulation | Where does the trained policy break when the scene changes? | `tracks/simulation/` |
| C — Data | How many demonstrations does it need, and can targeted new data repair a failure? | `tracks/data/` |

Simulation only. LeRobot's ACT implementation and the public human demonstrations are reused.

## Status (10 Nov 2026), checked on an Apple M4 Mac, macOS 26.5

- LeRobot and the ALOHA simulator install with Python 3.12.
- The published ACT checkpoint runs in the simulator on the Mac GPU (MPS): 2 of 3 episodes succeeded,
  about 7.5 s per episode, videos saved. Three episodes is a smoke test, not a result.
- The demonstration dataset loads (50 episodes, 20,000 frames) and training runs:
  about 1.15 s per step at batch size 8 on MPS. At that rate 80,000 steps is roughly 25 hours,
  so train on the RTX 4090, not the Mac.

Not checked: anything on the RTX 4090.

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install 'lerobot[aloha]' 'lerobot[training]'
```

On a headless Linux GPU machine also `export MUJOCO_GL=egl`.

## Evaluate the published checkpoint (do this first, to prove the harness)

The published checkpoint is in an older format and must be converted once:

```bash
python .venv/lib/python3.12/site-packages/lerobot/processor/migrate_policy_normalization.py \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
  --output-dir checkpoints/act_transfer_cube_published

lerobot-eval --policy.path=checkpoints/act_transfer_cube_published \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=100 --eval.batch_size=1 --eval.use_async_envs=false \
  --policy.device=mps --output_dir=outputs/eval/published_100
```

Videos land in `<output_dir>/videos/`, metrics in `<output_dir>/eval_info.json`.
Use `--policy.device=cuda` on the 4090.

## Train

```bash
lerobot-train --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --policy.device=cuda --policy.push_to_hub=false \
  --steps=80000 --batch_size=8 --save_freq=10000 \
  --trackio.enable=false --output_dir=outputs/train/act_transfer_cube
```

Then evaluate `outputs/train/act_transfer_cube/checkpoints/<step>/pretrained_model` with the eval command above.

Option names that differ from older tutorials: `--eval_steps` (not `--eval_freq`),
`--trackio.enable` (not `--wandb.enable`). Run `lerobot-train --help` for the full list.
