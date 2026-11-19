# Track A — Method

ACT training, chunk-length ablation, no-chunking baseline, inference-delay stress test. See
[PLAN.md](../../PLAN.md) and issues #4–#7. How the model works: [act_explained.md](act_explained.md).

## A1 (#4): reproduce the published run

**Reference recipe:** LeRobot 0.7.0's ACT defaults, unchanged. The 83.0% target (500 episodes,
80,000 steps) comes from the LeRobot model card
(`lerobot/act_aloha_sim_transfer_cube_human`), so we reproduce *that* recipe, not the paper's.
The full resolved config is pinned in [configs/a1_reproduce.json](configs/a1_reproduce.json).

### LeRobot defaults vs the paper

Paper values are from the hyperparameter table of Zhao et al. 2023 (arXiv 2304.13705). These are
known differences between the two recipes, not deviations of ours, so they are not in
`docs/deviations.md`.

| Setting | Paper | LeRobot 0.7.0 (what we run) | Note |
|---|---|---|---|
| Learning rate | 1e-5 | 1e-5 (backbone 1e-5), AdamW, weight decay 1e-4 | same |
| Batch size | 8 | 8 | same |
| Encoder layers | 4 | 4 | same |
| Decoder layers | **7** | **1** | A bug in the original ACT code means only the first of its 7 layers is used; LeRobot matches that behaviour (`configuration_act.py:108`, tonyzhaozh/act issue #25). |
| Feedforward dimension | 3200 | 3200 | same |
| Hidden dimension | 512 | 512 | same |
| Heads | 8 | 8 | same |
| Chunk size | 100 | 100 | same |
| KL weight (beta) | 10 | 10 | same |
| Dropout | 0.1 | 0.1 | same |
| CVAE latent / VAE encoder layers | not in table | 32 / 4 | |
| Backbone | ResNet-18 (ImageNet) | ResNet-18, `IMAGENET1K_V1` | same |
| Temporal ensembling | used, weights w_i = exp(−m·i) | **off**; executes all 100 actions of a chunk (`n_action_steps=100`) | Tested on/off in A2 (#5). |

### Running it

Smoke test on a Mac (about 4 minutes, MPS):

```bash
source .venv/bin/activate
STEPS=200 SAVE_FREQ=200 OUT=outputs/smoke/a1 tracks/method/train_a1.sh mps
```

Smoke result (18 Nov 2026, Apple M4, MPS): 200 steps, update time 1.13 s/step (loss 7.38,
L1 0.62, KL 0.68); checkpoint saved and loaded by `lerobot-eval`, 0/2 successes (expected
this early). This checks the plumbing, not the policy.

Full run on the 4090 server, in Docker:

```bash
docker build -t aloha-act:0.7.0 .
docker run --rm --gpus all aloha-act:0.7.0 nvidia-smi          # GPU visible in the container?
NAME=a1-main tracks/method/docker_run.sh -d tracks/method/train_a1.sh cuda
docker logs -f a1-main
```

Checkpoints land in `checkpoints/a1-reproduce-default/checkpoints/<step>/pretrained_model/` on the
server's disk (mounted, never in git). The first log line at step 200 reports the per-step update
time (`updt_s`); multiply by 80,000 for issue #2's estimate.

### Evaluation protocol (A1, and proposed for every Track A evaluation)

- **Seed 1000** for every evaluation (`lerobot-eval` default). Episode *i* uses seed 1000 + *i*, so
  every checkpoint faces the same cube positions.
- **100 episodes** per 10k-step checkpoint; **500 episodes** for the 80k checkpoint and for the
  published checkpoint (the control), matching the model card's 500.
- **Videos:** `lerobot-eval` renders the first 10 episodes.
- Report success with a 95% Wilson interval. Use stock `lerobot-eval` until the Track B harness
  (#8) exists, then rerun through it.

```bash
lerobot-eval \
  --policy.path=checkpoints/a1-reproduce-default/checkpoints/080000/pretrained_model \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=500 --eval.batch_size=1 --eval.use_async_envs=false --seed=1000 \
  --policy.device=cuda --output_dir=outputs/eval/a1-reproduce-default/080000
```
