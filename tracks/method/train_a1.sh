#!/usr/bin/env bash
# A1 (#4): reproduce the published LeRobot ACT run on AlohaTransferCube-v0.
# Recipe = LeRobot 0.7.0 ACT defaults (pinned in tracks/method/configs/a1_reproduce.json).
#
#   tracks/method/train_a1.sh cuda                    # full run on the 4090
#   STEPS=200 OUT=outputs/smoke/a1 tracks/method/train_a1.sh mps   # smoke test on the Mac
#
# Overrides (env vars): STEPS, OUT, SAVE_FREQ, SEED.
# Checkpoints land in $OUT/checkpoints/<step>/pretrained_model/ (LeRobot's layout).
# The first logged line (step 200) gives the per-step time used for issue #2.
set -euo pipefail

DEVICE="${1:?usage: train_a1.sh <cuda|mps|cpu>}"
STEPS="${STEPS:-80000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
SEED="${SEED:-1000}"
OUT="${OUT:-checkpoints/a1-reproduce-default}"

if [[ "$DEVICE" == "cuda" ]]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi

mkdir -p "$(dirname "$OUT")" outputs/logs
LOG="outputs/logs/$(basename "$OUT")-$(date +%Y%m%d-%H%M%S).log"
echo "start $(date -u +%FT%TZ) device=$DEVICE steps=$STEPS out=$OUT log=$LOG" | tee "$LOG"

# eval_steps=0: no in-training eval; checkpoints are evaluated separately (seed 1000, 100/500 episodes).
lerobot-train \
  --policy.type=act \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --steps="$STEPS" --batch_size=8 --seed="$SEED" \
  --save_freq="$SAVE_FREQ" --eval_steps=0 --log_freq=200 \
  --trackio.enable=false \
  --job_name="$(basename "$OUT")" \
  --output_dir="$OUT" 2>&1 | tee -a "$LOG"

echo "end $(date -u +%FT%TZ)" | tee -a "$LOG"
