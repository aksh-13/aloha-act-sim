#!/usr/bin/env bash
# A1 (#4): reproduce the published LeRobot ACT run on AlohaTransferCube-v0.
# The recipe is tracks/method/configs/a1_reproduce.json (LeRobot 0.7.0 ACT defaults, with
# in-training sim eval off). It is loaded directly; only per-machine settings are overridden here.
#
#   tracks/method/train_a1.sh cuda                                  # full run on the 4090
#   STEPS=200 SAVE_FREQ=200 OUT=outputs/smoke/a1 tracks/method/train_a1.sh mps   # smoke test
#   RESUME=1 tracks/method/train_a1.sh cuda                         # resume from $OUT's last checkpoint
#
# Overrides (env vars): STEPS, SAVE_FREQ, SEED, OUT, CONFIG, RESUME=1.
# Checkpoints land in $OUT/checkpoints/<step>/pretrained_model/ (LeRobot's layout).
set -euo pipefail
cd "$(dirname "$0")/../.."   # paths below are relative to the repo root

DEVICE="${1:?usage: train_a1.sh <cuda|mps|cpu>}"
CONFIG="${CONFIG:-tracks/method/configs/a1_reproduce.json}"
OUT="${OUT:-checkpoints/a1-reproduce-default}"

if [[ "$DEVICE" == "cuda" ]]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi

ARGS=(--policy.device="$DEVICE")
if [[ "${RESUME:-0}" == "1" ]]; then
  # Resume restores the saved run config (steps, seed, output dir) from the last checkpoint.
  LAST="$OUT/checkpoints/last/pretrained_model/train_config.json"
  [[ -f "$LAST" ]] || { echo "RESUME=1 but no checkpoint at $LAST" >&2; exit 1; }
  ARGS+=(--config_path="$LAST" --resume=true)
  [[ -n "${STEPS:-}" ]] && ARGS+=(--steps="$STEPS")   # e.g. to extend a finished run
else
  if [[ -e "$OUT" ]]; then
    echo "$OUT already exists. Use RESUME=1 to continue it, or choose another OUT." >&2
    exit 1
  fi
  ARGS+=(--config_path="$CONFIG" --output_dir="$OUT" --job_name="$(basename "$OUT")")
  [[ -n "${STEPS:-}" ]] && ARGS+=(--steps="$STEPS")
  [[ -n "${SAVE_FREQ:-}" ]] && ARGS+=(--save_freq="$SAVE_FREQ")
  [[ -n "${SEED:-}" ]] && ARGS+=(--seed="$SEED")
fi

mkdir -p "$(dirname "$OUT")" outputs/logs
LOG="outputs/logs/$(basename "$OUT")-$(date +%Y%m%d-%H%M%S).log"
echo "start $(date -u +%FT%TZ) device=$DEVICE out=$OUT resume=${RESUME:-0} log=$LOG" | tee "$LOG"

lerobot-train "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "end $(date -u +%FT%TZ)" | tee -a "$LOG"
