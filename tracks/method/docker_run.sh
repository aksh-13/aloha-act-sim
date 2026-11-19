#!/usr/bin/env bash
# Run any command inside the aloha-act image on the GPU server, with persistent
# checkpoints, outputs and Hugging Face cache mounted from the host.
#
#   tracks/method/docker_run.sh -d tracks/method/train_a1.sh cuda   # detached training
#   tracks/method/docker_run.sh nvidia-smi                          # interactive check
#
# -d runs detached (follow with: docker logs -f <name>). NAME overrides the container name.
set -euo pipefail

IMAGE="${IMAGE:-aloha-act:0.7.0}"
NAME="${NAME:-aloha-act-$(date +%Y%m%d-%H%M%S)}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

MODE=(--rm -it)
if [[ "${1:-}" == "-d" ]]; then
  MODE=(-d)
  shift
fi

mkdir -p "$REPO/checkpoints" "$REPO/outputs" "$HF_CACHE"

docker run "${MODE[@]}" --name "$NAME" --gpus all --shm-size=8g \
  -v "$REPO/checkpoints:/workspace/checkpoints" \
  -v "$REPO/outputs:/workspace/outputs" \
  -v "$HF_CACHE:/cache/huggingface" \
  "$IMAGE" "$@"

[[ "${MODE[0]}" == "-d" ]] && echo "Started $NAME. Follow with: docker logs -f $NAME"
