#!/usr/bin/env bash
# Run any command inside the aloha-act image on the GPU server, with persistent
# checkpoints, outputs and Hugging Face cache mounted from the host. The repo's
# tracks/ folder is mounted too, so script edits on the server apply without a rebuild
# (changes to requirements.txt or the Dockerfile still need `docker build`).
#
#   NAME=a1-main tracks/method/docker_run.sh -d tracks/method/train_a1.sh cuda   # detached
#   tracks/method/docker_run.sh nvidia-smi                                         # foreground
#
# -d runs detached; the container is kept after it exits so `docker logs` still works.
# A stopped container with the same NAME is removed first; a running one is never touched.
set -euo pipefail

IMAGE="${IMAGE:-aloha-act:0.7.0}"
NAME="${NAME:-aloha-act-$(date +%Y%m%d-%H%M%S)}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

DETACH=0
if [[ "${1:-}" == "-d" ]]; then
  DETACH=1
  shift
fi

if [[ "$DETACH" == "1" ]]; then
  MODE=(-d)
else
  MODE=(--rm -i)
  [[ -t 0 ]] && MODE+=(-t)   # only ask for a TTY when there is one (not under nohup/ssh cmd/cron)
fi

if state="$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null)"; then
  if [[ "$state" == "running" ]]; then
    echo "Container $NAME is still running. Stop it first or choose another NAME." >&2
    exit 1
  fi
  docker rm "$NAME" >/dev/null
fi

mkdir -p "$REPO/checkpoints" "$REPO/outputs" "$HF_CACHE"

docker run "${MODE[@]}" --name "$NAME" --gpus all --shm-size=8g \
  -v "$REPO/tracks:/workspace/tracks" \
  -v "$REPO/checkpoints:/workspace/checkpoints" \
  -v "$REPO/outputs:/workspace/outputs" \
  -v "$HF_CACHE:/cache/huggingface" \
  "$IMAGE" "$@"

if [[ "$DETACH" == "1" ]]; then
  echo "Started $NAME. Follow with: docker logs -f $NAME"
fi
