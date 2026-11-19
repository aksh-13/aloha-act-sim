"""Evaluation harness (issue B1): checkpoint in, one results JSON and videos out.

    python -m tracks.simulation.evaluate \
        --checkpoint checkpoints/act_transfer_cube_published --variant nominal --episodes 100

Episode i uses seed `--seed + i` for the cube start, so every variant and every checkpoint is
evaluated on the same episodes. The schema of the JSON file is in results/README.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from lerobot.utils.eval_stats import wilson_interval
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed

from tracks.simulation import failures
from tracks.simulation.perturbations import (
    VARIANTS,
    PerturbedTransferCubeEnv,
    distance_outside_training_range,
)

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
FPS = 50


def checkpoint_name(path: Path) -> str:
    """`checkpoints/a2-chunk-50/020000` -> `a2-chunk-50-020000`.

    A LeRobot training output, `outputs/train/<run>/checkpoints/<step>/pretrained_model`,
    becomes `<run>-<step>`.
    """
    parts = [part for part in path.resolve().parts if part != "pretrained_model"]
    if "checkpoints" not in parts:
        return parts[-1]
    i = len(parts) - 1 - parts[::-1].index("checkpoints")
    tail = parts[i + 1 :]
    if Path(*parts[:i]) != REPO_ROOT:
        tail = [parts[i - 1], *tail]
    return "-".join(tail)


def load_policy(checkpoint: Path, device: str):
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.pretrained_path = checkpoint
    policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def run_episode(env, policy, preprocessor, postprocessor, seed: int, keep_frames: bool) -> dict:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    observation, info = env.reset(seed=seed)
    frames = [observation["pixels"]["top"][::2, ::2]] if keep_frames else []
    trace = {key: [] for key in ("stage", "right_touch", "left_touch", "table_touch", "cube_z")}
    sum_reward = 0.0
    done = False
    while not done:
        batch = preprocess_observation(observation)
        batch["task"] = [env.task]
        with torch.inference_mode():
            action = postprocessor(policy.select_action(preprocessor(batch)))
        observation, reward, terminated, truncated, info = env.step(action[0].to("cpu").numpy())
        sum_reward += float(reward)
        for key in trace:
            trace[key].append(info[key])
        if keep_frames:
            frames.append(observation["pixels"]["top"][::2, ::2])
        done = terminated or truncated

    verdict = failures.classify(**trace)
    label = verdict.pop("label")
    x, y, _ = env.cube_start
    return {
        "seed": seed,
        "success": label == failures.SUCCESS,
        "max_stage": max(trace["stage"]),
        "final_stage": trace["stage"][-1],
        "sum_reward": sum_reward,
        "n_steps": len(trace["stage"]),
        "failure": None if label == failures.SUCCESS else label,
        "cube_start_xy": [round(float(x), 4), round(float(y), 4)],
        "cube_start_outside_training_range_m": round(distance_outside_training_range(x, y), 4),
        **verdict,
        "frames": frames,
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate(
    checkpoint: Path,
    variant: str,
    episodes: int,
    seed: int,
    device: str,
    videos: int,
    failure_videos: int,
    results_dir: Path,
    video_root: Path,
    name: str | None = None,
    episode_length: int = 400,
) -> Path:
    name = name or checkpoint_name(checkpoint)
    run_id = f"{name}__{variant}__seed{seed}_n{episodes}"
    video_dir = video_root / run_id
    set_seed(seed)
    policy, preprocessor, postprocessor = load_policy(checkpoint, device)
    env = PerturbedTransferCubeEnv(variant, episode_length=episode_length)

    start = time.time()
    records = []
    failure_videos_left = failure_videos
    for index in range(episodes):
        keep = index < videos or failure_videos_left > 0
        record = run_episode(env, policy, preprocessor, postprocessor, seed + index, keep)
        frames = record.pop("frames")
        record = {"episode": index, **record, "video": None}
        save = index < videos or (frames and not record["success"])
        if save:
            if index >= videos:
                failure_videos_left -= 1
            video_dir.mkdir(parents=True, exist_ok=True)
            path = video_dir / f"episode_{index:03d}.mp4"
            write_video(path, frames, FPS)
            record["video"] = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        records.append(record)
        n_success = sum(r["success"] for r in records)
        print(f"[{run_id}] episode {index + 1}/{episodes}  success so far {n_success}/{index + 1}", flush=True)

    n_success = sum(r["success"] for r in records)
    low, high = wilson_interval(n_success, episodes)
    config = policy.config
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {
            "name": name,
            "path": str(checkpoint),
            "policy_type": config.type,
            "chunk_size": getattr(config, "chunk_size", None),
            "n_action_steps": getattr(config, "n_action_steps", None),
            "temporal_ensemble_coeff": getattr(config, "temporal_ensemble_coeff", None),
        },
        "environment": {
            "task": "AlohaTransferCube-v0",
            "variant": variant,
            "episode_length": episode_length,
            "perturbation": VARIANTS[variant].to_dict(),
        },
        "seed": seed,
        "n_episodes": episodes,
        "n_success": n_success,
        "success_rate": n_success / episodes,
        "success_rate_ci95": [low, high],
        "ci_method": "wilson",
        "max_stage_counts": {str(s): sum(r["max_stage"] == s for r in records) for s in range(5)},
        "failure_counts": {
            category: sum(r["failure"] == category for r in records) for category in failures.FAILURE_CATEGORIES
        },
        "episodes": records,
        "device": device,
        "wall_time_s": round(time.time() - start, 1),
        "versions": {pkg: version(pkg) for pkg in ("lerobot", "gym-aloha", "mujoco", "torch")},
        "git_commit": git_commit(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    counts = Counter(r["failure"] for r in records if r["failure"])
    print(
        f"{run_id}: {n_success}/{episodes} = {100 * n_success / episodes:.1f}% "
        f"(95% Wilson {100 * low:.1f}-{100 * high:.1f}%)  failures: {dict(counts)}\n-> {out_path}"
    )
    return out_path


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="directory holding config.json and model.safetensors")
    parser.add_argument("--variant", default="nominal", choices=sorted(VARIANTS), metavar="VARIANT", help="environment variant; --list-variants shows them")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000, help="episode i uses seed + i (1000 matches lerobot-eval)")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--videos", type=int, default=10, help="save a video of the first N episodes")
    parser.add_argument("--failure-videos", type=int, default=0, help="also save up to N later failed episodes")
    parser.add_argument("--name", help="checkpoint name used in the result; derived from the path by default")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--video-root", type=Path, default=REPO_ROOT / "outputs" / "eval")
    parser.add_argument("--episode-length", type=int, default=400)
    parser.add_argument("--list-variants", action="store_true")
    args = parser.parse_args()
    if args.list_variants:
        for key, value in VARIANTS.items():
            print(f"{key:16s} {value.description}")
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    evaluate(
        args.checkpoint, args.variant, args.episodes, args.seed, args.device, args.videos,
        args.failure_videos, args.results_dir, args.video_root, args.name, args.episode_length,
    )


if __name__ == "__main__":
    main()
