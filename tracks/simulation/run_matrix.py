"""Evaluate one checkpoint on every variant, several variants at a time (issue B4).

    python -m tracks.simulation.run_matrix --checkpoint checkpoints/act_transfer_cube_published

Each variant is one call to the B1 harness in its own process. A variant whose result file
already exists is skipped, so an interrupted run can be restarted with the same command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tracks.simulation.evaluate import REPO_ROOT, checkpoint_name, default_device
from tracks.simulation.perturbations import VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), help="default: all")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument("--failure-videos", type=int, default=4)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--video-root", type=Path, default=REPO_ROOT / "outputs" / "eval")
    args = parser.parse_args()
    name = args.name or checkpoint_name(args.checkpoint)
    log_dir = args.video_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(variant: str) -> tuple[str, int]:
        run_id = f"{name}__{variant}__seed{args.seed}_n{args.episodes}"
        if (args.results_dir / f"{run_id}.json").exists():
            return variant, 0
        command = [
            sys.executable, "-m", "tracks.simulation.evaluate",
            "--checkpoint", str(args.checkpoint), "--name", name, "--variant", variant,
            "--episodes", str(args.episodes), "--seed", str(args.seed), "--device", args.device,
            "--videos", str(args.videos), "--failure-videos", str(args.failure_videos),
            "--results-dir", str(args.results_dir), "--video-root", str(args.video_root),
        ]  # fmt: skip
        with open(log_dir / f"{run_id}.log", "w") as log:
            code = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{'done' if code == 0 else 'FAILED'}  {variant}", flush=True)
        return variant, code

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failed = [variant for variant, code in pool.map(run, args.variants) if code != 0]
    if failed:
        sys.exit(f"failed variants: {failed}; logs in {log_dir}")


if __name__ == "__main__":
    main()
