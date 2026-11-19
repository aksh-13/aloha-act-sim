"""Robustness matrix figure and table, built from the JSON files in results/ only (issue B4).

    python -m tracks.simulation.plot_matrix

For every checkpoint with results, writes `results/figures/robustness_<checkpoint>.png` and
`results/robustness_<checkpoint>.md`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from tracks.simulation import failures
from tracks.simulation.evaluate import REPO_ROOT
from tracks.simulation.perturbations import VARIANTS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
# One-hue sequential ramp: light = low success, dark = high success.
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
FAILURE_COLOURS = dict(zip(failures.FAILURE_CATEGORIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], strict=True))
FAILURE_LABELS = {
    failures.NEVER_GRASPED: "never grasped",
    failures.GRASPED_THEN_DROPPED: "grasped then dropped",
    failures.LIFTED_NO_HANDOVER: "lifted, no handover",
    failures.HANDOVER_FAILED: "handover failed",
}
FAMILY_LABELS = {
    "cube_band": ("Cube start outside training range", ["0-1.5 cm", "1.5-3 cm", "3-6 cm"]),
    "cam_shift": ("Camera shifted sideways", ["1 cm", "3 cm", "6 cm"]),
    "cam_roll": ("Camera rolled", ["1°", "3°", "6°"]),
    "cam_tilt": ("Camera tilted", ["1°", "3°", "6°"]),
    "cube_colour": ("Cube colour", ["orange", "green", "blue"]),
    "table": ("Table appearance", ["mid grey", "brown", "stripes"]),
    "light": ("Lighting", ["70%", "40%", "160%"]),
    "mass": ("Cube mass", ["x2", "x4", "x8"]),
    "friction": ("Cube friction", ["x0.5", "x0.25", "x0.1"]),
    "push_table": ("Push on resting cube", ["~1 cm", "~3 cm", "~6 cm"]),
    "push_carry": ("Push while carried", ["2 N", "5 N", "10 N"]),
}


def load(results_dir: Path) -> dict[str, dict[str, dict]]:
    """checkpoint name -> variant -> result, keeping the largest run of each."""
    by_checkpoint: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text())
        if result.get("schema_version") != 1:
            continue
        runs = by_checkpoint[result["checkpoint"]["name"]]
        variant = result["environment"]["variant"]
        if variant not in runs or result["n_episodes"] > runs[variant]["n_episodes"]:
            runs[variant] = result
    return by_checkpoint


def percent(result: dict) -> tuple[float, float, float]:
    low, high = result["success_rate_ci95"]
    return 100 * result["success_rate"], 100 * low, 100 * high


def figure(name: str, runs: dict[str, dict], out: Path) -> None:
    families = [f for f in FAMILY_LABELS if any(f"{f}_{s}" in runs for s in (1, 2, 3))]
    rows = len(families)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, 0.62 * rows + 2.3), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.04}
    )
    fig.patch.set_facecolor(SURFACE)

    for row, family in enumerate(families):
        for column in range(3):
            result = runs.get(f"{family}_{column + 1}")
            if result is None:
                continue
            rate, low, high = percent(result)
            left.add_patch(
                plt.Rectangle((column + 0.02, row + 0.04), 0.96, 0.92, color=SEQUENTIAL(rate / 100), linewidth=0)
            )
            ink = "white" if rate >= 45 else INK
            level = FAMILY_LABELS[family][1][column]
            left.text(column + 0.5, row + 0.36, f"{rate:.0f}%", ha="center", va="center", color=ink, fontsize=13, fontweight="bold")
            left.text(column + 0.5, row + 0.74, f"{level}   [{low:.0f}-{high:.0f}]", ha="center", va="center", color=ink, fontsize=8.5)

            x = 0.0
            y = row + 0.2 + 0.27 * column
            for category in failures.FAILURE_CATEGORIES:
                count = 100 * result["failure_counts"][category] / result["n_episodes"]
                if count:
                    right.barh(y, count, left=x, height=0.2, color=FAILURE_COLOURS[category], edgecolor=SURFACE, linewidth=1.5)
                x += count
            right.text(-1.5, y, level, ha="right", va="center", fontsize=7.5, color=INK_MUTED)

    left.set_xlim(0, 3)
    left.set_ylim(rows, 0)
    left.set_xticks([0.5, 1.5, 2.5], ["severity 1", "severity 2", "severity 3"], color=INK_MUTED)
    left.set_yticks([r + 0.5 for r in range(rows)], [FAMILY_LABELS[f][0] for f in families], color=INK, fontsize=10.5)
    left.xaxis.tick_top()
    left.tick_params(length=0)
    left.set_title("Success rate, % of episodes [95% Wilson interval]", loc="left", fontsize=10.5, color=INK_MUTED, pad=24)

    right.set_xlim(-14, 100)
    right.set_ylim(rows, 0)
    right.set_yticks([])
    right.set_xticks([0, 25, 50, 75, 100])
    right.xaxis.tick_top()
    right.tick_params(length=0, colors=INK_MUTED)
    right.set_axisbelow(True)
    right.grid(axis="x", color="#e4e3df", linewidth=0.8)
    for row in range(1, rows):
        right.axhline(row, color="#e4e3df", linewidth=0.8)
    right.set_title("Failed episodes by type, % of episodes", loc="left", fontsize=10.5, color=INK_MUTED, pad=24)
    right.legend(
        handles=[Patch(color=FAILURE_COLOURS[c], label=FAILURE_LABELS[c]) for c in failures.FAILURE_CATEGORIES],
        loc="upper center", bbox_to_anchor=(0.45, -0.01), ncol=4, frameon=False, fontsize=9, labelcolor=INK,
    )  # fmt: skip
    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        for spine in axis.spines.values():
            spine.set_visible(False)

    title = f"Where the policy breaks: {name}"
    nominal = runs.get("nominal")
    if nominal:
        rate, low, high = percent(nominal)
        title += f"   (unperturbed: {rate:.0f}% [{low:.0f}-{high:.0f}], {nominal['n_episodes']} episodes)"
    fig.suptitle(title, x=0.012, ha="left", fontsize=13, color=INK, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def table(name: str, runs: dict[str, dict], out: Path) -> None:
    lines = [
        f"# Robustness matrix: {name}",
        "",
        "Generated by `python -m tracks.simulation.plot_matrix` from the JSON files in this folder.",
        "",
        "| Variant | What changed | Episodes | Success | 95% Wilson | never grasped | grasped then dropped | lifted, no handover | handover failed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variant in VARIANTS:
        result = runs.get(variant)
        if result is None:
            continue
        rate, low, high = percent(result)
        counts = " | ".join(str(result["failure_counts"][c]) for c in failures.FAILURE_CATEGORIES)
        lines.append(
            f"| `{variant}` | {VARIANTS[variant].description} | {result['n_episodes']} | {rate:.0f}% | {low:.0f}-{high:.0f}% | {counts} |"
        )
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    for name, runs in load(args.results_dir).items():
        figure(name, runs, args.results_dir / "figures" / f"robustness_{name}.png")
        table(name, runs, args.results_dir / f"robustness_{name}.md")
        print(f"{name}: {len(runs)} variants -> figure and table")


if __name__ == "__main__":
    main()
