"""Track C1: fixed demonstration subsets for the data-scaling curve.

Picks which of the 50 human demonstrations go into the 10- and 25-episode training sets,
once, with a fixed seed, and writes them to tracks/data/subsets.json so every run (and
every person) uses exactly the same episodes.

Subsets are nested (10 is inside 25, which is inside 50): the bigger set only ever adds
demonstrations. That way a change in success between sizes comes from the extra data, not
from swapping in a luckier or unluckier set of episodes.

    python tracks/data/define_subsets.py            # writes subsets.json, prints the train flags
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20261115
TOTAL = 50  # lerobot/aloha_sim_transfer_cube_human has episodes 0..49
SIZES = (10, 25, 50)


def main():
    order = np.random.default_rng(SEED).permutation(TOTAL).tolist()
    subsets = {f"n{n}": sorted(order[:n]) for n in SIZES}
    out = {
        "dataset": "lerobot/aloha_sim_transfer_cube_human",
        "seed": SEED,
        "nested": True,
        "how": "numpy default_rng(seed).permutation(50); subset n = first n of that order, sorted",
        "subsets": subsets,
        "train_flag": {k: "--dataset.episodes=[" + ",".join(map(str, v)) + "]" for k, v in subsets.items()},
    }
    path = Path(__file__).parent / "subsets.json"
    path.write_text(json.dumps(out, indent=1))
    for k, v in subsets.items():
        print(f"{k}: {v}")
    print(f"\nWrote {path}. Pass the matching flag to lerobot-train, e.g.\n  {out['train_flag']['n10']}")


if __name__ == "__main__":
    main()
