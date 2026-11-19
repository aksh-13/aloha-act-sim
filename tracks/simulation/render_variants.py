"""Render the first observation of every variant, for checking the perturbations by eye (issue B2).

    python -m tracks.simulation.render_variants --out outputs/variants

Writes one PNG per variant and a contact sheet per perturbation family. Push variants look
like the nominal scene at reset, so their frame is taken just after the push instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gym_aloha.constants import START_ARM_POSE
from PIL import Image, ImageDraw

from tracks.simulation.perturbations import VARIANTS, PerturbedTransferCubeEnv

HOLD = np.array(START_ARM_POSE[:6] + [1.0] + START_ARM_POSE[8:14] + [1.0], dtype=np.float32)


def first_frame(variant: str, seed: int) -> np.ndarray:
    env = PerturbedTransferCubeEnv(variant)
    observation, _ = env.reset(seed=seed)
    p = env.perturbation
    if p.push_force > 0 and not p.push_on_lift:
        for _ in range(p.push_at_step + p.push_steps + 40):
            observation, *_ = env.step(HOLD)
    return observation["pixels"]["top"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs/variants"))
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nominal = first_frame("nominal", args.seed)
    families: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, p in VARIANTS.items():
        frame = nominal if name == "nominal" else first_frame(name, args.seed)
        Image.fromarray(frame).save(args.out / f"{name}.png")
        if name != "nominal":
            families.setdefault(p.family, []).append((name, frame))

    for family, frames in families.items():
        tiles = [("nominal", nominal), *frames]
        sheet = Image.new("RGB", (640 * len(tiles), 480))
        for i, (name, frame) in enumerate(tiles):
            sheet.paste(Image.fromarray(frame), (640 * i, 0))
            ImageDraw.Draw(sheet).text((640 * i + 10, 10), f"{name}: {VARIANTS[name].description}", fill=(255, 255, 0))
        sheet.save(args.out / f"sheet_{family}.png")
    print(f"wrote {len(VARIANTS)} frames and {len(families)} contact sheets to {args.out}")


if __name__ == "__main__":
    main()
