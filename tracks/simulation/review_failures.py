"""Turn the saved failure videos of one result into frame strips for checking labels by eye (issue B3).

    python -m tracks.simulation.review_failures results/<run>.json --out outputs/review

One PNG per failed episode that has a video: 8 evenly spaced frames, captioned with the
automatic label and the events it was derived from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from tracks.simulation.evaluate import REPO_ROOT

FRAMES = 8


def strip(video: Path) -> Image.Image:
    with av.open(str(video)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    picks = np.linspace(0, len(frames) - 1, FRAMES).round().astype(int)
    height, width, _ = frames[0].shape
    sheet = Image.new("RGB", (width * FRAMES // 2, height * 2 + 16))
    draw = ImageDraw.Draw(sheet)
    for i, index in enumerate(picks):
        x, y = width * (i % (FRAMES // 2)), 16 + height * (i // (FRAMES // 2))
        sheet.paste(Image.fromarray(frames[index]), (x, y))
        draw.text((x + 4, y + 4), f"step {index}", fill=(255, 255, 0))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "review")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    out = args.out / result["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for episode in result["episodes"]:
        if episode["success"] or not episode["video"]:
            continue
        sheet = strip(REPO_ROOT / episode["video"])
        caption = (
            f"episode {episode['episode']}  label={episode['failure']}  max_stage={episode['max_stage']}  "
            f"first_lift={episode['first_lift_step']}  left_touch_after_lift={episode['first_left_touch_after_lift_step']}  "
            f"max_z={episode['max_cube_z']:.3f}  final_z={episode['final_cube_z']:.3f}"
        )
        ImageDraw.Draw(sheet).text((4, 2), caption, fill=(255, 255, 255))
        sheet.save(out / f"episode_{episode['episode']:03d}_{episode['failure']}.png")
        count += 1
    print(f"wrote {count} strips to {out}")


if __name__ == "__main__":
    main()
