"""Side-by-side video: one human demonstration next to the scripted expert on the same cube position.

The human dataset does not store where the cube started, so it is estimated from the human
demo itself: the right gripper's pad point at the moment it closes (forward kinematics on
the recorded joint state) sits right on the cube. The scripted expert is then run from that
cube position, and both are shown from the `top` camera (the only camera the policy sees).

    python tracks/data/make_comparison_video.py --human-episode 0 --out videos/human_vs_scripted_ep0.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import compare_datasets as cd  # noqa: E402
import scripted_expert as se  # noqa: E402

W, H = 640, 480


def label(frame: np.ndarray, title: str, sub: str) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    try:
        f1, f2 = ImageFont.truetype("arial.ttf", 22), ImageFont.truetype("arial.ttf", 16)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    d.rectangle([0, 0, W, 58], fill=(0, 0, 0))
    d.text((10, 6), title, fill=(255, 255, 255), font=f1)
    d.text((10, 34), sub, fill=(200, 200, 200), font=f2)
    return np.asarray(im)


STAGES = {0: "not touching", 1: "right touches cube", 2: "right lifted cube", 3: "left touches cube",
          4: "left holds cube (success)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human-episode", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7, help="seed for the expert's random variation")
    ap.add_argument("--out", type=Path, default=Path("videos/human_vs_scripted_ep0.mp4"))
    args = ap.parse_args()

    human = cd.load(cd.HUMAN, None)
    ds = human["ds"]
    ep_mask = human["ep"] == args.human_episode
    S, A = human["state"][ep_mask], human["action"][ep_mask]
    r_close = cd.first_drop(A[:, 13], int(np.argmax(A[:, 13][:250])))
    pad = cd.FK()(S[r_close])["right"]["pad"]
    cube = np.array([pad[0], pad[1], 0.05, 1, 0, 0, 0])
    print(f"human episode {args.human_episode}: estimated cube start x={pad[0]:.3f} y={pad[1]:.3f}")

    # human frames from the dataset video
    start = int(np.where(ep_mask)[0][0])
    human_frames = []
    for i in range(int(ep_mask.sum())):
        x = ds[start + i]["observation.images.top"]  # (3, H, W) float in [0, 1]
        human_frames.append((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    # scripted episode from the same cube position, recorded from the joint-space replay
    rng = np.random.default_rng(args.seed)
    ee = se.rollout_ee(cube, rng)
    script_frames = []
    rw = se.replay_joint(ee["joint_traj"], cube, render_top=True,
                         on_frame=lambda t, obs, a: script_frames.append(obs["top"].copy()))
    # stage shown on frame t is the stage after the previous step
    s_stage = np.concatenate([[0], rw[:-1]]).astype(int)
    print(f"scripted: max stage {int(rw.max())}, final stage {int(rw[-1])}")

    # human stage: replaying the human commands is not guaranteed to reproduce the episode
    # (estimated cube pose), so the human panel is labelled by gripper events instead
    l_close = cd.first_drop(A[:, 6], int(np.argmax(A[:, 6])))

    def human_event(t):
        if t < r_close:
            return "reaching for the cube"
        if t < l_close:
            return "right holds cube"
        return "left closes on cube"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(args.out), "w") as out:
        st = out.add_stream("libx264", rate=50)
        st.width, st.height, st.pix_fmt = 2 * W, H, "yuv420p"
        for t in range(min(len(human_frames), len(script_frames))):
            left = label(human_frames[t], f"HUMAN demo (episode {args.human_episode})", f"step {t}/400  |  {human_event(t)}")
            right = label(script_frames[t], "SCRIPTED expert (same cube start)", f"step {t}/400  |  {STAGES[s_stage[t]]}")
            frame = av.VideoFrame.from_ndarray(np.concatenate([left, right], axis=1), format="rgb24")
            for pkt in st.encode(frame):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
