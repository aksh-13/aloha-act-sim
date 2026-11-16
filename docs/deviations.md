# Deviations from the published recipe

Log every change from the LeRobot ACT transfer-cube recipe here, when it is made.

| Date | Run | Change | Reason |
|---|---|---|---|
| 2026-11-15 | C2 scripted dataset | New data, not a recipe change: scripted demonstrations generated with `tracks/data/generate_dataset.py` (same features, fps, 400-step episodes and task string as the human set). Episodes kept only if the handover succeeds and the cube is still held on the last step. | Track C2/C3 needs a second data source; filter avoids teaching a drop. |
| 2026-11-15 | C1 subsets | Human subsets of 10 and 25 episodes fixed in `tracks/data/subsets.json` (seed 20261115, nested). | Comparable data-scaling runs. |
| 2026-11-15 | Environment | Track C scripts run on Windows 11, Python 3.12, LeRobot 0.7.0, MuJoCo 3.8.1. `lerobot-train` fails at checkpoint save on Windows (symlink privilege); no effect on Mac/Linux. | Version note for issue #1. |
