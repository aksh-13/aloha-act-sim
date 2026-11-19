# Deviations from the published recipe

Log every change from the LeRobot ACT transfer-cube recipe here, when it is made.

| Date | Run | Change | Reason |
|---|---|---|---|
| 2026-11-18 | A1 (`a1-reproduce-default`), and every run launched from `configs/a1_reproduce.json` | `env_eval_freq: 0` (LeRobot default 20000): no simulator rollouts during training | Checkpoints are evaluated separately with a fixed protocol (seed 1000, 100/500 episodes). Avoids unplanned GPU time and a rendering failure killing the run. Training updates are unchanged. |
