# Results

One JSON file per evaluation, written by the Track B harness. Plots and tables are generated from these files only.

```bash
python -m tracks.simulation.evaluate --checkpoint <dir> --variant <name> --episodes 100
```

File name: `<checkpoint>__<variant>__seed<seed>_n<episodes>.json`.

## Schema (version 1)

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `run_id` | The file name without `.json` |
| `created_utc` | When the run finished |
| `checkpoint.name` | `<track>-<experiment>-<variant>-<step>`, derived from the checkpoint path or set with `--name` |
| `checkpoint.path`, `.policy_type`, `.chunk_size`, `.n_action_steps`, `.temporal_ensemble_coeff` | Read from the checkpoint's `config.json` |
| `environment.task` | `AlohaTransferCube-v0` |
| `environment.variant` | Variant name, `nominal` for the unmodified scene |
| `environment.perturbation` | Every parameter of the variant (see `tracks/simulation/perturbations.py`) |
| `environment.episode_length` | Step limit, 400 |
| `seed` | Episode `i` uses seed `seed + i` |
| `n_episodes`, `n_success`, `success_rate` | Rate is a fraction, 0 to 1 |
| `success_rate_ci95` | `[low, high]` as fractions, Wilson interval |
| `max_stage_counts` | Episodes by highest reward stage reached, keys `"0"` to `"4"` |
| `failure_counts` | Failed episodes by category |
| `episodes` | One record per episode, below |
| `device`, `wall_time_s`, `versions`, `git_commit` | Provenance |

Per-episode record:

| Field | Meaning |
|---|---|
| `episode`, `seed` | Index and the seed used |
| `success` | Reward stage 4 was reached |
| `max_stage`, `final_stage` | Reward stage, 0 to 4 |
| `sum_reward`, `n_steps` | Episodes stop at the first success |
| `failure` | `null` on success, else `never_grasped`, `grasped_then_dropped`, `lifted_no_handover` or `handover_failed` |
| `cube_start_xy`, `cube_start_outside_training_range_m` | Where the cube started |
| `first_right_touch_step`, `first_lift_step`, `first_left_touch_after_lift_step`, `max_cube_z`, `final_cube_z`, `held_by_right_at_end` | The simulator events the failure label was derived from |
| `video` | Path under `outputs/eval/`, or `null` if no video was kept |

Reward stages come from `gym_aloha`: 1 right gripper touches the cube, 2 right gripper holds it off the table,
3 left gripper touches it, 4 left gripper touches it off the table (success).

Videos are not committed. They show the camera the policy sees, at half resolution.
