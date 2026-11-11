# Project plan

## Goal

Train an Action Chunking Transformer (ACT) to make a simulated two-arm ALOHA robot hand a cube
from one gripper to the other, then answer three questions the original paper leaves open at
this scale:

1. **Method:** how much of ACT's performance comes from action chunking, and how does it cope with slow inference?
2. **Simulation:** where does the trained policy break when the scene changes?
3. **Data:** how many demonstrations does it need, and can targeted new data repair a failure?

The three questions join into one story: Track B finds a failure, Track C generates data to fix
it, Track A's training pipeline retrains, and Track B measures the recovery.

Paper: Zhao, Kumar, Levine, Finn, "Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware", 2023, arXiv 2304.13705.
Published target: 83.0% success over 500 episodes after 80,000 steps (LeRobot model card).

## What is already verified

On an Apple M4 Mac, LeRobot 0.7.0, Python 3.12:

- Install works; the published checkpoint runs in `gym-aloha` after a one-time format conversion.
- Evaluation: about 7.5 s per episode on the Mac GPU, videos and a 95% interval are produced.
- Training runs at about 1.15 s per step (batch 8) on the Mac, so 80,000 steps is roughly 25 hours there. Train on the RTX 4090.

Not verified: anything on the RTX 4090; the scripted demonstration dataset; the simulator edits below.

## Simulator facts the tracks build on

Read from the installed `gym_aloha` source:

- The cube starts at a random position with x in [0.0, 0.2] and y in [0.4, 0.6] (`gym_aloha/utils.py`, `sample_box_pose`).
- Reward is a stage counter from 0 to 4; an episode succeeds when it reaches 4 (`gym_aloha/tasks/sim.py`, `TransferCubeTask.get_reward`).
- The policy sees one overhead camera named `top`, 480x640.
- An end-effector-space version of the task exists (`gym_aloha/tasks/sim_end_effector.py`), which is what a scripted expert drives.

## Tracks

Each track has a build (code the owner writes), an experiment (runs and a figure), and a part
of the shared story. Each is sized for one person and uses a different skill.

### Track A — Method: the policy and its inference loop

Skill: model internals and control-loop engineering.

- **A1. Reproduce.** Train ACT for 80,000 steps; evaluate every 10,000 steps. Target 83%.
- **A2. Chunk-length ablation.** Train at chunk sizes 1, 10, 50, 100 on one shortened step budget. Add temporal ensembling on and off at inference.
- **A3. No-chunking baseline and compounding error.** Chunk size 1 is plain behaviour cloning. Measure how far its joint trajectory drifts from the nearest demonstration over an episode, against the chunked policy.
- **A4. Inference-delay stress test.** Modify the rollout loop so each new chunk arrives d steps late (the robot keeps executing the old chunk meanwhile). Plot success against delay. This is the problem real-time chunking papers address.

Headline figure: success against chunk length, and success against inference delay.

### Track B — Simulation: perturbations, evaluation and failure analysis

Skill: MuJoCo and simulator engineering, evaluation methodology.

- **B1. Evaluation harness.** One command: checkpoint in, JSON results out (success rate, 95% Wilson interval, per-episode stage reached, video paths). Validate on the published checkpoint with 100 episodes.
- **B2. Perturbation suite.** Environment variants, each switchable by a flag:
  - cube start range widened beyond the training range, in steps;
  - camera pose shifted and rotated;
  - visual changes: cube colour, table texture, lighting;
  - physical changes: cube mass and friction, an external push mid-episode.
- **B3. Automatic failure classification.** From simulator state per episode: never grasped, grasped then dropped, lifted but no handover, handover failed. Uses the reward stage plus contact and cube-height signals.
- **B4. Robustness matrix.** Published checkpoint and the team's checkpoints against every perturbation, with intervals.

Headline figure: robustness matrix (perturbation by severity), with a failure-type breakdown.

### Track C — Data: demonstrations, scripted expert and targeted repair

Skill: data engineering, classical robot control, dataset tooling.

- **C1. Data-scaling curve.** Train on 10, 25 and 50 demonstrations on the shared step budget. No published curve was found for this task.
- **C2. Scripted expert.** A waypoint policy in end-effector space that solves the transfer task, with noise injected so the demonstrations vary. Record its rollouts as a dataset in LeRobot format. Report the expert's own success rate.
- **C3. Human versus scripted.** Train on the scripted data and compare with the human-demonstration policy at equal size.
- **C4. Targeted repair.** Take the worst failure from B4 (for example, cube outside the training range). Generate scripted demonstrations covering it, retrain on human plus new data, and have Track B re-measure.

Headline figure: success against number of demonstrations; before-and-after bar for the repaired failure.

## Shared rules

- **One GPU queue.** Order: time 200 steps; A1 main run; then comparison runs alternating between tracks A and C. Track B needs no training runs of its own.
- **One shortened step budget** for every comparison run (set after the timing measurement), so runs are comparable.
- **Checkpoint naming:** `checkpoints/<track>-<experiment>-<variant>/<step>/`, for example `checkpoints/a2-chunk-50/020000/`.
- **Results:** every evaluation writes one JSON file under `results/`, produced by the B1 harness. Plots and tables are generated from those files only.
- **Evaluation size:** at least 100 episodes per reported number, fixed seeds, interval reported.
- **Deviations log:** any change from the published recipe goes in `docs/deviations.md` the moment it is made.

## Timeline

**Day one (base).** Environment on all machines; timing; A1 launched; B1 validated on the published
checkpoint; C1 subsets defined and queued; A2 configs ready; B2 first two perturbations; C2 expert
solving the nominal task. End of day: one trained policy, one success rate with interval, videos.

**Day two (simulation extension).** B2 complete and B4 run; A2, A3, A4 results; C2 dataset and C3;
C4 repair loop closed with Track B. End of day: three headline figures and the joined story.

## Writing

All prose (README, results tables, figure captions, the deviations log, resume lines) is written
with Claude from the JSON results. Each person must still be able to explain their own track aloud
without it.

## Honest framing

Simulation only. LeRobot's ACT implementation and the public human demonstrations are reused; the
perturbation suite, harness, failure classifier, scripted expert, generated datasets, inference-delay
loop and all experiments are the team's. Report the gap to the published number, and show failures
beside successes.
