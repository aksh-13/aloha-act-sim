# ACT, explained (as LeRobot 0.7.0 implements it)

Draft for issue #4's last checkbox: the owner must be able to explain the model aloud. Read it,
then close it and explain each part from memory. Rewrite anything you couldn't.

Source: Zhao et al. 2023, arXiv 2304.13705; code in
`.venv/lib/python3.12/site-packages/lerobot/policies/act/modeling_act.py`.

## The one-line version

Given one camera image and the robot's 14 joint positions, ACT predicts the **next 100 actions at
once** (a "chunk"), as a conditional VAE whose decoder is a transformer.

## The pieces

### 1. Inputs and outputs
- **Image:** the overhead `top` camera, 480×640 RGB.
- **State:** 14 numbers, i.e. per arm 6 joint angles + 1 gripper.
- **Output:** a 100×14 block of target joint positions, the next 100 control steps (2 seconds at 50 Hz).

### 2. Image backbone: ResNet-18
ImageNet-pretrained ResNet-18 with the classifier removed. It turns the 480×640 image into a
15×20 grid of feature vectors (32× downsampling). Each of those 300 cells becomes one token, with a
2D sine position embedding so the transformer knows where in the image it came from.

### 3. Transformer encoder (4 layers, width 512, 8 heads)
Input tokens: `[latent z] + [joint state] + [300 image tokens]`, so 302 tokens.
Self-attention mixes everything together: "what the scene looks like, where my arms are, and which
style of demonstration I'm imitating".

### 4. Transformer decoder (1 layer in practice)
100 learned **query** embeddings, one per future timestep, cross-attend to the encoder output. A
linear head maps each of the 100 outputs to 14 joint targets. The paper lists 7 decoder layers; a
bug in the original code means only the first is used, and LeRobot copies that (1 layer).

### 5. The latent variable z: why there's a VAE
Human demonstrations are inconsistent: the same scene can be handled in slightly different ways.
Plain regression would average those ways into a blurry, wrong action. So:
- **Training:** a separate small transformer (the "CVAE encoder", 4 layers) reads the *true* future
  100 actions plus the joint state, and outputs a 32-dim Gaussian (mean μ, log-variance). Sample
  `z` from it and feed it to the main encoder. `z` tells the decoder "which style this demo used".
  Notably, this encoder does **not** see the image.
- **Inference:** there are no future actions to encode, so `z` is set to **zero**, the mean of the
  prior. The policy acts deterministically, in the "average style".

### 6. The two loss terms
`loss = L1(predicted chunk, true chunk) + 10 · KL(q(z | actions, state) ‖ N(0, I))`
- **L1 reconstruction:** absolute error over all 100×14 predicted actions, ignoring padded steps
  past the end of an episode. L1 rather than L2 because it gave more precise actions in the paper.
- **KL term (weight β = 10):** pulls the latent distribution toward a standard normal, so that
  `z = 0` at inference is a sensible, in-distribution value.

## How it acts at run time

### Action chunking (what we run in A1)
Query the model once, then **execute all 100 actions open-loop** (`n_action_steps = 100`), then
query again. Fewer decisions per episode means fewer chances for errors to compound, which is the
paper's central claim and what A2/A3 test.

### Temporal ensembling (off in A1, tested in A2)
Query **every** step instead. At step t you then hold up to 100 overlapping predictions for that
step (from the chunks predicted at t, t−1, …). Average them with weights `w_i = exp(−m·i)`,
where `w_0` is the oldest prediction; the original work used m = 0.01. Smoother motion, but 100×
more inference, and LeRobot requires `n_action_steps = 1` for it.

## Questions to answer aloud without notes
1. Why predict 100 actions instead of 1? What problem does that reduce?
2. What does `z` encode, where does it come from during training, and what is it at test time?
3. Why does the CVAE encoder not need the image?
4. What would go wrong without the KL term?
5. How many tokens does the main encoder see, and where does each come from?
6. What's the cost of temporal ensembling, and why does it force `n_action_steps = 1`?
