# AlphaZero Gomoku 9x9

This repo contains an AlphaZero-style reinforcement learning system for 9x9 Gomoku, with trained checkpoints and a Pygame interface for playing against the final model.

## Environment Setup

There are three supported ways to create an environment.

### Option 1: GPU Conda Environment

Use this option for CUDA training and fastest inference:

```bash
conda env create -f gomoku-gpu.yml
conda activate gomoku-gpu
```

If the GPU environment already exists, update it instead:

```bash
conda env update -f gomoku-gpu.yml --prune
conda activate gomoku-gpu
```

### Option 2: CPU Conda Environment

Use this option on machines without an NVIDIA GPU:

```bash
conda env create -f gomoku-cpu.yml
conda activate gomoku-cpu
```

If the CPU environment already exists, update it instead:

```bash
conda env update -f gomoku-cpu.yml --prune
conda activate gomoku-cpu
```

### Option 3: uv Environment

Use this option if you prefer `uv` instead of conda:

```bash
uv sync
```

Run scripts through `uv`:

```bash
uv run python play.py
uv run python train.py
```

The `uv` configuration uses the PyTorch CUDA 12.4 wheel index on non-macOS platforms and the CPU wheel index on macOS.

### Check PyTorch Device

After setting up an environment, check whether PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Play Against The Final 800/40/200 Checkpoint

The included 800/40/200 checkpoint is:

```text
SUCCESS_800_40_200/model_final.pt
```

`800/40/200` means:

- `800` MCTS simulations per move
- `40` self-play games per training iteration
- `200` training iterations

Run the game:

```bash
python play.py
```

With `uv`, run:

```bash
uv run python play.py
```

`play.py` is already configured to load `SUCCESS_800_40_200/model_final.pt` and run with `800` MCTS simulations. When the window opens, click the left half to play first as Black, or the right half to play second as White.

## Implementation Notes

This is an AlphaZero adaptation for 9x9 Gomoku. It includes self-play, MCTS-guided policy targets, a residual policy-value network, checkpointing, evaluation against a baseline opponent, and an interactive GUI for testing the trained agent by hand.

The training loop uses batched MCTS self-play for better GPU utilization. Instead of running many small neural network calls one at a time, the search batches leaf evaluations across games so CUDA hardware is used more efficiently during training.

The training search also includes 50% randomised open-three threat detection. This gives the agent exposure to important tactical blocking positions without forcing every self-play game down the same defensive path. The goal is to address a self-play distribution gap where early agents may not naturally create enough dangerous open-three positions.

The saved experiment folders keep trained checkpoints and logs together. The final 800/40/200 checkpoint is included, so the trained agent can be run immediately without retraining.

## Run Training

Training settings are controlled by constants near the top of `train.py`.

For the 800/40/200 run, set:

```python
NUM_SIMULATIONS = 800
GAMES_PER_ITERATION = 40
NUM_ITERATIONS = 200
```

Then start training:

```bash
python train.py
```

With `uv`, run:

```bash
uv run python train.py
```

Training writes logs and checkpoints into `output/`:

```text
output/training_log.txt
output/model_iter_0010.pt
output/model_final.pt
```

After a successful 800/40/200 run, keep the result under a descriptive folder name such as `SUCCESS_800_40_200/`.
