# AlphaZero Gomoku 9x9

This repo contains an AlphaZero-style reinforcement learning system for 9x9 Gomoku, with trained checkpoints and a Pygame interface for playing against the final model.

## GPU Setup With Conda

Create the CUDA-enabled conda environment:

```bash
conda env create -f gomoku-gpu.yml
conda activate gomoku-gpu
```

If the environment already exists, update it instead:

```bash
conda env update -f gomoku-gpu.yml --prune
conda activate gomoku-gpu
```

Check that PyTorch can see the GPU:

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
conda activate gomoku-gpu
python play.py
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
conda activate gomoku-gpu
python train.py
```

Training writes logs and checkpoints into `output/`:

```text
output/training_log.txt
output/model_iter_0010.pt
output/model_final.pt
```

After a successful 800/40/200 run, keep the result under a descriptive folder name such as `SUCCESS_800_40_200/`.
