# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on learning submodular reward functions using Deep Q-Learning (DQN) in grid-world environments. The goal is to train neural networks that satisfy discrete Diminishing Returns (DR) submodularity constraints, comparing modular (linear) vs. submodular (non-modular) reward learning.

## Environment Setup

```bash
conda env create -f submodular_nn.yml
conda activate submodular_nn
```

Python 3.12.4, PyTorch 2.3.1 with Apple Silicon MPS backend, Gymnasium 1.0.0.

## Running

**Submodular net training (synthetic data):**
```bash
python3 submodular_net.py --no-modular   # submodular (diminishing returns)
python3 submodular_net.py --modular      # modular (linear)
```

**DQN in grid environment:**
```bash
python3 main.py -param subrl -env 1 -i 8
# -param: config file name under params/ (without .yaml)
# -env: environment variant ID
# -i: initial position index
```

Configuration lives in `params/subrl.yaml`. Multiple environment-type subdirectories exist under `params/` (entropy, coverage, steiner, gorilla, bimodal, two_rooms, GP), each with SRL/M/NM variants.

## Architecture

### Neural Network Hierarchy (`dqn.py`)

- **`IncreasingConcaveNet`**: Single φ network using ReLU to maintain concavity
- **`MonotoneSubmodularNet`**: Composed of multiple φ layers and m (feature mapping) networks; enforces monotone submodularity via composition: `φ[i](λ · Σm[i](x_j) + (1−λ) · φ[i-1](...))`
- **`PartialInputConcaveNN`**: Decomposes inputs into x/y components with separate concave branches
- **`DQN`**: Agent class wrapping online + target networks with ε-greedy exploration, soft target updates (τ=0.01), and experience replay

### Submodularity Enforcement

Two complementary mechanisms:
1. **Weight clamping** to `[0, ∞)` post-update to preserve monotonicity
2. **Concavity regularizer**: soft penalty `strength · Σ ReLU(−w)^power` on φ network weights, with strength increasing linearly over training epochs (ramps up through ~70% of epochs)

### Environment (`prize_grid_env.py`)

30×30 grid-world with configurable rewards, obstacles, and multi-agent support. Implements the Gymnasium interface. `GridWorld` is the raw engine; `GridWorldGym` wraps it as a Gym env.

### Supporting Modules

- **`environment.py`**: Graph generation + reward functions (log, facility location, graph cut, GP utility)
- **`metrics.py`**: Submodular function evaluation and dataset utilities for objectives (log, logdet, facility location, graph cut)
- **`replay_memory.py`**: Experience replay buffer (stores state/action/reward/next_state/done tuples)
- **`utils/visualization.py`**: Grid path plotting for trajectory analysis

### Training Data Flow

1. YAML config → environment initialization → agent setup
2. Per episode: reset → ε-greedy action selection → env step → store transition
3. Per learning step: sample batch → compute Double-DQN targets → MSE loss + concavity regularizer → backprop → weight clamp → soft target update

## Experiment Tracking

Uses Weights & Biases (`wandb`) for metric logging. Ensure you're logged in (`wandb login`) before running training scripts.
