# LLM + RL Cooperative Agent System

**Overview**
Real-time, multi-agent puzzle solver combining LLM reasoning with reinforcement learning; supports human-only, AI-only, and hybrid control.

**Architecture**

* Model: Claude Sonnet 4 (Anthropic API)
* Decision rate: 2 Hz (≈500 ms) with ≤150 tokens/decision
* I/O: JSON state in (positions, distances, switches, hazards) → JSON actions out (move, jump, wait)

**RL**
Reward: `R = 0.1·Δdist + 5·Δwalls + 10·Δswitch + 100·success - 50·death`
Logs `(s, a, r, s')`, reward traces, success metrics, human vs AI attribution; exports JSON/NPZ.

**Control Hierarchy**

1. LLM strategic policy → 2) rule-based fallbacks (pathfinding, hazard avoidance) → 3) anti-stagnation triggers.

**Features**

* Real-time LLM control and cooperative strategy
* Hybrid human+AI teamwork mode
* Live RL dataset generation and visualization (HUD, rewards, modes)
* Multi-level support with transfer across levels

**Use Cases**
Zero-shot control, human–AI collaboration studies, online reward tuning, behavioral cloning.

**Requirements**
Anthropic API key; Python 3.10+ with NumPy, Pygame; network access.

