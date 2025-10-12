# Fireboy & Watergirl: Collaborative Reinforcement Learning

A compact research & demo stack for **cooperative decision-making** across three paradigms: **Imitation Learning (BC)**, **LLM Planning**, and **Safety-Aware RL**. Built to show rapid iteration at a hackathon: record demos → clone behavior → add safe RL → compare against an LLM planner — all with persistent checkpoints and W&B tracking.

## Why this matters (hackathon-ready)

* **True cooperation**: Two agents (“Fire” & “Water”) must coordinate plates/switches to unlock exits.
* **Safety-first control**: A 100-step physics sim masks unsafe actions before they’re sampled.
* **Self-improvement loop**: BC seeds competent play; RL fine-tunes with partner-aware rewards; LLM plans zero-shot and logs traces for future training.
* **Human-in-the-loop**: Run human-only, AI-only, or hybrid (you drive one agent, AI drives the other).
* **Reproducible**: Full checkpointing of models/optimizers/replay + JSON logs + W&B.

## System at a glance

* **Behavior Cloning (IL)** — `env/maps/level_001BC.py`, `env/maps/train_il_agents.py`
  33-D state → 3-layer MLP → 6 discrete actions. Trains fast on curated demos; loads `models/*_best.pt`.

* **LLM Planner (Claude)** — `env/maps/level_001LLM.py`
  Structured JSON state → strict JSON action plan at ~2 Hz; heuristic fallback; exports RL-style traces.

* **Safety-Aware Cooperative RL (A2C)** — `env/maps/level_001RLv4.py`
  52-D partner-conditioned state (safety logits, hazards, partner features). Safety mask via trajectory sim; joint actor-critic with cooperation penalties; persistent replay & autosave.

## Real-World Applications (Industry)

* **Warehouse/AMR fleets**: Multi-robot aisle sharing, plate/switch ≈ interlocks/doors; safety mask ≈ “no-go” zones.
* **Manufacturing cells**: Robot-robot handoffs with shared fixtures; partner-aware rewards reduce idle/wait time.
* **Process & utilities**: Permit/lockout sequences mirror plate activation; safety layer prevents hazardous transitions.
* **Logistics yards/ports**: Crane–tug–ground crew coordination with collision-aware action masking.
* **AIOps / incident response**: Multi-agent runbooks where one agent isolates fault domains while another restores service.
* **Field robotics**: Teams navigating hazardous terrain; sim-to-policy transfer benefits from explicit hazard features.

## Benchmarks (tutorial map, typical)

* **BC**: 70–85% success after curated demos.
* **LLM**: Zero-shot planning at ~2 Hz with robust fallback.
* **RL**: >90% plate activation reliability; >70% end-to-end success ~1k episodes.

