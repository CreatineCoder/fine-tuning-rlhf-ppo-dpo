# RLHF-from-Scratch — Planning Document

## Goal

Build an RLHF stack (SFT → Reward Model → PPO → DPO) entirely by hand — no `trl` — and
run it end-to-end on a single consumer GPU (RTX 4050, 6GB) so every number in the
project summary is measured, not estimated:

1. Implemented the RLHF stack from scratch, hand-writing SFT, Bradley-Terry reward
   modeling, PPO and DPO losses instead of wrapping `trl`.
2. Built PPO with a clipped surrogate objective, GAE advantages and KL penalty,
   converging to **[KL]** against the frozen reference model.
3. Split trainable actor-critic from frozen reference-reward across two NCCL groups,
   measuring **[X]%** VRAM and **[Y]%** comms reduction.
4. Ran a distilgpt2 and bert-tiny toy pipeline end-to-end on one GPU, with the reward
   model ranking **[X]%** of held-out pairs correctly.

Reference implementation for architectural ideas (not copied): `../Improving-LLM-Models-with-RLHF-PPO-DPO/`.

## Hardware plan

- **Primary dev/training GPU:** RTX 4050 (6GB VRAM), local. Used for bullets 1, 2, 4
  in full — distilgpt2 (~82M params) + bert-tiny (~4.4M params) fit comfortably.
- **Bullet 3 (multi-GPU NCCL topology):** code is written and correctness-tested on
  the 4050 via single-GPU multi-process simulation throughout Phases 1-6. The final
  VRAM/comms numbers are captured in **Phase 7**, on a rented 2-GPU cloud instance
  (~1 hour, e.g. Vast.ai/RunPod spot). Until that run happens, this bullet stays
  labeled "designed + simulated, hardware-pending" — never a fabricated number.

## Models & data

- **Actor / policy:** `distilgpt2` (82M).
- **Reward model:** `bert-tiny` (4.4M) — Bradley-Terry pairwise loss.
- **Reference model:** frozen copy of the SFT'd `distilgpt2`.
- **Preference data:** Anthropic HH-RLHF, toy subset (~1K pairs) for fast iteration,
  larger held-out split for the final reward-ranking-accuracy number.

---

## Phase 0 — Scaffolding (this step)

- [x] Sibling repo `rlhf-ppo-dpo-scratch/` created.
- [x] Package skeleton: `src/rlhf_scratch/{models,data,training,distributed,utils}`.
- [x] `tests/`, `configs/`, `scripts/`, `docs/`, `results/`, `notebooks/` directories.
- [ ] `pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md` stub.
- [ ] `git init` + first commit.

**Exit criteria:** `pip install -e .` succeeds; package imports cleanly.

---

## Phase 1 — Data pipeline

- Download/cache Anthropic HH-RLHF via `datasets`.
- `src/rlhf_scratch/data/preference_dataset.py`: loads (prompt, chosen, rejected)
  triples, tokenizes for both distilgpt2 and bert-tiny tokenizers.
- Toy-mode flag: subsamples to ~1K pairs, with a fixed held-out split (~200 pairs)
  reserved untouched until the Phase 4 evaluation.
- Unit tests: dataset shapes, padding/truncation, no leakage between train/held-out.

**Exit criteria:** `pytest tests/test_data.py` green; a batch round-trips through both
tokenizers with correct shapes.

---

## Phase 2 — SFT (hand-written)

- `src/rlhf_scratch/training/sft.py`: plain PyTorch training loop (no `Trainer`,
  no `trl`) — causal LM loss on chosen responses, AMP, gradient accumulation for
  the 4050's 6GB budget.
- `scripts/train_sft.py` CLI entrypoint.
- Checkpoint saved to `results/sft/`.

**Exit criteria:** SFT loss curve decreasing and logged; checkpoint loadable by
Phase 3/4.

---

## Phase 3 — Reward model (Bradley-Terry, hand-written)

- `src/rlhf_scratch/models/reward_model.py`: bert-tiny + scalar head.
- `src/rlhf_scratch/training/reward.py`: Bradley-Terry pairwise loss
  `-log(sigmoid(r_chosen - r_rejected))`, written by hand (no `trl` reward trainer).
- Evaluation: accuracy = fraction of held-out pairs where `r_chosen > r_rejected`.
  This produces the real **[X]%** for bullet 4.

**Exit criteria:** held-out ranking accuracy computed and logged to
`results/reward/metrics.json`.

---

## Phase 4 — PPO engine (hand-written)

- `src/rlhf_scratch/training/ppo.py`:
  - Rollout generation from the actor (top-p/temperature sampling).
  - Reward scoring via the Phase 3 reward model.
  - Per-token KL penalty against the frozen reference model.
  - GAE advantage estimation (hand-written, not `trl`'s).
  - Clipped PPO surrogate objective + value loss + entropy bonus.
- Logs mean KL per PPO step; run until KL stabilizes near the target (e.g. 0.02-0.1
  nats, whatever the tuned target is) — that stabilized value becomes the real
  **[KL]** for bullet 2.
- `scripts/train_ppo.py` CLI entrypoint; metrics to `results/ppo/metrics.json`.

**Exit criteria:** KL trace plotted, converges (not diverging/collapsing); reward
trending upward over training.

---

## Phase 5 — DPO engine (hand-written)

- `src/rlhf_scratch/training/dpo.py`: reference-free-at-inference DPO loss
  (uses frozen reference only for the log-ratio term), hand-written per the DPO
  paper — no `trl`.
- Same preference data as reward model training, for a clean PPO-vs-DPO comparison.

**Exit criteria:** DPO run completes; win-rate vs. SFT baseline measured on held-out
prompts (secondary metric, supports bullet 1's "instead of wrapping trl" claim with
a working alternative).

---

## Phase 6 — Distributed topology (design + single-GPU simulation)

- `src/rlhf_scratch/distributed/topology.py`: two `dist.new_group()` groups —
  Group A (actor+critic, trainable), Group B (reference+reward, frozen/inference).
- `src/rlhf_scratch/distributed/comm_hooks.py`: minimal cross-group payload
  (rollout tokens, log-probs, scalar rewards only — never full gradients/optimizer
  state across groups).
- Validate on the 4050 via `torch.multiprocessing.spawn` with multiple processes
  sharing the one GPU (gloo or NCCL-on-one-device) — checks correctness and
  deadlock-freedom only. VRAM/comms numbers from this run are **not** used as the
  final bullet-3 metric (single GPU can't show a real reduction — see PLANNING
  rationale above); they're logged separately as "simulation, not final."

**Exit criteria:** topology tests pass (`tests/test_distributed.py`); no deadlocks
across ≥50 simulated steps.

---

## Phase 7 — Multi-GPU benchmark (rented hardware, final numbers)

- Rent a 2-GPU cloud instance for ~1 hour (Vast.ai/RunPod spot or similar).
- Baseline run: all 4 models in one process group on GPU0 (or replicated), measure
  peak VRAM (`torch.cuda.max_memory_allocated`) and NCCL bytes transferred.
- Split run: actor+critic on GPU0's group, reference+reward on GPU1's group, same
  measurement.
- Compute real **[X]% VRAM reduction** and **[Y]% comms reduction** from the two
  runs. Record raw logs in `results/distributed_benchmark/`.

**Exit criteria:** both real percentages recorded with raw before/after numbers,
not just the ratio (so the claim is auditable).

---

## Phase 8 — Integration, docs, final numbers

- `scripts/run_e2e_toy.py`: single command running SFT → Reward → PPO (or DPO) on
  the toy split end-to-end on the 4050, timed.
- Fill in the four project-summary bullets with the real measured values from
  Phases 3, 4, 7.
- `README.md`: quick start, validated-vs-designed table (mirroring the honesty
  style of the reference repo), results tables/plots in `results/`.
- `docs/`: short write-ups per component (PPO math, DPO math, topology rationale).

**Exit criteria:** fresh clone → `pip install -e .` → `scripts/run_e2e_toy.py`
reproduces the headline numbers within noise.

---

## Sequencing note

Phases 1-6 have no external dependency beyond the local 4050 and can run
back-to-back. Phase 7 is the only phase gated on renting hardware — everything
else is fully buildable and testable without it, and the code doesn't change
based on when Phase 7 happens.
