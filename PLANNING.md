# RLHF-from-Scratch — Planning Document

## Goal

Build an RLHF stack (SFT → Reward Model → PPO → DPO) entirely by hand — no `trl` — and
run it end-to-end on GPU so every number in the project summary is measured, not
estimated:

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

- **Dev machine (this one):** GTX 1650, driver too old for any CUDA build torch
  currently supports (`cuda.is_available()` is `False`). Used **CPU-only, code +
  correctness only** — every engine (SFT/reward/PPO/DPO) is written and unit-tested
  here against tiny synthetic data/models, never run as a real training job.
- **Training machine:** separate machine with an RTX 5080, used later to actually run
  the toy pipeline end-to-end and produce the real numbers for bullets 1, 2, 4.
- **Bullet 3 (multi-GPU NCCL topology):** a single GPU — even the 5080 — cannot be
  split into two independent devices for this (no MIG on consumer RTX cards; multiple
  processes on one GPU still share one VRAM pool/bus, so any measurement that way is
  a simulation, not a real result). Code is written and correctness-tested via
  single-GPU multi-process simulation throughout Phases 1-6. The final VRAM/comms
  numbers require a **second, separate GPU** alongside the 5080 (or a rented 2-GPU
  cloud instance) and are captured in **Phase 7**. Until that run happens, this
  bullet stays labeled "designed + simulated, hardware-pending" — never a fabricated
  number.

## Models & data

- **Actor / policy:** `distilgpt2` (82M).
- **Reward model:** `bert-tiny` (4.4M) — Bradley-Terry pairwise loss.
- **Reference model:** frozen copy of the SFT'd `distilgpt2`.
- **Preference data:** Anthropic HH-RLHF, toy subset (~1K pairs) for fast iteration,
  larger held-out split for the final reward-ranking-accuracy number.

---

## Phase 0 — Scaffolding ✅ COMPLETE

- [x] Package skeleton: `src/rlhf_scratch/{models,data,training,distributed,utils}`.
- [x] `tests/`, `configs/`, `scripts/`, `docs/`, `results/`, `notebooks/` directories.
- [x] `pyproject.toml`, `.gitignore`, `README.md` stub.
- [x] Repo already tracked in git with a GitHub remote (`fine-tuning-rlhf-ppo-dpo`).

**Exit criteria:** `pip install -e .` succeeds; package imports cleanly. ✅ Verified —
editable install succeeds, `import rlhf_scratch` + all five subpackages import clean.

---

## Phase 1 — Data pipeline ✅ COMPLETE

- [x] `src/rlhf_scratch/data/preference_dataset.py`: downloads/caches
  `Anthropic/hh-rlhf` via `datasets`, extracts (prompt, chosen, rejected) triples
  by diffing the shared prefix of `chosen`/`rejected` at the last `Assistant:`
  turn marker.
- [x] `load_preference_pairs(split, toy, seed)`: fixed seeded shuffle carves out a
  **200-pair held-out split first**, before any toy subsampling — so held-out stays
  identical across toy/full runs and is never trained on. `toy=True` subsamples the
  remainder to 1,000 pairs.
- [x] `PreferenceDataset` (torch `Dataset`) + `make_collate_fn(tokenizer)` — same
  dataset backs both the distilgpt2 actor and bert-tiny reward model, since
  tokenization happens per-tokenizer at collation time, not at dataset build time.
- [x] Unit tests (`tests/test_data.py`, 6 tests): prompt-extraction correctness,
  dataset `len`/`getitem`, collate_fn output shapes/keys, toy-vs-held-out
  disjointness, held-out seed-stability across calls.

**Exit criteria:** `pytest tests/test_data.py` green; a batch round-trips through both
tokenizers with correct shapes. ✅ Verified — `6 passed` against the real
`Anthropic/hh-rlhf` dataset (network download confirmed working), 41s runtime.

---

## Phase 2 — SFT (hand-written) ✅ COMPLETE (dev/correctness — real run pending on RTX 5080)

- [x] `data/preference_dataset.py::make_sft_collate_fn` — tokenizes (prompt +
  chosen) as one sequence, masks prompt + pad tokens to `-100` in `labels` so the
  causal LM loss only trains on response tokens.
- [x] `src/rlhf_scratch/training/sft.py`: plain PyTorch training loop (no
  `Trainer`, no `trl`) — `sft_loss()` (manual shifted cross-entropy, ignoring
  masked positions) + `SFTTrainer` (explicit AMP autocast/GradScaler, gradient
  accumulation, grad-norm clipping, checkpoint saving).
- [x] `scripts/train_sft.py` CLI entrypoint (typer) — loads `distilgpt2` +
  toy/full HH-RLHF data, runs `SFTTrainer`, saves to `results/sft/checkpoint`.
- [x] Unit tests (`tests/test_sft.py`, 4 tests) against a tiny randomly-initialized
  GPT-2 (2 layers, n_embd=16) — no model download needed, fast on CPU:
  masked-position loss correctness, prompt-masking in the collate_fn, an actual
  overfit-a-single-batch check (loss must decrease over epochs — proves the loop
  really trains, not just runs), and checkpoint save/reload shape.

**Exit criteria:** SFT loss curve decreasing and logged; checkpoint loadable by
Phase 3/4. ✅ Verified on CPU with the tiny synthetic model (`test_sft_trainer_reduces_loss_on_single_batch`).
⚠️ Not yet run as a real training job on `distilgpt2` + real toy data — that
happens on the RTX 5080 machine per the updated hardware plan above; this dev
machine (GTX 1650, CPU-only torch) is code + correctness only.

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
