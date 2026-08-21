# RLHF-from-Scratch — Planning Document

## Goal

Build an RLHF stack (SFT → Reward Model → PPO → DPO) entirely by hand — no `trl` — and
run it end-to-end on GPU so every number in the project summary is measured, not
estimated:

1. Implemented the RLHF stack from scratch, hand-writing SFT, Bradley-Terry reward
   modeling, PPO and DPO losses instead of wrapping `trl`.
2. Built PPO with a clipped surrogate objective, GAE advantages and KL penalty,
   converging to **[KL]** against the frozen reference model.
3. Split trainable actor-critic from frozen reference-reward across two NCCL process
   groups, measuring **[X]%** VRAM reduction and **[Y]%** fewer bytes exchanged
   across the group boundary vs. a single merged group.
4. Ran a distilgpt2 and bert-tiny toy pipeline end-to-end on one GPU, with the reward
   model ranking **[X]%** of held-out pairs correctly.

Reference implementation for architectural ideas (not copied): `../Improving-LLM-Models-with-RLHF-PPO-DPO/`.

## Hardware plan

- **Dev machine (this one):** GTX 1650, driver too old for any CUDA build torch
  currently supports (`cuda.is_available()` is `False`). Used **CPU-only, code +
  correctness only** — every engine (SFT/reward/PPO/DPO) is written and unit-tested
  here against tiny synthetic data/models, never run as a real training job.
- **Training machine:** separate machine with an RTX 5080 (16GB VRAM), used later to
  actually run the toy pipeline end-to-end and produce the real numbers for bullets
  1, 2, 4.
- **Bullet 3 (multi-GPU NCCL topology) — no rental needed:** two processes run on the
  *same* RTX 5080, each capped to its own VRAM budget via
  `torch.cuda.set_per_process_memory_fraction()` (e.g. ~4-6GB each). This is enough
  to get two of the three numbers honestly:
  - **VRAM reduction is real** even on one physical GPU — it's a per-process memory
    footprint measurement (Group A loads only actor+critic; Group B loads only
    frozen reference+reward), not something that requires separate devices to be
    true.
  - **Bytes exchanged across the group boundary is real** — a genuine count of what
    crosses from Group A to Group B (rollout tokens, log-probs, reward scalars),
    regardless of physical topology.
  - **What is *not* claimed:** wall-clock/bandwidth savings from avoiding real
    PCIe/NVLink traffic — two processes on one GPU share the same on-die path, so
    there's no real interconnect cost being avoided. The project summary's bullet 3
    is worded around "bytes exchanged," not communication time, specifically so it
    stays honest under this single-GPU setup. No cloud rental phase is needed.

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

## Phase 3 — Reward model (Bradley-Terry, hand-written) ✅ COMPLETE (dev/correctness — real run pending on RTX 5080)

- [x] `src/rlhf_scratch/models/reward_model.py`: `RewardModel` — bert-tiny encoder
  + a linear scalar head over the last real token's hidden state.
- [x] `src/rlhf_scratch/training/reward.py`: `bradley_terry_loss()` —
  `-log(sigmoid(r_chosen - r_rejected))`, hand-written (no `trl` reward trainer) —
  plus `RewardTrainer` (train loop with AMP/grad-accum/clipping, and
  `evaluate_ranking_accuracy()`: fraction of held-out pairs where
  `r_chosen > r_rejected`, the real number behind bullet 4's **[X]%**).
- [x] `scripts/train_reward.py` CLI entrypoint — trains on toy/full HH-RLHF,
  evaluates on the untouched held-out split, writes `results/reward/metrics.json`.
- [x] Unit tests (`tests/test_reward.py`, 5 tests): Bradley-Terry loss correctness
  (correctly-ranked pairs score lower loss than reversed; the symmetric r=r case
  equals log(2) exactly), forward-shape check, an overfit-a-batch training check,
  and checkpoint save/reload.

**⚠️ Known issue to fix before Phase 8:** this transformers version cannot build a
fast tokenizer for `prajjwal1/bert-tiny`'s repo (only ships a legacy `vocab.txt`,
no `tokenizer.json`) — raises `Couldn't instantiate the backend tokenizer`. Tests
route around it using `bert-base-uncased`'s tokenizer as a stand-in. Before the
real run: either pin an older `transformers`, use `use_fast=False` with a
`tokenizers`-compatible fallback, or switch to a tiny BERT repo that ships
`tokenizer.json` (e.g. a converted upload of bert-tiny).

**Exit criteria:** held-out ranking accuracy computed and logged to
`results/reward/metrics.json`. ✅ Loop + eval verified correct on CPU with a tiny
synthetic BERT; real accuracy number pending the RTX 5080 run (blocked on the
tokenizer issue above).

---

## Phase 4 — PPO engine (hand-written) ✅ COMPLETE (dev/correctness — real run pending on RTX 5080)

- [x] `src/rlhf_scratch/training/ppo.py`, fully hand-written (no `trl`):
  - `sequence_logprobs()` — teacher-forced per-token log-probs of the actual
    next tokens.
  - `compute_kl_penalty()` — per-token `KL(actor‖reference)` via the sampled
    log-ratio approximation.
  - `compute_gae()` — Generalized Advantage Estimation. **Caught a real bug via
    the unit tests**: bootstrapping/propagation must be gated by whether the
    *next* position is valid (`mask[t+1]`), not the current position
    (`mask[t]`) — the original version let a padded position's garbage value
    leak backward into real tokens' advantages. Fixed and now covered by an
    explicit masking test.
  - `ppo_clipped_surrogate_loss()` — the clipped PPO objective.
  - `value_loss()` — masked critic MSE.
  - `PPOStep` — orchestrates one PPO update (multiple `ppo_epochs` over one
    rollout batch): combines reward-model score + KL penalty into per-token
    reward, computes GAE, then clipped-surrogate + value loss, with actor and
    critic as the only trainable parameters (reference/reward frozen and only
    queried) — mirrors the two-process-group split from Phase 6/7.
  - `generate_rollouts()` — samples from the actor, scores with reference/critic/
    reward model, ready for `PPOStep.update`. Not deeply unit-tested (needs a
    real generate() loop); exercised for real on the RTX 5080 run.
- [x] `src/rlhf_scratch/models/critic.py`: `Critic` — causal-LM backbone +
  per-token scalar head.
- [x] `scripts/train_ppo.py` CLI entrypoint; writes KL trace to
  `results/ppo/metrics.json`. Has a known TODO: reward-model scoring currently
  reuses the actor's tokenizer output rather than re-decoding/re-tokenizing with
  the reward model's own tokenizer — needs fixing before the real run.
- [x] Unit tests (`tests/test_ppo.py`, 9 tests): exact numeric checks for
  `sequence_logprobs`, KL, GAE (single-step, two-step vs. manual recursion,
  masked-padding), both branches of the clip (clipped vs. unclipped), value
  loss, plus an integration test running `PPOStep.update` on tiny synthetic
  actor/critic models for 5 iterations — confirms losses stay finite and KL
  stays bounded rather than exploding (the closest a dev-only CPU test can get
  to validating the "converges" claim in bullet 2).

**Exit criteria:** KL trace plotted, converges (not diverging/collapsing); reward
trending upward over training. ✅ Math verified correct via unit tests (including
one real bug caught and fixed); the actual convergence trace and final **[KL]**
number are pending the real toy-data run on the RTX 5080.

---

## Phase 5 — DPO engine (hand-written) ✅ COMPLETE (dev/correctness — real run pending on RTX 5080)

- [x] `data/preference_dataset.py::make_dpo_collate_fn` — tokenizes (prompt +
  chosen) and (prompt + rejected) as two sequences, each prompt-masked to -100,
  reusing the SFT collate_fn's masking convention.
- [x] `src/rlhf_scratch/training/dpo.py`, hand-written per Rafailov et al. (no
  `trl`):
  - `compute_sequence_logps()` — sums per-token log-probs (via `ppo.sequence_logprobs`)
    over response tokens only.
  - `dpo_loss()` — `-log(sigmoid(beta * [(logpi_chosen - logpi_rejected) -
    (logref_chosen - logref_rejected)]))`, plus implicit per-sequence reward
    margins (`beta * log-ratio to reference`) for win-rate evaluation.
  - `DPOTrainer` — trains the policy against a frozen deep-copied reference
    (reference queried under `torch.no_grad()`, never optimized);
    `evaluate_win_rate()` — fraction of held-out pairs where the implicit
    reward ranks chosen above rejected, DPO's analogue of the reward model's
    ranking accuracy.
- [x] `scripts/train_dpo.py` CLI entrypoint — same toy/full HH-RLHF data as
  Phase 3, for a clean PPO-vs-DPO comparison on identical data.
- [x] Unit tests (`tests/test_dpo.py`, 6 tests): loss correctness (favoring
  chosen relative to reference lowers the loss; the zero-logratio-gap case
  equals `log(2)` exactly, mirroring the Bradley-Terry symmetric case), prompt
  masking in both chosen/rejected, an overfit-a-batch training check, and
  explicit verification that the reference model's parameters never change
  across training steps.

**Exit criteria:** DPO run completes; win-rate vs. SFT baseline measured on held-out
prompts (secondary metric, supports bullet 1's "instead of wrapping trl" claim with
a working alternative). ✅ Loop + math verified correct on CPU with a tiny synthetic
model; real win-rate number pending the RTX 5080 run.

---

## Phase 6 — Distributed topology + VRAM/bytes benchmark (single-GPU, RTX 5080)

- `src/rlhf_scratch/distributed/topology.py`: two `dist.new_group()` groups —
  Group A (actor+critic, trainable), Group B (reference+reward, frozen/inference) —
  launched as two processes on the one RTX 5080 via `torch.multiprocessing.spawn`,
  each capped with `torch.cuda.set_per_process_memory_fraction()`.
- `src/rlhf_scratch/distributed/comm_hooks.py`: minimal cross-group payload
  (rollout tokens, log-probs, scalar rewards only — never full gradients/optimizer
  state across groups) + a byte-counter wrapping every cross-group send.
- **Baseline run:** all 4 models (actor, critic, reference, reward) loaded into a
  single process/group; record peak VRAM (`torch.cuda.max_memory_allocated`).
- **Split run:** Group A loads only actor+critic, Group B loads only
  reference+reward; record peak VRAM per process, and total bytes crossing the
  group boundary over N rollout steps.
- Compute real **[X]% VRAM reduction** (split vs. baseline) and real **[Y]% fewer
  bytes exchanged** (only what crosses groups vs. what a merged single-group setup
  would move internally). Record raw before/after numbers in
  `results/distributed_benchmark/`, not just the ratio, so the claim is auditable.
- No wall-clock/bandwidth communication-time claim is made (see hardware plan
  above for why that specifically needs separate physical GPUs).

**Exit criteria:** topology tests pass (`tests/test_distributed.py`), no deadlocks
across ≥50 simulated steps; both real percentages (VRAM, bytes) recorded with raw
numbers on the RTX 5080. No cloud rental required.

---

## Phase 7 — Integration, docs, final numbers

- `scripts/run_e2e_toy.py`: single command running SFT → Reward → PPO (or DPO) on
  the toy split end-to-end on the RTX 5080, timed.
- Fill in the four project-summary bullets with the real measured values from
  Phases 3, 4, 6.
- `README.md`: quick start, validated-vs-designed table (mirroring the honesty
  style of the reference repo), results tables/plots in `results/`.
- `docs/`: short write-ups per component (PPO math, DPO math, topology rationale).

**Exit criteria:** fresh clone → `pip install -e .` → `scripts/run_e2e_toy.py`
reproduces the headline numbers within noise.

---

## Sequencing note

Phases 1-6 have no external dependency beyond the local dev machine (code +
correctness) and the RTX 5080 (real runs) — no cloud rental needed anywhere in
this plan. Phase 6's distributed benchmark runs entirely on the RTX 5080 as two
processes sharing the one GPU.
