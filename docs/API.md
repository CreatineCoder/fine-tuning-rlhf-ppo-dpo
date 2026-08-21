# API Documentation

Reference for the public surface of `rlhf_scratch` (Phases 1-5: data pipeline,
SFT, Bradley-Terry reward model, PPO, DPO). Phase 6 (`rlhf_scratch.distributed`)
is documented separately once built.

All modules live under `src/rlhf_scratch/`. Import from the top-level
subpackage (`rlhf_scratch.data`, `rlhf_scratch.models`, `rlhf_scratch.training`)
— each subpackage's `__init__.py` re-exports its public names.

---

## `rlhf_scratch.data`

Preference-pair dataset (Anthropic HH-RLHF) with toy-mode subsampling and a
fixed held-out split. Source: `src/rlhf_scratch/data/preference_dataset.py`.

### `PreferencePair`

```python
@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
```

One preference triple. `prompt` is the shared context; `chosen`/`rejected` are
the two candidate continuations.

### `extract_prompt(chosen: str, rejected: str) -> str`

Recovers the shared prompt from a raw HH-RLHF `(chosen, rejected)` pair by
diffing their common prefix and cutting at the last `"Assistant:"` turn
marker. Falls back to the raw common prefix if no marker is found.

### `load_preference_pairs(split="train", toy=False, seed=42) -> tuple[Dataset, Dataset]`

Downloads/caches `Anthropic/hh-rlhf` via 🤗 `datasets`, and returns
`(train_pool, held_out)`.

- A **200-pair held-out split is carved out first**, from a fixed seeded
  shuffle of the full `split` — identical across calls regardless of `toy`,
  and never touched during training.
- If `toy=True`, `train_pool` is subsampled to 1,000 pairs from the remainder.
- Both returned datasets have columns `prompt`, `chosen`, `rejected` (already
  passed through `extract_prompt`).

| Constant | Value | Meaning |
|---|---|---|
| `HELD_OUT_SIZE` | 200 | Size of the fixed held-out split |
| `TOY_TRAIN_SIZE` | 1000 | Size of the toy-mode training subsample |
| `SEED` | 42 | Default shuffle seed |

### `PreferenceDataset(hf_dataset)`

`torch.utils.data.Dataset` wrapper over a HF dataset with `prompt`/`chosen`/
`rejected` columns. `__getitem__` returns a `PreferencePair`.

### `make_collate_fn(tokenizer, max_length=512)`

Returns a `collate_fn` for **reward-model** training: tokenizes
`(prompt + chosen)` and `(prompt + rejected)` as full sequences, no masking.
Output dict:

```python
{
    "chosen_input_ids": LongTensor[B, T],
    "chosen_attention_mask": LongTensor[B, T],
    "rejected_input_ids": LongTensor[B, T],
    "rejected_attention_mask": LongTensor[B, T],
}
```

### `make_sft_collate_fn(tokenizer, max_length=512)`

Returns a `collate_fn` for **SFT**: tokenizes `(prompt + chosen)` as one
sequence and masks the prompt (and any pad) tokens to `-100` in `labels`, so
the causal LM loss only trains on the response. Output dict:
`{"input_ids", "attention_mask", "labels"}`, all `[B, T]`.

### `make_dpo_collate_fn(tokenizer, max_length=512)`

Returns a `collate_fn` for **DPO**: like `make_sft_collate_fn`, but applied to
both `chosen` and `rejected` independently. Output dict:

```python
{
    "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
    "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
}
```

---

## `rlhf_scratch.models`

### `RewardModel(base_model_name="prajjwal1/bert-tiny")`

`nn.Module`. Wraps a 🤗 `AutoModel` encoder with a linear scalar head.

```python
def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Returns per-sequence scalar reward, shape (B,)."""
```

Pools the encoder's hidden state at each sequence's **last real (non-pad)
token** (via `attention_mask.sum(dim=1) - 1`) before the scalar head.

⚠️ Known issue: the current `transformers` version can't build a fast
tokenizer for `prajjwal1/bert-tiny`'s repo (legacy `vocab.txt` only, no
`tokenizer.json`). See `PLANNING.md` Phase 3 for the fix needed before the
real training run.

### `Critic(base_model_name="distilgpt2")`

`nn.Module`. Causal-LM backbone (its `transformer`/decoder submodule) plus a
linear scalar head applied at every position — the PPO value function.

```python
def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
    """Returns per-token value estimate, shape (B, T)."""
```

---

## `rlhf_scratch.training` — SFT

Source: `src/rlhf_scratch/training/sft.py`.

### `sft_loss(model, input_ids, attention_mask, labels) -> Tensor`

Manual shifted cross-entropy: `logits[:, :-1]` predicts `labels[:, 1:]`,
`ignore_index=-100`. Assumes `labels` already has prompt/pad tokens masked
(see `make_sft_collate_fn`).

### `SFTConfig`

```python
@dataclass
class SFTConfig:
    lr: float = 5e-5
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    epochs: int = 1
    use_amp: bool | None = None  # None => True on CUDA, False on CPU
```

### `SFTTrainer(model, optimizer, device, config=None)`

- `train_epoch(dataloader) -> list[float]` — one pass over `dataloader`,
  gradient accumulation + AMP autocast/`GradScaler` + grad-norm clipping
  applied per `config`. Returns the per-batch loss values.
- `save_checkpoint(path)` — uses `model.save_pretrained(path)` if available
  (HF models), else `torch.save(model.state_dict(), path)`.

---

## `rlhf_scratch.training` — Reward model

Source: `src/rlhf_scratch/training/reward.py`.

### `bradley_terry_loss(chosen_rewards, rejected_rewards) -> Tensor`

`-log(sigmoid(chosen_rewards - rejected_rewards)).mean()` — the Bradley-Terry
pairwise preference NLL. Symmetric case (`chosen == rejected`) evaluates to
exactly `log(2)`.

### `RewardConfig`

```python
@dataclass
class RewardConfig:
    lr: float = 1e-4
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool | None = None
```

### `RewardTrainer(model, optimizer, device, config=None)`

- `train_epoch(dataloader) -> list[float]` — same shape as `SFTTrainer`, but
  scores `chosen_input_ids`/`rejected_input_ids` through `model` and applies
  `bradley_terry_loss`.
- `evaluate_ranking_accuracy(dataloader) -> float` — fraction of pairs where
  `r(chosen) > r(rejected)`. This is the metric behind bullet 4's held-out
  ranking accuracy number. Runs under `torch.no_grad()`, sets `model.eval()`.
- `save_checkpoint(path)` — `torch.save(model.state_dict(), path)`.

---

## `rlhf_scratch.training` — PPO

Source: `src/rlhf_scratch/training/ppo.py`. All hand-written, no `trl`.

### `sequence_logprobs(logits, input_ids) -> Tensor`

Teacher-forced per-token log-prob of the actual next token.
`logits: (B, T, V)`, `input_ids: (B, T)` → returns `(B, T-1)`, aligned with
`input_ids[:, 1:]`.

### `compute_kl_penalty(logprobs, ref_logprobs) -> Tensor`

`logprobs - ref_logprobs` — the standard sampled-log-ratio approximation of
per-token `KL(actor‖reference)`.

### `compute_gae(rewards, values, mask, gamma=1.0, lam=0.95) -> tuple[Tensor, Tensor]`

Generalized Advantage Estimation. `rewards`, `values`, `mask`: `(B, T)`.
Returns `(advantages, returns)`, same shape.

**Masking convention (load-bearing — see Phase 4 bug writeup in
`PLANNING.md`):** bootstrapping/propagation at step `t` is gated by whether
step `t+1` is a real token (`mask[:, t+1]`), **not** `mask[:, t]`. Getting this
backwards lets a padded position's garbage value leak into real tokens'
advantages — this was caught by `tests/test_ppo.py::test_compute_gae_respects_mask_padding`.

### `ppo_clipped_surrogate_loss(logprobs, old_logprobs, advantages, mask, clip_eps=0.2) -> Tensor`

The clipped PPO surrogate objective, negated for gradient descent:
`-min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`, masked-averaged.

### `value_loss(values, returns, mask) -> Tensor`

Masked MSE: `((values - returns)**2 * mask).sum() / mask.sum()`.

### `PPOConfig`

```python
@dataclass
class PPOConfig:
    lr: float = 1e-5
    clip_eps: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    kl_coef: float = 0.1
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
```

### `PPOStep(actor, critic, optimizer, device, config=None)`

One PPO update given a batch of already-generated, already-scored rollouts.
Actor + critic are trainable; reference/reward are only ever queried upstream
(in `generate_rollouts`), never optimized here.

- `compute_rewards_with_kl(env_rewards, logprobs, ref_logprobs, mask) -> tuple[Tensor, Tensor]`
  — combines `-kl_coef * KL` per token with the scalar reward-model score
  added at each sequence's last valid token. Returns
  `(per_token_rewards, mean_kl)`.
- `update(rollout_batch: dict) -> dict[str, float]` — runs `config.ppo_epochs`
  passes over one rollout batch: computes GAE once, normalizes advantages,
  then repeatedly computes clipped-surrogate + value loss and steps the
  optimizer. Returns `{"pg_loss", "value_loss", "mean_kl"}` from the last
  epoch.

  `rollout_batch` keys: `input_ids (B,T)`, `response_mask (B,T-1)`,
  `old_logprobs (B,T-1)`, `ref_logprobs (B,T-1)`, `old_values (B,T-1)`,
  `env_rewards (B,)`.
- `save_checkpoint(path)` — writes `actor.pt` and `critic.pt` under `path`.

### `generate_rollouts(actor, reference, critic, reward_model, prompt_input_ids, prompt_attention_mask, tokenizer, max_new_tokens=32, temperature=1.0, top_p=1.0) -> dict`

Samples responses from `actor.generate(...)`, then scores them with
`reference` (for KL), `critic` (for values), and `reward_model`. Returns a
dict shaped for `PPOStep.update`. Runs under `torch.no_grad()`.

⚠️ Known TODO (see `scripts/train_ppo.py`): reward-model scoring currently
reuses the actor's tokenizer output rather than re-decoding/re-tokenizing
generated text with the reward model's own tokenizer — fix before the real
training run if actor and reward-model tokenizers differ.

---

## `rlhf_scratch.training` — DPO

Source: `src/rlhf_scratch/training/dpo.py`. Hand-written per Rafailov et al.
(2023), no `trl`.

### `compute_sequence_logps(model, input_ids, attention_mask, labels) -> Tensor`

Sums `sequence_logprobs(...)` over response-token positions only (where
`labels[:, 1:] != -100`, matching `make_dpo_collate_fn`'s masking). Returns
`(B,)`.

### `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1) -> tuple[Tensor, Tensor, Tensor]`

```
logits = beta * [(policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)]
loss = -logsigmoid(logits).mean()
```

Returns `(loss, chosen_rewards, rejected_rewards)`, where the reward terms are
the detached implicit per-sequence rewards `beta * (policy_logp - ref_logp)`,
used for win-rate evaluation. When policy == reference for both sequences,
`loss == log(2)` exactly (mirrors `bradley_terry_loss`'s symmetric case).

### `DPOConfig`

```python
@dataclass
class DPOConfig:
    lr: float = 5e-6
    beta: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool | None = None
```

### `DPOTrainer(policy, reference, optimizer, device, config=None)`

`reference` is deep-copied by the caller and frozen internally
(`requires_grad_(False)` on all params, always queried under
`torch.no_grad()`).

- `train_epoch(dataloader) -> list[float]` — computes reference log-probs
  under `no_grad`, policy log-probs under autocast, backprops `dpo_loss`
  through the policy only.
- `evaluate_win_rate(dataloader) -> float` — fraction of pairs where the
  implicit reward ranks chosen above rejected; DPO's analogue of
  `RewardTrainer.evaluate_ranking_accuracy`.
- `save_checkpoint(path)` — `policy.save_pretrained(path)` if available, else
  `torch.save(policy.state_dict(), path)`.

---

## CLI entrypoints (`scripts/`)

Each wraps the corresponding trainer for a real run; not run on the CPU-only
dev machine (see `PLANNING.md` hardware plan) — intended for the RTX 5080.

| Script | Wraps | Key flags |
|---|---|---|
| `scripts/train_sft.py` | `SFTTrainer` | `--model-name`, `--toy`, `--epochs`, `--batch-size`, `--lr`, `--grad-accum-steps`, `--output-dir` |
| `scripts/train_reward.py` | `RewardTrainer` | `--base-model-name`, `--toy`, `--epochs`, `--batch-size`, `--lr`, `--output-dir` |
| `scripts/train_ppo.py` | `PPOStep` + `generate_rollouts` | `--actor-checkpoint`, `--reward-checkpoint`, `--toy`, `--steps`, `--batch-size`, `--max-new-tokens`, `--lr`, `--output-dir` |
| `scripts/train_dpo.py` | `DPOTrainer` | `--policy-checkpoint`, `--toy`, `--epochs`, `--batch-size`, `--lr`, `--beta`, `--output-dir` |

All are `typer` apps — run `python scripts/<name>.py --help` for the full
flag list.
