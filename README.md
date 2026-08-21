# RLHF from Scratch — SFT + Bradley-Terry Reward + PPO + DPO

Hand-written RLHF alignment stack — no `trl`. SFT, reward modeling, PPO, and DPO
losses are all implemented directly in PyTorch. Validated end-to-end on a single
consumer GPU (RTX 4050, 6GB) with a distilgpt2 actor and bert-tiny reward model.

> Status: in development. See [PLANNING.md](PLANNING.md) for the full phased plan
> and current progress. Headline numbers below are filled in as each phase lands —
> until then they stay as placeholders, never estimated.

## Summary

- Implemented the RLHF stack from scratch, hand-writing SFT, Bradley-Terry reward
  modeling, PPO and DPO losses instead of wrapping `trl`.
- Built PPO with a clipped surrogate objective, GAE advantages and KL penalty,
  converging to **[KL]** against the frozen reference model.
- Split trainable actor-critic from frozen reference-reward across two NCCL groups,
  measuring **[X]%** VRAM and **[Y]%** comms reduction.
- Ran a distilgpt2 and bert-tiny toy pipeline end-to-end on one GPU, with the
  reward model ranking **[X]%** of held-out pairs correctly.

## Quick start

```bash
pip install -e .

python scripts/train_sft.py --toy
python scripts/train_reward.py --toy
python scripts/train_ppo.py --toy
python scripts/train_dpo.py --toy
```

## Repo layout

```
src/rlhf_scratch/
  models/         reward model head, model loading helpers
  data/           preference dataset, tokenization, toy-mode splitting
  training/       sft.py, reward.py, ppo.py, dpo.py — the hand-written engines
  distributed/    two-NCCL-group topology (actor-critic vs frozen reference-reward)
  utils/          logging, seeding, metrics
scripts/          CLI entrypoints per phase
configs/          YAML configs
tests/            unit tests
results/          metrics, checkpoints, benchmark logs (gitignored beyond metrics)
docs/             short write-ups per component
```

See [PLANNING.md](PLANNING.md) for phase-by-phase details, including why the
distributed VRAM/comms benchmark needs a rented 2-GPU instance and how it's
sequenced relative to everything else.
