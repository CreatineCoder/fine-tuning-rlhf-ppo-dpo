"""CLI entrypoint for Phase 5 (DPO) — an alternative to PPO alignment on the
same preference data. Not intended to run on this dev machine (CPU-only, no
CUDA) — real runs happen on the RTX 5080 training machine.

    python scripts/train_dpo.py --toy --epochs 1
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import typer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlhf_scratch.data import PreferenceDataset, load_preference_pairs, make_dpo_collate_fn
from rlhf_scratch.training import DPOConfig, DPOTrainer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    policy_checkpoint: str = "results/sft/checkpoint",
    toy: bool = True,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 5e-6,
    beta: float = 0.1,
    max_length: int = 256,
    output_dir: str = "results/dpo",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(policy_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(policy_checkpoint)
    reference = copy.deepcopy(policy)

    train_pool, held_out = load_preference_pairs(split="train", toy=toy)
    collate_fn = make_dpo_collate_fn(tokenizer, max_length=max_length)

    train_loader = DataLoader(
        PreferenceDataset(train_pool), batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    held_out_loader = DataLoader(
        PreferenceDataset(held_out), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    optimizer = AdamW(policy.parameters(), lr=lr)
    trainer = DPOTrainer(policy, reference, optimizer, device, DPOConfig(lr=lr, beta=beta))

    for epoch in range(epochs):
        losses = trainer.train_epoch(train_loader)
        mean_loss = sum(losses) / len(losses)
        typer.echo(f"epoch={epoch} mean_loss={mean_loss:.4f}")

    win_rate = trainer.evaluate_win_rate(held_out_loader)
    typer.echo(f"held_out_win_rate={win_rate:.4f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(output_path / "checkpoint")
    tokenizer.save_pretrained(output_path / "checkpoint")
    (output_path / "metrics.json").write_text(json.dumps({"held_out_win_rate": win_rate}, indent=2))
    typer.echo(f"saved checkpoint + metrics to {output_path}")


if __name__ == "__main__":
    app()
