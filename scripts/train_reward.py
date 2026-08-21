"""CLI entrypoint for Phase 3 (Bradley-Terry reward model). Not intended to run
on this dev machine (CPU-only, no CUDA) — real runs happen on the RTX 5080
training machine.

    python scripts/train_reward.py --toy --epochs 1
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from rlhf_scratch.data import PreferenceDataset, load_preference_pairs, make_collate_fn
from rlhf_scratch.models import RewardModel
from rlhf_scratch.training import RewardConfig, RewardTrainer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    base_model_name: str = "prajjwal1/bert-tiny",
    toy: bool = True,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-4,
    max_length: int = 256,
    output_dir: str = "results/reward",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = RewardModel(base_model_name)

    train_pool, held_out = load_preference_pairs(split="train", toy=toy)
    collate_fn = make_collate_fn(tokenizer, max_length=max_length)

    train_loader = DataLoader(
        PreferenceDataset(train_pool), batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    held_out_loader = DataLoader(
        PreferenceDataset(held_out), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    trainer = RewardTrainer(model, optimizer, device, RewardConfig(lr=lr))

    for epoch in range(epochs):
        losses = trainer.train_epoch(train_loader)
        mean_loss = sum(losses) / len(losses)
        typer.echo(f"epoch={epoch} mean_loss={mean_loss:.4f}")

    accuracy = trainer.evaluate_ranking_accuracy(held_out_loader)
    typer.echo(f"held_out_ranking_accuracy={accuracy:.4f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(output_path / "checkpoint.pt")
    (output_path / "metrics.json").write_text(json.dumps({"held_out_ranking_accuracy": accuracy}, indent=2))
    typer.echo(f"saved checkpoint + metrics to {output_path}")


if __name__ == "__main__":
    app()
