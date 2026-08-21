"""CLI entrypoint for Phase 2 (SFT). Not intended to run on this dev machine
(CPU-only, no CUDA) — real runs happen on the RTX 5080 training machine.

    python scripts/train_sft.py --toy --epochs 1
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlhf_scratch.data import PreferenceDataset, load_preference_pairs, make_sft_collate_fn
from rlhf_scratch.training import SFTConfig, SFTTrainer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_name: str = "distilgpt2",
    toy: bool = True,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 5e-5,
    grad_accum_steps: int = 4,
    max_length: int = 256,
    output_dir: str = "results/sft",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    train_pool, _held_out = load_preference_pairs(split="train", toy=toy)
    dataset = PreferenceDataset(train_pool)
    collate_fn = make_sft_collate_fn(tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=lr)
    config = SFTConfig(lr=lr, grad_accum_steps=grad_accum_steps, epochs=epochs)
    trainer = SFTTrainer(model, optimizer, device, config)

    for epoch in range(epochs):
        losses = trainer.train_epoch(dataloader)
        mean_loss = sum(losses) / len(losses)
        typer.echo(f"epoch={epoch} mean_loss={mean_loss:.4f}")

    checkpoint_path = Path(output_dir) / "checkpoint"
    trainer.save_checkpoint(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    typer.echo(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    app()
