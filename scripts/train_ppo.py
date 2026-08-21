"""CLI entrypoint for Phase 4 (PPO). Not intended to run on this dev machine
(CPU-only, no CUDA) — real runs happen on the RTX 5080 training machine.

    python scripts/train_ppo.py --toy --steps 50
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

from rlhf_scratch.data import PreferenceDataset, load_preference_pairs
from rlhf_scratch.models import Critic, RewardModel
from rlhf_scratch.training import PPOConfig, PPOStep, generate_rollouts

app = typer.Typer(add_completion=False)


@app.command()
def main(
    actor_checkpoint: str = "results/sft/checkpoint",
    reward_checkpoint: str = "results/reward/checkpoint.pt",
    reward_base_model: str = "prajjwal1/bert-tiny",
    toy: bool = True,
    steps: int = 50,
    batch_size: int = 4,
    max_prompt_length: int = 64,
    max_new_tokens: int = 32,
    lr: float = 1e-5,
    output_dir: str = "results/ppo",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(actor_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor = AutoModelForCausalLM.from_pretrained(actor_checkpoint).to(device)
    reference = copy.deepcopy(actor).to(device).eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    critic = Critic(actor_checkpoint).to(device)

    reward_model = RewardModel(reward_base_model).to(device)
    reward_model.load_state_dict(torch.load(reward_checkpoint, map_location=device))
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_base_model)

    train_pool, _held_out = load_preference_pairs(split="train", toy=toy)
    dataset = PreferenceDataset(train_pool)

    def prompt_collate(batch):
        prompts = [p.prompt for p in batch]
        return tokenizer(
            prompts, padding=True, truncation=True, max_length=max_prompt_length, return_tensors="pt"
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=prompt_collate)

    optimizer = AdamW(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    ppo = PPOStep(actor, critic, optimizer, device, PPOConfig(lr=lr))

    kl_trace: list[float] = []
    data_iter = iter(dataloader)
    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        rollout_batch = generate_rollouts(
            actor,
            reference,
            critic,
            reward_model,
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            tokenizer,
            max_new_tokens=max_new_tokens,
        )
        # NOTE: reward model uses its own tokenizer/vocab in the real run; a
        # production version would re-decode + re-tokenize generated text with
        # reward_tokenizer before scoring. Left as a TODO for the Phase 8 run.
        metrics = ppo.update(rollout_batch)
        kl_trace.append(metrics["mean_kl"])
        typer.echo(f"step={step} pg_loss={metrics['pg_loss']:.4f} value_loss={metrics['value_loss']:.4f} mean_kl={metrics['mean_kl']:.4f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ppo.save_checkpoint(output_path / "checkpoint")
    (output_path / "metrics.json").write_text(
        json.dumps({"kl_trace": kl_trace, "final_kl": kl_trace[-1] if kl_trace else None}, indent=2)
    )
    typer.echo(f"saved checkpoint + metrics to {output_path}")


if __name__ == "__main__":
    app()
