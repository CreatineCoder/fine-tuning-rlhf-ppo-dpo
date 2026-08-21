"""Hand-written Bradley-Terry reward model training — no `trl` reward trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def bradley_terry_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """-log(sigmoid(r_chosen - r_rejected)) — the Bradley-Terry pairwise
    preference model's negative log-likelihood."""
    return -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()


@dataclass
class RewardConfig:
    lr: float = 1e-4
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool | None = None  # None => auto (True on CUDA, False on CPU)


class RewardTrainer:
    def __init__(self, model: nn.Module, optimizer: Optimizer, device: torch.device, config: RewardConfig | None = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.config = config or RewardConfig()
        use_amp = self.config.use_amp
        self.use_amp = (device.type == "cuda") if use_amp is None else use_amp
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)

    def train_epoch(self, dataloader: DataLoader) -> list[float]:
        self.model.train()
        losses: list[float] = []
        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_input_ids"].to(self.device, non_blocking=True)
            chosen_mask = batch["chosen_attention_mask"].to(self.device, non_blocking=True)
            rejected_ids = batch["rejected_input_ids"].to(self.device, non_blocking=True)
            rejected_mask = batch["rejected_attention_mask"].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                chosen_rewards = self.model(chosen_ids, chosen_mask)
                rejected_rewards = self.model(rejected_ids, rejected_mask)
                loss = bradley_terry_loss(chosen_rewards, rejected_rewards)

            self.scaler.scale(loss / self.config.grad_accum_steps).backward()

            if (step + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            losses.append(loss.item())

        return losses

    @torch.no_grad()
    def evaluate_ranking_accuracy(self, dataloader: DataLoader) -> float:
        """Fraction of pairs where r(chosen) > r(rejected) — the metric behind
        bullet 4's "[X]% of held-out pairs correctly" claim."""
        self.model.eval()
        correct = 0
        total = 0
        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_mask = batch["chosen_attention_mask"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_mask = batch["rejected_attention_mask"].to(self.device)

            chosen_rewards = self.model(chosen_ids, chosen_mask)
            rejected_rewards = self.model(rejected_ids, rejected_mask)
            correct += (chosen_rewards > rejected_rewards).sum().item()
            total += chosen_rewards.size(0)
        return correct / total if total else 0.0

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
