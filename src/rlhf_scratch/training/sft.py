"""Hand-written supervised fine-tuning loop — plain PyTorch, no HF `Trainer`,
no `trl`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def sft_loss(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Causal LM cross-entropy loss. `labels` must already have prompt/pad tokens
    masked to -100 (see `data.make_sft_collate_fn`) so only response tokens count."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100
    )


@dataclass
class SFTConfig:
    lr: float = 5e-5
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    epochs: int = 1
    use_amp: bool | None = None  # None => auto (True on CUDA, False on CPU)


class SFTTrainer:
    """Minimal training-loop object: gradient accumulation + AMP + grad clipping,
    all explicit so behavior is auditable rather than hidden in a framework."""

    def __init__(self, model: nn.Module, optimizer: Optimizer, device: torch.device, config: SFTConfig | None = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.config = config or SFTConfig()
        use_amp = self.config.use_amp
        self.use_amp = (device.type == "cuda") if use_amp is None else use_amp
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)

    def train_epoch(self, dataloader: DataLoader) -> list[float]:
        self.model.train()
        losses: list[float] = []
        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss = sft_loss(self.model, input_ids, attention_mask, labels)

            self.scaler.scale(loss / self.config.grad_accum_steps).backward()

            if (step + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            losses.append(loss.item())

        return losses

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), path)
