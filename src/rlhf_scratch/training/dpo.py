"""Hand-written Direct Preference Optimization (Rafailov et al., 2023) — no
`trl`. DPO replaces the reward-model + PPO loop with a single closed-form loss
computed directly against a frozen reference model; used here as an
alternative alignment path to PPO on the same preference data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from rlhf_scratch.training.ppo import sequence_logprobs


def compute_sequence_logps(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Sum of per-token log-probs over the response tokens only (labels == -100
    marks prompt/pad tokens, matching `data.make_dpo_collate_fn`). Returns (B,)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logprobs = sequence_logprobs(logits, input_ids)  # (B, T-1)
    response_mask = (labels[:, 1:] != -100).to(logprobs.dtype)
    return (logprobs * response_mask).sum(dim=1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO loss: -log(sigmoid(beta * [(logpi_chosen - logpi_rejected) -
    (logref_chosen - logref_rejected)])). Also returns the implicit per-sequence
    reward margins (beta * log-ratio to reference) for chosen/rejected, used for
    win-rate evaluation."""
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)

    loss = -nn.functional.logsigmoid(logits).mean()

    chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()
    return loss, chosen_rewards, rejected_rewards


@dataclass
class DPOConfig:
    lr: float = 5e-6
    beta: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool | None = None


class DPOTrainer:
    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        config: DPOConfig | None = None,
    ):
        self.policy = policy.to(device)
        self.reference = reference.to(device).eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.optimizer = optimizer
        self.device = device
        self.config = config or DPOConfig()
        use_amp = self.config.use_amp
        self.use_amp = (device.type == "cuda") if use_amp is None else use_amp
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)

    def _logps(self, model: nn.Module, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        input_ids = batch[f"{prefix}_input_ids"].to(self.device)
        attention_mask = batch[f"{prefix}_attention_mask"].to(self.device)
        labels = batch[f"{prefix}_labels"].to(self.device)
        return compute_sequence_logps(model, input_ids, attention_mask, labels)

    def train_epoch(self, dataloader: DataLoader) -> list[float]:
        self.policy.train()
        losses: list[float] = []

        for batch in dataloader:
            with torch.no_grad():
                ref_chosen_logps = self._logps(self.reference, batch, "chosen")
                ref_rejected_logps = self._logps(self.reference, batch, "rejected")

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                policy_chosen_logps = self._logps(self.policy, batch, "chosen")
                policy_rejected_logps = self._logps(self.policy, batch, "rejected")
                loss, _, _ = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.config.beta,
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(loss.item())

        return losses

    @torch.no_grad()
    def evaluate_win_rate(self, dataloader: DataLoader) -> float:
        """Fraction of pairs where the implicit reward (beta * log-ratio to
        reference) ranks chosen above rejected — DPO's analogue of the reward
        model's held-out ranking accuracy."""
        self.policy.eval()
        correct = 0
        total = 0
        for batch in dataloader:
            ref_chosen_logps = self._logps(self.reference, batch, "chosen")
            ref_rejected_logps = self._logps(self.reference, batch, "rejected")
            policy_chosen_logps = self._logps(self.policy, batch, "chosen")
            policy_rejected_logps = self._logps(self.policy, batch, "rejected")

            _, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=self.config.beta,
            )
            correct += (chosen_rewards > rejected_rewards).sum().item()
            total += chosen_rewards.size(0)
        return correct / total if total else 0.0

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.policy, "save_pretrained"):
            self.policy.save_pretrained(path)
        else:
            torch.save(self.policy.state_dict(), path)
