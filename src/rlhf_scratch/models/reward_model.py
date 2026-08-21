"""Reward model: a pretrained encoder (bert-tiny) plus a scalar value head,
trained with a hand-written Bradley-Terry pairwise loss (see
`training/reward.py`) — not the `trl` reward trainer."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class RewardModel(nn.Module):
    def __init__(self, base_model_name: str = "prajjwal1/bert-tiny"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns a scalar reward per sequence, shape (B,)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_idx, seq_lengths]  # hidden state of the last real token

        return self.value_head(pooled).squeeze(-1)
