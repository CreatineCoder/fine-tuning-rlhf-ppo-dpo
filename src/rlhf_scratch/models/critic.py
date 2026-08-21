"""Critic (value function): a causal-LM backbone plus a per-token scalar head,
used by the hand-written PPO engine (`training/ppo.py`)."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM


class Critic(nn.Module):
    def __init__(self, base_model_name: str = "distilgpt2"):
        super().__init__()
        causal_lm = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.backbone = causal_lm.get_decoder() if hasattr(causal_lm, "get_decoder") else causal_lm.transformer
        hidden_size = self.backbone.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Returns a per-token value estimate, shape (B, T)."""
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.value_head(hidden).squeeze(-1)
