"""Hand-written PPO engine: clipped surrogate objective, Generalized Advantage
Estimation, and a per-token KL penalty against a frozen reference model. No
`trl` — every piece of PPO math here is implemented directly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


def sequence_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-prob of the actual next token, teacher-forced.

    logits: (B, T, V), predicting token t+1 from position t.
    input_ids: (B, T).
    Returns (B, T-1), aligned with input_ids[:, 1:].
    """
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:].unsqueeze(-1)
    return log_probs.gather(-1, target).squeeze(-1)


def compute_kl_penalty(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """Per-token KL estimate KL(actor || reference) via the standard sampled
    log-ratio approximation: log(pi_actor) - log(pi_ref)."""
    return logprobs - ref_logprobs


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    rewards, values, mask: (B, T) — per-token reward, value estimate V(s_t),
    and a validity mask (1 = real response token, 0 = padding). The bootstrap
    value at T is implicitly 0 (episode ends at the last response token).

    Bootstrapping/propagation at step t is gated by whether step t+1 is a real
    token (mask[:, t+1]), not by mask[:, t] itself — otherwise the last valid
    token in a padded sequence would incorrectly bootstrap off garbage padded
    values, and its advantage would leak into earlier real positions.

    Returns (advantages, returns), each (B, T).
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(seq_len)):
        if t + 1 < seq_len:
            next_value = values[:, t + 1]
            next_mask = mask[:, t + 1]
        else:
            next_value = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)
            next_mask = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)

        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        last_gae = delta + gamma * lam * next_mask * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_clipped_surrogate_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Clipped PPO surrogate objective (negated for gradient descent)."""
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    surrogate = torch.min(unclipped, clipped)
    return -(surrogate * mask).sum() / mask.sum().clamp(min=1)


def value_loss(values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((values - returns) ** 2) * mask).sum() / mask.sum().clamp(min=1)


@dataclass
class PPOConfig:
    lr: float = 1e-5
    clip_eps: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    kl_coef: float = 0.1
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4


class PPOStep:
    """One PPO update given a batch of already-generated, already-scored
    rollouts. The actor+critic are trainable here; the reference/reward models
    are frozen and only ever queried upstream, never optimized — mirroring the
    two-process-group split built in Phase 6/7."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        config: PPOConfig | None = None,
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.optimizer = optimizer
        self.device = device
        self.config = config or PPOConfig()

    def compute_rewards_with_kl(
        self,
        env_rewards: torch.Tensor,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token reward = -kl_coef * KL, plus the scalar reward-model score
        added at each sequence's last valid token. Returns (per_token_rewards, mean_kl)."""
        kl = compute_kl_penalty(logprobs, ref_logprobs)
        per_token_rewards = -self.config.kl_coef * kl * mask

        seq_lengths = mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(per_token_rewards.size(0), device=per_token_rewards.device)
        per_token_rewards = per_token_rewards.clone()
        per_token_rewards[batch_idx, seq_lengths] += env_rewards

        mean_kl = (kl * mask).sum() / mask.sum().clamp(min=1)
        return per_token_rewards, mean_kl

    def update(self, rollout_batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """rollout_batch keys:
        - input_ids: (B, T) full prompt+response token ids
        - response_mask: (B, T-1) validity mask over predicted positions
        - old_logprobs, ref_logprobs: (B, T-1) from rollout time (no grad)
        - old_values: (B, T-1) critic estimate at rollout time
        - env_rewards: (B,) scalar reward-model score per sequence
        """
        input_ids = rollout_batch["input_ids"].to(self.device)
        response_mask = rollout_batch["response_mask"].to(self.device).to(torch.float32)
        old_logprobs = rollout_batch["old_logprobs"].to(self.device)
        ref_logprobs = rollout_batch["ref_logprobs"].to(self.device)
        old_values = rollout_batch["old_values"].to(self.device)
        env_rewards = rollout_batch["env_rewards"].to(self.device)

        per_token_rewards, mean_kl = self.compute_rewards_with_kl(
            env_rewards, old_logprobs, ref_logprobs, response_mask
        )
        advantages, returns = compute_gae(
            per_token_rewards, old_values, response_mask, self.config.gamma, self.config.lam
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics: dict[str, float] = {}
        for _ in range(self.config.ppo_epochs):
            logits = self.actor(input_ids=input_ids).logits
            logprobs = sequence_logprobs(logits, input_ids)
            values = self.critic(input_ids=input_ids)[:, :-1]

            pg_loss = ppo_clipped_surrogate_loss(
                logprobs, old_logprobs, advantages, response_mask, self.config.clip_eps
            )
            v_loss = value_loss(values, returns, response_mask)
            loss = pg_loss + self.config.value_coef * v_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), self.config.max_grad_norm
            )
            self.optimizer.step()

            metrics = {
                "pg_loss": pg_loss.item(),
                "value_loss": v_loss.item(),
                "mean_kl": mean_kl.item(),
            }
        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pt")
        torch.save(self.critic.state_dict(), path / "critic.pt")


@torch.no_grad()
def generate_rollouts(
    actor: nn.Module,
    reference: nn.Module,
    critic: nn.Module,
    reward_model: nn.Module,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    tokenizer,
    reward_tokenizer,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_p: float = 1.0,
    reward_max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """Samples responses from the actor, then scores them with the reference
    model (for KL), the critic (for values), and the reward model — producing
    a rollout batch ready for `PPOStep.update`.

    `tokenizer` (the actor's) and `reward_tokenizer` are almost always
    different vocabularies (e.g. GPT-2 BPE vs. BERT WordPiece) — generated
    token ids are decoded with `tokenizer` and re-encoded with
    `reward_tokenizer` before scoring, rather than feeding the actor's raw ids
    into the reward model. Feeding them directly would run without error
    (both are just int64 tensors within range) but score meaningless token
    ids, silently corrupting every reward signal PPO trains against.

    Not used in unit tests beyond a shape smoke-test — the real end-to-end
    generate-and-score loop is exercised on the RTX 5080 training run.
    """
    device = prompt_input_ids.device
    actor.eval()

    generated = actor.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    response_mask = torch.zeros(generated.shape[0], generated.shape[1] - 1, device=device)
    prompt_len = prompt_input_ids.shape[1]
    response_mask[:, prompt_len - 1 :] = (generated[:, prompt_len:] != tokenizer.pad_token_id).to(torch.float32)

    actor_logits = actor(input_ids=generated).logits
    old_logprobs = sequence_logprobs(actor_logits, generated)

    ref_logits = reference(input_ids=generated).logits
    ref_logprobs = sequence_logprobs(ref_logits, generated)

    old_values = critic(input_ids=generated)[:, :-1]

    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    reward_enc = reward_tokenizer(
        generated_texts, padding=True, truncation=True, max_length=reward_max_length, return_tensors="pt"
    ).to(device)
    env_rewards = reward_model(reward_enc["input_ids"], reward_enc["attention_mask"])

    actor.train()

    return {
        "input_ids": generated,
        "response_mask": response_mask,
        "old_logprobs": old_logprobs.detach(),
        "ref_logprobs": ref_logprobs.detach(),
        "old_values": old_values.detach(),
        "env_rewards": env_rewards.detach(),
    }
