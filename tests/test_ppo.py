import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, BertConfig, BertModel, GPT2Config, GPT2LMHeadModel

from rlhf_scratch.models import RewardModel
from rlhf_scratch.models.critic import Critic
from rlhf_scratch.training import (
    PPOConfig,
    PPOStep,
    compute_gae,
    generate_rollouts,
    compute_kl_penalty,
    ppo_clipped_surrogate_loss,
    sequence_logprobs,
    value_loss,
)


def test_sequence_logprobs_matches_manual_gather():
    vocab_size = 5
    logits = torch.zeros(1, 3, vocab_size)
    logits[0, 0] = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])  # predicts token 0 confidently
    logits[0, 1] = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0])  # predicts token 2 confidently

    input_ids = torch.tensor([[1, 0, 2]])  # positions 0,1 predict tokens at 1,2 => targets 0, 2
    logprobs = sequence_logprobs(logits, input_ids)

    assert logprobs.shape == (1, 2)
    # position 0 predicted target token 0 with near-certainty => logprob near 0
    assert logprobs[0, 0].item() > -0.01
    # position 1 predicted target token 2 with near-certainty => logprob near 0
    assert logprobs[0, 1].item() > -0.01


def test_compute_kl_penalty_is_simple_difference():
    logprobs = torch.tensor([[-1.0, -2.0]])
    ref_logprobs = torch.tensor([[-1.5, -2.5]])
    kl = compute_kl_penalty(logprobs, ref_logprobs)
    assert torch.allclose(kl, torch.tensor([[0.5, 0.5]]))


def test_compute_gae_single_step_matches_td_error():
    # single timestep episode: advantage should equal reward - value (delta),
    # since there is no bootstrap and no future term.
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.3]])
    mask = torch.tensor([[1.0]])

    advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    expected_advantage = rewards - values
    assert torch.allclose(advantages, expected_advantage, atol=1e-6)
    assert torch.allclose(returns, advantages + values, atol=1e-6)


def test_compute_gae_two_step_matches_manual_recursion():
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.2, 0.5]])
    mask = torch.tensor([[1.0, 1.0]])
    gamma, lam = 0.99, 0.9

    advantages, returns = compute_gae(rewards, values, mask, gamma=gamma, lam=lam)

    delta_1 = rewards[0, 1] + gamma * 0.0 - values[0, 1]
    gae_1 = delta_1
    delta_0 = rewards[0, 0] + gamma * values[0, 1] - values[0, 0]
    gae_0 = delta_0 + gamma * lam * gae_1

    assert torch.isclose(advantages[0, 1], gae_1, atol=1e-6)
    assert torch.isclose(advantages[0, 0], gae_0, atol=1e-6)


def test_compute_gae_respects_mask_padding():
    rewards = torch.tensor([[0.5, 0.0, 0.0]])
    values = torch.tensor([[0.1, 0.1, 0.1]])
    mask = torch.tensor([[1.0, 0.0, 0.0]])  # only first token is real

    advantages, _ = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    # masked-out positions contribute nothing to the recursion beyond position 0
    assert torch.isclose(advantages[0, 0], rewards[0, 0] - values[0, 0], atol=1e-6)


def test_ppo_clipped_surrogate_loss_clips_large_positive_ratio():
    logprobs = torch.tensor([[0.0]])  # ratio = exp(0 - (-2)) = e^2 ~ 7.39, far above 1+eps
    old_logprobs = torch.tensor([[-2.0]])
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1.0]])

    loss = ppo_clipped_surrogate_loss(logprobs, old_logprobs, advantages, mask, clip_eps=0.2)
    # with positive advantage, unclipped ratio*A > clipped (1.2)*A, min() picks clipped
    expected = -(1.2 * 1.0)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)


def test_ppo_clipped_surrogate_loss_no_clip_when_ratio_near_one():
    logprobs = torch.tensor([[-1.0]])
    old_logprobs = torch.tensor([[-1.0]])
    advantages = torch.tensor([[2.0]])
    mask = torch.tensor([[1.0]])

    loss = ppo_clipped_surrogate_loss(logprobs, old_logprobs, advantages, mask, clip_eps=0.2)
    assert torch.isclose(loss, torch.tensor(-2.0), atol=1e-5)


def test_value_loss_is_masked_mse():
    values = torch.tensor([[1.0, 5.0]])
    returns = torch.tensor([[2.0, 100.0]])
    mask = torch.tensor([[1.0, 0.0]])  # second position masked out

    loss = value_loss(values, returns, mask)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)  # (1-2)^2 / 1


def _tiny_actor_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size, n_positions=64, n_embd=16, n_layer=2, n_head=2
    )
    return GPT2LMHeadModel(config), tokenizer


def test_ppo_step_update_runs_and_produces_finite_bounded_kl():
    actor, tokenizer = _tiny_actor_and_tokenizer()
    critic = Critic.__new__(Critic)
    nn.Module.__init__(critic)
    critic.backbone = GPT2LMHeadModel(actor.config).transformer
    critic.value_head = nn.Linear(actor.config.n_embd, 1)

    device = torch.device("cpu")
    optimizer = AdamW(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)
    ppo = PPOStep(actor, critic, optimizer, device, PPOConfig(ppo_epochs=2, kl_coef=0.1))

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    response_mask = torch.ones(batch_size, seq_len - 1)

    with torch.no_grad():
        ref_logits = actor(input_ids=input_ids).logits  # reference == actor at init, so KL starts at 0
        old_logprobs = sequence_logprobs(ref_logits, input_ids)
        old_values = critic(input_ids=input_ids)[:, :-1]

    rollout_batch = {
        "input_ids": input_ids,
        "response_mask": response_mask,
        "old_logprobs": old_logprobs,
        "ref_logprobs": old_logprobs.clone(),
        "old_values": old_values,
        "env_rewards": torch.tensor([1.0, -1.0]),
    }

    kl_trace = []
    for _ in range(5):
        metrics = ppo.update(rollout_batch)
        assert torch.isfinite(torch.tensor(metrics["pg_loss"]))
        assert torch.isfinite(torch.tensor(metrics["value_loss"]))
        assert torch.isfinite(torch.tensor(metrics["mean_kl"]))
        kl_trace.append(metrics["mean_kl"])

    # KL penalty should keep divergence from the frozen reference bounded, not exploding
    assert max(abs(k) for k in kl_trace) < 5.0


def _tiny_reward_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    model = RewardModel.__new__(RewardModel)
    nn.Module.__init__(model)
    model.encoder = BertModel(config)
    model.value_head = nn.Linear(config.hidden_size, 1)
    return model, tokenizer


def test_generate_rollouts_scores_with_reward_tokenizer_not_actor_ids():
    """Regression test for a real bug: generated token ids are GPT-2 BPE ids and
    must be decoded + re-tokenized with the reward model's own (different)
    vocabulary before scoring, not fed to the reward model directly."""
    actor, actor_tokenizer = _tiny_actor_and_tokenizer()
    reference, _ = _tiny_actor_and_tokenizer()
    critic = Critic.__new__(Critic)
    nn.Module.__init__(critic)
    critic.backbone = GPT2LMHeadModel(actor.config).transformer
    critic.value_head = nn.Linear(actor.config.n_embd, 1)

    reward_model, reward_tokenizer = _tiny_reward_model_and_tokenizer()

    prompt = ["Human: hi\n\nAssistant:", "Human: yo\n\nAssistant:"]
    enc = actor_tokenizer(prompt, padding=True, return_tensors="pt")

    rollout_batch = generate_rollouts(
        actor,
        reference,
        critic,
        reward_model,
        enc["input_ids"],
        enc["attention_mask"],
        actor_tokenizer,
        reward_tokenizer,
        max_new_tokens=4,
    )

    assert rollout_batch["env_rewards"].shape == (2,)
    assert torch.isfinite(rollout_batch["env_rewards"]).all()
    # reward model's vocab is unrelated to the actor's -- if raw actor ids had
    # been fed in directly, any out-of-range id would crash the embedding
    # lookup; reaching here at all confirms decode/re-encode happened.
    assert rollout_batch["old_logprobs"].shape == rollout_batch["ref_logprobs"].shape
