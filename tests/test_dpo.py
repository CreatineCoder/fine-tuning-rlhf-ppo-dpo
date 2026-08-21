import copy

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from rlhf_scratch.data import PreferenceDataset, PreferencePair, make_dpo_collate_fn
from rlhf_scratch.training import DPOConfig, DPOTrainer, compute_sequence_logps, dpo_loss


def _tiny_model_and_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size, n_positions=64, n_embd=16, n_layer=2, n_head=2
    )
    model = GPT2LMHeadModel(config)
    return model, tokenizer


def test_dpo_loss_prefers_policy_that_favors_chosen_relative_to_reference():
    # Policy raises chosen logp and lowers rejected logp relative to reference
    # => policy_logratios - ref_logratios is positive => loss should be small.
    ref_chosen = torch.tensor([-5.0])
    ref_rejected = torch.tensor([-5.0])
    policy_chosen_good = torch.tensor([-2.0])  # much more likely under policy
    policy_rejected_good = torch.tensor([-8.0])  # much less likely under policy

    loss_good, chosen_r, rejected_r = dpo_loss(policy_chosen_good, policy_rejected_good, ref_chosen, ref_rejected, beta=0.1)

    policy_chosen_bad = torch.tensor([-8.0])
    policy_rejected_bad = torch.tensor([-2.0])
    loss_bad, _, _ = dpo_loss(policy_chosen_bad, policy_rejected_bad, ref_chosen, ref_rejected, beta=0.1)

    assert loss_good.item() < loss_bad.item()
    assert chosen_r.item() > rejected_r.item()


def test_dpo_loss_zero_logratio_gap_is_log2():
    # policy == reference for both => logits = 0 => -log(sigmoid(0)) = log(2)
    logps = torch.tensor([-3.0])
    loss, chosen_r, rejected_r = dpo_loss(logps, logps, logps, logps, beta=0.1)
    assert torch.isclose(loss, torch.log(torch.tensor(2.0)), atol=1e-5)
    assert torch.isclose(chosen_r, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(rejected_r, torch.tensor(0.0), atol=1e-6)


def test_compute_sequence_logps_only_sums_response_tokens():
    model, tokenizer = _tiny_model_and_tokenizer()
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 6))
    attention_mask = torch.ones_like(input_ids)

    labels_all_response = input_ids.clone()
    logps_all = compute_sequence_logps(model, input_ids, attention_mask, labels_all_response)

    labels_partial = input_ids.clone()
    labels_partial[:, :3] = -100  # mask out first half (prompt)
    logps_partial = compute_sequence_logps(model, input_ids, attention_mask, labels_partial)

    assert logps_all.shape == (1,)
    # summing over fewer (unmasked) tokens should generally differ from summing all
    assert not torch.isclose(logps_all, logps_partial)


def test_dpo_collate_fn_masks_prompt_in_both_chosen_and_rejected():
    _, tokenizer = _tiny_model_and_tokenizer()
    collate_fn = make_dpo_collate_fn(tokenizer, max_length=32)

    batch = [
        PreferencePair(prompt="Human: hi\n\nAssistant:", chosen=" hello there", rejected=" no thanks"),
        PreferencePair(prompt="Human: yo\n\nAssistant:", chosen=" good morning", rejected=" leave me alone"),
    ]
    out = collate_fn(batch)

    expected_keys = {
        "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
        "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
    }
    assert set(out.keys()) == expected_keys

    for i, pair in enumerate(batch):
        prompt_len = len(tokenizer(pair.prompt)["input_ids"])
        assert torch.all(out["chosen_labels"][i, :prompt_len] == -100)
        assert torch.all(out["rejected_labels"][i, :prompt_len] == -100)


def test_dpo_trainer_reduces_loss_on_single_batch():
    policy, tokenizer = _tiny_model_and_tokenizer()
    reference = copy.deepcopy(policy)
    collate_fn = make_dpo_collate_fn(tokenizer, max_length=32)

    pairs = [
        PreferencePair(prompt="Human: hi\n\nAssistant:", chosen=" hello there friend", rejected=" go away"),
        PreferencePair(prompt="Human: yo\n\nAssistant:", chosen=" good morning", rejected=" no comment"),
    ]

    class _ListDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, idx):
            return pairs[idx]

    dataloader = DataLoader(_ListDataset(), batch_size=2, collate_fn=collate_fn)

    device = torch.device("cpu")
    optimizer = AdamW(policy.parameters(), lr=1e-3)
    trainer = DPOTrainer(policy, reference, optimizer, device, DPOConfig(use_amp=False))

    first_losses = trainer.train_epoch(dataloader)
    later_losses = []
    for _ in range(15):
        later_losses = trainer.train_epoch(dataloader)

    assert sum(later_losses) / len(later_losses) < sum(first_losses) / len(first_losses)

    win_rate = trainer.evaluate_win_rate(dataloader)
    assert 0.0 <= win_rate <= 1.0


def test_dpo_trainer_reference_stays_frozen():
    policy, tokenizer = _tiny_model_and_tokenizer()
    reference = copy.deepcopy(policy)
    ref_params_before = [p.clone() for p in reference.parameters()]

    collate_fn = make_dpo_collate_fn(tokenizer, max_length=32)
    pairs = [PreferencePair(prompt="Human: hi\n\nAssistant:", chosen=" hello", rejected=" no")]

    class _ListDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return pairs[idx]

    dataloader = DataLoader(_ListDataset(), batch_size=1, collate_fn=collate_fn)
    optimizer = AdamW(policy.parameters(), lr=1e-2)
    trainer = DPOTrainer(policy, reference, optimizer, torch.device("cpu"), DPOConfig(use_amp=False))

    for _ in range(3):
        trainer.train_epoch(dataloader)

    for p_before, p_after in zip(ref_params_before, reference.parameters()):
        assert torch.equal(p_before, p_after)
        assert not p_after.requires_grad
