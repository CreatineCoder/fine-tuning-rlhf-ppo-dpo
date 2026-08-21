import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from rlhf_scratch.data import PreferenceDataset, PreferencePair, make_sft_collate_fn
from rlhf_scratch.training import SFTConfig, SFTTrainer, sft_loss


def _tiny_model_and_tokenizer():
    """A small randomly-initialized GPT-2 so tests run fast on CPU without
    downloading distilgpt2. The tokenizer itself is still the real GPT-2 BPE
    tokenizer (fast, no model weights to fetch)."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    return model, tokenizer


def test_sft_loss_ignores_masked_positions():
    model, tokenizer = _tiny_model_and_tokenizer()
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    labels_all_masked = torch.full_like(input_ids, -100)

    loss = sft_loss(model, input_ids, attention_mask, labels_all_masked)
    assert torch.isnan(loss), "cross_entropy over an all -100 target is NaN by definition"

    labels_partial = input_ids.clone()
    labels_partial[:, :5] = -100
    loss2 = sft_loss(model, input_ids, attention_mask, labels_partial)
    assert torch.isfinite(loss2)
    assert loss2.item() > 0


def test_sft_collate_fn_masks_prompt_tokens():
    _, tokenizer = _tiny_model_and_tokenizer()
    collate_fn = make_sft_collate_fn(tokenizer, max_length=32)

    batch = [
        PreferencePair(prompt="Human: hi\n\nAssistant:", chosen=" hello there", rejected=" bad"),
        PreferencePair(prompt="Human: yo\n\nAssistant:", chosen=" ok", rejected=" no"),
    ]
    out = collate_fn(batch)

    assert set(out.keys()) == {"input_ids", "attention_mask", "labels"}
    for i, pair in enumerate(batch):
        prompt_len = len(tokenizer(pair.prompt)["input_ids"])
        assert torch.all(out["labels"][i, :prompt_len] == -100)
        pad_positions = out["attention_mask"][i] == 0
        assert torch.all(out["labels"][i][pad_positions] == -100)


def test_sft_trainer_reduces_loss_on_single_batch():
    model, tokenizer = _tiny_model_and_tokenizer()
    collate_fn = make_sft_collate_fn(tokenizer, max_length=32)

    dataset = PreferenceDataset.__new__(PreferenceDataset)
    pairs = [
        PreferencePair(prompt="Human: hi\n\nAssistant:", chosen=" hello there friend", rejected=" x"),
        PreferencePair(prompt="Human: yo\n\nAssistant:", chosen=" good morning", rejected=" x"),
    ]

    class _ListDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, idx):
            return pairs[idx]

    dataloader = DataLoader(_ListDataset(), batch_size=2, collate_fn=collate_fn)

    device = torch.device("cpu")
    optimizer = AdamW(model.parameters(), lr=1e-3)
    config = SFTConfig(grad_accum_steps=1, use_amp=False)
    trainer = SFTTrainer(model, optimizer, device, config)

    first_epoch_losses = trainer.train_epoch(dataloader)
    later_epoch_losses = []
    for _ in range(10):
        later_epoch_losses = trainer.train_epoch(dataloader)

    assert sum(later_epoch_losses) / len(later_epoch_losses) < sum(first_epoch_losses) / len(first_epoch_losses)


def test_sft_trainer_save_checkpoint(tmp_path):
    model, tokenizer = _tiny_model_and_tokenizer()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    trainer = SFTTrainer(model, optimizer, torch.device("cpu"), SFTConfig(use_amp=False))

    ckpt_dir = tmp_path / "ckpt"
    trainer.save_checkpoint(ckpt_dir)

    assert (ckpt_dir / "config.json").exists()
    assert any(ckpt_dir.glob("*.safetensors")) or any(ckpt_dir.glob("*.bin"))
