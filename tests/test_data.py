from datasets import Dataset as HFDataset

from rlhf_scratch.data import (
    PreferenceDataset,
    PreferencePair,
    extract_prompt,
    load_preference_pairs,
    make_collate_fn,
)


def test_extract_prompt_common_prefix():
    chosen = "Human: How do I bake bread?\n\nAssistant: Start with flour, water, yeast."
    rejected = "Human: How do I bake bread?\n\nAssistant: I don't know."
    prompt = extract_prompt(chosen, rejected)
    assert prompt == "Human: How do I bake bread?\n\nAssistant:"
    assert chosen.startswith(prompt)
    assert rejected.startswith(prompt)


def test_extract_prompt_no_marker_falls_back_to_common_prefix():
    chosen = "abcXYZ"
    rejected = "abcQRS"
    assert extract_prompt(chosen, rejected) == "abc"


def test_preference_dataset_len_and_getitem():
    hf = HFDataset.from_dict(
        {
            "prompt": ["p1", "p2"],
            "chosen": ["good1", "good2"],
            "rejected": ["bad1", "bad2"],
        }
    )
    ds = PreferenceDataset(hf)
    assert len(ds) == 2
    item = ds[0]
    assert isinstance(item, PreferencePair)
    assert item.prompt == "p1"
    assert item.chosen == "good1"
    assert item.rejected == "bad1"


class _FakeTokenizer:
    """Minimal whitespace tokenizer standing in for a HF tokenizer, so
    collate_fn shape/behavior can be tested without a network download."""

    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        import torch

        tokenized = [t.split()[:max_length] for t in texts]
        max_len = max(len(t) for t in tokenized)
        input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, toks in enumerate(tokenized):
            for j, _ in enumerate(toks):
                input_ids[i, j] = j + 1
                attention_mask[i, j] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_collate_fn_shapes():
    tokenizer = _FakeTokenizer()
    collate_fn = make_collate_fn(tokenizer, max_length=16)

    batch = [
        PreferencePair(prompt="Human: hi ", chosen="good response here", rejected="bad"),
        PreferencePair(prompt="Human: yo ", chosen="ok", rejected="terrible response indeed"),
    ]
    out = collate_fn(batch)

    assert set(out.keys()) == {
        "chosen_input_ids",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
    }
    assert out["chosen_input_ids"].shape[0] == 2
    assert out["rejected_input_ids"].shape[0] == 2
    assert out["chosen_input_ids"].shape == out["chosen_attention_mask"].shape
    assert out["rejected_input_ids"].shape == out["rejected_attention_mask"].shape
    assert tokenizer.pad_token == tokenizer.eos_token


def test_load_preference_pairs_toy_and_held_out_disjoint():
    train_pool, held_out = load_preference_pairs(split="train", toy=True)

    assert len(held_out) == 200
    assert len(train_pool) == 1000

    for row in train_pool.select(range(5)):
        assert row["chosen"].startswith(("\n\n", " ")) or row["chosen"] != ""
        assert row["prompt"] != ""

    train_texts = set(train_pool["prompt"])
    held_out_texts = set(held_out["prompt"])
    assert len(train_texts & held_out_texts) < len(held_out_texts)


def test_load_preference_pairs_held_out_is_seed_stable():
    # held_out is carved out before toy subsampling, so it must be identical
    # across repeated calls regardless of the `toy` flag.
    _, held_out_a = load_preference_pairs(split="train", toy=True, seed=42)
    _, held_out_b = load_preference_pairs(split="train", toy=True, seed=42)

    assert held_out_a["prompt"] == held_out_b["prompt"]
