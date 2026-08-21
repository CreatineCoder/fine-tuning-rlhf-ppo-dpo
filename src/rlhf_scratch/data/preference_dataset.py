"""Preference-pair dataset (Anthropic HH-RLHF) with toy-mode subsampling and a
fixed held-out split that stays identical across toy and full runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset as HFDataset, load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

HELD_OUT_SIZE = 200
TOY_TRAIN_SIZE = 1000
SEED = 42


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


def extract_prompt(chosen: str, rejected: str) -> str:
    """HH-RLHF stores the shared prompt as the common prefix of chosen/rejected,
    ending at the last "Assistant:" turn marker."""
    common_len = 0
    for a, b in zip(chosen, rejected):
        if a != b:
            break
        common_len += 1
    prefix = chosen[:common_len]
    cut = prefix.rfind("Assistant:")
    return prefix[: cut + len("Assistant:")] if cut != -1 else prefix


def load_preference_pairs(
    split: str = "train",
    toy: bool = False,
    seed: int = SEED,
) -> tuple[HFDataset, HFDataset]:
    """Loads Anthropic/hh-rlhf and splits it into (train_pool, held_out).

    held_out is carved out first from a fixed seeded shuffle of the full split,
    so it is identical whether or not `toy` subsamples the remainder — it must
    never be touched during training, only for final evaluation (Phase 3/8).
    """
    raw = load_dataset("Anthropic/hh-rlhf", split=split)
    raw = raw.shuffle(seed=seed)

    held_out = raw.select(range(HELD_OUT_SIZE))
    remainder = raw.select(range(HELD_OUT_SIZE, len(raw)))

    if toy:
        remainder = remainder.select(range(min(TOY_TRAIN_SIZE, len(remainder))))

    def _map(example: dict) -> dict:
        prompt = extract_prompt(example["chosen"], example["rejected"])
        return {
            "prompt": prompt,
            "chosen": example["chosen"][len(prompt):],
            "rejected": example["rejected"][len(prompt):],
        }

    remainder = remainder.map(_map, remove_columns=raw.column_names)
    held_out = held_out.map(_map, remove_columns=raw.column_names)
    return remainder, held_out


class PreferenceDataset(Dataset):
    """Wraps a HF preference-pairs dataset (prompt/chosen/rejected columns)."""

    def __init__(self, hf_dataset: HFDataset):
        self._data = hf_dataset

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> PreferencePair:
        row = self._data[idx]
        return PreferencePair(prompt=row["prompt"], chosen=row["chosen"], rejected=row["rejected"])


def make_collate_fn(tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
    """Builds a collate_fn that tokenizes (prompt + response) for chosen/rejected
    with the given tokenizer. Same dataset works for any tokenizer this way, so
    the same PreferenceDataset backs both the distilgpt2 actor and bert-tiny
    reward model training loops."""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batch: list[PreferencePair]) -> dict[str, torch.Tensor]:
        chosen_text = [p.prompt + p.chosen for p in batch]
        rejected_text = [p.prompt + p.rejected for p in batch]

        chosen_enc = tokenizer(
            chosen_text, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        rejected_enc = tokenizer(
            rejected_text, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }

    return collate_fn
