"""Run this FIRST on the RTX 5080 training machine, before any real training
job — catches environment/platform issues cheaply (seconds) instead of
discovering them mid-run (minutes to hours in). Exits non-zero if any check
fails.

    python scripts/preflight_check.py
"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False)


def check_cuda() -> tuple[bool, str]:
    import torch

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False — no GPU visible to torch"
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return True, f"{name}, {vram_gb:.1f} GB"


def check_nccl_backend() -> tuple[bool, str]:
    import torch.distributed as dist

    if not dist.is_available():
        return False, "torch.distributed not built into this torch install"
    if not dist.is_nccl_available():
        return False, (
            "NCCL backend unavailable — NCCL does not run on native Windows. "
            "Either run under WSL2, or fall back to backend='gloo' in "
            "scripts/benchmark_topology.py / train_ppo.py's distributed setup "
            "(keeps VRAM and bytes-exchanged numbers real; drops the literal "
            "'NCCL' framing to 'process groups' in the writeup)."
        )
    return True, "NCCL backend available"


def check_amp_roundtrip() -> tuple[bool, str]:
    import torch

    if not torch.cuda.is_available():
        return False, "skipped — no CUDA"
    device = torch.device("cuda")
    x = torch.randn(4, 4, device=device, requires_grad=True)
    with torch.autocast(device_type="cuda"):
        y = (x @ x).sum()
    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(y).backward()
    return True, "autocast + GradScaler forward/backward OK"


def check_actor_tokenizer() -> tuple[bool, str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    return True, f"distilgpt2 tokenizer loads, vocab_size={tok.vocab_size}"


def check_reward_tokenizer() -> tuple[bool, str]:
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        return True, "prajjwal1/bert-tiny tokenizer loads"
    except Exception as e:  # noqa: BLE001 — preflight wants to report, not raise
        return False, (
            f"{type(e).__name__}: known issue (see PLANNING.md Phase 3) — this "
            "transformers version can't build a fast tokenizer from bert-tiny's "
            "legacy vocab.txt-only repo. Fix before scripts/train_reward.py: pin "
            "an older transformers, use_fast=False with a tokenizers-compatible "
            "fallback, or switch to a bert-tiny repo that ships tokenizer.json."
        )


def check_sentencepiece() -> tuple[bool, str]:
    import sentencepiece  # noqa: F401

    return True, "importable"


def check_hf_dataset_download() -> tuple[bool, str]:
    import socket

    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True, "huggingface.co reachable"
    except OSError as e:
        return False, f"cannot reach huggingface.co: {e} — Anthropic/hh-rlhf download will fail"


CHECKS: list[tuple[str, "callable"]] = [
    ("CUDA device", check_cuda),
    ("NCCL backend (needed for Phase 6 benchmark)", check_nccl_backend),
    ("AMP roundtrip", check_amp_roundtrip),
    ("distilgpt2 tokenizer", check_actor_tokenizer),
    ("bert-tiny tokenizer (known risk, Phase 3)", check_reward_tokenizer),
    ("sentencepiece", check_sentencepiece),
    ("network: huggingface.co", check_hf_dataset_download),
]


@app.command()
def main() -> None:
    results = []
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"raised {type(e).__name__}: {e}"
        typer.echo(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
        results.append(ok)

    if not all(results):
        typer.echo("\nOne or more checks failed — fix before running the real training phases.")
        raise typer.Exit(code=1)
    typer.echo("\nAll checks passed.")


if __name__ == "__main__":
    app()
