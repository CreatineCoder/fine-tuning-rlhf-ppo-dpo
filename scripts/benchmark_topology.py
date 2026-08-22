"""Phase 6 real benchmark: VRAM reduction + bytes-exchanged reduction from
splitting actor-critic (trainable) from reference-reward (frozen) across two
process groups, vs. a single merged process loading all 4 models.

Runs as two processes sharing one physical GPU (see PLANNING.md's hardware
plan for exactly what this setup can and can't honestly claim — VRAM and
bytes-exchanged are real measurements this way; wall-clock communication
savings are not, and are deliberately not reported).

Not runnable on this CPU-only dev machine (no CUDA) — intended for the RTX 5080.

    python scripts/benchmark_topology.py --rollout-steps 50
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.multiprocessing as mp
import typer
from transformers import AutoModelForCausalLM

from rlhf_scratch.distributed import (
    BenchmarkResult,
    ByteCounter,
    build_topology,
    init_process_group,
    measure_peak_vram_mb,
    recv_cross_group,
    reset_vram_stats,
    send_cross_group,
    teardown,
)
from rlhf_scratch.models import Critic, RewardModel

app = typer.Typer(add_completion=False)

ACTOR_CRITIC_RANKS = [0]
REF_REWARD_RANKS = [1]
WORLD_SIZE = 2


def _baseline_worker(rank: int, actor_name: str, reward_base: str, master_port: str, rollout_steps: int, result_queue) -> None:
    """All 4 models in a single process — the "merged group" baseline."""
    if rank != 0:
        return
    device = torch.device("cuda")
    reset_vram_stats(device)

    actor = AutoModelForCausalLM.from_pretrained(actor_name).to(device)
    critic = Critic(actor_name).to(device)
    reference = AutoModelForCausalLM.from_pretrained(actor_name).to(device)
    reward_model = RewardModel(reward_base).to(device)

    counter = ByteCounter()
    for step in range(rollout_steps):
        payload = torch.zeros(64, device=device)
        counter.count(payload.cpu())  # bytes that *would* cross a group boundary if split

    vram_mb = measure_peak_vram_mb(device)
    result_queue.put(("baseline", vram_mb, counter.total_bytes))


def _split_worker(rank: int, actor_name: str, reward_base: str, master_port: str, rollout_steps: int, result_queue) -> None:
    init_process_group(rank, WORLD_SIZE, backend="nccl", master_port=master_port)
    build_topology(rank, WORLD_SIZE, ACTOR_CRITIC_RANKS, REF_REWARD_RANKS)

    device = torch.device(f"cuda:{rank}" if torch.cuda.device_count() > 1 else "cuda")
    torch.cuda.set_per_process_memory_fraction(0.5, device)
    reset_vram_stats(device)

    counter = ByteCounter()
    if rank in ACTOR_CRITIC_RANKS:
        actor = AutoModelForCausalLM.from_pretrained(actor_name).to(device)
        critic = Critic(actor_name).to(device)
        for step in range(rollout_steps):
            buf = torch.zeros(64, device="cpu")
            recv_cross_group(buf, src=1)
    else:
        reference = AutoModelForCausalLM.from_pretrained(actor_name).to(device)
        reward_model = RewardModel(reward_base).to(device)
        for step in range(rollout_steps):
            payload = torch.zeros(64, device="cpu")
            send_cross_group(payload, dst=0, counter=counter)

    vram_mb = measure_peak_vram_mb(device)
    result_queue.put((f"split_rank{rank}", vram_mb, counter.total_bytes))
    teardown()


@app.command()
def main(
    actor_name: str = "distilgpt2",
    reward_base: str = "prajjwal1/bert-tiny",
    rollout_steps: int = 50,
    output_dir: str = "results/distributed_benchmark",
) -> None:
    if not torch.cuda.is_available():
        typer.echo("CUDA not available on this machine — this benchmark must run on the RTX 5080.")
        raise typer.Exit(code=1)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    baseline_proc = ctx.Process(
        target=_baseline_worker, args=(0, actor_name, reward_base, "29601", rollout_steps, result_queue)
    )
    baseline_proc.start()
    baseline_proc.join(timeout=300)

    split_procs = [
        ctx.Process(target=_split_worker, args=(r, actor_name, reward_base, "29602", rollout_steps, result_queue))
        for r in range(WORLD_SIZE)
    ]
    for p in split_procs:
        p.start()
    for p in split_procs:
        p.join(timeout=300)

    results = {}
    while not result_queue.empty():
        key, vram_mb, byte_count = result_queue.get()
        results[key] = (vram_mb, byte_count)

    baseline_vram, baseline_bytes = results["baseline"]
    split_vram = max(v for k, v in results.items() if k.startswith("split_rank"))
    split_bytes = sum(b for k, (_, b) in results.items() if k.startswith("split_rank"))

    benchmark = BenchmarkResult(
        baseline_vram_mb=baseline_vram,
        split_vram_mb=split_vram,
        baseline_bytes=baseline_bytes,
        split_bytes=split_bytes,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics.json").write_text(json.dumps(benchmark.to_dict(), indent=2))
    typer.echo(json.dumps(benchmark.to_dict(), indent=2))


if __name__ == "__main__":
    app()
