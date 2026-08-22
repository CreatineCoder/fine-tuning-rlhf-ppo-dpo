"""Two-process-group distributed topology: a trainable Group A (actor+critic)
and a frozen Group B (reference+reward), each created via `dist.new_group()`
so gradients/optimizer state never need to cross the boundary — only rollout
tokens, log-probs, and reward scalars do (see `comm_hooks.py`).

On the dev machine (no CUDA) this runs multi-process over `gloo`, for
correctness/deadlock testing. On the RTX 5080 training machine the same code
runs two processes on the one GPU (see PLANNING.md's hardware plan for why a
single physical GPU is sufficient for the VRAM and bytes-exchanged numbers,
but not for a real communication-time claim)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch.distributed as dist


@dataclass
class Topology:
    rank: int
    world_size: int
    actor_critic_ranks: list[int]
    ref_reward_ranks: list[int]
    actor_critic_group: dist.ProcessGroup
    ref_reward_group: dist.ProcessGroup

    @property
    def is_trainable(self) -> bool:
        """True if this rank belongs to the actor-critic (trainable) group."""
        return self.rank in self.actor_critic_ranks

    @property
    def my_group(self) -> dist.ProcessGroup:
        return self.actor_critic_group if self.is_trainable else self.ref_reward_group


def init_process_group(
    rank: int,
    world_size: int,
    backend: str = "gloo",
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
) -> None:
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def build_topology(
    rank: int,
    world_size: int,
    actor_critic_ranks: list[int],
    ref_reward_ranks: list[int],
) -> Topology:
    """Creates the two process groups.

    Must be called by *every* rank in `world_size`, including ranks in neither
    list if any exist — `dist.new_group()` is a collective call and will
    deadlock if any rank in the process group skips it.
    """
    if set(actor_critic_ranks) & set(ref_reward_ranks):
        raise ValueError("actor_critic_ranks and ref_reward_ranks must be disjoint")
    if rank not in actor_critic_ranks and rank not in ref_reward_ranks:
        raise ValueError(f"rank {rank} not assigned to either group")

    actor_critic_group = dist.new_group(ranks=actor_critic_ranks)
    ref_reward_group = dist.new_group(ranks=ref_reward_ranks)
    dist.barrier()

    return Topology(
        rank=rank,
        world_size=world_size,
        actor_critic_ranks=actor_critic_ranks,
        ref_reward_ranks=ref_reward_ranks,
        actor_critic_group=actor_critic_group,
        ref_reward_group=ref_reward_group,
    )


def teardown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
