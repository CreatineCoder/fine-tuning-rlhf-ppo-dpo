"""Minimal cross-group communication: only rollout tokens, log-probs, and
scalar rewards ever cross from the frozen reference-reward group to the
trainable actor-critic group — never full gradients or optimizer state. A
`ByteCounter` wraps every cross-group send so Phase 6's benchmark can report a
real, auditable "bytes exchanged across the group boundary" number."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ByteCounter:
    total_bytes: int = 0

    def count(self, tensor: torch.Tensor) -> None:
        self.total_bytes += tensor.element_size() * tensor.nelement()

    def reset(self) -> None:
        self.total_bytes = 0


def send_cross_group(tensor: torch.Tensor, dst: int, counter: ByteCounter | None = None) -> None:
    """Point-to-point send on the default (world) process group — subgroups
    created by `dist.new_group()` can't send()/recv() to ranks outside
    themselves, so cross-group traffic always goes through the world group.
    Counted if a `ByteCounter` is supplied.

    Caller and receiver must agree on tensor shape/dtype ahead of time (e.g.
    by padding rollouts to a fixed max length) — plain send/recv doesn't
    negotiate shape.
    """
    if counter is not None:
        counter.count(tensor)
    dist.send(tensor.contiguous(), dst=dst)


def recv_cross_group(tensor: torch.Tensor, src: int) -> torch.Tensor:
    """Receives into a pre-allocated `tensor` of the expected shape/dtype."""
    dist.recv(tensor, src=src)
    return tensor
