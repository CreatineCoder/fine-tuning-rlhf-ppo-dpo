"""Multi-process correctness tests for the Phase 6 topology, run over `gloo`
on CPU (this dev machine has no CUDA). Uses real `torch.distributed` process
groups spawned as separate OS processes — not mocked."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rlhf_scratch.distributed import ByteCounter, build_topology, init_process_group, recv_cross_group, send_cross_group, teardown

WORLD_SIZE = 4
ACTOR_CRITIC_RANKS = [0, 1]
REF_REWARD_RANKS = [2, 3]
N_STEPS = 50


def _topology_worker(rank: int, master_port: str, result_queue) -> None:
    init_process_group(rank, WORLD_SIZE, backend="gloo", master_port=master_port)
    topology = build_topology(rank, WORLD_SIZE, ACTOR_CRITIC_RANKS, REF_REWARD_RANKS)

    tensor = torch.tensor([float(rank)])
    dist.all_reduce(tensor, group=topology.my_group)

    result_queue.put((rank, topology.is_trainable, tensor.item()))
    teardown()


def test_topology_assigns_groups_and_isolates_all_reduce():
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(target=_topology_worker, args=(r, "29511", result_queue)) for r in range(WORLD_SIZE)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=60)
    assert all(p.exitcode == 0 for p in processes), "a rank crashed or hung"

    results = {}
    while not result_queue.empty():
        rank, is_trainable, reduced = result_queue.get()
        results[rank] = (is_trainable, reduced)

    assert len(results) == WORLD_SIZE
    for rank in ACTOR_CRITIC_RANKS:
        assert results[rank][0] is True
    for rank in REF_REWARD_RANKS:
        assert results[rank][0] is False

    # all_reduce(sum) within {0,1} => 0+1=1; within {2,3} => 2+3=5
    assert results[0][1] == 1.0
    assert results[1][1] == 1.0
    assert results[2][1] == 5.0
    assert results[3][1] == 5.0


def _cross_group_worker(rank: int, master_port: str, result_queue) -> None:
    init_process_group(rank, WORLD_SIZE, backend="gloo", master_port=master_port)
    build_topology(rank, WORLD_SIZE, ACTOR_CRITIC_RANKS, REF_REWARD_RANKS)

    counter = ByteCounter()
    if rank == 2:  # frozen reward group sends scalar rewards to the trainable group
        for step in range(N_STEPS):
            payload = torch.full((8,), float(step), dtype=torch.float32)
            send_cross_group(payload, dst=0, counter=counter)
    elif rank == 0:
        received_last = None
        for step in range(N_STEPS):
            buf = torch.zeros(8, dtype=torch.float32)
            recv_cross_group(buf, src=2)
            received_last = buf.clone()
        result_queue.put(("last_received", received_last.tolist()))

    if rank == 2:
        result_queue.put(("sender_bytes", counter.total_bytes))

    teardown()


def test_cross_group_send_recv_survives_many_steps_without_deadlock():
    """Exit criteria: no deadlocks across >=50 simulated cross-group steps."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(target=_cross_group_worker, args=(r, "29512", result_queue)) for r in range(WORLD_SIZE)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=90)
    assert all(p.exitcode == 0 for p in processes), "a rank crashed or hung across the cross-group loop"

    results = dict()
    while not result_queue.empty():
        key, value = result_queue.get()
        results[key] = value

    assert results["last_received"] == [float(N_STEPS - 1)] * 8
    # 50 steps * 8 float32 elements * 4 bytes = 1600 bytes total
    assert results["sender_bytes"] == N_STEPS * 8 * 4
