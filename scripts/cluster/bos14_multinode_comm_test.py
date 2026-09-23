"""Cross-node collective timing, at the payload sizes rl_games actually sends.

    torchrun ... scripts/cluster/bos14_multinode_comm_test.py

Measured against the pretrained checkpoint's own sizes:
  * all_reduce of the flattened actor gradient, 7.81 M float32 (31 MB), once per minibatch step
    (`a2c_common.py:379`); the central value net adds 2.04 M (8 MB);
  * gather_object of the full training state, once per epoch (`a2c_common.py:1603`). That state
    carries env_state + obs, ~150 MB per rank, and on a 1 GbE link it is the likely dominant cost.
Uses gloo, the backend train.py initialises.
"""

import os
import pickle
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def timed(fn, iters):
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters


def main():
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    host = [None] * world
    dist.all_gather_object(host, socket.gethostname())
    if rank == 0:
        print(f"[comm] world={world} hosts={host}", flush=True)

    results = {}
    for name, numel in (("actor_grad_31MB", 7_811_752), ("critic_grad_8MB", 2_037_769)):
        buf = torch.randn(numel, device=dev)

        def op(buf=buf):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

        results[name] = timed(op, 10)

    # ~150 MB python object, the size of get_full_state_weights() with env_state + obs attached.
    state = {"w": torch.randn(37_500_000).numpy()}
    payload_mb = len(pickle.dumps(state)) / 1e6

    def gather():
        out = [None] * world if rank == 0 else None
        dist.gather_object(state, out, dst=0)

    results[f"gather_object_{payload_mb:.0f}MB"] = timed(gather, 3)

    if rank == 0:
        for k, v in results.items():
            print(f"[comm] {k:<24} {v * 1000:8.1f} ms", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
