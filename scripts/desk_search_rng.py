"""Desk quantification of the `search_compose` device re-seed (R3), on CPU.

Two questions the [[search-compose-device-reseed]] pre-registration needs answered before any GPU
time is spent, neither of which needs a GPU:

  **A. How much device RNG does one Search gate consume, and where?** On CUDA the device generator
  is drawn from by exactly one thing in this code path: `nn.Dropout` in a rung's forward pass. On
  CPU those same draws come off the CPU generator, so they can be counted here by tallying
  `F.dropout`. The answer matters because it fixes how far the leak displaces the stream:

    * `SearchComposer` carries **no dropout** (Linear → GELU → Linear, plus the full-rank skip), so
      training the `budget` candidates draws NOTHING on the device. The device generator when
      `search_compose` returns is therefore *exactly* `manual_seed(last candidate index)` with zero
      draws consumed — not "that value plus the candidate's own training draws" as
      [[provisional-growth-code-review]] should-fix 3 assumed. The displacement is total and
      depends on `(budget, n_parents)` alone, through the last candidate's index.
    * The draws do happen, immediately afterwards and from that overwritten state: the GROW rung is
      a `ConceptModule` (`dropout=0.1`, `n_layers=2`) and so is every later `train_node`.

  Caveat, stated rather than hidden: CUDA's philox offset advances in a rounded quantum (bumped per
  launch as a function of the grid, not by the exact element count), so an element tally is an exact
  count of *what is drawn* and only a lower bound on *how far the offset moves*. Neither conclusion
  above depends on the quantum.

  **B. How does the post-gate device stream differ between run seeds, old code vs new?** Simulated
  against a stand-in `torch.Generator` wired in as the device generator (the
  `tests/test_shadow_refit_null.py` device: `torch.mps` cannot be forked at all on torch 2.0, and
  there is no CUDA here). Two runs that reach the first gate with different device states — which
  is what two `--seed` values give — are pushed through a faithful emulation of each discipline:

    old (candidate_seed_base=None): CPU-only fork; `torch.manual_seed(index)` fans out to the
        device generator (emulated explicitly, since the stand-in is not wired to torch's fan-out);
        the CPU half is restored, the device half is not; the candidate then TRAINS, drawing on the
        device outside the block.
    new (candidate_seed_base=int): device-covering fork; seed derived from (run seed, index); both
        halves restored; the candidate's training draws continue the run's own stream.

  What the table reports is the device generator's first draws after the gate, for two run seeds,
  under each discipline. Old: identical across seeds (the leak). New: distinct.

Run: `python scripts/desk_search_rng.py [--json out.json]`. CPU only, a few seconds.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
import torch.nn.functional as F

from concept_dag.training.kan_gate import (
    _candidate_init_seed,
    classification_task,
    search_compose,
)
from concept_dag.utils import rng as rng_utils
from concept_dag.utils.rng import fork_rng_all_devices

DIM = 64
N_CLASSES = 10


# ---------------------------------------------------------------------------
# A. How many device draws does one Search gate consume?
# ---------------------------------------------------------------------------


def count_dropout_draws(*, n: int, n_parents: int, budget: int, n_epochs: int,
                        val_fraction: float = 0.3) -> Dict[str, int]:
    """Tally every dropout mask ONE FULL GATE draws, split by whether it falls inside
    `search_compose` or after it.

    The gate is the real `decide_reuse_search_grow` with the published rung order
    (reuse → search → grow → null) and a `ConceptModule` grow probe, which is the only rung that
    carries dropout.
    """
    import concept_dag.training.kan_gate as kg
    from concept_dag.modules.concept_module import ConceptModule

    calls: List[int] = []
    real_dropout = F.dropout
    real_search = kg.search_compose
    span = {"enter": None, "exit": None}

    def _spy(input, p=0.5, training=True, inplace=False):
        if training and p > 0.0:
            calls.append(int(input.numel()))
        return real_dropout(input, p, training, inplace)

    def _search_spy(*args, **kwargs):
        if span["enter"] is None:
            span["enter"] = len(calls)
        out = real_search(*args, **kwargs)
        span["exit"] = len(calls)
        return out

    torch.manual_seed(0)
    X = torch.randn(n, n_parents, DIM)
    y = torch.randint(0, N_CLASSES, (n,))
    spec = classification_task(N_CLASSES)

    def _probe_factory():
        return ConceptModule(module_id="__probe__", in_dim=DIM, hidden_dim=DIM, out_dim=DIM,
                             n_layers=2, n_parents=n_parents, aggregation="mean", dropout=0.1)

    F.dropout = _spy
    kg.search_compose = _search_spy
    try:
        kg.decide_reuse_search_grow(
            X, y, _probe_factory, spec, concept_dim=DIM, n_parents=n_parents, device="cpu",
            n_epochs=n_epochs, lr=1e-3, val_fraction=val_fraction,
            search_budget=budget, search_skip=True)
    finally:
        F.dropout = real_dropout
        kg.search_compose = real_search

    inside = calls[span["enter"]:span["exit"]]
    after = calls[span["exit"]:]
    # The index the last candidate is seeded with, under the published seeding — the entire state
    # the device generator is left in.
    n_subsets = n_parents + (1 if n_parents > 1 else 0)
    last_index = (budget - 1) // n_subsets
    return {"n": n, "n_parents": n_parents, "budget": budget, "n_epochs": n_epochs,
            "gate_dropout_calls": len(calls), "gate_mask_elements": sum(calls),
            "inside_search_calls": len(inside), "inside_search_elements": sum(inside),
            "after_search_calls": len(after), "after_search_elements": sum(after),
            "published_last_candidate_index": last_index}


# ---------------------------------------------------------------------------
# B. The post-gate device stream, old vs new
# ---------------------------------------------------------------------------


def post_gate_stream(*, run_seed: int, fixed: bool, budget: int, n_parents: int,
                     device_draws_per_candidate: int = 64, k: int = 4) -> List[float]:
    """The device generator's first `k` draws after one emulated Search gate.

    `run_seed` sets the device state the gate is entered with — the state two `--seed` values
    genuinely differ in by the time the first gate is reached.
    """
    gen = torch.Generator().manual_seed(run_seed)

    def _fan_out(value: int) -> None:
        """`torch.manual_seed` reseeds the CPU generator AND every device generator."""
        torch.random.manual_seed(value)
        gen.manual_seed(value)

    # The candidate list `search_compose` builds: every subset once at restart 0, then again.
    subsets = [(i,) for i in range(n_parents)] + ([tuple(range(n_parents))] if n_parents > 1 else [])
    candidates: List[int] = []
    restart = 0
    while len(candidates) < budget:
        for _ in subsets:
            candidates.append(restart)
            if len(candidates) >= budget:
                break
        restart += 1

    base = run_seed * 1000 if fixed else None
    hooks = [("stand-in", gen.get_state, gen.set_state)]
    original_hooks = rng_utils._device_rng_hooks
    rng_utils._device_rng_hooks = lambda: hooks
    try:
        for sd in candidates:
            if fixed:
                with fork_rng_all_devices():
                    _fan_out(_candidate_init_seed(base, sd))
                    torch.randn(8)                      # the candidate's init, on the CPU
            else:
                with torch.random.fork_rng(devices=[]):
                    _fan_out(_candidate_init_seed(None, sd))
                    torch.randn(8)
            # ... and then the candidate TRAINS, outside the block, drawing dropout masks on the
            # device. This is the part the leak leaves standing on a stream nobody seeded from
            # `--seed`.
            torch.rand(device_draws_per_candidate, generator=gen)
    finally:
        rng_utils._device_rng_hooks = original_hooks

    return [round(v, 6) for v in torch.rand(k, generator=gen).tolist()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write the table as JSON here")
    args = ap.parse_args()

    out: Dict[str, object] = {}

    print("A. Device (dropout) draws in ONE gate, split at the Search rung; CPU tally")
    print(f"{'n':>7} {'par':>4} {'budget':>7} {'ep':>4} | {'inside search':>13} "
          f"{'elements':>10} | {'after search':>12} {'elements':>11} | {'sd_last':>7}")
    rows = []
    for n, n_parents, budget, n_epochs in ((200, 2, 6, 15), (400, 2, 6, 15), (400, 3, 6, 15),
                                           (2000, 3, 6, 15), (2000, 4, 6, 15)):
        r = count_dropout_draws(n=n, n_parents=n_parents, budget=budget, n_epochs=n_epochs)
        rows.append(r)
        print(f"{r['n']:>7} {r['n_parents']:>4} {r['budget']:>7} {r['n_epochs']:>4} | "
              f"{r['inside_search_calls']:>13} {r['inside_search_elements']:>10} | "
              f"{r['after_search_calls']:>12} {r['after_search_elements']:>11} | "
              f"{r['published_last_candidate_index']:>7}")
    out["device_draws_per_gate"] = rows

    print("\nB. The device generator's first draws AFTER one gate, by run seed")
    print(f"{'code':>6} {'run seed':>9}  first 4 post-gate device draws")
    brows = []
    for fixed in (False, True):
        for run_seed in (42, 43):
            s = post_gate_stream(run_seed=run_seed, fixed=fixed, budget=6, n_parents=2)
            brows.append({"code": "new" if fixed else "old", "run_seed": run_seed, "stream": s})
            print(f"{'new' if fixed else 'old':>6} {run_seed:>9}  {s}")
    out["post_gate_stream"] = brows

    old = [r["stream"] for r in brows if r["code"] == "old"]
    new = [r["stream"] for r in brows if r["code"] == "new"]
    out["old_streams_coincide_across_seeds"] = old[0] == old[1]
    out["new_streams_differ_across_seeds"] = new[0] != new[1]
    print(f"\nold: seeds 42 and 43 share the post-gate device stream: {old[0] == old[1]}")
    print(f"new: seeds 42 and 43 differ:                             {new[0] != new[1]}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
