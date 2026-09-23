"""RNG forking that actually covers the device the run trains on.

``torch.random.fork_rng`` saves and restores the CPU generator; the accelerator
generators it covers are named by its ``devices`` argument, and that argument
only ever means CUDA (torch <= 2.0 hard-codes ``torch.cuda`` inside it; MPS is
not reachable through it at all). Every call site in this repo passes
``devices=[]`` — chosen to avoid the multi-GPU initialisation warning — which
means the block protects the CPU generator and *nothing else*.

That is not a cosmetic gap. A block that calls ``torch.manual_seed`` inside such
a fork re-seeds **every** device generator (``torch.manual_seed`` fans out to
``torch.cuda.manual_seed_all`` and ``torch.mps.manual_seed``) and only the CPU
half is put back, so the device generator is left wherever the block's own work
left it. Dropout masks are drawn on the device, so all later training in the run
sees a different mask stream — an arm that is supposed to compute-and-not-act
(the P0b shadow refit) then changes the run's results. It is invisible on CPU,
which is why a CPU-only suite passes while a GPU run diverges.

``fork_rng_all_devices`` saves and restores the CPU generator plus every
initialised CUDA device and the MPS generator.

**MPS is not covered in the sense that matters.** On torch 2.0 the MPS
generator's ``get_rng_state`` / ``set_rng_state`` round-trip the 36-byte seed
state WITHOUT the philox offset, so restoring it rewinds the seed but not the
position in the stream. A block that draws on MPS therefore cannot be made a
no-op by this helper (or by any other), and a forked block on MPS is only
approximately neutral. That is why ``tests/test_shadow_refit_null.py``
deliberately uses a stand-in ``torch.Generator`` rather than ``torch.mps`` as
its device: an MPS run cannot certify the fork contract, so it cannot be used
to certify a null either. ``run_exp3a_kan`` refuses ``--provisional != off`` on
MPS for exactly this reason.

On CPU and CUDA a block wrapped in this helper IS a true no-op on the RNG
stream, which is what every identity/null claim in this project rests on.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import Callable, List, Tuple

import torch

# (label, getter, setter) for every device generator that exists on this machine.
RngHook = Tuple[str, Callable[[], torch.Tensor], Callable[[torch.Tensor], None]]


def _device_rng_hooks() -> List[RngHook]:
    """Every non-CPU generator whose state this process can read and write.

    Kept as a separate function so a CPU-only test can substitute a fake device
    and still assert the save/restore contract.
    """
    hooks: List[RngHook] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            hooks.append((
                f"cuda:{i}",
                (lambda i=i: torch.cuda.get_rng_state(i)),
                (lambda s, i=i: torch.cuda.set_rng_state(s, i)),
            ))
    try:
        # `torch.mps` is a submodule, not an attribute: it has to be imported by name.
        mps = importlib.import_module("torch.mps")
        if torch.backends.mps.is_available():
            hooks.append(("mps", mps.get_rng_state, mps.set_rng_state))
    except (ImportError, AttributeError):   # older torch without an MPS RNG API
        pass
    return hooks


@contextlib.contextmanager
def fork_rng_all_devices():
    """``torch.random.fork_rng`` over the CPU generator AND every device generator."""
    cpu_state = torch.get_rng_state()
    saved = [(label, setter, getter()) for label, getter, setter in _device_rng_hooks()]
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        for _label, setter, state in saved:
            setter(state)
