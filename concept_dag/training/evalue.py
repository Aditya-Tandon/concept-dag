"""Anytime-valid rung comparison: two one-sided betting e-processes over paired bit differences.

Why this exists
---------------
The ladder compares two rungs by the MEAN of the per-example held-out bit differences
``d_i = l_alt(x_i) - l_grow(x_i)`` and thresholds it once, at a fixed n. Four estimator loops
established that at CTrL's SVHN@400 the per-example SD is ~1.15 bits while the deciding margin is
0.002-0.05 bits, so the fixed-n comparison is a coin flip and no estimator fixes it
([[gate-h5-select-score-result]], [[prequential-grow-probe-stress-test]]).

[[theory-anytime-gate-regret-crystallisation]] §1 replaces the point comparison with a sequential
test. Betting on the paired differences gives a non-negative supermartingale under the null, so by
Ville's inequality a crossing of ``1/alpha`` controls the type-I error at ``alpha`` *at any stopping
time* — including "I ran out of held-out examples", which is exactly the state the gate never had.
``log2 E`` is the number of bits a bettor backing that rung saves, so the evidence for a rung is
measured in the same currency as the rung itself.

The construction (Waudby-Smith & Ramdas predictable plug-in, as in [[grunwald-2024-safe-testing]]):

    E_t = prod_{i<=t} (1 + lam_i * u_i),   u_i = clip(z_i, -B, B) / B  in [-1, 1]
    lam_i = clip( mu_hat_{i-1} / (var_hat_{i-1} + mu_hat_{i-1}^2), 0, lam_cap ),  lam_1 = 0

with ``lam_i`` predictable (a function of u_1..u_{i-1} only), so each factor has conditional
expectation <= 1 whenever ``E[u_i] <= 0``. ``B = log2(K)`` bits for a K-class task; the clip is a
modelling choice and is recorded.

This module is the reference implementation and is deliberately dependency-light (torch only for
the input tensors) so `tests/test_evalue.py` can replicate the archived desk numbers in
`h-theory-evalue.json` before anything new is run.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def eprocess_log2(u: Sequence[float], lam_cap: float = 0.5) -> List[float]:
    """log2 wealth path of ``prod (1 + lam_i u_i)`` with the predictable plug-in bet.

    ``u`` must already be scaled into [-1, 1] (i.e. clipped bit differences divided by B).
    Returns one log2 E per observation, in the order given — the ORDER IS PART OF THE TEST and
    must be fixed by the split RNG, never by anything that looks at the values.
    """
    out: List[float] = []
    w = 0.0
    n = 0
    s1 = 0.0
    s2 = 0.0
    mu = 0.0
    var = 1.0
    for ui in u:
        lam = min(lam_cap, max(0.0, mu / (var + mu * mu))) if n > 0 else 0.0
        w += math.log2(1.0 + lam * ui)
        out.append(w)
        n += 1
        s1 += ui
        s2 += ui * ui
        mu = s1 / n
        var = max(s2 / n - mu * mu, 1e-6)
    return out


def n_needed(mu_bits: float, sd_bits: float, B: float, alpha: float = 0.05,
             lam_cap: float = 0.5) -> float:
    """Held-out examples the Kelly-optimal fixed bet needs to reach 1/alpha, from (mu, sd) in bits.

    ``inf`` when the mean is on the wrong side of the null (no bet grows). This is the quantity
    that makes "noise-limited" a number: at CTrL SVHN@400 it is in the thousands against the 120
    examples the gate holds out.
    """
    if mu_bits <= 0:
        return float("inf")
    mu, var = mu_bits / B, (sd_bits / B) ** 2
    lam = min(lam_cap, mu / (var + mu * mu))
    growth = lam * mu - 0.5 * lam * lam * (var + mu * mu)      # nats/example, 2nd order
    return math.log(1.0 / alpha) / growth if growth > 0 else float("inf")


def decide(d_bits: Sequence[float], B: float, *, threshold_bits: float = 0.0,
           alpha: float = 0.05, lam_cap: float = 0.5) -> Dict[str, object]:
    """Three-valued rung test on paired per-example bit differences.

    ``d_bits[i] = l_alt(x_i) - l_grow(x_i)`` (positive => grow codes the example better).
    ``threshold_bits`` (``s``) is the ladder's own indifference margin, ``eps_grow * reducible``:
    the two processes test the SAME threshold from opposite sides, which is what makes them a
    coherent partition rather than two overlapping nulls —

        E_plus  bets on  E[d] >  s   (grow clears the ladder's bar)   from u = (d - s)/B
        E_minus bets on  E[d] <  s   (grow fails to clear it)         from u = (s - d)/B

    so both can never cross, and "neither crossed" is the honest third state. The asymmetry is the
    ladder's own: growth must beat reuse/search by ``eps`` of the reducible information, while the
    alternative wins by default — so the alternative's e-process is testing "grow did not clear the
    bar", not "the alternative is better in absolute terms".

    Returns ``{state, log2_e_plus, log2_e_minus, n, mean_d, sd_d, n_needed_plus, n_needed_minus,
    stopped_at, B, threshold_bits, alpha}`` with ``state`` in {"grow", "alt", "undetermined"}.
    """
    d = [float(x) for x in d_bits]
    n = len(d)
    empty = {"state": "undetermined", "log2_e_plus": 0.0, "log2_e_minus": 0.0, "n": n,
             "mean_d": 0.0, "sd_d": 0.0, "n_needed_plus": float("inf"),
             "n_needed_minus": float("inf"), "stopped_at": None, "B": B,
             "threshold_bits": threshold_bits, "alpha": alpha}
    if n == 0:
        return empty

    def _clip(v: float) -> float:
        return max(-B, min(B, v)) / B

    up = [_clip(x - threshold_bits) for x in d]
    um = [_clip(threshold_bits - x) for x in d]
    lp = eprocess_log2(up, lam_cap)
    lm = eprocess_log2(um, lam_cap)

    bound = math.log2(1.0 / alpha)
    stop_p = next((i for i, v in enumerate(lp) if v >= bound), None)
    stop_m = next((i for i, v in enumerate(lm) if v >= bound), None)
    if stop_p is not None and (stop_m is None or stop_p <= stop_m):
        state, stopped = "grow", stop_p + 1
    elif stop_m is not None:
        state, stopped = "alt", stop_m + 1
    else:
        state, stopped = "undetermined", None

    mean_d = sum(d) / n
    var_d = sum((x - mean_d) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    sd_d = math.sqrt(var_d)
    return {"state": state, "log2_e_plus": lp[-1], "log2_e_minus": lm[-1], "n": n,
            "mean_d": mean_d, "sd_d": sd_d,
            "n_needed_plus": n_needed(mean_d - threshold_bits, sd_d, B, alpha, lam_cap),
            "n_needed_minus": n_needed(threshold_bits - mean_d, sd_d, B, alpha, lam_cap),
            "stopped_at": stopped, "B": B, "threshold_bits": threshold_bits, "alpha": alpha}
