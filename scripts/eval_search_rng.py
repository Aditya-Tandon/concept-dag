#!/usr/bin/env python
"""eval_search_rng.py — the mechanical P0 / S1 / S2 / S3 evaluator for H10, the `search_compose`
device re-seed re-baseline (Central Library/Hypotheses/concept-dag/kan-gated-growth/
search-compose-device-reseed.md).

The note is the spec. This file implements its gate table and its branch chain verbatim, and is
committed and fixture-tested BEFORE the sweep, to the standard H8 and H9 were held to. It does
the evaluation ONLY: no plotting, no re-running experiments. The loaders, the seed-level collapse
and the bit-identity field-set comparison are imported from ``eval_provisional`` (H8) and
``eval_provisional_v2`` (H9) rather than duplicated, so "bit-identical" and "seed-level SE" keep
one definition across the three loops.

Stages
------
  --p0     P0a determinism (the 3 pre-flight duplicate pairs, bit-identical 3/3); P0b the device
           fork (i: the CUDA unit test RAN and PASSED on the pod, read from the pasted pytest
           output in the progress log; ii: every run carries `search_device_rng_fix: true` and
           every gated decision's `cand_seed_base == seed*1000 + t`); P0c the flag does something
           (each of the 9 pre-flight runs — at full evaluation, EVERY run — differs from its
           archived counterpart); P0d 5-Datasets 42 `shadow` bit-identical to 42 `off`.
  --signs  S1: the within-seed paired comparisons keep their signs (V2a on its 11 frozen
           archived-non-zero runs, V2c, V5a, V5b, V0b, the 5-Datasets evalue/off identity, R1b's
           absolute bars), each against its pre-registered pair count.
  --se     S2: seed-level SE old vs new on the same seeds, with a paired seed-label bootstrap
           interval on SE_new / SE_old, and the ordered tree on INTERVALS (judge change 3).
  --s3     S3: the re-run `eval_attn_root.py` / `eval_provisional_v2.py` gate JSONs read at the
           note's named paths against the archived ones, plus the decision-flip census.
  With no stage flag, all four run, plus the 124-run completeness check; every gate is computed
  and reported, and only then is the chain applied. The branch is named only when all four ran.

Branch chain (first match wins; each branch carries its gate_verdict, status and action)
---------------------------------------------------------------------------------------
  FLAG-INERT -> IDENTITY-BROKEN -> INCOMPLETE -> VERDICT-FLIPS -> SIGNS-BREAK ->
  SIGNS-HOLD-SE-{WIDENS, NARROWS, MIXED, UNCHANGED, UNRESOLVED}; UNMATCHED-COMBINATION otherwise,
  which after INCOMPLETE is reachable only when a stage was not computed at all (`not-run`).

Layout expected under --new (the sweep root, `results_rng/` on the pod)
-----------------------------------------------------------------------
    ctrl_<arm>/seed_<S>/exp_ctrl_<stream>/exp3a_kan_results.json   off, evalue, shadow, cls_off
    int_<arm>/seed_<S>/exp_ctrl_s_interleave/exp3a_kan_results.json  off, evalue, shadow
    5ds_<arm>/seed_<S>/exp5ds_kan/exp3a_kan_results.json            off, shadow, evalue, cls_off
    preflight_dup/{ctrl_off,int_evalue,5ds_off}/...                P0a's duplicates, OUTSIDE
                                                                   every evaluator's arm glob
    5ds_n8_on/, 5ds_n8_off/  (seeds 45-49)                         the n = 8 extension: attn_pool
                                                                   off, flag ON / flag OFF, same SHA
                                                                   (S2's powered 5-Datasets clause;
                                                                   P0b requires n8_off UNflagged;
                                                                   P0c pairs n8_on with n8_off)
    progress_rng*.log                                              (P0b(i): the pytest output)

Usage
-----
    python scripts/eval_search_rng.py --p0 --new <sweep> --old-attn <attn archive> \\
        --old-prov <v2 archive> --out preflight_gates.json      # exit 0 ok / 1 failed / 2 short
    python scripts/eval_search_rng.py --p0 --signs --se --s3 --new <sweep> \\
        --old-attn <attn archive> --old-prov <v2 archive> \\
        --s3-new-attn gates_h10_attn.json --s3-new-prov gates_h10_prov.json \\
        --bootstrap 2000 --out se_h10.json

Ambiguities resolved while implementing (conservative readings, reasoned at each site)
--------------------------------------------------------------------------------------
  * **A sign that goes to exactly 0** on a comparison whose archived value was non-zero has NOT
    kept its sign: it is counted as broken. An archived exact 0 is reported TIED, never tested.
  * **"R1b's absolute threshold"** is read as both of R1b's bars, per (seed) and per (seed,
    non-SVHN task): the attention arm's AA against 0.9113, and its per-task delta vs the
    `mlp_cls` arm at the same seed against -0.01. Each is a signed margin whose sign must hold.
  * **"CTrL t3 accuracy"** is `test_accs_final[3]` — the accuracy on task 3 (SVHN, the
    `--ctrl_middles` third middle) after the run's last consolidation; the 5-Datasets "SVHN root
    accuracy" is R1a's own observable, `test_accs[3]`.
  * **The bootstrap is exhaustive when n^n <= --bootstrap** (the note: "at n = 3 only 27
    distinct resamples exist and the interval is reported as such"); the interval is the 95 %
    percentile interval, and a resample whose OLD SE is 0 (every label the same seed) has no
    ratio and is counted in `n_undefined` — 3 of 27 at n = 3.
  * **An incomplete gate is not a failed gate.** A gate whose pre-registered inputs are missing
    is `incomplete` (mirrors H9's `_identity_gate`, except that a gate that is short AND already
    mismatching stays `fail`), and routes the chain to INCOMPLETE.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import random
import re
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_provisional as h8          # noqa: E402  (loaders + _seed_level)
import eval_provisional_v2 as v2       # noqa: E402  (compare_runs, _identity_gate, V2/V5/V0b)

CTRL_STREAMS = h8.CTRL_STREAMS
INT_STREAM = "s_interleave"
T3, T4 = 3, 4

CTRL_SEEDS = (42, 43, 44, 45, 46)
INT_SEEDS = tuple(range(42, 52))
FDS_SEEDS = (42, 43, 44)
N8_SEEDS = (45, 46, 47, 48, 49)               # the n = 8 extension's new seeds
#: Arm dirs run WITHOUT --search_device_rng_fix, by design: P0b(ii) requires them UNflagged.
UNFLAGGED_DIRS = ("5ds_n8_off",)

# --- thresholds, verbatim from the note (and, for R1b, from eval_attn_root.py) ---------------
S2_UNCHANGED_LO, S2_UNCHANGED_HI = 0.8, 1.25   # (1) SE-UNCHANGED point-ratio window
S2_CI = (0.025, 0.975)
R1B_AA_THRESHOLD = 0.9113                     # eval_attn_root.R1B_AA_THRESHOLD
R1B_TASK_TOL = 0.01                           # eval_attn_root.R1B_TASK_TOL
FDS_SVHN_TASK = 3                             # R1a's "SVHN root (t3)"

KEY = Tuple[Optional[str], int]               # (stream, seed); stream None for 5-Datasets


# ---------------------------------------------------------------------------
# The pre-registered run table (114 runs) — every stage's expectation is read from here
# ---------------------------------------------------------------------------

def _ctrl_keys(seeds=CTRL_SEEDS) -> Set[KEY]:
    return {(st, s) for s in seeds for st in CTRL_STREAMS}


RUN_TABLE: Dict[str, Set[KEY]] = {
    "ctrl_off": _ctrl_keys(),
    "ctrl_evalue": _ctrl_keys(),
    "ctrl_shadow": _ctrl_keys(),
    "ctrl_cls_off": _ctrl_keys(),
    "int_off": {(INT_STREAM, s) for s in INT_SEEDS},
    "int_evalue": {(INT_STREAM, s) for s in INT_SEEDS},
    "int_shadow": {(INT_STREAM, 42)},
    "5ds_off": {(None, s) for s in FDS_SEEDS},
    "5ds_shadow": {(None, s) for s in FDS_SEEDS},
    "5ds_evalue": {(None, 42)},
    "5ds_cls_off": {(None, s) for s in FDS_SEEDS},
    # the three pre-flight duplicates (P0a)
    "ctrl_off_dup": {("s_plus", 42)},
    "int_evalue_dup": {(INT_STREAM, 42)},
    "5ds_off_dup": {(None, 42)},
    # the coordinator's n = 8 extension (judge option 5, taken 2026-09-24): 5-Datasets attn_pool
    # `off` at seeds 45-49, flag ON and flag OFF, both at the sweep SHA. Their own directories,
    # which no other evaluator globs, so R1a/R1b/V0b/V4 keep their registered run sets.
    "5ds_n8_on": {(None, s) for s in N8_SEEDS},
    "5ds_n8_off": {(None, s) for s in N8_SEEDS},
}
RUN_TABLE_SIZE = 124

#: Where each logical arm lives under the sweep root. The duplicates are written OUTSIDE every
#: evaluator's `ctrl_<arm>/` / `int_<arm>/` / `5ds_<arm>/` glob (judge change 10), so no evaluator
#: can double-count or pair them; the first copy, in the arm directory, is canonical.
DUP_DIR = "preflight_dup"
ARM_PATH: Dict[str, str] = {d: (os.path.join(DUP_DIR, d[:-len("_dup")]) if d.endswith("_dup")
                                else d) for d in RUN_TABLE}

#: Stage 0: the 9 pre-flight runs, as (dir, key). Six belong to the sweep table, three are dups.
#: 5-Datasets seed 42 `shadow` is here (judge change 8, a re-sequence, count unchanged) so the
#: pre-flight spans the shadow-refit arm — the configuration H8 failed — via P0d. The seed-43 runs
#: are here because P0b(ii)'s `cand_seed_base == seed*1000 + t` must be seen at TWO seeds before a
#: base constant across seeds can be ruled out from the artefacts.
PREFLIGHT_RUNS: Tuple[Tuple[str, KEY], ...] = (
    ("5ds_off", (None, 42)), ("5ds_off_dup", (None, 42)), ("5ds_off", (None, 43)),
    ("5ds_shadow", (None, 42)),
    ("int_evalue", (INT_STREAM, 42)), ("int_evalue_dup", (INT_STREAM, 42)),
    ("ctrl_off", ("s_plus", 42)), ("ctrl_off_dup", ("s_plus", 42)), ("ctrl_off", ("s_plus", 43)),
)
#: P0d: the one shadow/off pair in stage 0 (5-Datasets 42, a 16,384-row cache).
P0D_PAIR: Tuple[str, str, KEY] = ("5ds_off", "5ds_shadow", (None, 42))

#: V2a's archived-non-zero runs, frozen (judge change 7: the note's parenthetical did not add up).
#: The archive (results_provisional_v2_2026-09-23, ctrl_off vs ctrl_evalue) has an exact-0 t3
#: regret improvement at every stream of seeds 42 and 44 and at seed 43 `s_plus`; these 11 are
#: the rest. S1 asserts the archive still yields exactly this set.
V2A_NONZERO_RUNS: Tuple[str, ...] = (
    "s_minus/43", "s_out/43", "s_in/43",
    "s_minus/45", "s_out/45", "s_plus/45", "s_in/45",
    "s_minus/46", "s_out/46", "s_plus/46", "s_in/46",
)

#: The pair counts each S1 clause is pre-registered to compare (judge change 1): fewer is
#: INCOMPLETE, never a vacuous pass. V5b's count depends on which runs defer at t4, so it has
#: none and is reported.
S1_EXPECTED = {"V2a": 11, "V2c": 20, "V5a": 30, "V0b": 24, "5ds-evalue-identity": 1,
               "R1b-bars": 15}
#: P0a: (primary dir, duplicate dir, key)
P0A_PAIRS: Tuple[Tuple[str, str, KEY], ...] = (
    ("5ds_off", "5ds_off_dup", (None, 42)),
    ("int_evalue", "int_evalue_dup", (INT_STREAM, 42)),
    ("ctrl_off", "ctrl_off_dup", ("s_plus", 42)),
)

#: Each new arm dir's archived counterpart: (archive, dir) in preference order. `prov` is
#: results_provisional_v2_2026-09-23, `attn` is results_attn_root_2026-09-09 (the note: 5-Datasets
#: `5ds_off` is bit-identical to `5ds_attn_pool`, CTrL and interleave come from the v2 archive).
COUNTERPART: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "ctrl_off": (("prov", "ctrl_off"), ("attn", "ctrl_attn_pool")),
    "ctrl_evalue": (("prov", "ctrl_evalue"),),
    "ctrl_shadow": (("prov", "ctrl_shadow"),),
    "ctrl_cls_off": (("attn", "ctrl_mlp_cls"),),
    "int_off": (("prov", "int_off"),),
    "int_evalue": (("prov", "int_evalue"),),
    "int_shadow": (("prov", "int_shadow"),),
    "5ds_off": (("prov", "5ds_off"), ("attn", "5ds_attn_pool")),
    "5ds_shadow": (("prov", "5ds_shadow"),),
    "5ds_evalue": (("prov", "5ds_evalue"),),
    "5ds_cls_off": (("attn", "5ds_mlp_cls"),),
    "ctrl_off_dup": (("prov", "ctrl_off"), ("attn", "ctrl_attn_pool")),
    "int_evalue_dup": (("prov", "int_evalue"),),
    "5ds_off_dup": (("prov", "5ds_off"), ("attn", "5ds_attn_pool")),
    # same-SHA: a flag-ON run's counterpart is the flag-OFF run at the same seed IN THE SWEEP
    "5ds_n8_on": (("new", "5ds_n8_off"),),
}


def check_run_table() -> Dict:
    """The table sums to the note's 124 (114 + the n = 8 extension's 10), and V0b's hard-coded 24-pair expectation
    (`eval_provisional_v2.expected_pairs`, `:1191` in the note) is exactly the shadow arms of this
    table, each with an `off` partner — a shadow arm short of it would MANUFACTURE a V0b failure,
    which is why the note runs all 20 CTrL shadow runs. Raises on any discrepancy."""
    n = sum(len(v) for v in RUN_TABLE.values())
    empty = {k: {} for k in ("baseline_ctrl", "baseline_5ds", "h8_ctrl", "h8_int", "h8_5ds")}
    v0b = v2.expected_pairs(empty, preflight=False)["V0b"]
    shadow = RUN_TABLE["ctrl_shadow"] | RUN_TABLE["int_shadow"] | RUN_TABLE["5ds_shadow"]
    off = RUN_TABLE["ctrl_off"] | RUN_TABLE["int_off"] | RUN_TABLE["5ds_off"]
    problems = []
    if n != RUN_TABLE_SIZE:
        problems.append(f"run table has {n} runs, the note has {RUN_TABLE_SIZE}")
    if len(v0b) != 24:
        problems.append(f"V0b expects {len(v0b)} pairs, the note has 24")
    if v0b != shadow:
        problems.append(f"V0b expectation != the table's shadow runs: "
                        f"{sorted(map(str, v0b ^ shadow))}")
    if not v0b <= off:
        problems.append(f"V0b pairs with no off partner: {sorted(map(str, v0b - off))}")
    pre = [k for k in PREFLIGHT_RUNS if k[1] not in RUN_TABLE.get(k[0], set())]
    if pre or len(PREFLIGHT_RUNS) != 9:
        problems.append(f"pre-flight runs outside the table: {pre}")
    if problems:
        raise AssertionError("; ".join(problems))
    return {"n_runs": n, "v0b_pairs": len(v0b)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

Tree = Dict[KEY, Dict]


def load_dir(root: Optional[str], dirname: str) -> Tree:
    """{(stream, seed): results} for one arm directory (a path relative to `root`), whatever its
    family — the family is read from the directory's own name."""
    if not root:
        return {}
    path = os.path.join(root, dirname)
    dirname = os.path.basename(dirname)
    if dirname.startswith("ctrl_"):
        t = h8._load_ctrl_tree(path)
        return {(st, s): r for s, streams in t.items() for st, r in streams.items()}
    if dirname.startswith("int_"):
        t = h8._load_single_stream_tree(path, "exp_ctrl_s_interleave")
        return {(INT_STREAM, s): r for s, r in t.items()}
    if dirname.startswith("5ds_"):
        t = h8._load_single_stream_tree(path, "exp5ds_kan")
        return {(None, s): r for s, r in t.items()}
    raise ValueError(f"unknown family for {dirname!r}")


def load_new(root: Optional[str]) -> Dict[str, Tree]:
    return {d: load_dir(root, ARM_PATH[d]) for d in RUN_TABLE}


def completeness(new: Dict[str, Tree]) -> Dict:
    """Every run of the 114-run table, present or not (judge change 1: an absent run routes the
    chain to INCOMPLETE rather than letting a thinner sweep pass)."""
    missing = [{"dir": ARM_PATH[d], "run": _keystr(k)} for d, ks in RUN_TABLE.items()
               for k in sorted(ks, key=lambda k: (k[1], str(k[0]))) if new[d].get(k) is None]
    # Runs OUTSIDE the table are reported, not dropped and not failed: P0b/P0c still read them
    # (an extra unflagged run is still an unflagged run), but no S-stage expectation counts them.
    unexpected = [{"dir": ARM_PATH[d], "run": _keystr(k)} for d, t in new.items()
                  for k in sorted(set(t) - RUN_TABLE[d], key=lambda k: (k[1], str(k[0])))]
    return {"n_expected": RUN_TABLE_SIZE, "n_present": RUN_TABLE_SIZE - len(missing),
            "missing": missing, "unexpected": unexpected,
            "verdict": "pass" if not missing else "incomplete"}


def load_old(old_attn: Optional[str], old_prov: Optional[str]) -> Dict[str, Dict[str, Tree]]:
    dirs = {"attn": set(), "prov": set()}
    for prefs in COUNTERPART.values():
        for arch, d in prefs:
            if arch in dirs:                   # "new" counterparts live in the sweep itself
                dirs[arch].add(d)
    roots = {"attn": old_attn, "prov": old_prov}
    return {arch: {d: load_dir(roots[arch], d) for d in sorted(ds)} for arch, ds in dirs.items()}


def counterpart(old: Dict[str, Dict[str, Tree]], new_dir: str, key: KEY,
                new: Optional[Dict[str, Tree]] = None) -> Tuple[Optional[Dict], Optional[str]]:
    for arch, d in COUNTERPART.get(new_dir, ()):
        src = (new or {}) if arch == "new" else old.get(arch, {})
        r = src.get(d, {}).get(key)
        if r is not None:
            return r, f"{arch}:{d}"
    return None, None


def _ctrl_nested(t: Tree) -> Dict[int, Dict[str, Dict]]:
    out: Dict[int, Dict[str, Dict]] = {}
    for (st, s), r in t.items():
        out.setdefault(s, {})[st] = r
    return out


def _single_nested(t: Tree) -> Dict[int, Dict]:
    return {s: r for (_st, s), r in t.items()}


def _keystr(key: KEY) -> str:
    st, s = key
    return f"{st or '5ds'}/{s}"


def _status(gate: Dict) -> str:
    return gate.get("verdict", "not-run")


def _finish(gate: Dict, failures: List, missing: List) -> Dict:
    """pass / fail / incomplete. A gate short of inputs AND already failing stays `fail`."""
    gate["failures"] = failures
    gate["missing"] = missing
    if failures:
        gate["verdict"] = "fail"
    elif missing:
        gate["verdict"] = "incomplete"
    else:
        gate["verdict"] = "pass"
    return gate


def _identity_status(g: Dict) -> Dict:
    """H9's `_identity_gate` reports a short gate as `fail` + `incomplete`; H10 separates them."""
    val = g.get("value") or {}
    if g.get("incomplete") and not val.get("n_mismatches"):
        g["verdict"] = "incomplete"
    return g


# ---------------------------------------------------------------------------
# P0 — the pre-flight
# ---------------------------------------------------------------------------

CUDA_TEST = "test_search_compose_restores_the_cuda_generator_when_the_seed_is_threaded"
_PYTEST_RE = re.compile(
    r"test_search_compose_device_rng\.py::" + CUDA_TEST +
    r"\s+(?P<outcome>PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)")


def gate_p0a(new: Dict[str, Tree]) -> Dict:
    pairs, expected = [], set()
    for prim, dup, key in P0A_PAIRS:
        expected.add(key)
        ra, rb = new[prim].get(key), new[dup].get(key)
        pairs.append((key[1], key[0], ra, rb))
    g = v2._identity_gate(
        "P0a", "the 3 pre-flight duplicate pairs: 5-Datasets 42 off, s_interleave 42 evalue, "
               "CTrL s_plus 42 off — each run twice at the sweep SHA",
        pairs, "first run", "duplicate", expected=expected)
    return _identity_status(g)


def parse_pytest_outcomes(paths: Sequence[str]) -> List[Dict]:
    out = []
    for p in paths or []:
        if not p or not os.path.isfile(p):
            continue
        with open(p, errors="replace") as f:
            for i, line in enumerate(f, 1):
                m = _PYTEST_RE.search(line)
                if m:
                    out.append({"file": p, "line": i, "outcome": m.group("outcome")})
    return out


def gate_p0b(new: Dict[str, Tree], progress: Sequence[str], required: Sequence[Tuple[str, KEY]]
             ) -> Dict:
    """(i) the CUDA unit test RAN and PASSED (every recorded outcome PASSED, at least one);
    (ii) every run present carries `search_device_rng_fix: true` and every gated decision
    (task >= 1) carries `cand_seed_base == seed * 1000 + task`. `required` are the runs that
    MUST be present (the 8 pre-flight runs, or the whole table)."""
    gate = {"gate": "P0b",
            "observable": "(i) pytest tests/test_search_compose_device_rng.py -v on the pod, "
                          "pasted into the progress log; (ii) every flagged run's artefacts",
            "threshold": f"(i) {CUDA_TEST} must RUN (not skip) and PASS; (ii) "
                         "search_device_rng_fix: true in every results JSON and "
                         "cand_seed_base == seed*1000 + t at every gated decision",
            "verdict": "not-run"}
    failures, missing = [], []

    outcomes = parse_pytest_outcomes(progress)
    bad = [o for o in outcomes if o["outcome"] != "PASSED"]
    i_verdict = ("fail" if bad else "pass" if outcomes else "incomplete")
    if bad:
        failures.append({"clause": "i", "detail": f"CUDA test outcome(s) {bad} — a SKIP means it "
                                                  f"never executed, which certifies nothing"})
    if not outcomes:
        missing.append({"clause": "i", "detail": f"no {CUDA_TEST} outcome in the progress log(s) "
                                                 f"{list(progress or [])}"})

    checked = 0
    for d, tree in new.items():
        for key, r in sorted(tree.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            checked += 1
            st, seed = key
            if d in UNFLAGGED_DIRS:
                # the n = 8 extension's flag-OFF arm must be genuinely unflagged: no key, no base
                if r.get("search_device_rng_fix") or any(
                        dec.get("cand_seed_base") is not None for dec in r.get("decisions", [])):
                    failures.append({"clause": "ii", "dir": d, "run": _keystr(key),
                                     "detail": "a flag-OFF run carries the flag or a seed base"})
                continue
            if r.get("search_device_rng_fix") is not True:
                failures.append({"clause": "ii", "dir": d, "run": _keystr(key),
                                 "detail": "search_device_rng_fix is not true — an unflagged run"})
            for dec in r.get("decisions", []):
                t = dec.get("task")
                if t is None or t < 1:
                    continue
                want = seed * 1000 + t
                got = dec.get("cand_seed_base")
                if got != want:
                    failures.append({"clause": "ii", "dir": d, "run": _keystr(key), "task": t,
                                     "cand_seed_base": got, "expected": want})
    for d, key in required:
        if new.get(d, {}).get(key) is None:
            missing.append({"clause": "ii", "dir": d, "run": _keystr(key)})
    gate["value"] = {"pytest_outcomes": outcomes, "i_verdict": i_verdict,
                     "n_runs_checked": checked}
    _finish(gate, failures, missing)
    gate["headline"] = (f"(i) {i_verdict} ({len(outcomes)} outcome lines); (ii) {checked} runs, "
                        f"{sum(1 for f in failures if f['clause'] == 'ii')} violations; "
                        f"{len(missing)} missing")
    return gate


def all_table_runs() -> List[Tuple[str, KEY]]:
    """Every (dir, key) of the table except the duplicates, which are P0a's business, and the
    unflagged n = 8 arm, which has no counterpart of its own — it IS the flag-ON arm's."""
    return [(d, k) for d, ks in RUN_TABLE.items()
            if not d.endswith("_dup") and d not in UNFLAGGED_DIRS
            for k in sorted(ks, key=lambda k: (k[1], str(k[0])))]


def gate_p0c(new: Dict[str, Tree], old: Dict[str, Dict[str, Tree]],
             runs: Sequence[Tuple[str, KEY]] = PREFLIGHT_RUNS) -> Dict:
    """Each listed run must NOT reproduce its archived counterpart — the pre-flight's 9 at stage 0,
    and every run of the sweep at full evaluation (judge change 9: the `mlp_cls`, `shadow` and
    `evalue` paths are otherwise never checked for flag wiring). ANY run identical to its archive
    is FLAG-INERT. Cross-archive, so the note's presence rule is relaxed (`strict_presence=False`)
    and the two new keys, which `compare_runs` never reads, are excluded by construction."""
    gate = {"gate": "P0c",
            "observable": f"each of {len(runs)} runs vs its archived (unfixed-code) counterpart",
            "threshold": "NOT identical — an inert flag would reproduce the archive",
            "verdict": "not-run"}
    rows, failures, missing = [], [], []
    for d, key in runs:
        rn = new[d].get(key)
        ro, src = counterpart(old, d, key, new)
        if rn is None or ro is None:
            missing.append({"dir": d, "run": _keystr(key),
                            "absent": "new" if rn is None else "archive"})
            continue
        mism = v2.compare_runs(ro, rn, key[1], key[0], "archive", "new", strict_presence=False)
        row = {"dir": d, "run": _keystr(key), "archive": src, "n_differences": len(mism),
               "first_difference": mism[0] if mism else None}
        rows.append(row)
        if not mism:
            failures.append(row)
    gate["value"] = {"rows": rows}
    _finish(gate, failures, missing)
    gate["headline"] = (f"{len(rows) - len(failures)}/{len(rows)} runs differ from the archive; "
                        f"{len(failures)} reproduce it; {len(missing)} missing")
    return gate


def gate_p0d(new: Dict[str, Tree]) -> Dict:
    """5-Datasets seed 42 `shadow` must be bit-identical to seed 42 `off` at the sweep SHA — the
    shadow refit must act on nothing, on the 16,384-row cache H8's P0b failed on."""
    off_d, sh_d, key = P0D_PAIR
    g = v2._identity_gate(
        "P0d", "5-Datasets seed 42 shadow vs off, in stage 0 (the configuration H8 failed)",
        [(key[1], key[0], new[off_d].get(key), new[sh_d].get(key))],
        "arm off", "arm shadow", expected={key})
    return _identity_status(g)


# ---------------------------------------------------------------------------
# S1 — signs
# ---------------------------------------------------------------------------

def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _sign_clause(name: str, old_rows: Dict, new_rows: Dict, what: str) -> Dict:
    """`*_rows` are {label: value}. A non-zero archived value must keep its sign; an archived
    exact 0 is reported TIED; a label present in the archive and absent from the sweep is
    missing."""
    rows, failures, missing, tied = [], [], [], []
    for label in sorted(old_rows, key=str):
        vo, vn = old_rows[label], new_rows.get(label)
        if vn is None:
            missing.append({"label": label, "old": vo})
            continue
        row = {"label": label, "old": vo, "new": vn}
        if vo == 0:
            row["status"] = "TIED"
            tied.append(row)
        else:
            row["status"] = "held" if _sign(vn) == _sign(vo) else "BROKEN"
            if row["status"] == "BROKEN":
                failures.append(row)
        rows.append(row)
    clause = {"clause": name, "observable": what, "rows": rows,
              "n_compared": sum(1 for r in rows if r["status"] != "TIED"),
              "n_tied": len(tied), "n_broken": len(failures)}
    return _finish(clause, failures, missing)


def _v2a_rows(off: Tree, ev: Tree) -> Dict:
    out = {}
    for key in sorted(set(off) & set(ev), key=lambda k: (k[1], str(k[0]))):
        (r_off, _), (r_ev, _) = v2._regret_final(off[key], T3), v2._regret_final(ev[key], T3)
        if r_off is not None and r_ev is not None:
            out[_keystr(key)] = r_off - r_ev           # eval_provisional_v2.py:358's sign
    return out


def _v2c_rows(off: Tree, ev: Tree) -> Dict:
    out = {}
    for key in sorted(set(off) & set(ev), key=lambda k: (k[1], str(k[0]))):
        (a_off, _), (a_ev, _) = h8.final_aa(off[key]), h8.final_aa(ev[key])
        if a_off is not None and a_ev is not None:
            out[_keystr(key)] = a_ev - a_off
    return out


def _v5_pairs(off_ctrl: Tree, ev_ctrl: Tree, off_int: Tree, ev_int: Tree) -> List[Tuple]:
    pairs = [(s, st, a, b) for s, st, a, b in h8.paired_ctrl(_ctrl_nested(off_ctrl),
                                                             _ctrl_nested(ev_ctrl))]
    pairs += [(k[1], INT_STREAM, off_int[k], ev_int[k])
              for k in sorted(set(off_int) & set(ev_int), key=lambda k: k[1])]
    return pairs


def _r1b_margins(treat: Tree, ctrl: Tree) -> Dict:
    """R1b's two bars as signed margins: AA - 0.9113 per seed; (T - C) + 0.01 per non-SVHN task."""
    out = {}
    for key in sorted(treat, key=lambda k: k[1]):
        t = treat[key]
        aa = t.get("average_accuracy")
        if aa is not None:
            out[f"seed {key[1]} AA - {R1B_AA_THRESHOLD}"] = aa - R1B_AA_THRESHOLD
        c = ctrl.get(key)
        if c is None:
            continue
        for i, (ta, ca) in enumerate(zip(t.get("test_accs", []), c.get("test_accs", []))):
            if i == FDS_SVHN_TASK:
                continue
            out[f"seed {key[1]} task {i} (T-C)+{R1B_TASK_TOL}"] = (ta - ca) + R1B_TASK_TOL
    return out


def _clause_n(name: str, c: Dict) -> int:
    """The number of pairs a clause actually compared, in the unit its S1_EXPECTED count uses:
    V2a its non-tied runs (11), V5a its off/evalue pairs, the identity clauses their compared
    runs, V2c and R1b-bars every paired row (tied or not)."""
    if name == "V2a":
        return c.get("n_compared", 0)
    if name == "V5a":
        return c.get("n_pairs", 0)
    if name in ("V0b", "5ds-evalue-identity"):
        return (c.get("value") or {}).get("n_runs_compared", 0)
    return len(c.get("rows", []))


def stage_s1(new: Dict[str, Tree], old: Dict[str, Dict[str, Tree]]) -> Dict:
    P, A = old["prov"], old["attn"]
    clauses: Dict[str, Dict] = {}

    # V2a — the PRIMARY clause, on the archived-non-zero runs only (11 of 20 in the archive).
    # Tested on the FROZEN 11 only (archived exact 0s ride along as TIED): a run outside the
    # frozen set is never sign-tested, and an archive that does not yield the frozen set makes
    # the clause unreadable (incomplete) rather than testing a set the note never registered.
    old_v2a = _v2a_rows(P["ctrl_off"], P["ctrl_evalue"])
    clauses["V2a"] = _sign_clause(
        "V2a", {k: v for k, v in old_v2a.items() if k in V2A_NONZERO_RUNS or v == 0},
        _v2a_rows(new["ctrl_off"], new["ctrl_evalue"]),
        "per-run t3 regret improvement off_regret - evalue_regret (post-consolidation), paired "
        "by (stream, seed), on the 11 frozen archived-non-zero runs; archived exact 0s are TIED")
    nonzero = sorted(k for k, v in old_v2a.items() if v != 0)
    clauses["V2a"]["n_archived_nonzero"] = len(nonzero)
    clauses["V2a"]["frozen_nonzero_runs"] = list(V2A_NONZERO_RUNS)
    if nonzero != sorted(V2A_NONZERO_RUNS):
        # The wrong archive, or an archive that changed under the note: the restriction the note
        # froze no longer describes it, so the clause cannot be read.
        clauses["V2a"]["archive_mismatch"] = {
            "archive_only": sorted(set(nonzero) - set(V2A_NONZERO_RUNS)),
            "frozen_only": sorted(set(V2A_NONZERO_RUNS) - set(nonzero))}
        clauses["V2a"]["missing"].append({"detail": "the archive's non-zero V2a runs are not the "
                                                    "frozen 11"})
        if clauses["V2a"]["verdict"] == "pass":
            clauses["V2a"]["verdict"] = "incomplete"

    clauses["V2c"] = _sign_clause(
        "V2c", _v2c_rows(P["ctrl_off"], P["ctrl_evalue"]),
        _v2c_rows(new["ctrl_off"], new["ctrl_evalue"]),
        "per-run average_accuracy_final difference evalue - off")

    # V5a / V5b — through H9's own gate so the split at the first deferral is its definition.
    g5_new = v2.gate_v5(_v5_pairs(new["ctrl_off"], new["ctrl_evalue"],
                                  new["int_off"], new["int_evalue"]))
    g5_old = v2.gate_v5(_v5_pairs(P["ctrl_off"], P["ctrl_evalue"], P["int_off"], P["int_evalue"]))
    a_new = ((g5_new.get("value") or {}).get("a_non_deferred_positions") or {})
    leaks = a_new.get("before_first_deferral_violations", [])
    v5_pairs_new = _v5_pairs(new["ctrl_off"], new["ctrl_evalue"], new["int_off"],
                             new["int_evalue"])
    v5a = {"clause": "V5a", "observable": "per-task pre-deferral deltas on test_accs (CTrL + "
                                          "s_interleave, off vs evalue) — must stay exactly 0",
           "json_path": "V5.value.a_non_deferred_positions.before_first_deferral_violations",
           "n_pairs": len(v5_pairs_new), "n_leaks": len(leaks), "leaks": leaks}
    clauses["V5a"] = _finish(v5a, leaks,
                             [] if g5_new.get("value") else [{"detail": "no off/evalue pairs"}])

    def _b_rows(g):
        rows = ((g.get("value") or {}).get("b_deferred_revisit") or {}).get("rows", [])
        return {f"{r['stream']}/{r['seed']}": r["diff"] for r in rows}

    b_old, b_new = _b_rows(g5_old), _b_rows(g5_new)
    v5b = _sign_clause("V5b", {k: v for k, v in b_old.items() if k in b_new},
                       b_new, "per-run deferred-revisit delta (CTrL t4, revisit_of == 0), on the "
                              "runs that defer at t4 in BOTH the archive and the sweep")
    # A run that defers at t4 on one side only has no paired delta: that is a decision change,
    # counted by S3's census, not a sign.
    v5b["unpaired_archive_only"] = sorted(set(b_old) - set(b_new))
    v5b["unpaired_sweep_only"] = sorted(set(b_new) - set(b_old))
    clauses["V5b"] = v5b

    # V0b — 24 shadow/off pairs, exactly identical, through H9's gate and its 24-pair expectation.
    empty = {k: {} for k in ("baseline_ctrl", "baseline_5ds", "h8_ctrl", "h8_int", "h8_5ds")}
    g0b = v2.gate_v0b(_ctrl_nested(new["ctrl_off"]), _single_nested(new["int_off"]),
                      _single_nested(new["5ds_off"]), _ctrl_nested(new["ctrl_shadow"]),
                      _single_nested(new["int_shadow"]), _single_nested(new["5ds_shadow"]),
                      expected=v2.expected_pairs(empty, preflight=False)["V0b"])
    clauses["V0b"] = _identity_status(g0b)

    # 5-Datasets evalue vs off, seed 42 — H9's P1b claim, exactly identical.
    k = (None, 42)
    gid = v2._identity_gate(
        "5ds-evalue-identity", "5-Datasets seed 42 evalue vs off (H9 P1b: the e-process never "
                               "returned UNDETERMINED at a 16k-sample position)",
        [(42, None, new["5ds_off"].get(k), new["5ds_evalue"].get(k))],
        "arm off", "arm evalue", expected={k})
    clauses["5ds-evalue-identity"] = _identity_status(gid)

    clauses["R1b-bars"] = _sign_clause(
        "R1b-bars", _r1b_margins(A["5ds_attn_pool"], A["5ds_mlp_cls"]),
        _r1b_margins(new["5ds_off"], new["5ds_cls_off"]),
        f"R1b's absolute bars as signed margins: attention-arm AA - {R1B_AA_THRESHOLD} per seed; "
        f"(attn - mlp_cls) + {R1B_TASK_TOL} per non-SVHN task per seed")

    # A clause that compared fewer pairs than it is pre-registered to compare is INCOMPLETE, not a
    # vacuous pass (judge change 1). A clause already failing stays failed.
    for name, want in S1_EXPECTED.items():
        c = clauses[name]
        got = _clause_n(name, c)
        c["n_expected"], c["n_got"] = want, got
        if got < want and c.get("verdict") == "pass":
            c["verdict"] = "incomplete"
            c.setdefault("missing", []).append(
                {"detail": f"{got} of {want} pre-registered pairs compared"})

    verdicts = {n: _status(c) for n, c in clauses.items()}
    stage = {"gate": "S1", "clauses": clauses, "clause_verdicts": verdicts,
             "not_re_evaluable": ["V2b (needs arm always)"],
             "exempt": ["the CTrL t3 decision LABEL (counted by S3's census)"]}
    stage["verdict"] = ("fail" if "fail" in verdicts.values() else
                        "pass" if all(v == "pass" for v in verdicts.values()) else "incomplete")
    stage["headline"] = ", ".join(f"{n}={v}" for n, v in verdicts.items())
    return stage


# ---------------------------------------------------------------------------
# S2 — the error bars
# ---------------------------------------------------------------------------

def _per_seed(rows: List[Dict]) -> Dict[int, float]:
    """eval_provisional.py::_seed_level's collapse: mean the per-(seed, stream) values WITHIN
    each seed first. Returned as {seed: mean}, the unit the bootstrap resamples."""
    sl = h8._seed_level(rows)
    return {int(s): v for s, v in sl["per_seed"].items()}


def _rows(tree: Tree, fn) -> List[Dict]:
    out = []
    for key, r in tree.items():
        v = fn(r)
        if v is not None:
            out.append({"seed": key[1], "diff": v})
    return out


def _aa(r: Dict) -> Optional[float]:
    return h8.final_aa(r)[0]


def _t3_acc(r: Dict) -> Optional[float]:
    accs = h8.final_accs(r)[0]
    return accs[T3] if accs and len(accs) > T3 else None


def _svhn_root(r: Dict) -> Optional[float]:
    accs = r.get("test_accs") or []                   # R1a's own observable (eval_attn_root)
    return accs[FDS_SVHN_TASK] if len(accs) > FDS_SVHN_TASK else None


def _paired_rows(off: Tree, ev: Tree, fn) -> List[Dict]:
    out = []
    for key in set(off) & set(ev):
        v = fn(off[key], ev[key])
        if v is not None:
            out.append({"seed": key[1], "diff": v})
    return out


def _regret_improvement(off_r: Dict, ev_r: Dict) -> Optional[float]:
    (a, _), (b, _) = v2._regret_final(off_r, T3), v2._regret_final(ev_r, T3)
    return None if a is None or b is None else a - b


def _aa_diff(off_r: Dict, ev_r: Dict) -> Optional[float]:
    a, b = _aa(off_r), _aa(ev_r)
    return None if a is None or b is None else b - a


def S2_METRICS(new: Dict[str, Tree], old: Dict[str, Dict[str, Tree]]) -> List[Dict]:
    P = old["prov"]
    return [
        {"metric": "s_interleave off AA", "primary": True,
         "old": _rows(P["int_off"], _aa), "new": _rows(new["int_off"], _aa)},
        {"metric": "5-Datasets off AA", "primary": False,
         "old": _rows(P["5ds_off"], _aa), "new": _rows(new["5ds_off"], _aa)},
        {"metric": "5-Datasets off SVHN root accuracy", "primary": False,
         "old": _rows(P["5ds_off"], _svhn_root), "new": _rows(new["5ds_off"], _svhn_root)},
        {"metric": "CTrL off seed-level AA", "primary": False,
         "old": _rows(P["ctrl_off"], _aa), "new": _rows(new["ctrl_off"], _aa)},
        {"metric": "CTrL off seed-level t3 accuracy", "primary": False,
         "old": _rows(P["ctrl_off"], _t3_acc), "new": _rows(new["ctrl_off"], _t3_acc)},
        {"metric": "CTrL paired t3 regret improvement off - evalue", "primary": False,
         "old": _paired_rows(P["ctrl_off"], P["ctrl_evalue"], _regret_improvement),
         "new": _paired_rows(new["ctrl_off"], new["ctrl_evalue"], _regret_improvement)},
        {"metric": "CTrL paired AA difference evalue - off", "primary": False,
         "old": _paired_rows(P["ctrl_off"], P["ctrl_evalue"], _aa_diff),
         "new": _paired_rows(new["ctrl_off"], new["ctrl_evalue"], _aa_diff)},
        # The n = 8 extension (coordinator decision 2026-09-24, the judge's option 5): flag OFF =
        # the archive's seeds 42-44 (the default path is byte-identical at the sweep SHA, asserted
        # by tests/test_mlp_cls_byte_identity.py) + the sweep's unflagged seeds 45-49; flag ON = the
        # sweep's 5ds_off 42-44 + 5ds_n8_on 45-49. Same SHA for 10 of the 16 runs; a secondary,
        # and the only 5-Datasets clause with the power to resolve (8^8 > --bootstrap, so 2,000
        # random paired label resamples).
        {"metric": "5-Datasets off AA, n = 8 (flag ON vs OFF)", "primary": False,
         "old": _rows({**P["5ds_off"], **new["5ds_n8_off"]}, _aa),
         "new": _rows({**new["5ds_off"], **new["5ds_n8_on"]}, _aa)},
    ]


def _se(vals: Sequence[float]) -> Optional[float]:
    return statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) >= 2 else None


def _quantile(xs: List[float], q: float) -> float:
    """Linear-interpolation quantile of a sorted list (numpy's default)."""
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def se_ratio(old_ps: Dict[int, float], new_ps: Dict[int, float], n_boot: int,
             rng_seed: int = 20260924) -> Dict:
    """SE_new / SE_old on the SAME seeds, with a paired seed-label bootstrap: each resample draws
    the seed LABELS once and applies them to both arms. Exhaustive when n^n <= n_boot."""
    seeds = sorted(set(old_ps) & set(new_ps))
    n = len(seeds)
    out = {"seeds": seeds, "n_seeds": n,
           "seeds_old_only": sorted(set(old_ps) - set(new_ps)),
           "seeds_new_only": sorted(set(new_ps) - set(old_ps))}
    se_o = _se([old_ps[s] for s in seeds])
    se_n = _se([new_ps[s] for s in seeds])
    out.update({"se_old": se_o, "se_new": se_n})
    if se_o is None or se_n is None or se_o == 0:
        out.update({"ratio": None, "defined": False,
                    "why_undefined": "n < 2" if n < 2 else "old SE is 0"})
        return out
    out.update({"ratio": se_n / se_o, "defined": True})

    exhaustive = n ** n <= n_boot
    if exhaustive:
        draws = itertools.product(seeds, repeat=n)
        n_draws = n ** n
    else:
        rng = random.Random(rng_seed)
        draws = ([rng.choice(seeds) for _ in range(n)] for _ in range(n_boot))
        n_draws = n_boot
    ratios, undefined = [], 0
    for labels in draws:
        so = _se([old_ps[s] for s in labels])
        sn = _se([new_ps[s] for s in labels])
        if not so:
            undefined += 1
            continue
        ratios.append(sn / so)
    ratios.sort()
    ci = ([_quantile(ratios, S2_CI[0]), _quantile(ratios, S2_CI[1])] if ratios else None)
    out["bootstrap"] = {"exhaustive": exhaustive, "n_resamples": n_draws,
                        "n_undefined": undefined, "n_used": len(ratios), "ci95": ci,
                        "note": (f"exhaustive over all {n_draws} ordered label resamples "
                                 f"(n = {n}); NOT a {n_boot}-resample quantity"
                                 if exhaustive else f"{n_boot} random paired resamples, "
                                                    f"seed {rng_seed}")}
    out["ci_contains_1"] = bool(ci and ci[0] <= 1.0 <= ci[1])
    return out


def metric_resolution(r: Dict) -> str:
    """A metric's SE change is RESOLVED only by its interval (judge change 3): `widens` when the
    95 % interval lies wholly above 1, `narrows` wholly below, otherwise `unresolved` — "not
    resolved at this n". A point ratio alone never resolves anything."""
    ci = (r.get("bootstrap") or {}).get("ci95") if r.get("defined") else None
    if not ci:
        return "undefined"
    if ci[0] > 1.0:
        return "widens"
    if ci[1] < 1.0:
        return "narrows"
    return "unresolved"


def transport(r: Dict, resolution: str) -> Dict:
    """What a downstream note may carry (judge change 4): a FACTOR only when the interval excludes
    1; otherwise the interval itself, declared as unresolved."""
    ci = (r.get("bootstrap") or {}).get("ci95")
    if resolution in ("widens", "narrows"):
        return {"kind": "factor", "factor": r.get("ratio"), "ci95": ci}
    return {"kind": "interval", "ci95": ci,
            "note": "not resolved at this n — transport the interval, not a factor"}


def classify_s2(metrics: List[Dict]) -> Dict:
    """The ordered tree, first match wins, on INTERVALS only (judge change 3). The primary
    (s_interleave, 10 seeds) is the only clause with power:
      SE-WIDENS     the primary's interval lies wholly above 1;
      SE-NARROWS    the primary's interval lies wholly below 1;
      SE-MIXED      the primary's interval sits inside [0.8, 1.25] AND at least one secondary's
                    interval excludes 1 — the only reading that licenses "stream-dependent";
      SE-UNCHANGED  the primary's interval sits inside [0.8, 1.25] and no secondary is resolved;
      SE-UNRESOLVED otherwise — not resolved at this n (e.g. a primary interval straddling 1 but
                    wider than the window, or an undefined primary)."""
    for m in metrics:
        m["resolution"] = metric_resolution(m["ratio"])
        m["transport"] = transport(m["ratio"], m["resolution"])
    primary = next((m for m in metrics if m["primary"]), None)
    secondary = [m for m in metrics if not m["primary"]]
    resolved_sec = [m["metric"] for m in secondary if m["resolution"] in ("widens", "narrows")]
    p_res = primary["resolution"] if primary else "undefined"
    p_ci = (primary["ratio"].get("bootstrap") or {}).get("ci95") if primary else None
    p_in_window = bool(p_ci and S2_UNCHANGED_LO <= p_ci[0] and p_ci[1] <= S2_UNCHANGED_HI)

    if p_res == "widens":
        outcome = "SE-WIDENS"
    elif p_res == "narrows":
        outcome = "SE-NARROWS"
    elif p_in_window and resolved_sec:
        outcome = "SE-MIXED"
    elif p_in_window:
        outcome = "SE-UNCHANGED"
    else:
        outcome = "SE-UNRESOLVED"
    return {"outcome": outcome,
            "primary_ratio": primary["ratio"].get("ratio") if primary else None,
            "primary_ci95": p_ci, "primary_resolution": p_res,
            "primary_ci_inside_window": p_in_window,
            "resolved_secondaries": resolved_sec,
            "n_secondary_defined": sum(1 for m in secondary if m["resolution"] != "undefined")}


def stage_s2(new: Dict[str, Tree], old: Dict[str, Dict[str, Tree]], n_boot: int) -> Dict:
    metrics = []
    for m in S2_METRICS(new, old):
        po, pn = _per_seed(m["old"]), _per_seed(m["new"])
        metrics.append({"metric": m["metric"], "primary": m["primary"],
                        "per_seed_old": po, "per_seed_new": pn,
                        "ratio": se_ratio(po, pn, n_boot)})
    cls = classify_s2(metrics)
    verdict = "pass" if metrics and any(m["ratio"]["defined"] for m in metrics) else "not-run"
    return {"gate": "S2", "metrics": metrics, **cls, "verdict": verdict,
            "quantity": "the COMBINED effect of seed-dependent candidate inits and the post-gate "
                        "device stream (e3c964d changes both; judge change 6)",
            "headline": f"{cls['outcome']} (primary ratio {cls['primary_ratio']}, CI "
                        f"{cls['primary_ci95']}; resolved secondaries "
                        f"{cls['resolved_secondaries']})"}


# ---------------------------------------------------------------------------
# S3 — verdicts, read at the note's named paths
# ---------------------------------------------------------------------------

#: (qualified name, which evaluator's JSON, path to the verdict-bearing record, field).
S3_PATHS: Tuple[Tuple[str, str, Tuple[str, ...], str], ...] = (
    ("adoption R1a", "attn", ("R1a",), "verdict"),
    ("adoption R1b", "attn", ("R1b",), "verdict"),
    ("adoption R1c", "attn", ("R1c",), "verdict"),
    ("adoption R1d", "attn", ("R1d",), "verdict"),
    ("adoption R2a", "attn", ("R2a",), "verdict"),
    ("adoption R2b", "attn", ("R2b",), "verdict"),
    ("adoption R2c", "attn", ("R2c",), "verdict"),
    ("adoption R2d", "attn", ("R2d",), "verdict"),
    ("adoption R3", "attn", ("R3",), "verdict"),
    ("adoption R4", "attn", ("R4",), "verdict"),
    ("H9 V0b", "prov", ("V0b",), "verdict"),
    ("H9 V2a", "prov", ("V2", "value", "a_regret_improvement"), "verdict"),
    ("H9 V2c", "prov", ("V2", "value", "c_collateral_aa"), "verdict"),
    ("H9 V3", "prov", ("V3",), "verdict"),
    ("H9 V4", "prov", ("V4",), "verdict"),
    ("H9 V5a", "prov", ("V5", "value", "a_non_deferred_positions"), "verdict"),
    ("H9 V5b", "prov", ("V5", "value", "b_deferred_revisit"), "verdict"),
    ("H9 V5c", "prov", ("V5", "value", "c_rolled_back_audit_entries"), "verdict"),
    ("H9 V6-roots", "prov", ("V6", "value"), "n_unresolved==0"),
)
S3_NOT_RE_EVALUABLE = ("adoption N1 (retired)", "adoption N2 (retired)", "H9 V0a-i", "H9 V0a-ii",
                       "H9 V1", "H9 V1b", "H9 V2b", "H9 R1", "H9 R2", "H9 V6 dump clause",
                       "H9 DA")


def _dig(doc: Optional[Dict], path: Sequence[str]) -> Optional[Dict]:
    cur = doc
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur if isinstance(cur, dict) else None


def _read_s3(doc: Optional[Dict], path: Sequence[str], field: str) -> Tuple[object, Optional[str]]:
    rec = _dig(doc, path)
    if rec is None:
        return None, "record absent"
    if field == "n_unresolved==0":
        n = rec.get("n_unresolved")
        return (None, "n_unresolved absent") if n is None else (n == 0, None)
    v = rec.get(field)
    note = rec.get("note")
    if v == "not-run":
        return v, "not-run"
    if isinstance(note, str) and "missing" in note.lower():
        return v, f"note names a missing input: {note}"
    return v, None


def decision_census(new: Dict[str, Tree], old: Dict[str, Dict[str, Tree]]) -> Dict:
    """How many runs change ANY decision vs the archive, and at which task. Reported, not gated.
    The note's "40 CTrL" are the attention arm's `off` + `evalue` (judge ambiguity 7); `mlp_cls`
    and `shadow` are reported separately so they never pad that count."""
    out = {}
    groups = (("ctrl", ("ctrl_off", "ctrl_evalue")), ("int", ("int_off", "int_evalue")),
              ("ctrl_mlp_cls", ("ctrl_cls_off",)), ("ctrl_shadow", ("ctrl_shadow",)))
    for label, dirs in groups:
        runs, by_task, changed = 0, {}, []
        for d in dirs:
            for key, rn in sorted(new[d].items(), key=lambda kv: (kv[0][1], str(kv[0][0]))):
                ro, _src = counterpart(old, d, key)
                if ro is None:
                    continue
                runs += 1
                tasks = []
                for dn in rn.get("decisions", []):
                    do = h8.dec_at(ro, dn.get("task"))
                    if do is None or do.get("decision") != dn.get("decision"):
                        tasks.append({"task": dn.get("task"),
                                      "old": do.get("decision") if do else None,
                                      "new": dn.get("decision")})
                if tasks:
                    changed.append({"dir": d, "run": _keystr(key), "changes": tasks})
                    for t in tasks:
                        by_task[str(t["task"])] = by_task.get(str(t["task"]), 0) + 1
        out[label] = {"n_runs_compared": runs, "n_runs_changed": len(changed),
                      "changes_by_task": by_task, "runs": changed}
    return out


def stage_s3(new_attn: Optional[Dict], new_prov: Optional[Dict], old_attn: Optional[Dict],
             old_prov: Optional[Dict], census: Dict) -> Dict:
    docs = {"attn": (old_attn, new_attn), "prov": (old_prov, new_prov)}
    rows, flipped, not_computed = [], [], []
    for name, which, path, field in S3_PATHS:
        o_doc, n_doc = docs[which]
        vo, why_o = _read_s3(o_doc, path, field)
        vn, why_n = _read_s3(n_doc, path, field)
        row = {"gate": name, "path": ".".join(path) + "." + field, "old": vo, "new": vn}
        if why_o or why_n:
            row["status"] = "NOT-COMPUTED"
            row["why"] = {"old": why_o, "new": why_n}
            not_computed.append(name)
        elif vo == vn:
            row["status"] = "held"
        else:
            row["status"] = "FLIPPED"
            row["direction"] = f"{vo} -> {vn}"
            flipped.append(row)
        rows.append(row)
    # adoption R3 is descriptive, but it names its own sub-branch; a change there is reported.
    r3o, r3n = (_dig(old_attn, ("R3",)) or {}), (_dig(new_attn, ("R3",)) or {})
    stage = {"gate": "S3", "rows": rows, "flipped": flipped, "not_computed": not_computed,
             "not_re_evaluable": list(S3_NOT_RE_EVALUABLE),
             "adoption_R3_branch": {"old": r3o.get("r3_branch"), "new": r3n.get("r3_branch")},
             "census": census}
    computed = [r for r in rows if r["status"] != "NOT-COMPUTED"]
    # A flip is a flip even if other gates are short; a re-evaluable gate that came back
    # NOT-COMPUTED makes the stage INCOMPLETE (judge change 1) — every gate in S3_PATHS is one the
    # note lists as re-evaluable on this sweep, so none may be quietly absent.
    if new_attn is None and new_prov is None:
        stage["verdict"] = "not-run"
        stage["note"] = "no re-run gate JSONs supplied (--s3-new-attn / --s3-new-prov)"
    elif flipped:
        stage["verdict"] = "fail"
    elif not_computed:
        stage["verdict"] = "incomplete"
    else:
        stage["verdict"] = "pass"
    stage["headline"] = (f"{len(computed) - len(flipped)}/{len(computed)} computed gates held, "
                         f"{len(flipped)} flipped "
                         f"({', '.join(r['gate'] + ' ' + r['direction'] for r in flipped)}), "
                         f"{len(not_computed)} NOT-COMPUTED; census ctrl "
                         f"{census['ctrl']['n_runs_changed']}/{census['ctrl']['n_runs_compared']}"
                         f", int {census['int']['n_runs_changed']}/"
                         f"{census['int']['n_runs_compared']}")
    return stage


# ---------------------------------------------------------------------------
# Branch precedence, and each branch's gate record and pre-registered action (judge change 12)
# ---------------------------------------------------------------------------

S2_BRANCH = {"SE-WIDENS": "SIGNS-HOLD-SE-WIDENS", "SE-NARROWS": "SIGNS-HOLD-SE-NARROWS",
             "SE-MIXED": "SIGNS-HOLD-SE-MIXED", "SE-UNCHANGED": "SIGNS-HOLD-SE-UNCHANGED",
             "SE-UNRESOLVED": "SIGNS-HOLD-SE-UNRESOLVED"}

_HOLD = {"H10.1": "accepted", "H10.3": "accepted", "H10.4": "accepted"}
BRANCH_RECORD: Dict[str, Dict] = {
    "FLAG-INERT": {
        "gate_verdict": "inconclusive", "status": "needs-review", "hypotheses": {},
        "action": "STOP. At stage 0: stage 1 is not released. At full evaluation: the sweep is "
                  "void. Fix the flag's wiring (a new SHA, a new tag, a new pre-flight); no "
                  "downstream note is touched."},
    "IDENTITY-BROKEN": {
        "gate_verdict": "inconclusive", "status": "needs-review",
        "hypotheses": {"H10.4": "rejected"},
        "action": "STOP. Stage 1 is not released. Diagnose the determinism / device-fork / "
                  "shadow-refit failure before any re-run; no downstream note is touched."},
    "INCOMPLETE": {
        "gate_verdict": "inconclusive", "status": "needs-review", "hypotheses": {},
        "action": "Re-run the missing runs or re-stage the evaluators named in the reason, then "
                  "re-evaluate. No branch below may be read and no downstream note is touched "
                  "until the evidence is complete."},
    "VERDICT-FLIPS": {
        "gate_verdict": "rejected", "status": "refuted",
        "hypotheses": {"H10.3": "rejected"},
        "action": "Apply the note's 'What changes downstream if the branch is VERDICT-FLIPS', "
                  "per note, for the gates named in the reason."},
    "SIGNS-BREAK": {
        "gate_verdict": "rejected", "status": "refuted",
        "hypotheses": {"H10.1": "rejected", "H10.3": "accepted", "H10.4": "accepted"},
        "action": "Every published claim resting on a broken paired sign is flagged "
                  "needs-review in its note; the verdicts stand (S3 held)."},
    "SIGNS-HOLD-SE-WIDENS": {
        "gate_verdict": "accepted", "status": "validated",
        "hypotheses": {**_HOLD, "H10.2": "accepted"},
        "action": "Replace R3's caveat with the measured factor on the streams whose interval "
                  "excludes 1; transport intervals, not factors, elsewhere."},
    "SIGNS-HOLD-SE-NARROWS": {
        "gate_verdict": "accepted", "status": "validated",
        "hypotheses": {**_HOLD, "H10.2": "rejected"},
        "action": "Record that the fix NARROWED the powered stream's spread (the opposite of "
                  "H10.2); transport as for SE-WIDENS."},
    "SIGNS-HOLD-SE-MIXED": {
        "gate_verdict": "accepted", "status": "validated",
        "hypotheses": {**_HOLD, "H10.2": "accepted (stream-dependent)"},
        "action": "Narrow R3's caveat per stream: a factor where the interval excludes 1, the "
                  "interval elsewhere."},
    "SIGNS-HOLD-SE-UNCHANGED": {
        "gate_verdict": "accepted", "status": "validated",
        "hypotheses": {**_HOLD, "H10.2": "rejected"},
        "action": "R3's caveat is retired for the powered stream; the n = 3..5 streams carry "
                  "their intervals — SE-UNCHANGED is a statement about n, not a vindication."},
    "SIGNS-HOLD-SE-UNRESOLVED": {
        "gate_verdict": "accepted", "status": "validated",
        "hypotheses": {**_HOLD, "H10.2": "inconclusive"},
        "action": "Signs and verdicts stand; R3's caveat is kept and each archived SE carries "
                  "this sweep's interval, not a factor. Not resolved at this n."},
    "UNMATCHED-COMBINATION": {
        "gate_verdict": "inconclusive", "status": "needs-review", "hypotheses": {},
        "action": "Report the combination and the non-passing gates; the coordinator decides. "
                  "Reached only when a stage was not computed at all (not-run: not requested, "
                  "or its inputs could not be loaded)."},
}


def determine_branch(g: Dict[str, Dict]) -> Dict:
    """FLAG-INERT -> IDENTITY-BROKEN -> INCOMPLETE -> VERDICT-FLIPS -> SIGNS-BREAK ->
    SIGNS-HOLD-SE-{WIDENS, NARROWS, MIXED, UNCHANGED, UNRESOLVED}, first match wins; anything
    else is UNMATCHED-COMBINATION with the non-passing gates named — never a fall-through to a
    pass. `g` may carry a `completeness` record for the 114-run table."""
    v = {k: _status(x) for k, x in g.items()}
    not_passing = sorted(k for k, s in v.items() if s != "pass")

    def br(name, reason):
        return {"branch": name, "reason": reason, "verdicts": v, **BRANCH_RECORD[name]}

    if v.get("P0c") == "fail":
        return br("FLAG-INERT", "P0c: a flagged run reproduces its archived counterpart")
    if any(v.get(k) == "fail" for k in ("P0a", "P0b", "P0d")):
        return br("IDENTITY-BROKEN",
                  f"P0a={v.get('P0a')}, P0b={v.get('P0b')}, P0d={v.get('P0d')}")
    short = sorted(k for k, s in v.items() if s == "incomplete")
    s3_nc = (g.get("S3") or {}).get("not_computed") or []
    if short or s3_nc:
        why = f"incomplete: {short}" + (f"; S3 NOT-COMPUTED: {s3_nc}" if s3_nc else "")
        return br("INCOMPLETE", why)
    stages = ("P0a", "P0b", "P0c", "P0d", "S1", "S2", "S3")
    if any(v.get(k) != "pass" and not (k in ("S1", "S3") and v.get(k) == "fail")
           for k in stages):
        return br("UNMATCHED-COMBINATION", f"no branch matches; not passing: {not_passing}")
    if v["S3"] == "fail":
        flips = [f"{r['gate']} ({r['direction']})" for r in g["S3"]["flipped"]]
        return br("VERDICT-FLIPS", "S3 flipped: " + "; ".join(flips))
    if v["S1"] == "fail":
        broken = [n for n, s in g["S1"]["clause_verdicts"].items() if s == "fail"]
        return br("SIGNS-BREAK", f"S1 clauses broken with S3 holding: {broken}")
    outcome = g["S2"]["outcome"]
    return br(S2_BRANCH[outcome], f"S1 and S3 hold; S2 = {outcome}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_json(path: Optional[str]) -> Optional[Dict]:
    return h8._read_json(path) if path and os.path.isfile(path) else None


def _first_existing(*paths: Optional[str]) -> Optional[str]:
    return next((p for p in paths if p and os.path.isfile(p)), None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="H10 evaluator (search_compose device re-seed)")
    p.add_argument("--p0", action="store_true", help="P0a/P0b/P0c, the pre-flight")
    p.add_argument("--signs", action="store_true", help="S1, paired signs")
    p.add_argument("--se", action="store_true", help="S2, the SE ratios")
    p.add_argument("--s3", action="store_true", help="S3, verdicts at the named paths + census")
    p.add_argument("--new", required=True, help="the sweep root (results_rng/)")
    p.add_argument("--old-attn", default=None, help="results_attn_root_2026-09-09")
    p.add_argument("--old-prov", default=None, help="results_provisional_v2_2026-09-23")
    p.add_argument("--progress", nargs="*", default=None,
                   help="progress logs holding the pasted pytest output "
                        "(default: <new>/progress_rng*.log)")
    p.add_argument("--s3-new-attn", default=None,
                   help="eval_attn_root.py's JSON on the sweep (default <new>/gates_h10_attn.json)")
    p.add_argument("--s3-new-prov", default=None, help="eval_provisional_v2.py's JSON on the "
                                                       "sweep (default <new>/gates_h10_prov.json)")
    p.add_argument("--s3-old-attn", default=None, help="default <old-attn>/gates.json")
    p.add_argument("--s3-old-prov", default=None, help="default <old-prov>/gates_h9.json")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--out", default="se_h10.json")
    return p


def _print(gates: Dict[str, Dict]) -> None:
    print(f"{'gate':13s}{'verdict':12s}headline")
    print("-" * 120)
    for k, g in gates.items():
        head = g.get("headline") or g.get("note") or (
            f"{g.get('n_present')}/{g.get('n_expected')} runs present" if k == "Completeness"
            else "")
        print(f"{k:13s}{str(g.get('verdict')):12s}{str(head)[:95]}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    table = check_run_table()
    stages = [s for s in ("p0", "signs", "se", "s3") if getattr(args, s)] or [
        "p0", "signs", "se", "s3"]
    preflight_only = stages == ["p0"]

    new = load_new(args.new)
    old = load_old(args.old_attn, args.old_prov)
    progress = (args.progress if args.progress is not None
                else sorted(glob.glob(os.path.join(args.new, "progress_rng*.log"))))

    # Every gate is computed, reported in the note's order, and only then is the chain applied
    # (judge change 2): VERDICT-FLIPS vs SIGNS-BREAK needs S3 even after S1 has failed.
    gates: Dict[str, Dict] = {}
    if not preflight_only:
        gates["Completeness"] = completeness(new)
    if "p0" in stages:
        required = (list(PREFLIGHT_RUNS) if preflight_only else
                    [(d, k) for d, ks in RUN_TABLE.items() for k in ks])
        gates["P0a"] = gate_p0a(new)
        gates["P0b"] = gate_p0b(new, progress, required)
        # stage 0: the 9 pre-flight runs; full evaluation: every run of the sweep (judge change 9)
        gates["P0c"] = gate_p0c(new, old, PREFLIGHT_RUNS if preflight_only else all_table_runs())
        gates["P0d"] = gate_p0d(new)
    if "signs" in stages:
        gates["S1"] = stage_s1(new, old)
    if "se" in stages:
        gates["S2"] = stage_s2(new, old, args.bootstrap)
    if "s3" in stages:
        new_attn = _load_json(args.s3_new_attn or _first_existing(
            os.path.join(args.new, "gates_h10_attn.json")))
        new_prov = _load_json(args.s3_new_prov or _first_existing(
            os.path.join(args.new, "gates_h10_prov.json")))
        old_attn = _load_json(args.s3_old_attn or (args.old_attn and os.path.join(
            args.old_attn, "gates.json")))
        old_prov = _load_json(args.s3_old_prov or (args.old_prov and os.path.join(
            args.old_prov, "gates_h9.json")))
        gates["S3"] = stage_s3(new_attn, new_prov, old_attn, old_prov, decision_census(new, old))

    out: Dict = {"run_table": table, "stages": stages, "progress_logs": progress, **gates}
    _print(gates)
    p0 = {k: gates[k] for k in ("P0a", "P0b", "P0c", "P0d") if k in gates}
    failed = [k for k, g in p0.items() if _status(g) == "fail"]
    short = [k for k, g in p0.items() if _status(g) == "incomplete"]
    if len(stages) == 4:
        branch = determine_branch(gates)
        out["branch"], out["branch_detail"] = branch["branch"], branch
        print(f"\n=== branch: {branch['branch']} (gate_verdict {branch['gate_verdict']}) ===\n"
              f"  reason: {branch['reason']}\n  action: {branch['action']}")
    elif p0:
        out["preflight"] = {"failed_gates": failed, "incomplete_gates": short}
        if failed:
            pre = determine_branch({k: p0[k] for k in p0})
            out["preflight"]["branch"] = pre["branch"]
            print(f"\n=== PREFLIGHT FAILED: {failed} -> {pre['branch']} — do not release "
                  f"stage 1 ===")
        elif short:
            print(f"\n=== PREFLIGHT INCONCLUSIVE: {short} short of pre-registered inputs ===")
        else:
            print("\n=== PREFLIGHT OK: P0a, P0b, P0c, P0d pass — stage 1 may be released on the "
                  "coordinator's go ===")
    code = 1 if failed else 2 if short else 0
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
