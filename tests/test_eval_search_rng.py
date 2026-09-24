"""Fixture tests for the H10 evaluator, written BEFORE the sweep.

ref: Hypotheses/concept-dag/kan-gated-growth/search-compose-device-reseed.md ("P0, S1 and S2 all
     need new evaluator code ... committed and fixture-tested before the sweep")

Every stage gets a synthetic pass case and a synthetic fail case, built as the results dicts the
runner writes, and the branch chain is driven to each of its ends — so a mis-implemented clause
cannot wait for 114 GPU runs to be discovered. On top of that the script is run end to end with
the unfixed v2 archive standing in for the sweep: it must name FLAG-INERT, since an archive run is
by construction what an inert flag would produce.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
SCRIPT = os.path.join(SCRIPTS, "eval_search_rng.py")
RESULTS = os.path.abspath(os.path.join(REPO, "..", "concept-dag-results"))
OLD_ATTN = os.path.join(RESULTS, "results_attn_root_2026-09-09")
OLD_PROV = os.path.join(RESULTS, "results_provisional_v2_2026-09-23")


@pytest.fixture(scope="module")
def ev():
    sys.path.insert(0, SCRIPTS)
    spec = importlib.util.spec_from_file_location("eval_search_rng", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic runs and trees
# ---------------------------------------------------------------------------


def mkrun(seed, *, accs=(0.9, 0.8, 0.7, 0.6, 0.5), accs_final=None, flag=True, base="ok",
          l_grow=1.0,
          oracle=0.7, decisions=("grow", "grow", "grow", "grow", "reuse")):
    """A results dict in the shape `run_exp3a_kan` writes. `base`: "ok" writes seed*1000+t,
    "const" a seed-independent base, None none."""
    decs = []
    for t, d in enumerate(decisions):
        rec = {"task": t, "decision": d}
        if t >= 1:
            rec.update({"L_grow_bits": l_grow, "L_reuse_bits": 2.0, "gate_cache_n": 400})
            if base == "ok":
                rec["cand_seed_base"] = seed * 1000 + t
            elif base == "const":
                rec["cand_seed_base"] = 42_000 + t
        if t == 3:
            rec["oracle_accs"] = {"grow": oracle, "reuse": oracle - 0.1}
        decs.append(rec)
    accs = list(accs)
    fin = list(accs_final) if accs_final is not None else list(accs)
    r = {"decisions": decs, "test_accs": accs, "test_accs_final": fin,
         "average_accuracy": sum(accs) / len(accs),
         "average_accuracy_final": sum(fin) / len(fin),
         "param_curve": [1, 2, 3, 4, 5], "param_curve_total": [1, 2, 3, 4, 5],
         "consolidation_passes": [{"at_task": 4, "ops": [], "final": True}]}
    if flag:
        r["search_device_rng_fix"] = True
    return r


def write(root, dirname, key, run):
    stream, seed = key
    base = os.path.basename(dirname)
    if base.startswith("ctrl_"):
        p = os.path.join(root, dirname, f"seed_{seed}", f"exp_ctrl_{stream}")
    elif base.startswith("int_"):
        p = os.path.join(root, dirname, f"seed_{seed}", "exp_ctrl_s_interleave")
    else:
        p = os.path.join(root, dirname, f"seed_{seed}", "exp5ds_kan")
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "exp3a_kan_results.json"), "w") as f:
        json.dump(run, f)


def acc_for(key, *, shift=0.0, jitter=True):
    """Per-seed values that vary, so every SE is defined; `shift` moves the whole run."""
    st, s = key
    j = (s - 42) * 0.003 + (hash(st) % 7) * 0.0001 if jitter else 0.0
    return tuple(round(a + j + shift, 6) for a in (0.9, 0.8, 0.7, 0.6, 0.5))


def build_world(ev, tmp_path, *, new_l_grow=1.5, pytest_outcome="PASSED", mutate=None):
    """Old archives (v2 + attn) with the unfixed discipline, and a COMPLETE 124-run sweep whose
    every flagged run differs from its archive (L_grow_bits), whose duplicates reproduce, and whose
    n = 8 flag-OFF arm is genuinely unflagged."""
    old_prov, old_attn, new = tmp_path / "prov", tmp_path / "attn", tmp_path / "new"
    for d, keys in ev.RUN_TABLE.items():
        if d.endswith("_dup"):
            continue
        if d.startswith("5ds_n8"):
            unflag = d in ev.UNFLAGGED_DIRS
            for key in keys:
                write(new, d, key, mkrun(key[1], accs=acc_for(key), flag=not unflag,
                                         base=None if unflag else "ok",
                                         l_grow=1.0 if unflag else new_l_grow))
            continue
        for key in keys:
            # evalue is better than off AFTER consolidation only: before any deferral the two
            # arms share every draw, so `test_accs` must be identical (V5a).
            tied = key[1] in (42, 44) or key == ("s_plus", 43)
            shift = 0.01 if ("evalue" in d and not (d == "ctrl_evalue" and tied)) else 0.0
            o = mkrun(key[1], accs=acc_for(key), accs_final=acc_for(key, shift=shift),
                      flag=False, base=None)
            for arch, od in ev.COUNTERPART[d][:1]:
                write(old_prov if arch == "prov" else old_attn, od, key, o)
            if d in ("ctrl_off", "5ds_off"):                 # the attn fallback archive too
                write(old_attn, ev.COUNTERPART[d][1][1], key, o)
            n = mkrun(key[1], accs=acc_for(key), accs_final=acc_for(key, shift=shift),
                      l_grow=new_l_grow)
            if d == "5ds_cls_off" or d == "ctrl_cls_off":
                n = mkrun(key[1], accs=acc_for(key, shift=-0.005), l_grow=new_l_grow)
                write(old_attn, ev.COUNTERPART[d][0][1], key,
                      mkrun(key[1], accs=acc_for(key, shift=-0.005), flag=False, base=None))
            write(new, d, key, n)
    for prim, dup, key in ev.P0A_PAIRS:
        with open(_path(new, prim, key)) as f:
            write(new, DUPS[dup], key, json.load(f))
    # 5-Datasets evalue must equal off (S1's identity clause): overwrite with the off run.
    with open(_path(new, "5ds_off", (None, 42))) as f:
        write(new, "5ds_evalue", (None, 42), json.load(f))
    with open(_path(old_prov, "5ds_off", (None, 42))) as f:
        write(old_prov, "5ds_evalue", (None, 42), json.load(f))
    # shadow must equal off (V0b)
    for fam in ("ctrl", "int", "5ds"):
        for key in ev.RUN_TABLE[f"{fam}_shadow"]:
            with open(_path(new, f"{fam}_off", key)) as f:
                write(new, f"{fam}_shadow", key, json.load(f))
    if pytest_outcome:
        (new / "progress_rnga.log").write_text(
            "tests/test_search_compose_device_rng.py::"
            f"{ev.CUDA_TEST} {pytest_outcome} [ 50%]\n")
    if mutate:
        mutate(new, old_prov, old_attn)
    return str(new), str(old_attn), str(old_prov)


DUPS = {"ctrl_off_dup": "preflight_dup/ctrl_off", "int_evalue_dup": "preflight_dup/int_evalue",
        "5ds_off_dup": "preflight_dup/5ds_off"}


def _path(root, d, key):
    """`d` is a logical arm; the duplicates live under preflight_dup/ (judge change 10)."""
    d = DUPS.get(d, d)
    st, s = key
    base = os.path.basename(d)
    sub = (f"exp_ctrl_{st}" if base.startswith("ctrl_") else
           "exp_ctrl_s_interleave" if base.startswith("int_") else "exp5ds_kan")
    return os.path.join(root, d, f"seed_{s}", sub, "exp3a_kan_results.json")


def edit(root, d, key, fn):
    p = _path(root, d, key)
    with open(p) as f:
        r = json.load(f)
    fn(r)
    with open(p, "w") as f:
        json.dump(r, f)


def load(ev, new, old_attn, old_prov):
    return ev.load_new(new), ev.load_old(old_attn, old_prov)


# ---------------------------------------------------------------------------
# The run table and V0b's 24-pair expectation
# ---------------------------------------------------------------------------


def test_run_table_is_the_notes_124_and_v0b_expects_its_24_shadow_pairs(ev):
    assert ev.check_run_table() == {"n_runs": 124, "v0b_pairs": 24}


def test_the_n8_flag_off_arm_must_be_unflagged_and_its_flag_on_twin_must_differ(ev, tmp_path):
    prog = lambda new: [os.path.join(new, "progress_rnga.log")]
    new, oa, op = build_world(ev, tmp_path / "ok")
    n, o = load(ev, new, oa, op)
    assert ev.gate_p0b(n, prog(new), [])["verdict"] == "pass"
    p0c = ev.gate_p0c(n, o, ev.all_table_runs())
    assert p0c["verdict"] == "pass" and len(p0c["value"]["rows"]) == 116   # 111 + 5 same-SHA
    assert {r["archive"] for r in p0c["value"]["rows"] if r["dir"] == "5ds_n8_on"} == {
        "new:5ds_n8_off"}

    new, oa, op = build_world(ev, tmp_path / "flagged")
    edit(new, "5ds_n8_off", (None, 47), lambda r: r.__setitem__("search_device_rng_fix", True))
    g = ev.gate_p0b(ev.load_new(new), prog(new), [])
    assert g["verdict"] == "fail" and "flag-OFF run carries" in g["failures"][0]["detail"]

    new, oa, op = build_world(ev, tmp_path / "inert")
    with open(_path(new, "5ds_n8_off", (None, 48))) as f:
        off48 = json.load(f)
    edit(new, "5ds_n8_on", (None, 48), lambda r: (r.clear(), r.update(
        {**off48, "search_device_rng_fix": True})))
    g = ev.gate_p0c(*load(ev, new, oa, op), ev.all_table_runs())
    assert g["verdict"] == "fail" and g["failures"][0]["run"] == "5ds/48"


def test_s2_carries_the_n8_metric_on_eight_seeds(ev, tmp_path):
    s2 = ev.stage_s2(*load(ev, *build_world(ev, tmp_path)), 200)
    m = next(x for x in s2["metrics"] if x["metric"].startswith("5-Datasets off AA, n = 8"))
    assert m["ratio"]["n_seeds"] == 8 and not m["ratio"]["bootstrap"]["exhaustive"]
    assert len(s2["metrics"]) == 8


def test_a_short_shadow_arm_is_caught_before_it_manufactures_a_v0b_failure(ev, monkeypatch):
    short = dict(ev.RUN_TABLE)
    short["ctrl_shadow"] = {k for k in short["ctrl_shadow"] if k[1] == 42}   # the old 5-run arm
    monkeypatch.setattr(ev, "RUN_TABLE", short)
    with pytest.raises(AssertionError, match="V0b"):
        ev.check_run_table()


# ---------------------------------------------------------------------------
# P0
# ---------------------------------------------------------------------------


def test_p0_passes_on_a_clean_preflight(ev, tmp_path):
    new, oa, op = build_world(ev, tmp_path)
    n, o = load(ev, new, oa, op)
    prog = [os.path.join(new, "progress_rnga.log")]
    assert ev.gate_p0a(n)["verdict"] == "pass"
    assert ev.gate_p0b(n, prog, ev.PREFLIGHT_RUNS)["verdict"] == "pass"
    assert ev.gate_p0c(n, o)["verdict"] == "pass"


def test_p0a_fails_when_a_duplicate_differs_and_is_incomplete_when_one_is_missing(ev, tmp_path):
    new, oa, op = build_world(ev, tmp_path)
    edit(new, "int_evalue_dup", ("s_interleave", 42),
         lambda r: r["decisions"][2].__setitem__("L_grow_bits", 9.9))
    g = ev.gate_p0a(ev.load_new(new))
    assert g["verdict"] == "fail" and g["value"]["first_mismatch"]["field"] == "L_grow_bits"

    os.remove(_path(new, "5ds_off_dup", (None, 42)))
    edit(new, "int_evalue_dup", ("s_interleave", 42),
         lambda r: r["decisions"][2].__setitem__("L_grow_bits", 1.5))
    assert ev.gate_p0a(ev.load_new(new))["verdict"] == "incomplete"


@pytest.mark.parametrize("outcome, verdict", [("PASSED", "pass"), ("SKIPPED", "fail"),
                                              ("FAILED", "fail"), (None, "incomplete")])
def test_p0b_i_the_cuda_test_must_run_and_pass(ev, tmp_path, outcome, verdict):
    new, *_ = build_world(ev, tmp_path, pytest_outcome=outcome)
    g = ev.gate_p0b(ev.load_new(new), [os.path.join(new, "progress_rnga.log")],
                    ev.PREFLIGHT_RUNS)
    assert g["verdict"] == verdict


def test_p0b_ii_an_unflagged_run_or_a_mis_threaded_base_fails(ev, tmp_path):
    prog = lambda new: [os.path.join(new, "progress_rnga.log")]
    new, *_ = build_world(ev, tmp_path / "a")
    edit(new, "ctrl_off", ("s_plus", 43), lambda r: r.pop("search_device_rng_fix"))
    g = ev.gate_p0b(ev.load_new(new), prog(new), ev.PREFLIGHT_RUNS)
    assert g["verdict"] == "fail" and "unflagged" in g["failures"][0]["detail"]

    new, *_ = build_world(ev, tmp_path / "b")
    # a base constant across seeds — the mis-threading the note says only the base can reveal
    edit(new, "ctrl_off", ("s_plus", 43), lambda r: [d.__setitem__("cand_seed_base", 42_000 + d[
        "task"]) for d in r["decisions"] if d["task"] >= 1])
    g = ev.gate_p0b(ev.load_new(new), prog(new), ev.PREFLIGHT_RUNS)
    assert g["verdict"] == "fail"
    assert {f["expected"] - f["cand_seed_base"] for f in g["failures"]} == {1000}


def test_p0c_fails_when_a_flagged_run_reproduces_the_archive(ev, tmp_path):
    new, oa, op = build_world(ev, tmp_path, new_l_grow=1.0)     # new == old on every value
    g = ev.gate_p0c(*load(ev, new, oa, op))
    assert g["verdict"] == "fail" and len(g["failures"]) == 9


def test_p0c_is_cross_archive_so_new_only_fields_do_not_count_as_a_difference(ev, tmp_path):
    """The two new keys and fields the archive predates must not make an inert flag look live."""
    new, oa, op = build_world(ev, tmp_path, new_l_grow=1.0)
    for d, key in ev.PREFLIGHT_RUNS:
        edit(new, d, key, lambda r: r.__setitem__("param_curve_total_extra", 1))
        edit(new, d, key, lambda r: r["decisions"][1].__setitem__("L_extra_bits", 3.0))
    assert ev.gate_p0c(*load(ev, new, oa, op))["verdict"] == "fail"


def test_compare_runs_default_presence_rule_is_unchanged(ev):
    import eval_provisional_v2 as v2
    a, b = mkrun(42), mkrun(42)
    del b["param_curve_total"]
    assert v2.compare_runs(a, b, 42, None, "a", "b")[0]["detail"] == "present on one side only"
    assert v2.compare_runs(a, b, 42, None, "a", "b", strict_presence=False) == []


# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------


def test_s1_passes_when_every_sign_holds(ev, tmp_path):
    s1 = ev.stage_s1(*load(ev, *build_world(ev, tmp_path)))
    assert s1["verdict"] == "pass", s1["clause_verdicts"]


def test_s1_v2a_a_flipped_improvement_breaks_and_an_archived_zero_is_tied(ev, tmp_path):
    def flip(key):
        # sweep: evalue now WORSE than off at t3 for this run
        return lambda new, p, a: edit(new, "ctrl_evalue", key,
                                      lambda r: r["test_accs_final"].__setitem__(3, 0.2))

    s1 = ev.stage_s1(*load(ev, *build_world(ev, tmp_path, mutate=flip(("s_minus", 45)))))
    v2a = s1["clauses"]["V2a"]
    assert v2a["verdict"] == "fail" and v2a["failures"][0]["label"] == "s_minus/45"
    assert s1["verdict"] == "fail"

    # seed 44 is an archived exact 0: reported TIED, never tested
    s1 = ev.stage_s1(*load(ev, *build_world(ev, tmp_path / "t", mutate=flip(("s_minus", 44)))))
    row = next(r for r in s1["clauses"]["V2a"]["rows"] if r["label"] == "s_minus/44")
    assert row["status"] == "TIED" and s1["clauses"]["V2a"]["verdict"] == "pass"
    assert s1["clauses"]["V2a"]["n_got"] == 11


def test_s1_v2a_an_archive_that_is_not_the_frozen_11_is_incomplete(ev, tmp_path):
    def untie(new, prov, attn):
        edit(prov, "ctrl_evalue", ("s_minus", 42),
             lambda r: r["test_accs_final"].__setitem__(3, 0.9))
    s1 = ev.stage_s1(*load(ev, *build_world(ev, tmp_path, mutate=untie)))
    v2a = s1["clauses"]["V2a"]
    assert v2a["verdict"] == "incomplete" and v2a["archive_mismatch"]["archive_only"] == [
        "s_minus/42"]


def test_s1_a_clause_short_of_its_pre_registered_pairs_is_incomplete(ev, tmp_path):
    def drop(new, prov, attn):
        os.remove(_path(new, "ctrl_evalue", ("s_out", 46)))
    s1 = ev.stage_s1(*load(ev, *build_world(ev, tmp_path, mutate=drop)))
    assert s1["clauses"]["V2c"]["verdict"] == "incomplete"
    assert s1["clauses"]["V2c"]["n_got"] == 19
    assert s1["verdict"] == "incomplete"


def test_s1_a_sign_that_goes_to_zero_is_broken(ev):
    c = ev._sign_clause("x", {"a": 0.1}, {"a": 0.0}, "")
    assert c["verdict"] == "fail"


@pytest.mark.parametrize("d, key, clause", [
    ("ctrl_shadow", ("s_in", 44), "V0b"),
    ("5ds_evalue", (None, 42), "5ds-evalue-identity"),
])
def test_s1_identity_clauses_fail_on_any_difference(ev, tmp_path, d, key, clause):
    mut = lambda new, p, a: edit(new, d, key, lambda r: r.__setitem__("average_accuracy", 0.1))
    s1 = ev.stage_s1(*load(ev, *build_world(ev, tmp_path, mutate=mut)))
    assert s1["clauses"][clause]["verdict"] == "fail"


def test_s1_v5a_a_pre_deferral_leak_fails(ev, tmp_path):
    mut = lambda new, p, a: edit(new, "ctrl_evalue", ("s_out", 43),
                                 lambda r: r["test_accs"].__setitem__(1, 0.123))
    s1 = ev.stage_s1(*load(ev, *build_world(ev, tmp_path, mutate=mut)))
    assert s1["clauses"]["V5a"]["verdict"] == "fail"


def test_s1_r1b_bar_crossed_fails(ev, tmp_path):
    # archived seed-43 AA sits above 0.9113 on the attention arm; the sweep's falls below it
    def mut(new, p, a):
        edit(a, "5ds_attn_pool", (None, 43), lambda r: r.__setitem__("average_accuracy", 0.93))
        edit(new, "5ds_off", (None, 43), lambda r: r.__setitem__("average_accuracy", 0.90))
    s1 = ev.stage_s1(*load(ev, *build_world(ev, tmp_path, mutate=mut)))
    assert s1["clauses"]["R1b-bars"]["verdict"] == "fail"


# ---------------------------------------------------------------------------
# S2
# ---------------------------------------------------------------------------


def test_se_ratio_is_exhaustive_at_n3_and_paired(ev):
    old = {42: 1.0, 43: 2.0, 44: 3.0}
    new = {42: 1.0, 43: 3.0, 44: 5.0}                  # exactly twice the spread
    r = ev.se_ratio(old, new, 2000)
    assert r["ratio"] == pytest.approx(2.0)
    b = r["bootstrap"]
    assert b["exhaustive"] and b["n_resamples"] == 27
    assert b["n_undefined"] == 3                        # the three all-one-label resamples
    # paired labels: new is an affine map of old, so EVERY defined resample gives exactly 2
    assert b["ci95"] == pytest.approx([2.0, 2.0])
    assert not r["ci_contains_1"]


def test_se_ratio_is_undefined_below_two_seeds(ev):
    assert ev.se_ratio({42: 1.0}, {42: 2.0}, 2000)["defined"] is False


def _metric(primary, ci, ratio=1.0):
    return {"metric": "m", "primary": primary,
            "ratio": {"defined": ci is not None, "ratio": ratio,
                      "bootstrap": {"ci95": ci} if ci is not None else {}}}


WIN, WIDE, UP, DOWN = [0.85, 1.2], [0.5, 2.0], [1.1, 1.9], [0.4, 0.9]


@pytest.mark.parametrize("primary, secondaries, outcome", [
    (UP, [WIDE] * 6, "SE-WIDENS"),
    (DOWN, [WIDE] * 6, "SE-NARROWS"),
    (WIN, [UP] + [WIDE] * 5, "SE-MIXED"),        # stream-dependent: powered flat, one resolved
    (WIN, [DOWN] + [WIDE] * 5, "SE-MIXED"),
    (WIN, [WIDE] * 6, "SE-UNCHANGED"),
    (WIDE, [UP] * 6, "SE-UNRESOLVED"),           # the powered interval is not inside the window
    (None, [UP] * 6, "SE-UNRESOLVED"),
    ([0.8, 1.3], [WIDE] * 6, "SE-UNRESOLVED"),   # straddles 1 but leaves the window
])
def test_s2_decision_tree_is_on_intervals_only(ev, primary, secondaries, outcome):
    ms = [_metric(True, primary)] + [_metric(False, c) for c in secondaries]
    assert ev.classify_s2(ms)["outcome"] == outcome


def test_s2_point_ratios_alone_never_resolve_anything(ev):
    """At n = 3..5 a point ratio above 1 is a coin flip; only an interval resolves a metric."""
    ms = [_metric(True, WIN, ratio=1.2)] + [_metric(False, WIDE, ratio=3.0)] * 6
    c = ev.classify_s2(ms)
    assert c["outcome"] == "SE-UNCHANGED" and c["resolved_secondaries"] == []


def test_s2_transports_a_factor_only_when_the_interval_excludes_1(ev):
    ms = [_metric(True, WIN), _metric(False, UP, ratio=1.5), _metric(False, WIDE, ratio=1.5)]
    ev.classify_s2(ms)
    assert ms[1]["transport"] == {"kind": "factor", "factor": 1.5, "ci95": UP}
    assert ms[2]["transport"]["kind"] == "interval"


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------


def _attn(**v):
    base = {g: {"verdict": "pass"} for g in ("R1a", "R1b", "R1c", "R2a", "R2c", "R2d")}
    base.update({"R1d": {"verdict": "descriptive"}, "R2b": {"verdict": "fail"},
                 "R3": {"verdict": "descriptive", "r3_branch": "X"},
                 "R4": {"verdict": "descriptive"}})
    for k, val in v.items():
        base[k] = {"verdict": val}
    return base


def _prov(v2_top="pass", v2a="pass", n_unresolved=0, **v):
    d = {"V0b": {"verdict": "pass"}, "V3": {"verdict": "pass"}, "V4": {"verdict": "pass"},
         "V2": {"verdict": v2_top, "value": {"a_regret_improvement": {"verdict": v2a},
                                             "c_collateral_aa": {"verdict": "pass"}}},
         "V5": {"verdict": "pass", "value": {"a_non_deferred_positions": {"verdict": "pass"},
                                             "b_deferred_revisit": {"verdict": "pass"},
                                             "c_rolled_back_audit_entries":
                                                 {"verdict": "descriptive"}}},
         "V6": {"verdict": "pass", "value": {"n_unresolved": n_unresolved}}}
    for k, val in v.items():
        d[k] = {"verdict": val}
    return d


CENSUS = {"ctrl": {"n_runs_changed": 0, "n_runs_compared": 40},
          "int": {"n_runs_changed": 0, "n_runs_compared": 20}}


def test_s3_holds_and_reads_v2a_at_its_path_not_the_top_level_verdict(ev):
    # the sweep has no `always` arm, so gates.V2.verdict is "not-run" — that must NOT flip
    s3 = ev.stage_s3(_attn(), _prov(v2_top="not-run"), _attn(), _prov(), CENSUS)
    assert s3["verdict"] == "pass" and not s3["flipped"]


def test_s3_a_flip_is_named_with_its_direction(ev):
    s3 = ev.stage_s3(_attn(R2b="pass"), _prov(n_unresolved=2), _attn(), _prov(), CENSUS)
    assert s3["verdict"] == "fail"
    assert {r["gate"]: r["direction"] for r in s3["flipped"]} == {
        "adoption R2b": "fail -> pass", "H9 V6-roots": "True -> False"}


def test_s3_not_run_is_not_computed_not_a_flip(ev):
    s3 = ev.stage_s3(_attn(R1b="not-run"), _prov(), _attn(), _prov(), CENSUS)
    assert not s3["flipped"] and "adoption R1b" in s3["not_computed"]
    assert s3["verdict"] == "incomplete"      # a re-evaluable gate may not be quietly absent


def test_s3_census_counts_decision_changes(ev, tmp_path):
    def mut(new, p, a):
        edit(new, "ctrl_off", ("s_plus", 44),
             lambda r: r["decisions"][3].__setitem__("decision", "search"))
    n, o = load(ev, *build_world(ev, tmp_path, mutate=mut))
    c = ev.decision_census(n, o)
    assert c["ctrl"]["n_runs_compared"] == 40 and c["int"]["n_runs_compared"] == 20
    assert c["ctrl"]["n_runs_changed"] == 1 and c["ctrl"]["changes_by_task"] == {"3": 1}


# ---------------------------------------------------------------------------
# The branch chain, end to end
# ---------------------------------------------------------------------------


def G(**v):
    out = {k: {"verdict": s} for k, s in v.items()}
    out.setdefault("S1", {"verdict": "pass"})["clause_verdicts"] = {"V2a": v.get("S1", "pass")}
    out.setdefault("S3", {"verdict": "pass"})["flipped"] = [{"gate": "adoption R2b",
                                                             "direction": "fail -> pass"}]
    out["S3"].setdefault("not_computed", [])
    return out


ALL_PASS = dict(Completeness="pass", P0a="pass", P0b="pass", P0c="pass", P0d="pass",
                S1="pass", S2="pass", S3="pass")


@pytest.mark.parametrize("over, s2, branch", [
    (dict(P0c="fail", P0a="fail"), "SE-MIXED", "FLAG-INERT"),       # FLAG-INERT outranks all
    (dict(P0b="fail"), "SE-MIXED", "IDENTITY-BROKEN"),
    (dict(P0d="fail"), "SE-MIXED", "IDENTITY-BROKEN"),
    (dict(P0a="fail", S3="fail"), "SE-MIXED", "IDENTITY-BROKEN"),
    (dict(Completeness="incomplete"), "SE-MIXED", "INCOMPLETE"),
    (dict(P0b="incomplete", S3="fail"), "SE-MIXED", "INCOMPLETE"),  # before VERDICT-FLIPS
    (dict(S1="incomplete"), "SE-MIXED", "INCOMPLETE"),
    (dict(S3="fail", S1="fail"), "SE-MIXED", "VERDICT-FLIPS"),
    (dict(S1="fail"), "SE-MIXED", "SIGNS-BREAK"),
    ({}, "SE-WIDENS", "SIGNS-HOLD-SE-WIDENS"),
    ({}, "SE-NARROWS", "SIGNS-HOLD-SE-NARROWS"),
    ({}, "SE-MIXED", "SIGNS-HOLD-SE-MIXED"),
    ({}, "SE-UNCHANGED", "SIGNS-HOLD-SE-UNCHANGED"),
    ({}, "SE-UNRESOLVED", "SIGNS-HOLD-SE-UNRESOLVED"),
    (dict(S2="not-run"), "SE-MIXED", "UNMATCHED-COMBINATION"),
    (dict(S3="not-run"), "SE-MIXED", "UNMATCHED-COMBINATION"),
])
def test_branch_chain(ev, over, s2, branch):
    g = G(**{**ALL_PASS, **over})
    g["S2"]["outcome"] = s2
    b = ev.determine_branch(g)
    assert b["branch"] == branch, b
    assert b["gate_verdict"] in ("accepted", "rejected", "inconclusive") and b["action"]
    if branch == "UNMATCHED-COMBINATION":
        assert "not passing" in b["reason"]


def test_s3_not_computed_routes_to_incomplete(ev):
    g = G(**ALL_PASS)
    g["S2"]["outcome"] = "SE-MIXED"
    g["S3"]["not_computed"] = ["adoption R1b"]
    assert ev.determine_branch(g)["branch"] == "INCOMPLETE"


def test_every_branch_has_a_gate_record(ev):
    names = set(ev.S2_BRANCH.values()) | {"FLAG-INERT", "IDENTITY-BROKEN", "INCOMPLETE",
                                          "VERDICT-FLIPS", "SIGNS-BREAK", "UNMATCHED-COMBINATION"}
    assert names == set(ev.BRANCH_RECORD)


def _s3_files(tmp_path, new_attn=None, new_prov=None):
    for name, doc in (("gates_h10_attn.json", new_attn or _attn()),
                      ("gates_h10_prov.json", new_prov or _prov(v2_top="not-run"))):
        (tmp_path / name).write_text(json.dumps(doc))
    for name, doc in (("old_attn_gates.json", _attn()), ("old_prov_gates.json", _prov())):
        (tmp_path / name).write_text(json.dumps(doc))
    return ["--s3-new-attn", str(tmp_path / "gates_h10_attn.json"),
            "--s3-new-prov", str(tmp_path / "gates_h10_prov.json"),
            "--s3-old-attn", str(tmp_path / "old_attn_gates.json"),
            "--s3-old-prov", str(tmp_path / "old_prov_gates.json")]


def test_main_full_synthetic_sweep_names_a_signs_hold_branch(ev, tmp_path):
    new, oa, op = build_world(ev, tmp_path)
    out = tmp_path / "se.json"
    code = ev.main(["--new", new, "--old-attn", oa, "--old-prov", op, "--bootstrap", "200",
                    "--out", str(out)] + _s3_files(tmp_path))
    d = json.loads(out.read_text())
    assert code == 0
    assert d["branch"].startswith("SIGNS-HOLD-SE-"), d["branch_detail"]


def test_main_full_sweep_with_a_flipped_gate_is_verdict_flips(ev, tmp_path):
    new, oa, op = build_world(ev, tmp_path)
    out = tmp_path / "se.json"
    ev.main(["--new", new, "--old-attn", oa, "--old-prov", op, "--bootstrap", "200",
             "--out", str(out)] + _s3_files(tmp_path, new_attn=_attn(R2b="pass")))
    assert json.loads(out.read_text())["branch"] == "VERDICT-FLIPS"


@pytest.mark.parametrize("scenario, code", [("ok", 0), ("inert", 1), ("broken", 1),
                                            ("short", 2)])
def test_preflight_exit_codes(ev, tmp_path, scenario, code):
    new, oa, op = build_world(ev, tmp_path, new_l_grow=1.0 if scenario == "inert" else 1.5,
                              pytest_outcome="SKIPPED" if scenario == "broken" else "PASSED")
    if scenario == "short":
        os.remove(_path(new, "ctrl_off_dup", ("s_plus", 42)))
    out = tmp_path / "pre.json"
    assert ev.main(["--p0", "--new", new, "--old-attn", oa, "--old-prov", op,
                    "--out", str(out)]) == code
    d = json.loads(out.read_text())
    if scenario == "inert":
        assert d["preflight"]["branch"] == "FLAG-INERT"
    if scenario == "broken":
        assert d["preflight"]["branch"] == "IDENTITY-BROKEN"


# ---------------------------------------------------------------------------
# observable_r3 is flag-aware
# ---------------------------------------------------------------------------


def test_observable_r3_reads_the_flag_from_the_runs():
    sys.path.insert(0, SCRIPTS)
    import eval_provisional_v2 as v2
    flagged = [(42, "s_minus", mkrun(42)), (43, "s_minus", mkrun(43))]
    plain = [(42, "s_minus", mkrun(42, flag=False))]
    assert v2.observable_r3(flagged)["value"]["fixed"] is True
    assert "FIXED" in v2.observable_r3(flagged)["headline"]
    assert v2.observable_r3(plain)["value"]["fixed"] is False
    assert v2.observable_r3(flagged + plain)["value"]["fixed"] == "mixed"


# ---------------------------------------------------------------------------
# End to end on the real archive: the unfixed runs ARE what an inert flag produces
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (os.path.isdir(OLD_ATTN) and os.path.isdir(OLD_PROV)),
                    reason="concept-dag-results archives not present")
def test_the_unfixed_archive_standing_in_for_the_sweep_is_flag_inert(ev, tmp_path):
    out = tmp_path / "se.json"
    code = ev.main(["--new", OLD_PROV, "--old-attn", OLD_ATTN, "--old-prov", OLD_PROV,
                    "--s3-new-attn", os.path.join(OLD_ATTN, "gates.json"),
                    "--s3-new-prov", os.path.join(OLD_PROV, "gates_h9.json"),
                    "--bootstrap", "200", "--out", str(out)])
    d = json.loads(out.read_text())
    assert code == 1 and d["branch"] == "FLAG-INERT"
    # and the archived numbers the note quotes are what S1/S2 read
    assert d["S1"]["clauses"]["V2a"]["n_archived_nonzero"] == 11
    se = {m["metric"]: m["ratio"]["se_old"] for m in d["S2"]["metrics"]}
    assert se["s_interleave off AA"] == pytest.approx(0.00502, abs=5e-6)
    assert se["5-Datasets off AA"] == pytest.approx(0.00029, abs=5e-6)
    assert se["CTrL off seed-level AA"] == pytest.approx(0.01295 / 5 ** 0.5, abs=5e-6)
    assert se["CTrL paired t3 regret improvement off - evalue"] == pytest.approx(0.02917,
                                                                                 abs=5e-5)
    assert se["CTrL paired AA difference evalue - off"] == pytest.approx(0.00611, abs=5e-5)
