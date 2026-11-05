"""Tests for the timed-window CONTENT contract (window.toml).

The two-window timing contract settles how a reading is taken; this one settles
what work sits inside each side's window. An undeclared boundary — in either
direction — is a fixed tax on one side that no kernel lever removes and no
number reveals; so the file is machine-checked at start, both decompositions
are banked, and the declared extra work is priced at confirmation next to
(never instead of) the raw ratio — and only when the pricing can be trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgekernel.baseline import ROLE_ARCHITECTURE, RefBaseline, load_baseline, save_baseline
from forgekernel.inner.ncu import kernel_table, kernels_from_ncu, parse_ncu_csv
from forgekernel.inner.trajectory import IterationRecord, append_record, load_records
from forgekernel.window_contract import (
    COVERAGE_BAND,
    EXIT_WINDOW_UNDECLARED,
    WINDOW_FILENAME,
    WindowContract,
    credited_ratio,
    load_window_contract,
    parse_window_contract,
    render_window_ledger,
    window_credit,
    window_findings,
    window_warnings,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "ncu_sample.csv"

DECLARED = """
schema_version = 1
[reference]
window = "bwd only"
inputs = ["q", "k", "v", "g_cumsum", "A", "beta", "do"]
outputs = ["dq", "dk", "dv", "dbeta", "dg"]
[student]
window = "bwd from raw inputs"
inputs = ["q", "k", "v", "g", "beta", "do"]
outputs = ["dq", "dk", "dv", "dg", "dbeta"]
[[asymmetry]]
ref_input = "A"
class = "price"
produced_by = "forward"
derived_from = "k, beta"
student_kernel_regex = "Preprocess"
note = "intra-chunk solve"
[[asymmetry]]
ref_input = "g_cumsum"
class = "accept"
produced_by = "forward"
derived_from = "g"
"""

# The other direction: the student receives a derived tensor the reference
# computes in-window, and skips an output the reference must produce.
STUDENT_FAVOURED = """
[reference]
inputs = ["q", "g"]
outputs = ["o", "lse"]
[student]
inputs = ["q", "g_cumsum"]
outputs = ["o"]
[[asymmetry]]
student_input = "g_cumsum"
class = "price"
derived_from = "g"
ref_kernel_regex = "cumsum"
[[asymmetry]]
ref_output = "lse"
class = "price"
ref_kernel_regex = "lse_pass"
"""


def _write(op_dir: Path, text: str = DECLARED) -> Path:
    op_dir.mkdir(parents=True, exist_ok=True)
    p = op_dir / WINDOW_FILENAME
    p.write_text(text, encoding="utf-8")
    return p


# --- parsing + findings ------------------------------------------------------

def test_declared_contract_parses_and_is_consistent(tmp_path):
    _write(tmp_path)
    c = load_window_contract(tmp_path)
    assert c is not None and c.declared
    assert c.missing_on_student == ["g_cumsum", "A"]
    assert c.extra_on_student == ["g"]          # explained by g_cumsum's derived_from
    assert window_findings(c) == []
    assert [a.name for a in c.priced_entries] == ["A"]
    assert c.unpriced_entries == []
    assert c.asymmetry_for("A").payer == "student"


def test_absent_file_is_none_and_a_named_finding(tmp_path):
    assert load_window_contract(tmp_path) is None
    (finding,) = window_findings(None)
    assert WINDOW_FILENAME in finding and "forge run" in finding


def test_an_empty_contract_is_undeclared_not_malformed():
    """Empty lists parse but are UNDECLARED: a fresh card must refuse until the
    boundaries are declared — vacuous consistency would defeat the gate."""
    text = ('schema_version = 1\n[reference]\nwindow = ""\ninputs = []\noutputs = []\n'
            '[student]\nwindow = ""\ninputs = []\noutputs = []\n')
    c = parse_window_contract(text)
    assert not c.declared
    findings = window_findings(c)
    for where in ("[reference].inputs", "[student].inputs",
                  "[reference].outputs", "[student].outputs"):
        assert any(f.startswith(f"{where} is empty") for f in findings), where


def test_malformed_toml_raises(tmp_path):
    _write(tmp_path, "[reference\ninputs = [")
    with pytest.raises(ValueError, match="not valid TOML"):
        load_window_contract(tmp_path)


def test_wrong_schema_raises(tmp_path):
    _write(tmp_path, "schema_version = 7\n[reference]\ninputs=['q']\n[student]\ninputs=['q']\n")
    with pytest.raises(ValueError, match="schema_version"):
        load_window_contract(tmp_path)


def test_entry_keyed_by_none_or_several_kinds_raises():
    with pytest.raises(ValueError, match="exactly one of"):
        parse_window_contract("[[asymmetry]]\nclass='accept'\n")
    with pytest.raises(ValueError, match="exactly one of"):
        parse_window_contract("[[asymmetry]]\nref_input='A'\nstudent_input='B'\nclass='accept'\n")


_BOTH = """
[reference]
inputs = ["q", "A"]
outputs = ["o"]
[student]
inputs = ["q"]
outputs = ["o"]
"""


def test_undeclared_reference_input_is_a_finding():
    (finding,) = window_findings(parse_window_contract(_BOTH))
    assert "reference input 'A'" in finding and "ref_input" in finding


def test_the_other_direction_is_held_to_the_same_rule():
    """A student input the reference does not receive, and an output only one
    side produces, are findings — not information. The reference computes the
    extra input inside its window; the skipped output is work the student never
    pays for. Both bias the ratio toward the student."""
    c = parse_window_contract("""
[reference]
inputs = ["q"]
outputs = ["o", "lse"]
[student]
inputs = ["q", "g_cumsum"]
outputs = ["o", "aux"]
""")
    findings = window_findings(c)
    assert any("student input 'g_cumsum'" in f and "student_input" in f for f in findings)
    assert any("reference output 'lse'" in f and "ref_output" in f for f in findings)
    assert any("student output 'aux'" in f and "student_output" in f for f in findings)
    assert len(findings) == 3
    assert window_findings(parse_window_contract(STUDENT_FAVOURED)) == []


def test_derived_from_explains_the_raw_input_on_the_other_side():
    c = parse_window_contract("""
[reference]
inputs = ["q", "g"]
outputs = ["o"]
[student]
inputs = ["q", "g_cumsum"]
outputs = ["o"]
[[asymmetry]]
student_input = "g_cumsum"
class = "accept"
derived_from = "g"
""")
    assert window_findings(c) == []


@pytest.mark.parametrize("entry, needle", [
    ('ref_input = "A"\nclass = "bogus"', "class='bogus'"),
    ('ref_input = "zzz"\nclass = "accept"', "not in [reference].inputs"),
    ('ref_input = "q"\nclass = "price"', "both sides receive it"),
    ('ref_input = "A"\nclass = "symmetrize"', "only one side receives it"),
    ('ref_input = "A"\nclass = "accept"\nstudent_kernel_regex = "x"', "only price-class"),
    ('ref_input = "A"\nclass = "price"\nstudent_kernel_regex = "("', "not a valid regex"),
    ('ref_input = "A"\nclass = "price"\nref_kernel_regex = "x"', "use student_kernel_regex"),
    ('ref_input = "A"\nclass = "price"\nproduced_by = "elsewhere"', "produced_by"),
    ('ref_output = "o"\nclass = "accept"', "both sides produce it"),
    ('ref_output = "o"\nclass = "symmetrize"', "outputs cannot be symmetrize"),
    ('student_output = "o"\nclass = "accept"', "both sides produce it"),
])
def test_inconsistent_entries_are_named(entry, needle):
    c = parse_window_contract(_BOTH + f"[[asymmetry]]\n{entry}\n")
    findings = window_findings(c)
    assert any(needle in f for f in findings), findings


def test_regex_on_the_side_that_does_not_pay_is_a_finding():
    c = parse_window_contract(STUDENT_FAVOURED.replace('ref_kernel_regex = "cumsum"',
                                                        'student_kernel_regex = "cumsum"'))
    assert any("use ref_kernel_regex" in f for f in window_findings(c))


def test_duplicate_entry_is_a_finding():
    c = parse_window_contract(_BOTH + """
[[asymmetry]]
ref_input = "A"
class = "accept"
[[asymmetry]]
ref_input = "A"
class = "price"
""")
    assert any("declared twice" in f for f in window_findings(c))


def test_symmetrize_requires_both_sides_and_is_not_a_tax():
    c = parse_window_contract("""
[reference]
inputs = ["q", "lse"]
outputs = ["o"]
[student]
inputs = ["q", "lse"]
outputs = ["o"]
[[asymmetry]]
ref_input = "lse"
class = "symmetrize"
produced_by = "forward"
shape = [8, 128]
dtype = "fp32"
""")
    assert window_findings(c) == []
    assert c.missing_on_student == []
    assert window_credit(c, [{"name": "k", "time_ns": 1.0}]) is None


def test_price_without_regex_is_a_warning_not_a_finding():
    c = parse_window_contract(_BOTH + '[[asymmetry]]\nref_input = "A"\nclass = "price"\n')
    assert window_findings(c) == []
    (warn,) = window_warnings(c)
    assert "student_kernel_regex" in warn
    assert [a.name for a in c.unpriced_entries] == ["A"]


# --- per-kernel table --------------------------------------------------------

def test_kernel_table_keeps_every_kernel_by_share():
    rows = kernels_from_ncu(FIXTURE.read_text())
    assert "BwdMainKernel" in rows[0]["name"] and "Preprocess" in rows[1]["name"]
    assert rows[0]["pct"] > rows[1]["pct"]
    assert abs(sum(r["pct"] for r in rows) - 100.0) < 0.01
    assert all(r["launches"] >= 1 for r in rows)
    assert kernel_table(parse_ncu_csv("")) == []


def test_baseline_archives_and_reloads_the_kernel_table(tmp_path):
    rows = kernels_from_ncu(FIXTURE.read_text())
    saved = RefBaseline(op="demo", ms=2.0, ms_device=1.5, kernels=rows)
    save_baseline(tmp_path, saved)
    assert load_baseline(tmp_path) == saved
    text = saved.render()
    assert "every launch inside the reference's timed window" in text and "FlashAttnBwdPreprocess" in text


def test_pre_field_baseline_loads_with_an_empty_table(tmp_path):
    from forgekernel.baseline import SCHEMA_VERSION, baseline_path
    baseline_path(tmp_path).write_text(
        json.dumps({"op": "demo", "ms": 1.0, "schema_version": SCHEMA_VERSION}),
        encoding="utf-8")
    assert load_baseline(tmp_path).kernels == []


def test_iteration_record_carries_kernels_and_older_rows_load(tmp_path):
    status = tmp_path / "status.jsonl"
    append_record(status, IterationRecord(iter=1, decision_id="D0.1", change="x",
                                          correctness=True, verdict="KEEP",
                                          device_median_ms=1.0,
                                          kernels=[{"name": "a", "launches": 1,
                                                    "time_ns": 5.0, "pct": 100.0}]))
    with open(status, "a") as fh:  # a row written before the field existed
        fh.write(json.dumps({"iter": 2, "decision_id": "D0.2", "change": "y",
                             "correctness": True, "verdict": "KEEP"}) + "\n")
    recs = load_records(status)
    assert recs[0].kernels[0]["name"] == "a"
    assert recs[1].kernels == []


# --- pricing -----------------------------------------------------------------

_KERNELS = [
    {"name": "kernel_cutlass__k3_main", "launches": 4, "time_ns": 800.0, "pct": 80.0},
    {"name": "kernel_cutlass__k1_Preprocess_A", "launches": 4, "time_ns": 150.0, "pct": 15.0},
    {"name": "reduce_heads", "launches": 8, "time_ns": 50.0, "pct": 5.0},
]
_REF_KERNELS = [
    {"name": "fused_bwd", "launches": 4, "time_ns": 900.0, "pct": 90.0},
    {"name": "gate_cumsum", "launches": 4, "time_ns": 60.0, "pct": 6.0},
    {"name": "lse_pass", "launches": 4, "time_ns": 40.0, "pct": 4.0},
]
# A digest whose per-call kernel total equals the timed reading (coverage 1.0):
# total_time_ns over dominant_launches == window_ms * 1e6.
_PROF_OK = {"total_time_ns": 3_200_000.0, "dominant_launches": 4}          # 0.8 ms/call
_REF_PROF_OK = {"total_time_ns": 2_720_000.0, "dominant_launches": 4}      # 0.68 ms/call


def test_credit_is_the_matched_share_per_side():
    c = parse_window_contract(DECLARED)
    credit = window_credit(c, _KERNELS, _REF_KERNELS)
    assert credit["usable"] and credit["why"] == []
    assert credit["student"]["fraction"] == pytest.approx(0.15)
    assert credit["student"]["matched"] == ["kernel_cutlass__k1_Preprocess_A"]
    assert credit["reference"]["priced"] == [] and credit["reference"]["fraction"] == 0.0
    cr = credited_ratio(credit, student_ms=0.80, ref_ms=0.68)
    assert cr["raw_ratio"] == pytest.approx(0.85)
    assert cr["credited_student_ms"] == pytest.approx(0.68)
    assert cr["credited_ref_ms"] == pytest.approx(0.68)
    assert cr["credited_ratio"] == pytest.approx(1.0)


def test_reference_side_extras_credit_the_reference_reading():
    """The student is favoured: the reference does the cumsum and the lse pass
    inside its window. The credit comes off the REFERENCE reading and the
    credited ratio moves against the student."""
    c = parse_window_contract(STUDENT_FAVOURED)
    credit = window_credit(c, _KERNELS, _REF_KERNELS)
    assert credit["usable"]
    assert credit["reference"]["fraction"] == pytest.approx(0.10)
    assert sorted(credit["reference"]["matched"]) == ["gate_cumsum", "lse_pass"]
    cr = credited_ratio(credit, student_ms=0.68, ref_ms=0.68)
    assert cr["raw_ratio"] == pytest.approx(1.0)
    assert cr["credited_ratio"] == pytest.approx(0.9)


def test_a_regex_that_matches_nothing_makes_the_credit_unusable_not_zero():
    """Fusing the rebuild into the main kernel (or renaming it) is a legitimate
    design move; it must surface as 'cannot price', never as 'no tax'."""
    c = parse_window_contract(DECLARED.replace('"Preprocess"', '"no_such_kernel"'))
    credit = window_credit(c, _KERNELS, _REF_KERNELS)
    assert not credit["usable"]
    assert credit["student"]["unmatched"] == ["A"]
    assert any("matched nothing for A" in w for w in credit["why"])
    assert credited_ratio(credit, 0.8, 0.68) is None


def test_a_missing_table_on_the_paying_side_makes_the_credit_unusable():
    c = parse_window_contract(STUDENT_FAVOURED)
    credit = window_credit(c, _KERNELS, None)
    assert not credit["usable"] and "no reference per-kernel table" in credit["why"][0]
    assert credited_ratio(credit, 0.8, 0.68) is None


def test_no_priced_entry_prices_nothing():
    assert window_credit(None, _KERNELS) is None
    accept_only = WindowContract(reference_inputs=["q", "A"], student_inputs=["q"],
                                 reference_outputs=["o"], student_outputs=["o"],
                                 asymmetries=[])
    assert window_credit(accept_only, _KERNELS) is None


def test_credit_reconciles_the_table_with_the_timed_window():
    """A table that also holds the bench's setup launches (or does not cover
    the window) dilutes / inflates every share: the same per-call arithmetic
    the mis-aimed-profile alarm uses refuses the credit outside the band."""
    c = parse_window_contract(DECLARED)
    ok = window_credit(c, _KERNELS, _REF_KERNELS,
                       student_profiler=_PROF_OK, student_ms=0.8,
                       ref_profiler=_REF_PROF_OK, ref_ms=0.68)
    assert ok["usable"] and ok["reconciliation"] == {"student": 1.0}
    # 40% more profiled kernel time per call than the window reading: setup
    # launches rode into the table
    diluted = dict(_PROF_OK, total_time_ns=_PROF_OK["total_time_ns"] * 1.4)
    bad = window_credit(c, _KERNELS, _REF_KERNELS,
                        student_profiler=diluted, student_ms=0.8)
    assert not bad["usable"]
    assert bad["reconciliation"]["student"] == pytest.approx(1.4)
    assert any("does not reconcile" in w for w in bad["why"])
    assert credited_ratio(bad, 0.8, 0.68) is None
    # inside the band: still usable
    close = dict(_PROF_OK, total_time_ns=_PROF_OK["total_time_ns"] * (1 + COVERAGE_BAND / 2))
    assert window_credit(c, _KERNELS, _REF_KERNELS,
                         student_profiler=close, student_ms=0.8)["usable"]
    # the reference side is reconciled only when the reference pays
    both = window_credit(parse_window_contract(STUDENT_FAVOURED), _KERNELS, _REF_KERNELS,
                         ref_profiler=dict(_REF_PROF_OK, total_time_ns=1.0), ref_ms=0.68)
    assert not both["usable"] and "reference NCU table" in both["why"][0]


# --- rendering ---------------------------------------------------------------

def test_ledger_states_absence_out_loud():
    text = render_window_ledger(None)
    assert "Not declared" in text and WINDOW_FILENAME in text


def test_ledger_renders_both_tables_and_the_credit():
    c = parse_window_contract(DECLARED)
    text = render_window_ledger(
        c, ref_kernels=_REF_KERNELS, ref_ms=0.68, ref_label="architecture job 1",
        student_kernels=_KERNELS, student_ms=0.80, student_label="best KEEP iter 9")
    assert "A: class=price, student pays" in text and "g_cumsum: class=accept" in text
    assert "gate_cumsum" in text and "k1_Preprocess_A" in text
    assert "≈" in text  # per-call estimates when the window reading is known
    assert "raw ratio 0.8500 → tax-credited ratio 1.0000" in text
    assert "still use the raw ratio" in text


def test_ledger_names_the_other_direction_and_an_unusable_credit():
    c = parse_window_contract(STUDENT_FAVOURED)
    text = render_window_ledger(c, ref_kernels=_REF_KERNELS, ref_ms=0.68,
                                student_kernels=_KERNELS, student_ms=0.68)
    assert "Inputs the student gets for free and the reference must compute inside its window" in text
    assert "Outputs the reference window must produce and the student does not" in text
    assert "Reference-side pricing" in text and "tax-credited ratio 0.9000" in text
    fused = parse_window_contract(DECLARED.replace('"Preprocess"', '"gone"'))
    text = render_window_ledger(fused, ref_kernels=_REF_KERNELS, ref_ms=0.68,
                                student_kernels=_KERNELS, student_ms=0.8)
    assert "Pricing unavailable" in text and "matched nothing" in text
    assert "→ tax-credited ratio" not in text


def test_ledger_shows_findings_instead_of_tables_when_inconsistent():
    c = parse_window_contract(_BOTH)
    text = render_window_ledger(c, student_kernels=_KERNELS)
    assert "Declaration incomplete" in text and "'A'" in text and "k1_Preprocess_A" not in text


# --- loop start preflight ----------------------------------------------------

def _cfg_with_ref():
    return {"loop": {"ref_cmd": "python3 measure.py", "ref_name": "lib"}}


def test_preflight_refuses_a_fresh_start_without_a_declaration(tmp_path):
    from forgekernel.loop.cli import _window_preflight
    state, contract, problems = _window_preflight(tmp_path, _cfg_with_ref(), resuming=False)
    assert state == "refuse" and contract is None and problems


def test_preflight_only_warns_on_resume(tmp_path):
    from forgekernel.loop.cli import _window_preflight
    state, _, problems = _window_preflight(tmp_path, _cfg_with_ref(), resuming=True)
    assert state == "warn" and problems


def test_preflight_is_not_applicable_without_a_reference(tmp_path):
    from forgekernel.loop.cli import _window_preflight
    state, _, problems = _window_preflight(tmp_path, {}, resuming=False)
    assert state == "n/a" and problems == []


def test_preflight_passes_a_declared_card_and_surfaces_notes(tmp_path):
    from forgekernel.loop.cli import _window_preflight
    _write(tmp_path, DECLARED.replace('student_kernel_regex = "Preprocess"\n', ""))
    state, contract, problems = _window_preflight(tmp_path, _cfg_with_ref(), resuming=False)
    assert state == "ok" and contract.declared
    assert len(problems) == 1 and "student_kernel_regex" in problems[0]


def test_preflight_refuses_a_malformed_file_even_on_resume(tmp_path):
    from forgekernel.loop.cli import _window_preflight
    _write(tmp_path, "[reference\n")
    state, contract, problems = _window_preflight(tmp_path, _cfg_with_ref(), resuming=True)
    assert state == "refuse" and contract is None and "not valid TOML" in problems[0]


def test_preflight_holds_declared_arch_refs_to_the_contract(tmp_path):
    from forgekernel.loop.cli import _window_preflight
    state, *_ = _window_preflight(tmp_path, {"loop": {"arch_ref_mode": "flashref"}},
                                  resuming=False)
    assert state == "refuse"
    state, *_ = _window_preflight(tmp_path, {"loop": {"arch_ref_device_ms": 0.5}},
                                  resuming=False)
    assert state == "refuse"


def test_loop_start_exit_code_is_reserved():
    """13 — the codes are hand-assigned and must not collide (see cli._main)."""
    assert EXIT_WINDOW_UNDECLARED == 13


# --- driver: binding, student note, confirmation credit ------------------------

def _driver_ctx(tmp_path, contract):
    from test_loop_driver import FakeBackend, _ctx
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.gate_window = "device"
    ctx.window_contract = contract
    ctx.arch_baseline = RefBaseline(
        op="demo", ms=1.0, ms_device=0.68, role=ROLE_ARCHITECTURE, ref_name="lib",
        ncu=dict(_REF_PROF_OK, tensor_pct=30.0), kernels=_REF_KERNELS)
    return ctx


def _keep(iter_no=2, sha="abc123", profiler=None):
    return IterationRecord(iter=iter_no, decision_id=f"D0.{iter_no}", change="x",
                           correctness=True, verdict="KEEP", device_median_ms=0.8, sha=sha,
                           kernels=_KERNELS, profiler=profiler if profiler is not None else _PROF_OK)


def test_binding_and_student_note_read_the_contract(tmp_path):
    from forgekernel.loop.driver import _bindings, _student_prompt
    ctx = _driver_ctx(tmp_path, parse_window_contract(DECLARED))
    append_record(ctx.status_path, _keep())
    block = _bindings(ctx)["{WINDOW_CONTRACT}"]
    assert "fused_bwd" in block and "best KEEP iter 2" in block
    assert "tax-credited ratio" in block and "reconciled" in block
    prompt = _student_prompt(ctx, 3)
    assert "Timed-window contract" in prompt and "`Preprocess`" in prompt and "g_cumsum" in prompt
    # measurement convenience must not steer the design
    assert "fused into the main kernel is decided by performance" in prompt
    assert "don't fold it into the main kernel" not in prompt


def test_student_note_only_lists_what_the_student_pays(tmp_path):
    from forgekernel.loop.driver import _student_prompt
    ctx = _driver_ctx(tmp_path, parse_window_contract(STUDENT_FAVOURED))
    assert "Timed-window contract" not in _student_prompt(ctx, 1)   # the reference pays both


def test_binding_says_undeclared_and_student_note_stays_silent(tmp_path):
    from forgekernel.loop.driver import _bindings, _student_prompt
    ctx = _driver_ctx(tmp_path, None)
    assert "Not declared" in _bindings(ctx)["{WINDOW_CONTRACT}"]
    assert "Timed-window contract" not in _student_prompt(ctx, 1)


def test_confirmation_credit_lands_next_to_the_raw_ratio(tmp_path):
    from forgekernel.loop.driver import apply_window_credit
    ctx = _driver_ctx(tmp_path, parse_window_contract(DECLARED))
    append_record(ctx.status_path, _keep())
    confirm = {"sha": "abc123", "median_ms": 0.80, "median_ratio": 0.85,
               "ref_median_ms": 0.68, "jobs": []}
    credit = apply_window_credit(ctx, confirm)
    assert credit["usable"] and credit["credited_ratio"] == pytest.approx(1.0)
    assert confirm["median_ratio"] == 0.85  # the verdict currency is untouched
    assert confirm["window_credit"]["source_iter"] == 2
    assert confirm["window_credit"]["reconciliation"] == {"student": 1.0}

    # same-job path: the ref median comes from the samples
    confirm2 = {"sha": "abc123", "median_ms": 0.80, "median_ratio": 0.85,
                "jobs": [{"ref_ms": 0.70}, {"ref_ms": 0.66}, {"ref_ms": None}]}
    assert apply_window_credit(ctx, confirm2)["ref_ms"] == pytest.approx(0.68)


def test_confirmation_credit_is_marked_unusable_never_zero(tmp_path):
    from forgekernel.loop.driver import apply_window_credit
    # regex matches nothing (the rebuild was fused into the main kernel)
    ctx = _driver_ctx(tmp_path, parse_window_contract(DECLARED.replace('"Preprocess"', '"gone"')))
    append_record(ctx.status_path, _keep())
    confirm = {"sha": "abc123", "median_ms": 0.80, "ref_median_ms": 0.68, "jobs": []}
    credit = apply_window_credit(ctx, confirm)
    assert credit is not None and not credit["usable"]
    assert "credited_ratio" not in credit and "matched nothing" in credit["why"][0]
    assert confirm["window_credit"]["usable"] is False
    # table does not reconcile with the timed window (setup launches rode in)
    ctx.window_contract = parse_window_contract(DECLARED)
    ctx.status_path.write_text("")
    append_record(ctx.status_path,
                  _keep(profiler=dict(_PROF_OK, total_time_ns=_PROF_OK["total_time_ns"] * 1.5)))
    credit = apply_window_credit(ctx, dict(confirm))
    assert not credit["usable"] and "does not reconcile" in credit["why"][0]


def test_confirmation_credit_is_absent_when_unpriced_or_untabled(tmp_path):
    from forgekernel.loop.driver import apply_window_credit
    unpriced = parse_window_contract(DECLARED.replace('student_kernel_regex = "Preprocess"\n', ""))
    ctx = _driver_ctx(tmp_path, unpriced)
    confirm = {"sha": "abc123", "median_ms": 0.80, "jobs": []}
    assert apply_window_credit(ctx, confirm) is None and "window_credit" not in confirm
    ctx.window_contract = parse_window_contract(DECLARED)
    credit = apply_window_credit(ctx, confirm)  # no per-kernel table on record
    assert credit is not None and not credit["usable"]
    assert "no student per-kernel table" in credit["why"][0]


def test_iteration_banks_the_kernel_table_off_the_ncu_job(tmp_path):
    """The student's decomposition rides the NCU job into IterationRecord.kernels
    and stays OUT of the flat profiler digest."""
    from test_loop_driver import EVAL, REC, FakeBackend, _ctx
    from forgekernel.loop.driver import run_inner_iteration
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), FIXTURE.read_text()])
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 0, metric_before=3.0)
    (rec,) = load_records(ctx.status_path)
    names = sorted(r["name"] for r in rec.kernels)
    assert len(names) == 2 and "BwdMainKernel" in names[0] and "Preprocess" in names[1]
    assert "_kernels" not in rec.profiler and "kernel_count" in rec.profiler


# --- the prep step: symmetrize, mechanically ----------------------------------
#
# A symmetrized input reaches the student's timed call from the bench's prep
# step outside every window. That step is the one place work legitimately leaves
# the window, so it is fenced: declared shape/dtype, a bench that runs it, and a
# per-iteration check of what it produced.

SYMMETRIZED = """
[reference]
window = "bwd only (autograd path)"
inputs = ["q", "k", "v", "g_cumsum", "A", "beta", "do"]
outputs = ["dq", "dk", "dv", "dbeta", "dg"]
[student]
window = "bwd only; A and g_cumsum from the prep step"
inputs = ["q", "k", "v", "g_cumsum", "A", "beta", "do"]
outputs = ["dq", "dk", "dv", "dg", "dbeta"]
[[asymmetry]]
ref_input = "A"
class = "symmetrize"
produced_by = "forward"
derived_from = "k, beta"
shape = [4, 32, 64]
dtype = "bf16"
[[asymmetry]]
ref_input = "g_cumsum"
class = "symmetrize"
produced_by = "forward"
derived_from = "g"
shape = [4, 32]
dtype = "fp32"
"""

PREP_OK = {"outputs": {"A": {"shape": [4, 32, 64], "dtype": "bfloat16"},
                       "g_cumsum": {"shape": [4, 32], "dtype": "torch.float32"}},
           "median_ms": 0.09, "device_median_ms": 0.08}


def test_symmetrized_contract_declares_prep_entries_and_is_consistent():
    from forgekernel.window_contract import check_prep_report
    c = parse_window_contract(SYMMETRIZED)
    assert window_findings(c) == []
    assert [a.name for a in c.prep_entries] == ["A", "g_cumsum"]
    assert c.prep_entries[0].shape == [4, 32, 64] and c.prep_entries[0].dtype == "bf16"
    assert check_prep_report(c, PREP_OK) == []      # aliases: bfloat16 / torch.float32


def test_symmetrize_without_shape_or_dtype_is_a_finding():
    text = SYMMETRIZED.replace('shape = [4, 32, 64]\n', "").replace('dtype = "fp32"\n', "")
    findings = window_findings(parse_window_contract(text))
    assert any("'A'" in f and "no shape" in f for f in findings), findings
    assert any("'g_cumsum'" in f and "no dtype" in f for f in findings), findings


def test_shape_on_a_priced_entry_is_a_finding_and_a_bad_shape_is_malformed():
    c = parse_window_contract(DECLARED.replace('note = "intra-chunk solve"',
                                               'note = "x"\nshape = [1, 2]'))
    assert any("not a prep-produced symmetrize" in f for f in window_findings(c))
    with pytest.raises(ValueError, match="shape must be a list of ints"):
        parse_window_contract(SYMMETRIZED.replace("[4, 32, 64]", '["4", 32]'))


def test_a_cache_symmetrize_is_not_held_to_a_prep_declaration():
    c = parse_window_contract("""
[reference]
inputs = ["x", "weights"]
outputs = ["y"]
[student]
inputs = ["x", "weights"]
outputs = ["y"]
[[asymmetry]]
ref_input = "weights"
class = "symmetrize"
produced_by = "cache"
""")
    assert window_findings(c) == [] and c.prep_entries == []


@pytest.mark.parametrize("report, needle", [
    (None, "no prep block"),
    ({"median_ms": 0.1}, "no prep block"),
    ({"outputs": {"A": PREP_OK["outputs"]["A"]}}, "did not produce 'g_cumsum'"),
    ({"outputs": dict(PREP_OK["outputs"], A={"shape": [4, 32, 32], "dtype": "bf16"})},
     "shape [4, 32, 32], declared [4, 32, 64]"),
    ({"outputs": dict(PREP_OK["outputs"], A={"shape": [4, 32, 64], "dtype": "float16"})},
     "dtype 'float16', declared 'bf16'"),
    ({"outputs": dict(PREP_OK["outputs"], h_states={"shape": [4, 128, 128], "dtype": "fp32"})},
     "undeclared tensor 'h_states'"),
    (dict(PREP_OK, mutated=["A"]), "mutated the prep tensor(s) A"),
])
def test_prep_report_is_held_to_the_declaration(report, needle):
    from forgekernel.window_contract import check_prep_report
    problems = check_prep_report(parse_window_contract(SYMMETRIZED), report)
    assert any(needle in p for p in problems), problems


def test_prep_check_is_vacuous_without_symmetrized_inputs():
    from forgekernel.window_contract import check_prep_report
    assert check_prep_report(parse_window_contract(DECLARED), None) == []
    assert check_prep_report(None, PREP_OK) == []


BENCH_OK = """
import torch
from oracle import make_inputs
from kernel import gdn_prefill_bwd

def prep_saved():
    k, beta, g = make_inputs()
    return {"A": torch.linalg.solve_triangular(k, beta), "g_cumsum": torch.cumsum(g, 1)}

win = measure_windows(lambda saved: gdn_prefill_bwd(saved["A"]), prep=prep_saved)
"""

NCU_OK = """
import torch
from bench import prep_saved
from kernel import gdn_prefill_bwd

saved = prep_saved()
torch.cuda.profiler.start()
gdn_prefill_bwd(saved["A"])
torch.cuda.profiler.stop()
"""


@pytest.mark.parametrize("src, needle", [
    ("win = measure_windows(_run)\n", "never passes prep="),
    ("win = measure_windows(_run, prep=None)\n", "never passes prep="),
    ("win = other(_run, prep=prep_saved)\n", "never passes prep="),
    ("win = measure_windows(_run, prep=lambda: {'A': a})\n", "plain name"),
    ("win = measure_windows(_run, prep=timing.prep)\n", "plain name"),
    ("from kernel import prep_a\nwin = measure_windows(_run, prep=prep_a)\n",
     "not a module-level def"),
    ("from kernel import solve_a\ndef prep_saved():\n    return {'A': solve_a()}\n"
     "win = measure_windows(_run, prep=prep_saved)\n", "references the kernel module (solve_a)"),
    ("import kernel as kk\ndef prep_saved():\n    return {'A': kk.solve()}\n"
     "win = measure_windows(_run, prep=prep_saved)\n", "references the kernel module (kk)"),
    ("def prep_saved():\n    from kernel import solve_a\n    return {'A': solve_a()}\n"
     "win = measure_windows(_run, prep=prep_saved)\n", "references the kernel module (kernel, solve_a)"),
    ("def f(:\n", "does not parse"),
])
def test_bench_prep_rules_are_static_findings(src, needle):
    from forgekernel.window_contract import bench_prep_analysis, bench_uses_prep
    (finding,) = bench_prep_analysis(src)["findings"]
    assert needle in finding, finding
    assert not bench_uses_prep(src)


def test_a_conforming_bench_prep_names_the_function():
    from forgekernel.window_contract import bench_prep_analysis, bench_uses_prep
    assert bench_prep_analysis(BENCH_OK) == {"prep": "prep_saved", "findings": []}
    assert bench_uses_prep(BENCH_OK)


@pytest.mark.parametrize("src, needle", [
    ("import torch\ntorch.cuda.profiler.start()\n", "does not import and call"),
    ("from bench import prep_saved\nimport torch\ntorch.cuda.profiler.start()\n",
     "does not import and call"),
    ("from bench import prep_saved\nsaved = prep_saved()\n", "never calls torch.cuda.profiler.start"),
    ("import torch\nfrom bench import prep_saved\ntorch.cuda.profiler.start()\n"
     "saved = prep_saved()\n", "after torch.cuda.profiler.start()"),
    ("import torch\nfrom bench import prep_saved\nsaved = prep_saved()\n"
     "torch.cuda.profiler.start()\nsaved = prep_saved()\n", "after torch.cuda.profiler.start()"),
])
def test_ncu_run_must_run_the_same_prep_before_the_profiler_window(src, needle):
    from forgekernel.window_contract import ncu_run_prep_findings
    (finding,) = ncu_run_prep_findings(src, "prep_saved")
    assert needle in finding, finding


def test_ncu_run_conforming_forms_pass():
    from forgekernel.window_contract import ncu_run_prep_findings
    assert ncu_run_prep_findings(NCU_OK, "prep_saved") == []
    aliased = NCU_OK.replace("from bench import prep_saved", "from bench import prep_saved as p"
                             ).replace("saved = prep_saved()", "saved = p()")
    assert ncu_run_prep_findings(aliased, "prep_saved") == []
    module_form = NCU_OK.replace("from bench import prep_saved", "import bench"
                                 ).replace("saved = prep_saved()", "saved = bench.prep_saved()")
    assert ncu_run_prep_findings(module_form, "prep_saved") == []


def test_bench_prep_findings_reads_bench_then_ncu_run(tmp_path):
    from forgekernel.window_contract import bench_prep_findings
    c = parse_window_contract(SYMMETRIZED)
    (tmp_path / "eval").mkdir()
    bench = tmp_path / "eval" / "bench.py"
    bench.write_text("win = measure_windows(_run)\n", encoding="utf-8")
    (finding,) = bench_prep_findings(tmp_path, c)
    assert "never passes prep=" in finding and "A, g_cumsum" in finding
    bench.write_text(BENCH_OK, encoding="utf-8")
    assert bench_prep_findings(tmp_path, c) == []          # no ncu_run.py: nothing to hold
    ncu = tmp_path / "eval" / "ncu_run.py"
    ncu.write_text(NCU_OK.replace("saved = prep_saved()\ntorch.cuda.profiler.start()",
                                  "torch.cuda.profiler.start()\nsaved = prep_saved()"),
                   encoding="utf-8")
    (finding,) = bench_prep_findings(tmp_path, c)
    assert "after torch.cuda.profiler.start()" in finding
    ncu.write_text(NCU_OK, encoding="utf-8")
    assert bench_prep_findings(tmp_path, c) == []
    assert bench_prep_findings(tmp_path, parse_window_contract(DECLARED)) == []


def test_preflight_refuses_a_symmetrized_card_whose_bench_runs_no_prep(tmp_path):
    from forgekernel.loop.cli import _window_preflight
    _write(tmp_path, SYMMETRIZED)
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "bench.py").write_text("win = measure_windows(_run)\n", encoding="utf-8")
    state, contract, problems = _window_preflight(tmp_path, _cfg_with_ref(), resuming=False)
    assert state == "refuse" and "never passes prep=" in problems[0]
    (tmp_path / "eval" / "bench.py").write_text(BENCH_OK, encoding="utf-8")
    (tmp_path / "eval" / "ncu_run.py").write_text(NCU_OK, encoding="utf-8")
    state, contract, problems = _window_preflight(tmp_path, _cfg_with_ref(), resuming=False)
    assert state == "ok" and problems == []
    assert len(contract.prep_entries) == 2


def test_parse_eval_reads_the_prep_block():
    from forgekernel.loop.parse import parse_eval
    perf = json.dumps({"bwd_median_ms": 1.0, "device_median_ms": 0.9, "prep": PREP_OK})
    ev = parse_eval('=== JSON ===\n{"correctness": {"pass": true}, "performance": %s}' % perf)
    assert ev.prep == PREP_OK
    top = ('=== JSON ===\n{"correctness": {"pass": true}, "performance": {"bwd_median_ms": 1.0},'
           ' "prep": %s}' % json.dumps(PREP_OK))
    assert parse_eval(top).prep == PREP_OK
    from test_loop_driver import EVAL
    assert parse_eval(EVAL(2.0)).prep is None


def _eval_with_prep(ms, prep):
    return ('=== JSON ===\n{"correctness": {"pass": true}, "performance": {"bwd_median_ms": %r, '
            '"device_median_ms": %r, "bwd_mfu_pct": 32.0, "prep": %s}}'
            % (ms, ms, json.dumps(prep)))


def test_iteration_holds_the_prep_report_to_the_contract(tmp_path):
    """A prep step that widens what the timed call receives is a REVERT with the
    mismatch as its error; a conforming one banks the prep block on the record."""
    from test_loop_driver import NCU, REC, FakeBackend, _ctx
    from forgekernel.loop.driver import run_inner_iteration
    smuggled = {"outputs": dict(PREP_OK["outputs"],
                                h_states={"shape": [4, 128, 128], "dtype": "fp32"}),
                "median_ms": 0.5, "device_median_ms": 0.45}
    be = FakeBackend([REC("D2.3", 2.6), REC("D2.4", 2.5)],
                     [_eval_with_prep(2.6, smuggled), NCU, _eval_with_prep(2.5, PREP_OK), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.window_contract = parse_window_contract(SYMMETRIZED)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "REVERT" and "prep contract" in rec.error
    assert "undeclared tensor 'h_states'" in rec.error
    rec2 = run_inner_iteration(ctx, 1, metric_before=3.0)
    assert rec2.verdict == "KEEP" and rec2.prep == PREP_OK
    recs = load_records(ctx.status_path)
    assert recs[1].prep["device_median_ms"] == pytest.approx(0.08)


def test_iteration_without_a_prep_block_fails_when_the_contract_symmetrizes(tmp_path):
    from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx
    from forgekernel.loop.driver import run_inner_iteration
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.window_contract = parse_window_contract(SYMMETRIZED)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "REVERT" and "no prep block" in rec.error
    # an unsymmetrized contract holds nothing: the same eval is a plain KEEP
    be2 = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    (tmp_path / "b").mkdir()
    ctx2 = _ctx(tmp_path / "b", be2)
    ctx2.window_contract = parse_window_contract(DECLARED)
    assert run_inner_iteration(ctx2, 0, metric_before=3.0).verdict == "KEEP"


def test_inner_loop_stops_on_a_structural_prep_defect(tmp_path):
    """No prep block at all is the bench's defect, not the kernel's: stop with
    its own reason on the first sighting (gate_unmeasured's family). A mismatch
    that repeats verbatim stops too; one that a student edit clears does not."""
    from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx
    from forgekernel.loop.driver import run_inner_loop
    be = FakeBackend([REC("D1", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.window_contract = parse_window_contract(SYMMETRIZED)
    assert run_inner_loop(ctx, start_iter=0) == "prep_unreported"
    (rec,) = load_records(ctx.status_path)
    assert rec.verdict == "REVERT" and "no prep block" in rec.error

    wrong = {"outputs": dict(PREP_OK["outputs"], A={"shape": [4, 32, 32], "dtype": "bf16"}),
             "median_ms": 0.1, "device_median_ms": 0.1}
    (tmp_path / "rep").mkdir()
    be = FakeBackend([REC("D1", 2.6), REC("D2", 2.5)],
                     [_eval_with_prep(2.6, wrong), NCU, _eval_with_prep(2.5, wrong), NCU])
    ctx = _ctx(tmp_path / "rep", be)
    ctx.window_contract = parse_window_contract(SYMMETRIZED)
    assert run_inner_loop(ctx, start_iter=0) == "prep_contract"
    assert [r.verdict for r in load_records(ctx.status_path)] == ["REVERT", "REVERT"]

    (tmp_path / "heal").mkdir()
    be = FakeBackend([REC("D1", 2.6), REC("D2", 2.5)],
                     [_eval_with_prep(2.6, wrong), NCU, _eval_with_prep(2.5, PREP_OK), NCU])
    ctx = _ctx(tmp_path / "heal", be, target_ms=2.55)
    ctx.window_contract = parse_window_contract(SYMMETRIZED)
    assert run_inner_loop(ctx, start_iter=0) == "target"
    assert [r.verdict for r in load_records(ctx.status_path)] == ["REVERT", "KEEP"]


def test_student_note_and_ledger_show_the_prep_step(tmp_path):
    from forgekernel.loop.driver import _bindings, _student_prompt
    from forgekernel.window_contract import render_prep_lines
    ctx = _driver_ctx(tmp_path, parse_window_contract(SYMMETRIZED))
    prompt = _student_prompt(ctx, 1)
    assert "produced by the bench's prep step" in prompt and "(4, 32, 64) bf16" in prompt and "REVERT" in prompt
    block = _bindings(ctx)["{WINDOW_CONTRACT}"]
    assert "prep" in block and "no prep reading banked yet" in block
    rec = _keep()
    rec.prep = PREP_OK
    append_record(ctx.status_path, rec)
    block = _bindings(ctx)["{WINDOW_CONTRACT}"]
    assert "device 0.0800 ms/call" in block and "produced (4, 32, 64) bfloat16" in block
    assert render_prep_lines(parse_window_contract(DECLARED), PREP_OK) == []


def test_older_records_load_with_an_empty_prep_block(tmp_path):
    status = tmp_path / "status.jsonl"
    with open(status, "a") as fh:
        fh.write(json.dumps({"iter": 2, "decision_id": "D0.2", "change": "y",
                             "correctness": True, "verdict": "KEEP"}) + "\n")
    assert load_records(status)[0].prep == {}
