"""Tests for the inner-loop trajectory (status.jsonl) schema + digest builder."""

import pytest

from forgekernel.inner import (
    IterationRecord,
    append_record,
    build_digest,
    gate_ms,
    load_records,
)


def _rec(**over):
    base = dict(
        iter=0,
        decision_id="D2.3",
        change="causal-skip m_block_min=n_block",
        correctness=True,
        verdict="KEEP",
        bwd_median_ms=3.363,
        mfu=25.8,
    )
    base.update(over)
    return IterationRecord(**base)


def test_record_roundtrip_jsonl(tmp_path):
    p = tmp_path / "status.jsonl"
    append_record(p, _rec(iter=0))
    append_record(p, _rec(iter=1, decision_id="D2.1", bwd_median_ms=2.677, mfu=32.4))
    append_record(p, _rec(iter=2, bwd_median_ms=0.2835, vs_ref=0.9679,
                          device_median_ms=0.2731, ref_device_ms=0.2491,
                          vs_ref_device=0.9121))
    recs = load_records(p)
    assert [r.iter for r in recs] == [0, 1, 2]
    assert recs[1].decision_id == "D2.1"
    # the device-window trio survives the roundtrip, and a single-window record
    # reads as None rather than inheriting anything
    assert recs[2].device_median_ms == 0.2731 and recs[2].vs_ref_device == 0.9121
    assert recs[1].device_median_ms is None


def test_record_validate_requires_core_fields():
    with pytest.raises((TypeError, ValueError)):
        IterationRecord(iter=0)  # missing required fields


def test_build_digest_shape_and_achieved():
    target = {"metric": "bwd_median_ms", "value": 1.90, "mfu": 45.7}
    recs = [
        _rec(iter=0, bwd_median_ms=3.363, verdict="KEEP"),
        _rec(iter=1, decision_id="D2.1", bwd_median_ms=2.677, verdict="KEEP"),
        _rec(iter=2, decision_id="D2.1s2", bwd_median_ms=None, verdict="REVERT",
             correctness=False, error="correctness FAIL: dq max_diff=1.13",
             profiler={"bankconf_st": 72_000_000}),
        _rec(iter=3, decision_id="D3.1", bwd_median_ms=2.419, verdict="KEEP",
             profiler={"spill_st_sectors": 32_000_000}),
    ]
    d = build_digest(recs, target, gate_window="e2e")  # an e2e-gated run's digest
    assert d["target"]["value"] == 1.90
    # achieved = best (min) among KEEP records
    assert d["achieved"]["bwd_median_ms"] == 2.419
    assert d["achieved"]["iter"] == 3
    assert d["achieved"]["gap_pct"] == pytest.approx((2.419 - 1.90) / 1.90 * 100, rel=1e-3)
    # curve has one point per record
    assert len(d["curve"]) == 4
    # regressions carry the OBJECTIVE error signature, not subjective prose
    assert len(d["regressions"]) == 1
    assert d["regressions"][0]["id"] == "D2.1s2"
    assert "max_diff" in d["regressions"][0]["error"]
    # profiler recorded for BOTH kept and reverted iters, tagged with verdict
    assert {p["iter"]: p["verdict"] for p in d["profiler"]} == {2: "REVERT", 3: "KEEP"}
    # no subjective field leaks into decisions
    assert all("binding_constraint" not in dec for dec in d["decisions"])


def test_record_rejects_subjective_fields():
    with pytest.raises(TypeError):
        IterationRecord(iter=0, decision_id="D", change="x", correctness=True,
                        verdict="KEEP", binding_constraint="my theory")


def test_load_records_drops_unknown_legacy_keys(tmp_path):
    import json
    p = tmp_path / "status.jsonl"
    p.write_text(json.dumps({
        "iter": 0, "decision_id": "D2.3", "change": "causal-skip", "correctness": True,
        "verdict": "KEEP", "bwd_median_ms": 3.3, "mfu": 25.8,
        "hypothesis": "legacy", "binding_constraint": "legacy", "regression": "legacy",
    }) + "\n")
    recs = load_records(p)
    assert recs[0].decision_id == "D2.3" and recs[0].bwd_median_ms == 3.3


def test_build_digest_empty():
    d = build_digest([], {"metric": "bwd_median_ms", "value": 1.90})
    assert d["achieved"] is None
    assert d["curve"] == []


def test_gate_ms_selects_the_window():
    r = _rec(bwd_median_ms=0.2835, device_median_ms=0.2731)
    assert gate_ms(r) == 0.2731  # device is the default window
    assert gate_ms(r, "e2e") == 0.2835
    assert gate_ms(r, "device") == 0.2731
    # a single-window record reads None in the other window, never a substitute
    assert gate_ms(_rec(device_median_ms=None), "device") is None
    with pytest.raises(ValueError):
        gate_ms(r, "wallclock")


def test_build_digest_device_gate_ranks_by_device_window():
    # e2e and device disagree about which iteration is best: iter 0 wins e2e
    # (thin host path), iter 1 wins device (better kernel). The digest must
    # agree with whichever window the driver gates on.
    target = {"metric": "device_median_ms", "value": 0.25}
    recs = [
        _rec(iter=0, bwd_median_ms=0.280, device_median_ms=0.270),
        _rec(iter=1, decision_id="D2.1", bwd_median_ms=0.290, device_median_ms=0.260),
    ]
    d = build_digest(recs, target, gate_window="device")
    assert d["achieved"]["iter"] == 1
    assert d["achieved"]["gate_window"] == "device"
    assert d["achieved"]["device_median_ms"] == 0.260
    assert d["achieved"]["gap_pct"] == pytest.approx((0.260 - 0.25) / 0.25 * 100, rel=1e-3)
    # decisions' metric_after reads the gate window (metric_before is written in it)
    assert [dec["metric_after"] for dec in d["decisions"]] == [0.270, 0.260]
    # curve rows carry the device reading alongside e2e
    assert d["curve"][1]["device_median_ms"] == 0.260
    # the default window is device, so an unspecified digest ranks the same way
    assert build_digest(recs, {"value": 0.27})["achieved"]["iter"] == 1
    # an explicit e2e gate still ranks by the e2e window
    assert build_digest(recs, {"value": 0.27}, gate_window="e2e")["achieved"]["iter"] == 0


def test_record_carries_commit_sha(tmp_path):
    # the commit sha of the landed iteration is recorded so the meta loop can
    # reset the workspace to a specific decision's checkpoint (prefix reuse).
    p = tmp_path / "status.jsonl"
    append_record(p, _rec(iter=0, sha="abc123"))
    recs = load_records(p)
    assert recs[0].sha == "abc123"


def test_record_sha_defaults_empty():
    # correctness-broken / probe-blocked iters may have no reusable sha
    assert _rec().sha == ""


_DIFF_A = ("correctness FAIL: dq: max_diff=4.504025 tol=0.07852; "
           "dk: max_diff=7.560892 tol=0.080503; dv: max_diff=7.823140 tol=0.156463")
_DIFF_B = ("correctness FAIL: dq: max_diff=0.913311 tol=0.07852; "
           "dk: max_diff=7.560892 tol=0.080503; dv: max_diff=7.823140 tol=0.156463")


def _fail(i, error=_DIFF_A):
    return _rec(iter=i, correctness=False, verdict="REVERT",
                bwd_median_ms=None, mfu=None, error=error)


def test_constant_output_flag_fires_on_bit_identical_diffs():
    """Bit-identical per-tensor max_diff across distinct failing attempts means the
    scored outputs never changed — a broken handoff, not a numeric bug. Rows
    without a diff (probe-blocked, build errors) neither extend nor break the
    streak: the live pattern (bwd-mha, four alternating iterations) interleaved both."""
    from forgekernel.inner.trajectory import constant_output_flag

    records = [
        _fail(9),
        _rec(iter=10, correctness=False, verdict="REVERT", bwd_median_ms=None,
             mfu=None, error="could not certify notes/probe.py in 8 probe rounds"),
        _fail(11),
        _fail(13),
        _fail(15),
    ]
    flag = constant_output_flag(records)
    assert flag is not None and flag["kind"] == "constant_output"
    assert flag["iters"] == [9, 11, 13, 15]
    # and the digest carries it where student and meta prompts can see it
    digest = build_digest(records, target={"value": 2.3})
    assert digest["flags"] and digest["flags"][0]["kind"] == "constant_output"


def test_constant_output_flag_stays_quiet_when_diffs_move():
    """Diffs that change with the code are a live numeric bug — the ordinary,
    debuggable case. A KEEP also clears the streak: outputs demonstrably respond."""
    from forgekernel.inner.trajectory import constant_output_flag

    assert constant_output_flag([_fail(1), _fail(2, _DIFF_B), _fail(3)]) is None
    assert constant_output_flag([_fail(1), _fail(2)]) is None  # below min repeats
    assert constant_output_flag(
        [_fail(1), _fail(2), _rec(iter=3), _fail(4)]) is None  # KEEP resets
    assert build_digest([_fail(1), _fail(2, _DIFF_B)], target={})["flags"] == []
