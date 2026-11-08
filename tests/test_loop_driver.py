"""Tests for the two-layer driver orchestration core (with a fake backend).

The backend (agent spawn / GPU submit / git) is faked so the loop control flow —
probe sub-loop, relaxed-revert, plateau->meta transition, prefix-reuse resume — is
verified deterministically without an agent or a cluster.
"""

from __future__ import annotations

from pathlib import Path

import json
import pytest

from forgekernel.baseline import RefBaseline
from forgekernel.loop.driver import (
    LoopContext,
    MetaResult,
    run_best_confirmation,
    run_inner_iteration,
    run_inner_loop,
    run_kb_certify,
    run_meta_step,
    run_two_layer,
)
from forgekernel.resume import Decision
from forgekernel.inner.trajectory import load_records

REPO = Path(__file__).resolve().parents[1]

REC = lambda did, ms, kf="flash_attn_dsl/flash_bwd.py": (  # noqa: E731
    f"decision_id: {did}\nchange: did {did}\nhypothesis: h\n"
    f"binding_constraint: bc\nfiles_touched: {kf}\n"
)
EVAL = lambda ms, ok=True: (  # noqa: E731  # a dual-window bench (the template contract)
    '=== JSON ===\n{"correctness": {"pass": %s}, "performance": {"bwd_median_ms": %r, '
    '"device_median_ms": %r, "bwd_mfu_pct": 32.0}}'
    % ("true" if ok else "false", ms, ms)
)
NCU = "(no csv)"  # profiler parse yields {} -> profiler stays empty, must not crash

# a minimal §9-conformant milestone (passes validate_milestone) with a §3 DAG
_MILESTONE = """# demo milestone

## 0. Problem statement and target
x

## 1. Problem contract (Math Spec)
x

## 2. Decision DAG + phase order
D1.1 -> D2.3

## 3. Decision details
**D1.1 base**
- Action: a
- Purpose: b
- Trigger condition: c
- Cost: d
- Depends on: —
- Handoff protocol: [SYNC: none] (single-role demo)
- Feed form: [GATHER: none]
- Consumer phase: [PHASE: none]

M1 Gate:
- End-to-end performance line: ≤2 ms [ESTIMATE]
- Mechanism-verification line: tensor pipe active % ≥60%

**D2.3 next**
- Action: a
- Purpose: b
- Trigger condition: c
- Cost: d
- Depends on: D1.1

M2 Gate:
- End-to-end performance line: ≤1 ms [ESTIMATE]
- Mechanism-verification line: tensor pipe active % ≥80%

## 4. Known dead ends
x

## 5. Global correctness / performance gate
x

## 6. Rules
x

## Worked Example
x
"""


class FakeBackend:
    def __init__(self, agent_outputs, gpu_logs):
        self.agent_outputs = list(agent_outputs)
        self.gpu_logs = list(gpu_logs)
        self.calls = []
        self.prompts = []
        self._sha = 0

    def run_agent(self, prompt: str) -> str:
        self.calls.append(("agent",))
        self.prompts.append(prompt)
        out = self.agent_outputs.pop(0)
        # A callable stands in for an agent that touches the filesystem. The default
        # canned-string agent cannot, which is exactly why a patch that is reported but
        # never written used to look identical to one that landed.
        return out(prompt) if callable(out) else out

    def submit_gpu(self, branch: str, mode: str) -> str:
        self.calls.append(("gpu", mode))
        return self.gpu_logs.pop(0)

    def commit(self, message: str) -> str:
        self._sha += 1
        self.calls.append(("commit", message[:12]))
        return f"sha{self._sha}"

    def push_branch(self, branch: str) -> None:
        self.calls.append(("push", branch))

    def reset(self, ref: str) -> None:
        self.calls.append(("reset", ref))

    def head(self) -> str:
        return f"sha{self._sha}"


_REF = RefBaseline(op="demo", ms=2.0, ms_device=2.0, job="633122", ref_name="cuDNN",
                   ncu={"tensor_pct": 61.0, "warp_active_pct": 42.0})


def _ctx(tmp_path, backend, *, target_ms=1.9, deadline=1e18, dag=None,
         ref_baseline=_REF) -> LoopContext:
    status = tmp_path / "status.jsonl"
    status.write_text("")
    # a conformant milestone on disk: absent counts as a violation (it is the
    # pre-distillation state), which would put the post-patch gate into re-prompt
    milestone = tmp_path / "m.md"
    if not milestone.exists():
        milestone.write_text(_MILESTONE, encoding="utf-8")
    # kb_dir is tmp_path and the run's tree is kb_dir/<kb_arch> (sm90 by default):
    # the CLI refuses to launch without that tree, and the certify / harvest lint
    # gates fail closed on a missing one, so the fixture provides it the way a
    # real launch guarantees it.
    (tmp_path / "sm90").mkdir(exist_ok=True)
    return LoopContext(
        op="demo", workspace=tmp_path, milestone=milestone,
        kb_dir=tmp_path, status_path=status, meta_dir=REPO / "meta",
        meta_status_path=tmp_path / "meta_status.jsonl",
        kernel_file="flash_attn_dsl/flash_bwd.py", branch_prefix="fs-demo",
        target_ms=target_ms, deadline_epoch=deadline, backend=backend,
        now=lambda: 0.0, decisions_dag=dag or [], ncu_retries=0,
        ref_baseline=ref_baseline,
        # The forced M0 audit is on by default in production; these fixtures predate
        # it and script exact agent-reply sequences, so they opt out (the --no-m0-audit
        # flow). The audit's own behavior is covered by the test_m0_audit_* group.
        m0_audit=False,
        # Same story for stall pricing: on in production, but it spends a GPU log per
        # plateau diagnosis and adds a fourth recon evidence channel, which the scripted
        # sequences here predate. Covered by the test_stall_pricing_* group.
        stall_pricing=False,
        # And for the lever pricing round (one design turn + one GPU job before a
        # plateau diagnosis), which every scripted plateau sequence predates.
        # Covered by the test_lever_pricing_* group.
        lever_pricing_rounds=0,
        # And for the no-reopen-without-a-price rule: these fixtures script bare
        # "cause_class: R1" reopens from before it existed. Covered by the
        # test_r1_* / test_reopen_price_* / *unpriced_reopen* group.
        reopen_requires_price=False,
        # And for suspicious-watermark confirmation (extra GPU logs on big jumps),
        # which these exact-sequence fixtures also predate. Covered by the
        # test_watermark_* group.
        watermark_margin=0,
    )


def test_inner_iteration_happy_path(tmp_path):
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec is not None and rec.decision_id == "D2.3"
    assert rec.verdict == "KEEP" and rec.bwd_median_ms == 2.6
    # committed + pushed + ran all + ran ncu
    modes = [c for c in be.calls if c[0] == "gpu"]
    assert ("gpu", "all") in modes and ("gpu", "ncu") in modes
    assert len(load_records(ctx.status_path)) == 1
    # the KEEP record carries the commit sha so the meta loop can reset to it
    assert rec.sha and load_records(ctx.status_path)[0].sha == rec.sha


def test_inner_iteration_probe_subloop(tmp_path):
    # student asks for a probe, it PASSes, student is re-prompted and returns a record
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", REC("D2.4", 2.5)],
        ["PROBE PASS max_diff=0.0", EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 6, metric_before=2.66)
    assert rec.decision_id == "D2.4"
    gpu_modes = [c[1] for c in be.calls if c[0] == "gpu"]
    assert gpu_modes[0].startswith("cmd:")  # probe ran first
    assert "all" in gpu_modes and "ncu" in gpu_modes


def test_probe_certifying_on_a_bogus_max_diff_is_not_accepted(tmp_path):
    # a PASS whose max_diff cannot be an error norm must not end the probe sub-loop:
    # the student is re-prompted with the rejection instead of proceeding to eval.
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D3.1", 2.5)],
        ["PROBE PASS max_diff=80000000000.0", "PROBE PASS max_diff=0.0",
         EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 7, metric_before=2.66)
    assert rec.decision_id == "D3.1"
    assert "did NOT certify" in be.prompts[1] and "ceiling" in be.prompts[1]
    # the bogus round cost a probe run, not an eval
    assert [c[1] for c in be.calls if c[0] == "gpu"][:2] == ["cmd:python3 notes/p.py"] * 2


def test_probe_with_no_verdict_is_not_reported_as_a_numerical_failure(tmp_path):
    # a job that died before printing a verdict used to reach the student as
    # "PROBE FAIL (max_diff=None)", sending it to hunt a numerical bug it did not have.
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", REC("D3.1", 2.5)],
        ["ptxas error: unresolved extern function\nJob killed", EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 8, metric_before=2.66)
    feedback = be.prompts[1]
    assert "NO verdict" in feedback and "NOT a numerical failure" in feedback
    assert "ptxas error" in feedback  # the log tail is handed back


def test_probe_fail_feedback_carries_the_log_tail(tmp_path):
    # a FAILing probe used to reach the student as verdict counts alone — every
    # [diag] line the probe printed (sanitizer reports, shape dumps) was discarded
    # in exactly the case the student needed it. The FAIL branch now hands back a
    # bounded log tail like the no-verdict branch always did.
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", REC("D3.1", 2.5)],
        ["[diag] Invalid __global__ write at kernel.py:27\n"
         "[diag] Address 0x2af7 is out of bounds\n"
         "PROBE FAIL max_diff=0.5", EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 9, metric_before=2.66)
    feedback = be.prompts[1]
    assert "PROBE FAIL" in feedback
    assert "Invalid __global__ write" in feedback  # the probe's own diagnostics
    assert "out of bounds" in feedback


def test_inner_iteration_reprompts_on_unparseable_then_records(tmp_path):
    # first agent reply is prose with no record block -> re-prompt -> valid record
    be = FakeBackend(["I analyzed the kernel but forgot the block", REC("D1", 2.6)],
                     [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec is not None and rec.decision_id == "D1" and rec.verdict == "KEEP"


def test_inner_iteration_all_unparseable_is_blocked_revert(tmp_path):
    be = FakeBackend(["garbage"] * 4, [])
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 4
    rec = run_inner_iteration(ctx, 0, metric_before=2.6, last_good_ref="sha-prev")
    assert rec.verdict == "REVERT" and rec.decision_id == "unparseable"
    assert "parseable" in rec.error.lower()
    assert ("reset", "sha-prev") in be.calls


def test_probe_pass_on_the_last_round_gets_a_delivery_turn(tmp_path):
    """The budget counts agent turns, probe turns included; a probe that PASSes
    on the last one has no turn left to deliver the decision it certified
    (observed on dsa_sparse: r20 PASS 2/2 -> "could not certify" -> two hours
    reset). One delivery-only turn closes the hole."""
    be = FakeBackend(["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py",
                      REC("D1", 2.6)],
                     ["PROBE FAIL max_diff=9", "PROBE PASS max_diff=0.0", EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    rec = run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev")
    assert rec is not None and rec.decision_id == "D1" and rec.verdict == "KEEP"
    assert "This turn can only deliver" in be.prompts[2]
    assert ("reset", "sha-prev") not in be.calls


def test_probe_pass_undelivered_is_named_and_the_probe_survives_the_reset(tmp_path):
    """If even the delivery turn lands no record, the row must say the fight
    was WON (not "could not certify" — that line once sent the next iteration past a
    certified probe into a -17.9% commit), and the probe file must survive
    the workspace reset so the next iteration continues from the PASS."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('rig v3')\n")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", "garbage"],
                     ["PROBE FAIL max_diff=9", "PROBE PASS max_diff=0.0"])
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    rec = run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev")
    assert rec.verdict == "REVERT" and rec.decision_id == "probe-undelivered"
    assert "PASSED on the last probe round" in rec.error
    assert "could not certify" not in rec.error
    assert ("reset", "sha-prev") in be.calls
    assert (tmp_path / "notes" / "p.py").read_text() == "print('rig v3')\n"


def test_probe_blocked_keeps_the_probe_file_and_says_it_did(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('rig v2')\n")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py"] * 2, ["PROBE FAIL max_diff=9"] * 2)
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    rec = run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev")
    assert rec.decision_id == "probe-blocked" and "kept in the workspace" in rec.error
    assert (tmp_path / "notes" / "p.py").read_text() == "print('rig v2')\n"


def test_inner_iteration_correctness_fail_reverts_and_resets(tmp_path):
    be = FakeBackend([REC("D9.9", None)], [EVAL(0.0, ok=False), NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 3, metric_before=2.66, last_good_ref="sha-prev")
    assert rec.verdict == "REVERT" and rec.correctness is False
    assert ("reset", "sha-prev") in be.calls  # bad edit rolled back
    assert rec.error  # objective failure signature recorded
    # NCU IS run on a correctness-REVERT too — negative iters carry hardware truth
    assert ("gpu", "ncu") in be.calls


def test_inner_loop_stops_on_target(tmp_path):
    be = FakeBackend([REC("D1", 1.8)], [EVAL(1.8), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    reason = run_inner_loop(ctx, start_iter=0)
    assert reason == "target"


# a dual-window eval: e2e reading + pipelined-batch device reading
EVAL_DW = lambda ms, dev: (  # noqa: E731
    '=== JSON ===\n{"correctness": {"pass": true}, "performance": '
    '{"bwd_median_ms": %r, "bwd_mfu_pct": 32.0, "device_median_ms": %r}}' % (ms, dev)
)


def test_device_gate_stops_on_the_device_window(tmp_path):
    # e2e 2.5 (host tax) / device 1.8: past target in the device window only.
    # The same eval must stop a device-gated loop and keep an e2e-gated one going.
    be = FakeBackend([REC("D1", 2.5)], [EVAL_DW(2.5, 1.8), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.gate_window = "device"
    assert run_inner_loop(ctx, start_iter=0) == "target"
    rec = load_records(ctx.status_path)[0]
    assert rec.bwd_median_ms == 2.5 and rec.device_median_ms == 1.8

    be2 = FakeBackend([REC("D1", 2.5)], [EVAL_DW(2.5, 1.8), NCU])
    ctx2 = _ctx(tmp_path, be2, target_ms=1.9)
    ctx2.gate_window = "e2e"
    ctx2.status_path.write_text("")
    ctx2.max_iters = 1
    assert run_inner_loop(ctx2, start_iter=0) == "max_iters"


# an e2e-only bench (violates the dual-window template contract on a device-gated run)
EVAL_E2E_ONLY = lambda ms: (  # noqa: E731
    '=== JSON ===\n{"correctness": {"pass": true}, "performance": '
    '{"bwd_median_ms": %r, "bwd_mfu_pct": 32.0}}' % ms
)


def test_device_gate_over_a_single_window_bench_refuses_loudly(tmp_path):
    # gate_window="device" but the bench only times e2e (structural: it measured
    # bwd_median_ms, so the perf path ran and simply never times the device
    # window): every KEEP would carry no gate reading, so target/plateau could
    # never fire and the loop would grind to its wall-clock budget steering on
    # nothing. Must stop with its own reason — "backend_error" reads as GPU
    # flake — and only AFTER the record landed, so the trajectory agrees with
    # the pushed commit on resume.
    be = FakeBackend([REC("D1", 1.8)], [EVAL_E2E_ONLY(1.8), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.gate_window = "device"
    assert run_inner_loop(ctx, start_iter=0) == "gate_unmeasured"
    recs = load_records(ctx.status_path)
    assert len(recs) == 1 and recs[0].verdict == "KEEP" and recs[0].sha


EVAL_NO_PERF = ('=== JSON ===\n'
                '{"correctness": {"pass": true}, "performance": null}')


def test_resolve_target_trusts_only_a_window_matched_ssot(tmp_path):
    # §0's target_ms is a bare float and the e2e-era prompts teach the teacher to
    # derive it from the e2e reading — on a device-gated run that number gates
    # ~one host tax loose, so the driver only takes a §0 target whose
    # target_window annotation matches the run's gate window.
    from forgekernel.loop.driver import _resolve_target
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=1.7)
    ctx.gate_window = "e2e"

    ctx.milestone.write_text("## 0. t\ntarget_ms: 2.4\n", encoding="utf-8")
    assert _resolve_target(ctx) == 2.4          # unannotated = e2e by construction
    ctx.gate_window = "device"
    assert _resolve_target(ctx) == 1.7          # mismatch -> CLI-derived seed

    ctx.milestone.write_text("## 0. t\ntarget_ms: 2.4\ntarget_window: device\n",
                             encoding="utf-8")
    assert _resolve_target(ctx) == 2.4          # matching annotation -> SSOT wins
    ctx.gate_window = "e2e"
    assert _resolve_target(ctx) == 1.7          # device-anchored §0 on an e2e run


def test_seed_target_writes_the_window_annotation(tmp_path):
    from forgekernel.loop.driver import _seed_target_ms
    from forgekernel.milestone import parse_target_ms, parse_target_window
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=1.7)
    ctx.gate_window = "device"
    ctx.milestone.write_text("## 0. Problem statement and target\nx\n", encoding="utf-8")
    _seed_target_ms(ctx)
    text = ctx.milestone.read_text(encoding="utf-8")
    assert parse_target_ms(text) == 1.7
    assert parse_target_window(text) == "device"


def test_a_wholly_missing_perf_block_stays_tolerated_in_both_windows(tmp_path):
    # correctness passed but the perf phase died (job killed, transient): before
    # gate_window existed this was a KEEP row with no metric and the loop went
    # on. It must stay that way in BOTH windows — the structural guard is for a
    # bench that measured e2e and skipped device, not for a job that lost its
    # whole perf block.
    for window in ("e2e", "device"):
        be = FakeBackend([REC("D1", None)], [EVAL_NO_PERF, NCU])
        ctx = _ctx(tmp_path, be, target_ms=1.9)
        ctx.status_path.write_text("")
        ctx.gate_window = window
        ctx.max_iters = 1
        assert run_inner_loop(ctx, start_iter=0) == "max_iters"
        rec = load_records(ctx.status_path)[0]
        assert rec.verdict == "KEEP" and rec.bwd_median_ms is None


def test_inner_loop_advances_the_rollback_anchor_to_the_last_keep(tmp_path):
    """The production half of the revert contract: run_inner_loop must supply the
    anchor run_inner_iteration resets to. It never did — last_good_ref stayed None,
    the reset branch was dead code, and a REVERTed commit stayed in the tree as the
    next iteration's foundation (observed as back-to-back REVERTs of the same
    decision in a bwd run). A KEEP moves the anchor; the next correctness
    break resets to exactly that sha."""
    be = FakeBackend([REC("D1.1", 2.5), REC("D2.3", None)],
                     [EVAL(2.5), NCU, EVAL(0.0, ok=False), NCU])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 2
    assert run_inner_loop(ctx, start_iter=0) == "max_iters"
    assert ("reset", "sha1") in be.calls  # sha1 = the KEEP's commit


def test_a_green_segments_entry_head_is_the_first_anchor(tmp_path):
    # a green segment starts on a verified tree: a break on the very first
    # iteration restores the entry state, not nothing
    be = FakeBackend([REC("D1.1", None)], [EVAL(0.0, ok=False), NCU])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 1
    run_inner_loop(ctx, start_iter=0)  # initial_green defaults to True
    assert ("reset", "sha0") in be.calls  # sha0 = HEAD at segment entry


def test_scaffolding_failures_never_reset(tmp_path):
    # pre-green there is no verified state to restore; a failed gate row leaves the
    # half-built tree in place for the student to keep repairing
    be = FakeBackend([REC("M0-gate", None)], [EVAL(0.0, ok=False), NCU])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 1
    run_inner_loop(ctx, start_iter=0, initial_green=None)
    assert not any(c[0] == "reset" for c in be.calls)


def test_inner_loop_stops_on_exhausted(tmp_path):
    be = FakeBackend(["EXHAUSTED: no more decisions"], [])
    ctx = _ctx(tmp_path, be)
    reason = run_inner_loop(ctx, start_iter=0)
    assert reason == "exhausted"


def test_inner_loop_stops_on_plateau(tmp_path):
    be = FakeBackend(
        [REC("A", 2.66), REC("B", 2.70), REC("C", 2.68), REC("D", 2.69)],
        [EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.68), NCU, EVAL(2.69), NCU],
    )
    ctx = _ctx(tmp_path, be, target_ms=1.0)
    reason = run_inner_loop(ctx, start_iter=0)
    assert reason == "plateau"


def test_plateau_is_scoped_to_post_resume_segment(tmp_path):
    # regression: a meta patch resumes with start_iter advanced; a SINGLE non-improving
    # iteration must NOT immediately re-plateau on the pre-meta history. The segment
    # (records[start_iter:]) needs > window points before plateau can fire, so each
    # patch gets a real proving window instead of meta firing every iteration.
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    be = FakeBackend([REC("E", 2.61)], [EVAL(2.61), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.0)  # _ctx wipes status; populate AFTER
    for i, ms in enumerate([2.66, 2.62, 2.61, 2.60]):
        append_record(ctx.status_path, IR(iter=i, decision_id=f"P{i}", change="x",
                                           correctness=True, verdict="KEEP", bwd_median_ms=ms, sha=f"s{i}"))
    ctx.max_iters = 1  # run exactly one iteration then bail via max_iters, not plateau
    reason = run_inner_loop(ctx, start_iter=4)
    assert reason == "max_iters"  # NOT "plateau" — the segment has only 1 point


def test_plateau_fires_after_full_window_in_segment(tmp_path):
    # after a resume, plateau still fires once the SEGMENT accumulates a full stalled window
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    be = FakeBackend([REC("E", 2.70), REC("F", 2.71), REC("G", 2.72), REC("H", 2.73)],
                     [EVAL(2.70), NCU, EVAL(2.71), NCU, EVAL(2.72), NCU, EVAL(2.73), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.0)  # _ctx wipes status; populate AFTER
    for i, ms in enumerate([2.66, 2.60]):
        append_record(ctx.status_path, IR(iter=i, decision_id=f"P{i}", change="x",
                                           correctness=True, verdict="KEEP", bwd_median_ms=ms, sha=f"s{i}"))
    reason = run_inner_loop(ctx, start_iter=2)
    assert reason == "plateau"


PKGSTEP = lambda did, k, n: (  # noqa: E731  # a package-step delivery: the decl opens the change
    f"decision_id: {did}\nchange: part {k} of {n}. exchange face step\n"
    f"hypothesis: h\nbinding_constraint: bc\nfiles_touched: flash_attn_dsl/flash_bwd.py\n"
)


def test_plateau_defers_to_declared_package_in_flight(tmp_path):
    # The stop lane reads the same plea the judgment lane HOLDs on: a standing
    # plateau whose segment tail is an unfinished declared package step must NOT
    # stop the loop (an n-step package's measured intermediates all but guarantee
    # one mid-package plateau, and stopping burns a meta step on construction).
    # Construction continues until the tail moves on — then the standing plateau
    # routes to meta as usual.
    be = FakeBackend(
        [REC("A", 2.66), PKGSTEP("B", 1, 4), PKGSTEP("C", 2, 4), PKGSTEP("D", 3, 4),
         REC("E", 2.67)],  # a plain KEEP: the tail moved on -> plea forfeited
        [EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.68), NCU, EVAL(2.69), NCU,
         EVAL(2.67), NCU],
    )
    ctx = _ctx(tmp_path, be, target_ms=1.0)
    reason = run_inner_loop(ctx, start_iter=0)
    assert reason == "plateau"
    # 5 agent turns: the plateau standing after iteration 3 (mid-package) did not stop the loop
    assert sum(1 for c in be.calls if c[0] == "agent") == 5


def test_plateau_fires_when_the_package_is_finished(tmp_path):
    # k == n is a finished package, not an in-flight one: normal plateau routing
    be = FakeBackend(
        [REC("A", 2.66), PKGSTEP("B", 1, 3), PKGSTEP("C", 2, 3), PKGSTEP("D", 3, 3)],
        [EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.68), NCU, EVAL(2.69), NCU],
    )
    ctx = _ctx(tmp_path, be, target_ms=1.0)
    reason = run_inner_loop(ctx, start_iter=0)
    assert reason == "plateau"
    assert sum(1 for c in be.calls if c[0] == "agent") == 4


def test_plateau_fires_when_the_package_budget_is_spent(tmp_path):
    # more declared KEEPs than the declared n: the plea's row budget is spent,
    # the deferral is void, and the standing plateau stops the loop
    be = FakeBackend(
        [REC("A", 2.66), PKGSTEP("B", 1, 2), PKGSTEP("C", 1, 2), PKGSTEP("D", 1, 2)],
        [EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.68), NCU, EVAL(2.69), NCU],
    )
    ctx = _ctx(tmp_path, be, target_ms=1.0)
    reason = run_inner_loop(ctx, start_iter=0)
    assert reason == "plateau"
    assert sum(1 for c in be.calls if c[0] == "agent") == 4


def _patching(tmp_path, addition, reply):
    """A patch agent that edits the milestone the way a real one must, then reports it.

    The milestone is seeded here rather than in _ctx so the callable can close over the
    path before the backend is built.
    """
    milestone = tmp_path / "m.md"
    if not milestone.exists():
        milestone.write_text(_MILESTONE, encoding="utf-8")

    def _run(_prompt):
        milestone.write_text(milestone.read_text(encoding="utf-8") + addition, encoding="utf-8")
        return reply
    return _run


def test_meta_step_parses_patch_and_computes_resume(tmp_path):
    dag = [Decision(id="D2.4", depends_on=["D1.1"]),
           Decision(id="D5.2", depends_on=["D0.9"])]
    diagnosis = "cause_class: R1\ndeficient_locus: D1.1"
    patch = _patching(tmp_path, "\n### D0.9 loosened\nx\n",
                      "patch_class: structural\nchanged_entries: [D0.9]\npatch: ...\n")
    be = FakeBackend([diagnosis, patch], [])
    ctx = _ctx(tmp_path, be, dag=dag)
    res = run_meta_step(ctx)
    assert isinstance(res, MetaResult)
    assert res.patch_class == "structural"
    assert res.changed_entries == ["D0.9"]
    assert res.resume_after == "D2.4"  # only D5.2 (depends on D0.9) invalidated
    assert res.invalidated == ["D5.2"]


def test_a_patch_that_is_reported_but_never_written_is_caught(tmp_path):
    # The observed failure (two consecutive meta steps): the patch agent read the
    # milestone, emitted `patch:` text in its reply, and edited nothing. Nothing applies
    # that text, so the milestone was unchanged -- yet it stayed schema-clean and the
    # loop spent a full inner segment on a patch that did not exist.
    diagnosis = "cause_class: R2\ndeficient_locus: M1"
    reported = "patch_class: additive\nchanged_entries: [D1.4]\npatch: <the new decision>\n"
    be = FakeBackend([diagnosis, reported, reported, reported], [])
    ctx = _ctx(tmp_path, be)
    before = ctx.milestone.read_text(encoding="utf-8")
    res = run_meta_step(ctx)
    assert res.milestone_violations, "an unwritten patch must not pass the gate"
    assert "never reached the file" in res.milestone_violations[0]
    assert ctx.milestone.read_text(encoding="utf-8") == before


def test_an_unwritten_patch_is_re_prompted_before_it_is_given_up_on(tmp_path):
    # Same shape, but the agent fixes itself on the second round: the gate must let the
    # corrected patch through rather than condemn the whole meta step on round one.
    diagnosis = "cause_class: R1\ndeficient_locus: D1.1"
    reply = "patch_class: structural\nchanged_entries: [D0.9]\npatch: ...\n"
    be = FakeBackend([diagnosis, reply, _patching(tmp_path, "\n### D0.9 loosened\nx\n", reply)],
                     [])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert not res.milestone_violations
    assert "never reached the file" in be.prompts[-1], "the re-prompt must say what was wrong"


def test_a_patch_claiming_a_decision_the_file_does_not_have_is_caught(tmp_path):
    # Weaker than an unwritten patch and easier to miss: the agent did edit, but the
    # entry it reported is nowhere in the file, so the student can never act on it.
    diagnosis = "cause_class: R2\ndeficient_locus: M1"
    reply = "patch_class: additive\nchanged_entries: [D9.9]\npatch: ...\n"
    agent = _patching(tmp_path, "\n### D7.1 something else\nx\n", reply)
    be = FakeBackend([diagnosis, agent, agent, agent], [])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert any("D9.9" in v for v in res.milestone_violations)


def test_the_ledger_records_whether_the_patch_landed(tmp_path):
    from forgekernel.meta_trajectory import load_meta_records
    diagnosis = "cause_class: R1\ndeficient_locus: D1.1"
    reply = "patch_class: structural\nchanged_entries: [D0.9]\npatch: ...\n"
    be = FakeBackend([diagnosis, _patching(tmp_path, "\n### D0.9 loosened\nx\ny\n", reply)], [])
    ctx = _ctx(tmp_path, be)
    run_meta_step(ctx)
    rec = load_meta_records(ctx.meta_status_path)[0]
    assert rec.patch_landed is True
    assert rec.milestone_delta == "+4/-0 lines"  # the appended block, nothing removed


def test_meta_step_records_meta_trajectory(tmp_path):
    # a milestone patch appends a PENDING MetaRecord carrying the diagnosis + patch
    from forgekernel.meta_trajectory import load_meta_records
    diagnosis = ("cause_class: R1\ndeficient_locus: D1.4\n"
                 "patch_direction: shrink tile_m 128->64\n")
    patch = _patching(tmp_path, "\n### D1.4 shrunk\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend([diagnosis, patch], [])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    run_meta_step(ctx, best_before=2.66, milestone_before="OLD MILESTONE TEXT")
    recs = load_meta_records(ctx.meta_status_path)
    assert len(recs) == 1
    assert recs[0].cause_class == "R1" and recs[0].deficient_locus == "D1.4"
    assert recs[0].changed_entries == ["D1.4"]
    assert "shrink tile_m" in recs[0].patch_direction
    assert recs[0].outcome == "PENDING"
    assert recs[0].best_before == 2.66
    assert recs[0].milestone_before == "OLD MILESTONE TEXT"


def test_meta_step_g3_records_certified_outcome(tmp_path):
    from forgekernel.meta_trajectory import load_meta_records
    be = FakeBackend(
        ["cause_class: G3\ndeficient_locus: KB:swizzle\npatch_direction: certify flat-tv",
         "PROBE_REQUEST: notes/kb_probe.py", "RECIPE_CERTIFIED: hw_kb/cutedsl/sm90/swizzle.md"],
        ["PROBE PASS max_diff=0.0"],
    )
    ctx = _ctx(tmp_path, be)
    run_meta_step(ctx)
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].outcome == "KB_CERTIFIED" and recs[0].certified is True


def test_meta_step_injects_reflection_into_diagnose(tmp_path):
    # a prior REVERT_PATCH is surfaced to the diagnose agent so it does not repeat it
    from forgekernel.meta_trajectory import MetaRecord, append_meta_record
    append_meta_record(tmp_path / "meta_status.jsonl",
                       MetaRecord(meta_step=0, cause_class="R1", deficient_locus="D1.4",
                                  patch_class="structural", changed_entries=["D1.4"],
                                  patch_direction="shrink tile_m 128->64",
                                  outcome="REVERT_PATCH"))
    be = FakeBackend(["cause_class: R2\ndeficient_locus: D6\npatch_direction: add decision",
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")], [])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    run_meta_step(ctx)
    diagnose_prompt = be.prompts[0]  # first agent call is the diagnose step
    assert "REVERT_PATCH" in diagnose_prompt and "shrink tile_m" in diagnose_prompt


def test_meta_step_flags_malformed_milestone(tmp_path):
    # a structural (non-G3) patch that leaves the milestone free-form is flagged
    # (schema gate re-prompts up to max_patch_rounds; the fake never fixes it).
    # The agent does edit the file every round, so what is under test here is the
    # schema gate and not the landing gate.
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone.write_text("# free-form milestone, no schema sections", encoding="utf-8")
    edits = [_patching(tmp_path, f"\nround {i}\n",
                       "patch_class: structural\nchanged_entries: []\n") for i in range(3)]
    ctx.backend = FakeBackend(["cause_class: R1\ndeficient_locus: D1", *edits], [])
    res = run_meta_step(ctx)
    assert res.milestone_violations  # malformed -> violations recorded (schema enforced)
    assert not any("never reached the file" in v for v in res.milestone_violations)


class _SchemaFake(FakeBackend):
    """A patch fake that writes milestone text to the file on each patch agent call,
    so the driver's schema gate sees the file the agent 'wrote'."""

    def __init__(self, agent_outputs, milestone_path, milestone_writes):
        super().__init__(agent_outputs, [])
        self.milestone_path = milestone_path
        self.milestone_writes = list(milestone_writes)

    def run_agent(self, prompt: str) -> str:
        out = super().run_agent(prompt)
        if self.milestone_writes:
            text = self.milestone_writes.pop(0)
            if text is not None:
                self.milestone_path.write_text(text, encoding="utf-8")
        return out


def test_meta_step_schema_gate_reprompts_then_accepts(tmp_path):
    # first patch leaves the milestone malformed -> gate re-prompts with the
    # violations -> second patch is schema-clean -> accepted, no violations recorded.
    diag = "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: x"
    patch = "patch_class: structural\nchanged_entries: [D1.1]\n"
    ctx_ms = tmp_path / "m.md"
    be = _SchemaFake([diag, patch, patch], ctx_ms,
                     milestone_writes=[None, "# bad, no sections",
                                       _MILESTONE + "\n### D1.1 loosened\nx\n"])
    ctx = _ctx(tmp_path, be)
    ctx.milestone = ctx_ms
    res = run_meta_step(ctx)
    assert res.milestone_violations == []  # accepted after re-prompt
    # the second patch prompt carried the schema violations back to the agent
    assert any("Violations" in p or "missing required section" in p for p in be.prompts[2:])


def test_inner_iteration_retries_ncu_when_empty_on_passing_kernel(tmp_path):
    # NCU is best-effort on the GPU side; a KEEP kernel that comes back with an empty
    # profiler means NCU itself failed (e.g. a lost job) -> retry so key points keep truth.
    valid_csv = (REPO / "tests" / "fixtures" / "ncu_sample.csv").read_text()
    be = FakeBackend([REC("D1", 0.21)],
                     [EVAL(0.21), "(no csv)", valid_csv])  # all, ncu(empty), ncu(retry ok)
    ctx = _ctx(tmp_path, be)
    ctx.ncu_retries = 2
    rec = run_inner_iteration(ctx, 0, metric_before=0.30)
    assert rec.verdict == "KEEP" and rec.profiler  # profiler recovered on retry
    ncu_calls = [c for c in be.calls if c == ("gpu", "ncu")]
    assert len(ncu_calls) == 2  # retried once


def test_inner_iteration_does_not_retry_ncu_on_correctness_break(tmp_path):
    # a correctness-broken kernel legitimately yields no NCU (bench crashes) -> no retry
    be = FakeBackend([REC("D9", None)], [EVAL(0.0, ok=False), "(no csv)"])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=0.30, last_good_ref="s0")
    assert rec.verdict == "REVERT"
    ncu_calls = [c for c in be.calls if c == ("gpu", "ncu")]
    assert len(ncu_calls) == 1  # not retried — empty NCU is expected here


def test_kb_certify_captures_recipe_path(tmp_path):
    be = FakeBackend(
        ["PROBE_REQUEST: notes/kb.py", "RECIPE_CERTIFIED: hw_kb/cutedsl/sm90/stmatrix_epilogue.md"],
        ["PROBE PASS max_diff=0.0"],
    )
    res = run_kb_certify(_ctx(tmp_path, be), diagnosis="cause_class: G3")
    assert res.certified is True and res.recipe == "hw_kb/cutedsl/sm90/stmatrix_epilogue.md"


def test_student_prompt_injects_fresh_recipe(tmp_path):
    # after a G3 KB certify, the student must be actively pointed at the new recipe
    # (passive "look in KB_DIR" was not enough — the live run's student ignored it).
    from forgekernel.loop.driver import _student_prompt
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.fresh_recipe = "hw_kb/cutedsl/sm90/stmatrix_epilogue.md"
    prompt = _student_prompt(ctx, iter_no=9)
    assert "hw_kb/cutedsl/sm90/stmatrix_epilogue.md" in prompt
    assert "Read and apply it first" in prompt and "just certified" in prompt


def test_two_layer_points_student_at_certified_recipe(tmp_path):
    # end-to-end: G3 certify -> the next student iteration's prompt names the recipe
    ms = tmp_path / "m.md"
    ms.write_text(_MILESTONE, encoding="utf-8")
    be = _E2EFake(
        agent_outputs=[
            REC("A", 2.66), REC("B", 2.66),                                  # seg1 -> plateau
            "cause_class: G3\ndeficient_locus: KB:stmatrix\npatch_direction: certify stmatrix",
            "PROBE_REQUEST: notes/kb.py",                                    # teacher recipe+probe
            "RECIPE_CERTIFIED: hw_kb/cutedsl/sm90/stmatrix_epilogue.md",             # finalize
            REC("C", 2.50), REC("D", 2.50),                                  # seg2 (post-certify)
        ],
        gpu_logs=[EVAL(2.66), NCU, EVAL(2.66), NCU, "PROBE PASS max_diff=0.0",
                  EVAL(2.50), NCU, EVAL(2.50), NCU],
        milestone_path=ms, patch_writes=[])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.milestone = ms
    ctx.plateau_window = 1
    run_two_layer(ctx, max_meta=1)
    # a post-certify student prompt names the certified recipe path
    assert any("stmatrix_epilogue.md" in p for p in be.prompts)


def test_parse_cause_class_extracts_enum_even_with_trailing_prose(tmp_path):
    # the meta agent sometimes editorializes in the field ("R1 (atypical: …"); the strict
    # extractor must pull the clean enum so routing + records stay clean.
    from forgekernel.loop.driver import _parse_cause_class
    assert _parse_cause_class("cause_class: R1（atypical: the taxonomy has no target cell）") == "R1"
    assert _parse_cause_class("cause_class: G3\ndeficient_locus: KB:x") == "G3"
    assert _parse_cause_class("cause_class: R2 missing optimization") == "R2"
    assert _parse_cause_class("cause_class: I think the foundation is wrong") == ""  # no valid enum


def test_meta_step_records_clean_enum_cause_class(tmp_path):
    # even when the agent adds prose, the recorded MetaRecord.cause_class is the enum
    from forgekernel.meta_trajectory import load_meta_records
    be = FakeBackend(["cause_class: R1 (atypical: target unreachable)\ndeficient_locus: D1.1\n"
                      "patch_direction: cut regs",
                      _patching(tmp_path, "\n### D1.1 loosened\nx\n",
                                "patch_class: structural\nchanged_entries: [D1.1]\n")], [])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    run_meta_step(ctx)
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].cause_class == "R1"  # clean enum, not the prose-laden field


def test_meta_step_reprompts_diagnose_on_invalid_cause_class(tmp_path):
    # a diagnosis with no valid enum -> re-prompt once with a format reminder, then accept
    be = FakeBackend(["cause_class: I think the foundation is wrong\ndeficient_locus: D1.1",  # invalid
                      "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: x",  # fixed
                      _patching(tmp_path, "\n### D1.1 loosened\nx\n",
                                "patch_class: structural\nchanged_entries: [D1.1]\n")], [])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    run_meta_step(ctx)
    # the second diagnose prompt carried the enum-format reminder
    assert any("cause_class" in p and ("enum" in p or "G1" in p) for p in be.prompts[1:])


def test_language_binding_and_student_prompt_injection(tmp_path):
    # {LANGUAGE} is bound and the student prompt tells the student which language to write in
    from forgekernel.loop.driver import _bindings, _student_prompt
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.language = "cutedsl"
    assert _bindings(ctx)["{LANGUAGE}"] == "cutedsl"
    prompt = _student_prompt(ctx, iter_no=0)
    assert "cutedsl" in prompt.lower()


def test_language_defaults_to_cutedsl_in_context(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    assert ctx.language == "cutedsl"  # LoopContext default


def test_resolve_target_reads_milestone_then_cli(tmp_path):
    # target SSOT: a machine-readable target_ms in the milestone wins; the CLI
    # --target-ms is only the seed/fallback. This makes meta's §0 target edit LIVE
    # (it was inert before — the loop only ever read ctx.target_ms).
    from forgekernel.loop.driver import _resolve_target
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.40)  # CLI seed
    ctx.milestone.write_text(
        "# op\n## 0. target\ntarget_ms: 0.1852\ntarget_window: device\n",
        encoding="utf-8")
    assert _resolve_target(ctx) == 0.1852  # milestone SSOT wins
    ctx.milestone.write_text("# op\n## 0. target\nmeasured performance of the original op.\n", encoding="utf-8")
    assert _resolve_target(ctx) == 0.40  # no marker -> CLI fallback


def test_inner_loop_target_from_milestone(tmp_path):
    # the loop's stop condition honors the milestone target, not just ctx.target_ms
    be = FakeBackend([REC("D1", 0.30)], [EVAL(0.30), NCU])
    ctx = _ctx(tmp_path, be, target_ms=0.10)  # CLI says 0.10 (not hit by 0.30)
    ctx.milestone.write_text(
        "# op\n## 0.\ntarget_ms: 0.35\ntarget_window: device\n",
        encoding="utf-8")  # milestone: 0.35
    reason = run_inner_loop(ctx, start_iter=0)
    assert reason == "target"  # 0.30 <= milestone 0.35 -> target hit


    # the diagnose prompt must carry the rendered NCU report (hardware truth) inline,
    # so the meta agent diagnoses from measured metrics — not from perf numbers alone.
    # (the meta agent runs in the student workspace and cannot fetch it via ./forge.)
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    be = FakeBackend(["cause_class: R1\ndeficient_locus: D1.1\npatch_direction: cut regs",
                      _patching(tmp_path, "\n### D1.1 loosened\nx\n",
                                "patch_class: structural\nchanged_entries: [D1.1]\n")], [])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    # a record carrying the register-pressure signature the meta agent should see
    append_record(ctx.status_path, IR(iter=0, decision_id="D1.1", change="x",
                                      correctness=True, verdict="KEEP", bwd_median_ms=0.21, mfu=64.0,
                                      sha="s0", profiler={"regs_per_thread": 232,
                                      "spill_st_sectors": 1_113_088, "warp_active_pct": 11.8,
                                      "tensor_pct": 79.4}))
    run_meta_step(ctx)
    diagnose_prompt = be.prompts[0]
    # the injected NCU report names the bottleneck verdict + the flagged metric
    assert "spill" in diagnose_prompt.lower() or "register" in diagnose_prompt.lower()
    assert "232" in diagnose_prompt or "warp_active" in diagnose_prompt


def test_meta_step_g3_routes_to_kb_certify(tmp_path):
    # diagnose says G3 -> the meta step certifies a hw_kb recipe (teacher probe),
    # NOT a milestone patch.
    be = FakeBackend(
        ["cause_class: G3\ndeficient_locus: KB:swizzle",  # diagnose
         "PROBE_REQUEST: notes/kb_probe.py",              # teacher writes recipe + probe
         "RECIPE_CERTIFIED: hw_kb/cutedsl/sm90/swizzle.md"],      # finalize after PASS
        ["PROBE PASS max_diff=0.0"],
    )
    res = run_meta_step(_ctx(tmp_path, be))
    assert res.patch_class == "kb-example"
    assert res.certified is True
    assert any(c[0] == "gpu" and c[1].startswith("cmd:") for c in be.calls)


def test_kb_certify_probe_pass_first_try(tmp_path):
    be = FakeBackend(
        ["PROBE_REQUEST: notes/kb.py", "RECIPE_CERTIFIED: hw_kb/x.md"],
        ["PROBE PASS max_diff=0.0"],
    )
    res = run_kb_certify(_ctx(tmp_path, be), diagnosis="cause_class: G3")
    assert res.certified is True and res.probe == "notes/kb.py"


def test_kb_certify_logs_and_hands_over_the_certifying_job_id(tmp_path, capsys):
    """The certifying job is the entry's provenance pointer: it is on the
    verdict log line (round + job), in the finalize turn's instruction (so the
    teacher cites it in the status cell), and on the result."""
    be = FakeBackend(
        ["PROBE_REQUEST: notes/kb.py", "PROBE_REQUEST: notes/kb.py", "RECIPE_CERTIFIED: hw_kb/x.md"],
        ["PROBE FAIL max_diff=9", "PROBE PASS max_diff=0.0"],
    )
    jobs = iter(["881001", "881002"])

    def submit(branch, mode, _orig=be.submit_gpu):
        be.last_job_id = next(jobs)
        return _orig(branch, mode)
    be.submit_gpu = submit
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 3
    res = run_kb_certify(ctx, diagnosis="cause_class: G3")
    assert res.certified is True and res.job == "881002" and res.rounds == 2
    lines = [l for l in capsys.readouterr().out.splitlines() if "recipe probe notes/kb.py ->" in l]
    assert "(round 1, job 881001)" in lines[0] and "(round 2, job 881002)" in lines[1]
    assert "job 881002" in be.prompts[2] and "round 2 of this certification" in be.prompts[2]


def test_kb_certify_fail_then_pass(tmp_path):
    be = FakeBackend(
        ["PROBE_REQUEST: notes/kb.py", "PROBE_REQUEST: notes/kb.py", "RECIPE_CERTIFIED: ok"],
        ["PROBE FAIL max_diff=9", "PROBE PASS max_diff=0.0"],
    )
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 4
    assert run_kb_certify(ctx, diagnosis="G3").certified is True


def test_kb_certify_all_fail_is_uncertified(tmp_path):
    be = FakeBackend(["PROBE_REQUEST: notes/kb.py"] * 3, ["PROBE FAIL max_diff=9"] * 3)
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 3
    assert run_kb_certify(ctx, diagnosis="G3").certified is False


def test_kb_certify_lint_bounces_leak_back_to_teacher(tmp_path):
    # the teacher's fresh recipe cites the reference -> the certify-time lint
    # bounces it back with findings; once cleaned in place, certification stands
    (tmp_path / "sm90").mkdir(exist_ok=True)

    def write_leaky(prompt):
        (tmp_path / "sm90" / "r.md").write_text("schedule order extracted from the teacher oracle\n")
        return "RECIPE_CERTIFIED: sm90/r.md"

    def fix_in_place(prompt):
        assert "KB LEAK LINT FAILED" in prompt and "sm90/r.md" in prompt
        (tmp_path / "sm90" / "r.md").write_text("# clean mechanism body\n")
        return "fixed"

    be = FakeBackend(["PROBE_REQUEST: notes/kb.py", write_leaky, fix_in_place],
                     ["PROBE PASS max_diff=0.0"])
    res = run_kb_certify(_ctx(tmp_path, be), diagnosis="G3")
    assert res.certified is True and res.recipe == "sm90/r.md"


def test_kb_certify_quarantines_a_persistent_leak(tmp_path):
    # findings surviving the fix rounds: the entry must leave the tree the
    # student reads (moved, not deleted), and certification must fail
    kb = tmp_path / "repo" / "hw_kb" / "cutedsl"
    (kb / "sm90").mkdir(parents=True)

    def write_leaky(prompt):
        (kb / "sm90" / "r.md").write_text("schedule order extracted from the teacher oracle\n")
        return "RECIPE_CERTIFIED: sm90/r.md"

    be = FakeBackend(
        ["PROBE_REQUEST: notes/kb.py", write_leaky,
         lambda p: "still leaky", lambda p: "still leaky"],
        ["PROBE PASS max_diff=0.0"])
    ctx = _ctx(tmp_path, be)
    ctx.kb_dir = kb
    res = run_kb_certify(ctx, diagnosis="G3")
    assert res.certified is False and res.recipe is None
    assert not (kb / "sm90" / "r.md").exists()
    quarantined = tmp_path / "repo" / "kb_quarantine" / "r.md"
    assert quarantined.is_file() and "oracle" in quarantined.read_text()


def test_inner_iteration_probe_budget_exhausted_is_graceful_revert(tmp_path):
    # the student keeps probing past the budget -> record a REVERT, do NOT crash
    be = FakeBackend(["PROBE_REQUEST: notes/p.py"] * 4, ["PROBE FAIL max_diff=1.0"] * 4)
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 4
    rec = run_inner_iteration(ctx, 0, metric_before=2.66, last_good_ref="sha-prev")
    assert rec is not None and rec.verdict == "REVERT" and rec.correctness is False
    assert "probe" in rec.error.lower()
    assert ("reset", "sha-prev") in be.calls
    assert len(load_records(ctx.status_path)) == 1


def test_inner_loop_stops_stuck_on_consecutive_reverts(tmp_path):
    be = FakeBackend(
        [REC("A", 0.0), REC("B", 0.0), REC("C", 0.0)],
        [EVAL(0.0, ok=False), NCU, EVAL(0.0, ok=False), NCU, EVAL(0.0, ok=False), NCU],
    )
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_consecutive_reverts = 3
    reason = run_inner_loop(ctx, start_iter=0)
    assert reason == "stuck"


def test_inner_loop_stops_cleanly_on_agent_failure(tmp_path):
    from forgekernel.loop.driver import AgentError

    class Boom(FakeBackend):
        def run_agent(self, prompt):
            raise AgentError("boom")

    ctx = _ctx(tmp_path, Boom([], []))
    assert run_inner_loop(ctx, start_iter=0) == "agent_failed"


def test_inner_loop_stops_cleanly_on_backend_failure(tmp_path):
    from forgekernel.loop.driver import BackendError

    class Boom(FakeBackend):
        def push_branch(self, branch):
            raise BackendError("git push exit 128")

    be = Boom([REC("A", 2.6)], [])
    assert run_inner_loop(_ctx(tmp_path, be), start_iter=0) == "backend_error"


def test_inner_loop_safety_net_on_unexpected_error(tmp_path):
    class Boom(FakeBackend):
        def run_agent(self, prompt):
            raise KeyError("some unexpected bug")

    assert run_inner_loop(_ctx(tmp_path, Boom([], [])), start_iter=0) == "error"


def test_resolve_resume_additive_keeps_head(tmp_path):
    from forgekernel.loop.driver import MetaResult, _resolve_resume
    from forgekernel.inner.trajectory import IterationRecord as IR
    recs = [IR(iter=0, decision_id="D1.1", change="x", correctness=True, verdict="KEEP",
               bwd_median_ms=2.6, sha="shaKEEP")]
    be = FakeBackend([], [])
    be._sha = 7  # head() -> sha7
    ctx = _ctx(tmp_path, be)
    meta = MetaResult("d", "additive", [], None, [])
    # additive: nothing invalidated -> keep all student state (current HEAD)
    assert _resolve_resume(ctx, meta, recs, baseline_ref="shaBASE") == "sha7"


def test_resolve_resume_foundation_restart_to_baseline(tmp_path):
    from forgekernel.loop.driver import MetaResult, _resolve_resume
    from forgekernel.inner.trajectory import IterationRecord as IR
    recs = [IR(iter=0, decision_id="D1.1", change="x", correctness=True, verdict="KEEP",
               bwd_median_ms=2.6, sha="shaKEEP")]
    ctx = _ctx(tmp_path, FakeBackend([], []))
    # structural patch whose earliest affected decision is the first -> None -> baseline
    meta = MetaResult("d", "structural", ["D1.1"], resume_after=None, invalidated=["D1.1"])
    assert _resolve_resume(ctx, meta, recs, baseline_ref="shaBASE") == "shaBASE"


def test_resolve_resume_structural_to_prefix_sha(tmp_path):
    from forgekernel.loop.driver import MetaResult, _resolve_resume
    from forgekernel.inner.trajectory import IterationRecord as IR
    recs = [IR(iter=0, decision_id="D1.1", change="x", correctness=True, verdict="KEEP",
               bwd_median_ms=2.6, sha="shaA"),
            IR(iter=1, decision_id="D2.3", change="y", correctness=True, verdict="KEEP",
               bwd_median_ms=2.5, sha="shaB")]
    ctx = _ctx(tmp_path, FakeBackend([], []))
    # resume_after=D1.1 -> reset to D1.1's checkpoint sha (prefix reuse)
    meta = MetaResult("d", "structural", ["D2.3"], resume_after="D1.1", invalidated=["D2.3"])
    assert _resolve_resume(ctx, meta, recs, baseline_ref="shaBASE") == "shaA"


def test_best_keep_profiler_is_strict_about_which_keep_it_reads():
    # NCU is best-effort: the sitting best KEEP can land with profiler={} after
    # retries. The reconciliation must then stay silent (None), never fall back
    # to an older profiled KEEP — that kernel no longer exists, and stamping
    # OVERLAP-BOUND on it would steer the diagnosis at a ghost.
    from forgekernel.loop.driver import _best_keep_profiler
    from forgekernel.inner.trajectory import IterationRecord as IR
    old = IR(iter=3, decision_id="D1.1", change="x", correctness=True, verdict="KEEP",
             bwd_median_ms=2.9, device_median_ms=2.9,
             profiler={"tensor_pct": 40.0, "time_ns": 2_900_000})
    best = IR(iter=7, decision_id="D2.2", change="y", correctness=True, verdict="KEEP",
              bwd_median_ms=2.5, device_median_ms=2.5, profiler={})
    assert _best_keep_profiler([old, best], "device") is None
    # ...and when the best KEEP does carry one, it is that record's, with its iter.
    best_profiled = IR(iter=9, decision_id="D2.3", change="z", correctness=True,
                       verdict="KEEP", bwd_median_ms=2.4, device_median_ms=2.4,
                       profiler={"tensor_pct": 71.0, "time_ns": 2_400_000})
    assert _best_keep_profiler([old, best_profiled], "device") == (
        best_profiled.profiler, 9)


def test_meta_step_parses_milestone_dag_when_ctx_dag_empty(tmp_path):
    # ctx.decisions_dag empty -> run_meta_step parses the milestone §4 to compute resume
    diag = "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: loosen the tile"
    patch = _patching(tmp_path, "\n### D1.1 loosened\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.1]\n")
    be = FakeBackend([diag, patch], [])
    ctx = _ctx(tmp_path, be)  # dag defaults to []
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")  # D1.1 is the first decision
    res = run_meta_step(ctx)
    # D1.1 is the first decision -> earliest affected is first -> resume_after None
    assert res.resume_after is None and res.invalidated == ["D1.1", "D2.3"]


def test_two_layer_inner_plateau_meta_then_target(tmp_path):
    # inner plateaus (4 flat), meta runs (diag+patch), resume, inner hits target
    dag = [Decision(id="A", depends_on=[])]
    agent = [REC("A", 2.66), REC("B", 2.70), REC("C", 2.68), REC("D", 2.69),  # plateau
             "cause_class: R1",
             _patching(tmp_path, "\n### patched\nx\n",
                       "patch_class: structural\nchanged_entries: []\n"),  # meta
             REC("E", 1.8)]  # post-patch student breaks through
    gpu = [EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.68), NCU, EVAL(2.69), NCU,
           EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9, dag=dag)
    summary = run_two_layer(ctx, max_meta=2)
    assert summary["stop"] == "target"
    assert summary["meta_steps"] == 1
    assert any(c[0] == "reset" for c in be.calls)  # resume reset happened


class _E2EFake(FakeBackend):
    """Full two-layer fake: student iters + meta steps. On each patch agent call
    (detected by the patch prompt marker) it writes the next queued milestone text,
    so the driver's schema gate + transactional rollback act on a real file.

"""

    def __init__(self, agent_outputs, gpu_logs, milestone_path, patch_writes):
        super().__init__(agent_outputs, gpu_logs)
        self.milestone_path = milestone_path
        self.patch_writes = list(patch_writes)

    def run_agent(self, prompt: str) -> str:
        out = super().run_agent(prompt)
        # a patch turn OR the initial distill turn -> the agent 'writes' the milestone
        # (a None entry = a patch turn that reports but never writes)
        if ("Meta-Agent — Patch step" in prompt or "Initial distillation authoring step" in prompt) and self.patch_writes:
            w = self.patch_writes.pop(0)
            if w is not None:
                self.milestone_path.write_text(w, encoding="utf-8")
        return out


def test_a_patch_that_never_lands_costs_no_inner_segment(tmp_path):
    # The saving, not just the detection: when the patch agent reports a patch it never
    # wrote, the milestone is unchanged, so there is nothing for the student to act on.
    # The rollback happens straight after the patch rounds, and the loop retries the
    # META step (or stops on budget) instead of spending an inner segment rediscovering
    # "exhausted" on the restored state — the redundant turn observed in a past run.
    from forgekernel.meta_trajectory import load_meta_records
    ms = tmp_path / "m.md"
    ms.write_text(_MILESTONE, encoding="utf-8")
    be = _E2EFake(
        agent_outputs=["EXHAUSTED: nothing left",                       # seg1 -> meta
                       "cause_class: R2\ndeficient_locus: M1\npatch_direction: add one",
                       *["patch_class: additive\nchanged_entries: [D1.4]\n"] * 3,
                       "EXHAUSTED: still nothing"],                     # must NOT be consumed
        gpu_logs=[], milestone_path=ms, patch_writes=[])                # writes nothing
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.milestone = ms
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["stop"] == "meta_budget"
    rec = load_meta_records(ctx.meta_status_path)[0]
    assert rec.outcome == "REVERT_PATCH" and rec.patch_landed is False
    assert ms.read_text(encoding="utf-8") == _MILESTONE
    # the patch is never credited with an inner segment of its own, and no student
    # turn runs on the rolled-back state either: the seg2 reply stays unconsumed
    assert be.agent_outputs == ["EXHAUSTED: still nothing"]


def test_an_intercepted_patch_retries_the_meta_step_not_the_segment(tmp_path):
    # with budget left, the loop goes straight from the interception back to diagnose
    # (which sees the REVERT_PATCH through the reflection) — no student turn between;
    # the inner segment runs only once a patch actually lands
    from forgekernel.meta_trajectory import load_meta_records
    ms = tmp_path / "m.md"
    ms.write_text(_MILESTONE, encoding="utf-8")
    good = _MILESTONE + "\n<!-- GOOD_PATCH_MARKER -->\n"
    be = _E2EFake(
        agent_outputs=["EXHAUSTED: nothing left",                        # seg1 -> meta 1
                       "cause_class: R2\ndeficient_locus: M1\npatch_direction: add one",
                       *["patch_class: additive\nchanged_entries: [D1.4]\n"] * 3,  # never writes
                       "cause_class: R2\ndeficient_locus: D2.3\npatch_direction: other",
                       "patch_class: additive\nchanged_entries: []\n",   # writes good
                       "EXHAUSTED: done"],                               # seg2 on the landed patch
        gpu_logs=[], milestone_path=ms,
        patch_writes=[None, None, None, good])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.milestone = ms
    run_two_layer(ctx, max_meta=2)
    assert be.agent_outputs == []          # seg2 DID run once the second patch landed
    # prompts: [0]=student seg1, [1]=diagnose1, [2..4]=patch rounds, [5]=diagnose2 —
    # diagnose2 follows the interception directly and carries its REVERT_PATCH
    assert "REVERT_PATCH" in be.prompts[5]
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].patch_landed is False and recs[1].patch_landed is True


def test_resume_ref_finds_a_decision_inside_a_bundle(tmp_path):
    # delivery granularity is the agent's choice: a structural patch resuming after
    # D2.3 must find the bundle commit that carried it — the exact-match miss fell
    # back to baseline_ref, a silent full restart discarding every kept iteration
    from forgekernel.inner.trajectory import IterationRecord as IR
    from forgekernel.loop.driver import _ref_for_decision
    rows = [IR(iter=0, decision_id="D1.1", change="x", correctness=True,
               verdict="KEEP", bwd_median_ms=2.7, sha="s0"),
            IR(iter=1, decision_id="D2.3+D2.4", change="x", correctness=True,
               verdict="KEEP", bwd_median_ms=2.6, sha="s1")]
    assert _ref_for_decision(rows, "D2.3") == "s1"
    assert _ref_for_decision(rows, "D2.4") == "s1"
    assert _ref_for_decision(rows, "D1.1") == "s0"
    assert _ref_for_decision(rows, "D9.9") is None


def test_two_layer_reverts_unhelpful_patch(tmp_path):
    # a schema-valid patch that yields no measured improvement is rolled back:
    # milestone restored to the pre-patch snapshot + outcome recorded REVERT_PATCH.
    from forgekernel.meta_trajectory import load_meta_records
    ms = tmp_path / "m.md"
    bad = _MILESTONE + "\n<!-- BAD_PATCH_MARKER -->\n"
    be = _E2EFake(
        agent_outputs=[REC("A", 2.66), REC("B", 2.66),                     # seg1 -> plateau
                       "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: tweak X",
                       "patch_class: additive\nchanged_entries: []\n",     # patch (writes bad)
                       REC("C", 2.70), REC("C2", 2.71)],                   # seg2: no gain -> plateau
        gpu_logs=[EVAL(2.66), NCU, EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.71), NCU],
        milestone_path=ms, patch_writes=[bad])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.milestone = ms
    ms.write_text(_MILESTONE, encoding="utf-8")
    ctx.plateau_window = 1
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["meta_steps"] == 1
    # milestone rolled back: the bad patch's marker is gone
    assert "BAD_PATCH_MARKER" not in ms.read_text()
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].outcome == "REVERT_PATCH"
    assert recs[0].best_after is not None


def test_a_sub_noise_patch_gain_is_not_attribution(tmp_path):
    # 2.66 -> 2.655 is a 0.19% "gain" — inside the ~1.37% measurement noise. A
    # useless patch must not earn KEEP_PATCH (whose milestone edit is permanent)
    # off one lucky draw: it reverts, milestone restored, workspace reset.
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    from forgekernel.loop.driver import _judge_patch, _PendingPatch
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    append_record(ctx.status_path, IR(iter=0, decision_id="D", change="x",
                                      correctness=True, verdict="KEEP",
                                      bwd_median_ms=2.655, sha="s0"))
    out = _judge_patch(ctx, _PendingPatch(best_before=2.66,
                                          milestone_before="snapshot", good_ref="good"))
    assert out == "REVERT_PATCH"
    assert ("reset", "good") in be.calls


def test_a_supra_noise_patch_gain_is_kept(tmp_path):
    # 2.66 -> 2.60 is 2.26% — distinguishable from noise, so the patch is credited
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    from forgekernel.loop.driver import _judge_patch, _PendingPatch
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    append_record(ctx.status_path, IR(iter=0, decision_id="D", change="x",
                                      correctness=True, verdict="KEEP",
                                      bwd_median_ms=2.60, device_median_ms=2.60,
                                      sha="s0"))
    out = _judge_patch(ctx, _PendingPatch(best_before=2.66,
                                          milestone_before="snapshot", good_ref="good"))
    assert out == "KEEP_PATCH"
    assert not any(c[0] == "reset" for c in be.calls)


def _ratio_rows(ctx, rows):
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    for i, (ms, ratio) in enumerate(rows):
        append_record(ctx.status_path, IR(iter=i, decision_id=f"D{i}", change="x",
                                          correctness=True, verdict="KEEP",
                                          device_median_ms=ms, vs_ref_device=ratio,
                                          sha=f"s{i}"))


def test_patch_judged_on_same_job_ratio_not_a_lucky_ms_best(tmp_path):
    """The live case: the pre-patch best was a draw 1.5% fast (2.05934 ms,
    ratio 1.0398); the rebuild lifted the same-job ratio to 1.0747 but read only
    0.14% faster in ms than that draw. Judged in ratio it is a gain."""
    from forgekernel.loop.driver import _judge_patch, _PendingPatch
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    _ratio_rows(ctx, [(2.05934, 1.0398), (2.09524, 1.0436),     # pre-patch
                      (2.08910, 1.0486), (2.05642, 1.0747)])    # post-patch segment
    out = _judge_patch(ctx, _PendingPatch(best_before=2.05934, milestone_before="snap",
                                          good_ref="s0", records_before=2))
    assert out == "KEEP_PATCH"
    assert not any(c[0] == "reset" for c in be.calls)


def test_patch_ms_gain_on_a_fast_node_is_not_a_ratio_gain(tmp_path):
    """The other direction: a segment that read 3% faster in ms because its job
    drew a fast node, with the same-job ratio flat, is refuted."""
    from forgekernel.loop.driver import _judge_patch, _PendingPatch
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    _ratio_rows(ctx, [(2.10, 1.040), (2.037, 1.043)])
    out = _judge_patch(ctx, _PendingPatch(best_before=2.10, milestone_before="snap",
                                          good_ref="s0", records_before=1))
    assert out == "REVERT_PATCH"
    assert ("reset", "s0") in be.calls


def test_patch_ratio_bar_ignores_rows_of_a_reverted_segment(tmp_path):
    """Rows inside a segment an earlier REVERT_PATCH voided describe a tree that
    is gone; their ratio does not set the bar for the next patch."""
    from forgekernel.loop.driver import _judge_patch, _PendingPatch
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    _ratio_rows(ctx, [(2.10, 1.040), (2.00, 1.080),     # row 1: voided segment
                      (2.05, 1.070)])                     # post-patch
    ctx.voided_ranges.append((1, 2))
    out = _judge_patch(ctx, _PendingPatch(best_before=2.00, milestone_before="snap",
                                          good_ref="s0", records_before=2))
    assert out == "KEEP_PATCH"                            # 1.070 vs 1.040, not vs 1.080


def test_two_layer_reflects_then_lands_good_patch(tmp_path):    # THE end-to-end goal: meta step 1's patch does not help -> REVERT_PATCH; meta step 2
    # SEES that revert in its diagnose prompt (reflection) and lands a DIFFERENT patch
    # that improves -> KEEP_PATCH. Proves the meta agent correctly patches AND reflects.
    from forgekernel.meta_trajectory import load_meta_records
    ms = tmp_path / "m.md"
    bad = _MILESTONE + "\n<!-- BAD_PATCH_MARKER -->\n"
    good = _MILESTONE + "\n<!-- GOOD_PATCH_MARKER -->\n"
    be = _E2EFake(
        agent_outputs=[
            REC("A", 2.66), REC("B", 2.66),                                 # seg1 -> plateau
            "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: shrink tile_m 128->64",
            "patch_class: additive\nchanged_entries: []\n",                 # patch1 (bad)
            REC("C", 2.70), REC("C2", 2.71),                               # seg2: no gain -> plateau
            "cause_class: R2\ndeficient_locus: D2.3\npatch_direction: add pipelining decision",
            "patch_class: additive\nchanged_entries: []\n",                 # patch2 (good)
            REC("D", 2.50), REC("E", 2.50),                                 # seg3: improves -> plateau
        ],
        gpu_logs=[EVAL(2.66), NCU, EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.71), NCU,
                  EVAL(2.50), NCU, EVAL(2.50), NCU],
        milestone_path=ms, patch_writes=[bad, good])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.milestone = ms
    ms.write_text(_MILESTONE, encoding="utf-8")
    ctx.plateau_window = 1
    summary = run_two_layer(ctx, max_meta=2)

    # step 1 reverted, step 2 kept
    recs = load_meta_records(ctx.meta_status_path)
    assert [r.outcome for r in recs] == ["REVERT_PATCH", "KEEP_PATCH"]
    # the GOOD patch landed and the BAD one was rolled back
    final = ms.read_text()
    assert "GOOD_PATCH_MARKER" in final and "BAD_PATCH_MARKER" not in final
    # REFLECTION: meta step 2's diagnose prompt carried step 1's REVERT_PATCH direction
    # (seg2 now runs 2 iters before plateau, so step-2 diagnose is prompts[6])
    diagnose2_prompt = be.prompts[6]
    assert "REVERT_PATCH" in diagnose2_prompt and "shrink tile_m" in diagnose2_prompt
    assert summary["meta_steps"] == 2


# --- C6: initial distillation step (meta authors milestone v1 from the oracle) -----

_SKELETON = """# gemm milestone
## 0. target and equivalence judge
x
## 1. Problem contract (Math Spec)
x
State the constraint on every item.
## 3. Decision DAG + phase order
(to be written by meta)
## 3. Decision details
Every Dx.y uses the five-tuple (Action / Purpose / Trigger condition / Cost / Depends on).
## 4. Known dead ends
## 6. Correctness / Performance Gate
x
## Worked Example
x
"""


def test_needs_authoring_detects_empty_skeleton(tmp_path):
    from forgekernel.loop.driver import _needs_authoring
    skel = tmp_path / "m.md"
    skel.write_text(_SKELETON, encoding="utf-8")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone = skel
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// open source kernel", encoding="utf-8")
    assert _needs_authoring(ctx) is True  # no Dx.y decisions -> skeleton
    # a fully-authored milestone does not need authoring
    skel.write_text(_MILESTONE, encoding="utf-8")
    assert _needs_authoring(ctx) is False


def test_needs_authoring_when_milestone_does_not_exist(tmp_path):
    """A scaffolded operator has no milestone file at all — that is the entry state.

    Nothing writes a placeholder any more (the distill step creates the file), so if
    a missing file read as "conformant" the loop would skip distillation and hand the
    student a milestone that is not there.
    """
    from forgekernel.loop.driver import _needs_authoring
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone = tmp_path / "never-written.md"
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// open source kernel", encoding="utf-8")
    assert not ctx.milestone.exists()
    assert _needs_authoring(ctx) is True


def test_missing_milestone_is_a_violation_not_a_pass(tmp_path):
    """The post-patch gate must not read a deleted milestone as schema-clean."""
    from forgekernel.loop.driver import _validate_milestone_file
    assert _validate_milestone_file(tmp_path / "gone.md")


def test_distill_refuses_without_a_ref_baseline(tmp_path):
    """No measured ref means every gate would be an [ESTIMATE], target_ms included.

    Refusing is the point of measuring first: a run that distills anyway produces a
    milestone whose stop threshold is a guess from oracle source.
    """
    be = FakeBackend(["should never be called"], [])
    ctx = _ctx(tmp_path, be, ref_baseline=None)
    ctx.milestone = tmp_path / "never-written.md"
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["stop"] == "no_ref_baseline"
    assert not be.prompts, "no agent should be spawned without a baseline"


@pytest.mark.parametrize("compact", [False, True])
def test_distill_prompt_carries_the_measured_baseline(tmp_path, compact):
    """The teacher can only anchor gates to numbers it is actually given."""
    from forgekernel.loop.driver import run_distill_step
    ms = tmp_path / "distilled.md"
    be = _E2EFake(agent_outputs=["authored"], gpu_logs=[],
                  milestone_path=ms, patch_writes=[_MILESTONE])
    ctx = _ctx(tmp_path, be)
    ctx.milestone = ms
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    if compact:
        ctx.source_brief = "Compute a normalized row in FP32."
        ctx.arch_baseline = _REF
    assert not run_distill_step(ctx).milestone_violations
    prompt = be.prompts[0]
    assert "2.0 ms" in prompt and "633122" in prompt      # the measured ref + its job
    assert "tensor_pct: 61.0" in prompt                    # its NCU signature
    assert ("Compute a normalized row in FP32." in prompt) == compact


def test_needs_authoring_false_without_oracle(tmp_path):
    from forgekernel.loop.driver import _needs_authoring
    skel = tmp_path / "m.md"
    skel.write_text(_SKELETON, encoding="utf-8")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone = skel
    ctx.oracle_path = None  # cannot distill without the source
    assert _needs_authoring(ctx) is False


def test_blackbox_distills_without_an_oracle_path(tmp_path):
    """Denying source must not silently deny distillation.

    Before oracle_mode existed the only question asked was "is there an oracle path",
    so forbidding the teacher to read source also turned distillation off — and the
    loop then ran the student against whatever milestone happened to be on disk, which
    looks exactly like a normal resume. A run whose whole premise is "the teacher
    never read the reference" would have produced no milestone at all.
    """
    from forgekernel.loop.driver import _needs_authoring
    skel = tmp_path / "m.md"
    skel.write_text(_SKELETON, encoding="utf-8")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone = skel
    ctx.oracle_path = None
    ctx.oracle_mode = "blackbox"
    assert _needs_authoring(ctx) is True
    ctx.oracle_mode = "none"
    assert _needs_authoring(ctx) is False


def test_blackbox_binding_says_withheld_not_missing(tmp_path):
    """A teacher told the oracle is "not provided" reads that as a missing input and
    either stalls on it or invents a substitute. It is not missing — it is withheld,
    and the reference is still there to be run and profiled."""
    from forgekernel.loop.driver import _bindings
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.oracle_path = "/ref/gemm_sm90.py"
    ctx.oracle_mode = "blackbox"
    oracle = _bindings(ctx)["{ORACLE}"]
    assert "/ref/gemm_sm90.py" not in oracle, "blackbox leaked the oracle path into the prompt"
    assert "withheld" in oracle


def test_arch_ref_absent_reads_differently_under_blackbox(tmp_path):
    """Absence of an architecture reference has two causes that need opposite advice.

    Where the reference IS the library — a GEMM against cuBLAS — the two coincide and
    the library baseline is exactly what the run reproduces; template A.4 asks that the
    coincidence be stated. Telling that teacher the target "is NOT what this run
    reproduces" is simply false, and invites it to invent a second anchor.
    """
    from forgekernel.loop.driver import _bindings
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.arch_baseline = None
    ctx.oracle_mode = "blackbox"
    assert "coincide" in _bindings(ctx)["{ARCH_REF}"]
    ctx.oracle_mode = "source"
    assert "understates" in _bindings(ctx)["{ARCH_REF}"]


def test_run_distill_step_seeds_target_ms(tmp_path):
    # after authoring, if the milestone has no machine-readable target_ms, distill
    # seeds it from the CLI target so the SSOT lives in the milestone from the start.
    from forgekernel.loop.driver import run_distill_step
    from forgekernel.milestone import parse_target_ms
    ms = tmp_path / "m.md"
    ms.write_text(_SKELETON, encoding="utf-8")
    be = _E2EFake(agent_outputs=["authored"], gpu_logs=[],
                  milestone_path=ms,
                  patch_writes=[_MILESTONE])  # authored milestone has no target_ms line
    ctx = _ctx(tmp_path, be, target_ms=0.40)
    ctx.milestone = ms
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    run_distill_step(ctx)
    assert parse_target_ms(ms.read_text()) == 0.40  # seeded from CLI


def test_run_distill_step_authors_and_records(tmp_path):
    from forgekernel.loop.driver import run_distill_step
    from forgekernel.meta_trajectory import load_meta_records
    ms = tmp_path / "distilled.md"  # does not exist: nothing scaffolds a milestone
    # the distill agent 'writes' a full §9 milestone (fake writes it on the author call)
    be = _E2EFake(agent_outputs=["milestone authored: see file"],
                  gpu_logs=[], milestone_path=ms, patch_writes=[_MILESTONE])
    # _E2EFake writes on 'Meta-Agent — Patch step'; the distill prompt has its own marker -> use author marker
    ctx = _ctx(tmp_path, be)
    ctx.milestone = ms
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    res = run_distill_step(ctx)
    assert res.milestone_violations == []  # authored a schema-clean milestone
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].outcome == "AUTHORED" and recs[0].cause_class == "DISTILL"
    # the distill prompt was given the oracle path
    assert str(ctx.oracle_path) in be.prompts[0]


def test_distill_that_writes_nothing_is_reported_not_passed(tmp_path):
    """A distill agent that never writes the file must not read as a clean v1.

    Observed in this failure mode: 5 rounds in a row ended with no milestone on
    disk because each agent session died before its Write (API timeout / truncated
    stream — the CLI exits 0 with the error as its final text, so the backend's
    rc-based retry never fires). Re-prompting those rounds with "file unreadable"
    burns the schema budget for zero information, so the second consecutive
    no-file round must raise AgentError (transient, watchdog-restartable) instead
    of grinding on toward distill_malformed.
    """
    from forgekernel.loop.driver import AgentError, run_distill_step
    be = FakeBackend(["API Error: upstream stream ended without a terminal frame"] * 4, [])
    ctx = _ctx(tmp_path, be)
    ctx.milestone = tmp_path / "never-written.md"
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    with pytest.raises(AgentError):
        run_distill_step(ctx)
    assert len(be.prompts) == 2  # gave the agent one retry, then stopped burning rounds
    assert not ctx.milestone.exists()  # and never fabricated a milestone


def test_distill_survives_agent_deleting_milestone_mid_revision(tmp_path):
    """Observed on dsa_indexer_fwd: during schema re-prompt the agent deleted milestone.md,
    the next validation hit FileNotFoundError, and the run died distill_malformed.
    The draft snapshot must restore the previous round's file and keep re-prompting —
    the delete round is a wasted round, not a dead run."""
    from forgekernel.loop.driver import run_distill_step

    def writes_draft(prompt):
        (tmp_path / "resilient.md").write_text("# draft, not §9-conformant", encoding="utf-8")
        return "wrote a draft"

    def deletes_it(prompt):
        (tmp_path / "resilient.md").unlink()
        return "oops, removed the file"

    def writes_clean(prompt):
        (tmp_path / "resilient.md").write_text(_MILESTONE, encoding="utf-8")
        return "authored"

    be = FakeBackend([writes_draft, deletes_it, writes_clean], [])
    ctx = _ctx(tmp_path, be)
    ctx.milestone = tmp_path / "resilient.md"
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    res = run_distill_step(ctx)
    assert res.milestone_violations == []  # recovered: round 3 authored a clean v1
    assert ctx.milestone.exists()
    # the post-delete re-prompt carried the DRAFT's violations, not "file unreadable" —
    # proof the snapshot was restored before validation rather than validated as missing
    assert "unreadable" not in be.prompts[2]


def test_two_layer_authors_before_first_inner_loop(tmp_path):
    # THE distillation flow, from the real entry state: no milestone file exists at all
    # (nothing scaffolds one) -> meta authors it from the oracle FIRST, then the inner
    # loop runs against the authored milestone.
    from forgekernel.meta_trajectory import load_meta_records
    ms = tmp_path / "distilled.md"
    be = _E2EFake(
        agent_outputs=["authored milestone",              # distill (writes _MILESTONE)
                       REC("D1.1", 1.8)],                  # first student iter hits target
        gpu_logs=[EVAL(1.8), NCU],
        milestone_path=ms, patch_writes=[_MILESTONE])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.milestone = ms
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    summary = run_two_layer(ctx, max_meta=1)
    # authored first (prompts[0] is the distill step, not a student iter)
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].outcome == "AUTHORED"
    # the authored milestone is what the student then ran against
    assert "D1.1" in ms.read_text() or "D2.3" in ms.read_text()
    assert summary["stop"] == "target"


def test_exhausted_milestone_routes_to_meta(tmp_path):
    # the phase order ends at Mk, so the student has no free lever left: running out of
    # milestone decisions below target must hand control to the meta loop (extend the
    # decision graph) rather than end the run with the meta budget unspent
    from forgekernel.meta_trajectory import load_meta_records
    ms = tmp_path / "m.md"
    ms.write_text(_MILESTONE, encoding="utf-8")
    be = _E2EFake(
        agent_outputs=["EXHAUSTED: no further phase-order decision",   # seg1 -> exhausted
                       "cause_class: R2\ndeficient_locus: D2.3\npatch_direction: add a decision",
                       "patch_class: additive\nchanged_entries: []\n",
                       "EXHAUSTED: still nothing"],                    # seg2
        gpu_logs=[], milestone_path=ms, patch_writes=[_MILESTONE + "\n**D2.4 added**\nx\n"])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.milestone = ms
    summary = run_two_layer(ctx, max_meta=1)
    assert [r.cause_class for r in load_meta_records(ctx.meta_status_path)] == ["R2"]
    assert summary["stop"] == "meta_budget" and summary["meta_steps"] == 1


# --- conformance gate ------------------------------------------------------

def test_malformed_v1_refuses_to_start(tmp_path):
    # v1 is the only milestone write with nothing to roll back to, and every later
    # patch edits it in place — so a v1 that never passed its gate must stop the run
    # rather than burn a GPU budget optimizing against it
    ms = tmp_path / "m.md"
    ms.write_text(_SKELETON, encoding="utf-8")
    be = _E2EFake(agent_outputs=["authored"] * 4, gpu_logs=[],
                  milestone_path=ms, patch_writes=["# still free-form, no sections"] * 4)
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.milestone = ms
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["stop"] == "distill_malformed"
    assert summary["violations"]
    assert not any(c[0] == "gpu" for c in be.calls)  # never reached the GPU


def test_conformant_v1_starts_the_inner_loop(tmp_path):
    # the mirror: a v1 that passes the gate proceeds normally
    ms = tmp_path / "m.md"
    ms.write_text(_SKELETON, encoding="utf-8")
    be = _E2EFake(agent_outputs=["authored", REC("D1.1", 1.8)], gpu_logs=[EVAL(1.8), NCU],
                  milestone_path=ms, patch_writes=[_MILESTONE])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.milestone = ms
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    assert run_two_layer(ctx, max_meta=1)["stop"] == "target"


def test_gate_flags_missing_mechanism_line_per_milestone(tmp_path):
    # one gate for the whole file is not enough: each structural milestone needs its own
    from forgekernel.milestone import validate_milestone
    broken = _MILESTONE.replace("- Mechanism-verification line: tensor pipe active % ≥60%\n", "")
    v = validate_milestone(broken)
    assert any("milestone D1" in x and "mechanism-verification line" in x for x in v)
    assert not any("milestone D2" in x for x in v)  # D2 still has its own


def test_gate_flags_dangling_dependency(tmp_path):
    # a dangling dep silently reads as "no dependency" -> prefix reuse under-invalidates
    from forgekernel.milestone import validate_milestone
    broken = _MILESTONE.replace("- Depends on: D1.1", "- Depends on: D9.9")
    assert any("depends on undefined decision D9.9" in x for x in validate_milestone(broken))


def test_gate_flags_unsourced_perf_number(tmp_path):
    from forgekernel.milestone import validate_milestone
    broken = _MILESTONE.replace("≤1 ms [ESTIMATE]", "≤1 ms")
    assert any("performance number without" in x for x in validate_milestone(broken))


def test_gate_flags_out_of_order_sections(tmp_path):
    from forgekernel.milestone import validate_milestone
    swapped = _MILESTONE.replace("## 4. Known dead ends\nx\n", "").replace(
        "## 6. Rules", "## 4. Known dead ends\nx\n\n## 6. Rules")
    assert any("out of order" in x for x in validate_milestone(swapped))


# --- NCU aim check ------------------------------------------------------------
# A profile of the wrong kernel is indistinguishable from a profile of a fast one,
# which is how a 27-iteration run reported the same 117us preprocess pass while the
# kernel under test went from 2871 ms to 4 ms. The only cheap tell is that a kernel
# said to dominate cannot be a sliver of the time it dominates.

def test_mis_aimed_profile_is_flagged():
    from forgekernel.loop.driver import _check_profile_covers_the_kernel
    # the real numbers from that run: preprocess captured, main kernel absent
    warn = _check_profile_covers_the_kernel({"time_ns": 117216, "kernel_count": 1}, 4.0097)
    assert warn and "mis-aimed" in warn
    assert "eval/ncu_run.py" in warn      # names where to look, not the kernel


def test_mis_aimed_profile_is_flagged_in_the_shape_profiler_from_ncu_emits():
    """The same incident through the digest a real profile produces.

    The test above feeds a legacy two-key dict, which no longer resembles the
    output of profiler_from_ncu: that always carries total_time_ns (a window total
    over every launch) and dominant_launches. Passing the window total against a
    per-call clock scales the coverage by the launch count — the template's
    ref_ncu_cmd profiles 24 — which turns this very incident from 0.029 into 0.70
    and silences the alarm. So the regression has to be stated in the real shape.
    """
    from forgekernel.loop.driver import _check_profile_covers_the_kernel
    digest = {"time_ns": 117216, "kernel_count": 1,
              "dominant_launches": 24, "total_launches": 24,
              "total_time_ns": 117216 * 24}
    warn = _check_profile_covers_the_kernel(digest, 4.0097)
    assert warn and "mis-aimed" in warn
    # and the message quotes the per-call time, not the 24x window total
    assert "2.813 ms/call" not in warn and "0.117 ms/call" in warn


def test_a_healthy_multi_launch_profile_reads_near_one():
    """A kernel that really does fill the window must not read as 22x coverage."""
    from forgekernel.timing_audit import ncu_coverage
    digest = {"time_ns": 950_000, "kernel_count": 1,
              "dominant_launches": 24, "total_launches": 24,
              "total_time_ns": 950_000 * 24}
    assert 0.9 < ncu_coverage(digest, 1.0) < 1.0


def test_a_dominant_kernel_profile_is_not_flagged():
    from forgekernel.loop.driver import _check_profile_covers_the_kernel
    # the architecture reference's own split: 2.077 ms of a 2.107 ms backward
    assert _check_profile_covers_the_kernel(
        {"time_ns": 2077216, "kernel_count": 3}, 2.1074) is None


def test_the_check_is_silent_without_both_numbers():
    """A correctness-broken iteration has no timing and legitimately no profile;
    neither is evidence of mis-aiming."""
    from forgekernel.loop.driver import _check_profile_covers_the_kernel
    assert _check_profile_covers_the_kernel({}, 4.0) is None
    assert _check_profile_covers_the_kernel({"time_ns": 117216}, None) is None


# --- scaffold mode: the declaration-driven eval schedule ----------------------

def test_scaffold_mode_skips_eval_until_the_gate_row_goes_green(tmp_path):
    """POLICY §3 build-phase semantics, executed: once a full eval has shown the lineage
    is not green, sub-decision commits are recorded NEUTRAL without touching the GPU;
    the gate row always evaluates, and its PASS restores evaluate-everything."""
    agent = [REC("D0.1", None), REC("D0.2", None), REC("D0.3", None),
             REC("M0-gate", 4.8), REC("D1.1", 4.5)]
    gpu = [EVAL(0.0, ok=False), NCU,   # iter0: green unknown -> full eval, broken
           EVAL(4.8), NCU,             # gate row: always evaluated
           EVAL(4.5), NCU]             # post-green: back to evaluate-everything
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 5
    reason = run_inner_loop(ctx, start_iter=0, initial_green=None)
    recs = load_records(ctx.status_path)
    assert [r.verdict for r in recs] == ["REVERT", "NEUTRAL", "NEUTRAL", "KEEP", "KEEP"]
    # scaffold rows: not evaluated (correctness None) but still a resume checkpoint
    assert recs[1].correctness is None and recs[1].sha
    assert recs[1].bwd_median_ms is None
    # exactly three full evals were paid: the probe, the gate, the post-green iter
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 3
    assert reason == "max_iters"


def test_a_failed_gate_row_returns_to_scaffolding(tmp_path):
    # gate rows always evaluate; a FAIL on an unknown lineage marks it not-green,
    # so the next sub-decision is scaffold-committed, not evaluated
    agent = [REC("M0-gate", None), REC("D0.5", None)]
    gpu = [EVAL(0.0, ok=False), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 2
    assert run_inner_loop(ctx, start_iter=0, initial_green=None) == "max_iters"
    assert [r.verdict for r in load_records(ctx.status_path)] == ["REVERT", "NEUTRAL"]
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 1


def test_direct_inner_loop_callers_keep_evaluate_everything(tmp_path):
    # the default initial_green=True preserves the old behavior: a correctness break
    # stays a hard REVERT and the next decision still gets the full eval
    agent = [REC("A", None), REC("B", 2.5)]
    gpu = [EVAL(0.0, ok=False), NCU, EVAL(2.5), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 2
    run_inner_loop(ctx, start_iter=0)
    assert [r.verdict for r in load_records(ctx.status_path)] == ["REVERT", "KEEP"]
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 2


def test_green_seeding_from_the_resumed_checkpoint(tmp_path):
    from forgekernel.inner.trajectory import IterationRecord
    from forgekernel.loop.driver import _green_at_ref
    records = [
        IterationRecord(iter=0, decision_id="D0.1", change="c", correctness=None,
                        verdict="NEUTRAL", sha="sha1"),
        IterationRecord(iter=1, decision_id="M0-gate", change="c", correctness=True,
                        verdict="KEEP", bwd_median_ms=4.8, sha="sha2"),
    ]
    assert _green_at_ref(records, "sha1") is False   # mid-construction checkpoint
    assert _green_at_ref(records, "sha2") is True    # a passing kernel
    assert _green_at_ref(records, "sha0") is None    # baseline / unowned ref


def test_recon_runs_the_teacher_script_and_feeds_the_output_back(tmp_path):
    """The teacher decides what to measure; the driver owns submission.

    Same division as the student probe loop and KB certification, for the same reason:
    lost-job retry, bounded polling and archiver lag are already solved once in the
    backend, and an agent reimplementing them gets them wrong.
    """
    from forgekernel.loop.driver import run_recon_step

    def write_up(prompt):  # artifacts must be written THIS session to count
        (tmp_path / "recon-report.md").write_text(
            "observations\n"
            "SASS-WAIVED: launch-config probe only in this scenario\n"
            "SCHED-WAIVED: no SASS page in this scenario\n"
            "LIVE-WAIVED: no SASS page in this scenario")
        (tmp_path / "candidates.md").write_text("A|B")
        return "report written, nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/launch_cfg.py", write_up], [])
    recon = FakeBackend([], ["Block Size (384,1,1)  Grid Size (2,1,66)"])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "launch_cfg.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.rounds == 1 and result.finished is True
    # the script was committed before submission: delivery is `git archive`, which
    # ships committed files only, so an uncommitted script arrives as absence
    assert any(c[0] == "commit" for c in recon.calls)
    assert ("gpu", "cmd:python3 notes/launch_cfg.py") in recon.calls
    # and the job's output reached the next prompt rather than being dropped
    assert "Grid Size (2,1,66)" in teacher.prompts[1]


def test_recon_request_for_a_missing_script_costs_no_job(tmp_path):
    """A teacher once echoed the prompt's own `notes/<name>.py` placeholder and a GPU
    job was spent learning "can't open file". The driver re-prompts instead; the
    rejected request never reaches the reconnaissance backend."""
    from forgekernel.loop.driver import run_recon_step

    def write_up(prompt):  # fresh write-up so the request-less finish is accepted
        (tmp_path / "recon-report.md").write_text(
            "observations\nSASS-WAIVED: no probe ran in this scenario\n"
            "SCHED-WAIVED: no probe ran\nLIVE-WAIVED: no probe ran")
        (tmp_path / "candidates.md").write_text("A|B")
        return "giving a report instead"

    teacher = FakeBackend(["RECON_REQUEST: notes/<name>.py", write_up], [])
    recon = FakeBackend([], [])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path  # exists, but the requested file does not
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    # no submission: the toolkit-seeding commit is the only backend traffic allowed
    assert not [c for c in recon.calls if c[0] == "gpu"]
    assert "was rejected" in teacher.prompts[1] and "notes/<name>.py" in teacher.prompts[1]


def test_recon_write_up_turns_fire_until_the_artifacts_exist(tmp_path):
    """Observed live: 12 rounds of probing, zero artifacts, and distillation went on
    to read a report nobody had written. Probing over means writing begins — the
    teacher gets write-up turns until both files exist, and no GPU job is spent."""
    from forgekernel.loop.driver import run_recon_step

    def write_up(prompt):
        assert "can no longer send RECON_REQUEST" in prompt
        assert "SASS-WAIVED" in prompt      # budget gone: waiving is the only legal state
        (tmp_path / "recon-report.md").write_text(
            "observations [MEASURED, job 1]\nSASS-WAIVED: budget exhausted before a probe\n"
            "SCHED-WAIVED: budget exhausted\nLIVE-WAIVED: budget exhausted")
        (tmp_path / "candidates.md").write_text("| cand A | cand B |")
        return "artifacts written"

    # a request-less turn with no artifacts is re-prompted, not finished, so the
    # write-up guard is reached by exhausting the round budget
    teacher = FakeBackend(["nothing further from me", write_up], [])
    recon = FakeBackend([], [])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    ctx.recon_path = tmp_path / "recon-report.md"
    ctx.max_recon_rounds = 1
    result = run_recon_step(ctx)
    assert result.finished is False
    assert (tmp_path / "candidates.md").exists()
    # write-up costs agent turns, never jobs (seeding the toolkit is the one commit)
    assert not [c for c in recon.calls if c[0] == "gpu"]


def test_recon_finish_rejects_stale_artifacts_from_a_previous_run(tmp_path):
    """The default recon path ships with a populated report in-tree (a previous
    operator's), so existence alone must not let a request-less turn finish
    reconnaissance — the write-up has to be (re)written this session."""
    from forgekernel.loop.driver import run_recon_step

    # stale residue: written BEFORE the recon session starts, waivers included —
    # under the old existence-only gate this satisfied a finish outright
    (tmp_path / "recon-report.md").write_text(
        "previous operator's observations\nSASS-WAIVED: stale\n"
        "SCHED-WAIVED: stale\nLIVE-WAIVED: stale")
    (tmp_path / "candidates.md").write_text("stale A|B")

    def refresh(prompt):  # second turn: the teacher re-affirms the write-up
        (tmp_path / "recon-report.md").write_text(
            "this operator's observations\nSASS-WAIVED: fresh\n"
            "SCHED-WAIVED: fresh\nLIVE-WAIVED: fresh")
        (tmp_path / "candidates.md").write_text("fresh A|B")
        return "report refreshed, nothing further"

    teacher = FakeBackend(["nothing further from me", refresh], [])
    recon = FakeBackend([], [])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    ctx.recon_path = tmp_path / "recon-report.md"
    ctx.max_recon_rounds = 3
    result = run_recon_step(ctx)
    # turn 1 (request-less, stale artifacts) was re-prompted, not accepted
    assert result.finished is True and result.rounds == 1
    assert "did not end in a legal way" in teacher.prompts[1]
    assert "fresh" in (tmp_path / "recon-report.md").read_text()


def test_recon_artifacts_written_freshness_floor(tmp_path):
    from types import SimpleNamespace
    from forgekernel.loop.driver import _recon_artifacts_written
    ctx = SimpleNamespace(recon_path=tmp_path / "report.md")
    assert not _recon_artifacts_written(ctx)            # missing files
    ctx.recon_path.write_text("r")
    (tmp_path / "candidates.md").write_text("c")
    assert _recon_artifacts_written(ctx)                # exist, no floor asked
    written_at = ctx.recon_path.stat().st_mtime
    assert _recon_artifacts_written(ctx, since=written_at - 1)
    assert not _recon_artifacts_written(ctx, since=written_at + 1)  # stale


def test_a_truncated_turn_is_not_a_finish(tmp_path):
    """Observed live: a teacher ended a turn with "Writing probe3:" — no request, no
    artifacts — and the loop read it as a deliberate finish, forfeiting 10 of 12
    budget rounds. A request-less turn only finishes once both artifacts exist;
    otherwise it is re-prompted and the reconnaissance continues."""
    from forgekernel.loop.driver import run_recon_step

    def finish(prompt):
        (tmp_path / "recon-report.md").write_text("observations [MEASURED, job 1]")
        (tmp_path / "candidates.md").write_text("| cand A | cand B |")
        return "report and candidates written, nothing further"

    teacher = FakeBackend(["Writing probe3:",              # truncated — no marker
                           "RECON_REQUEST: notes/probe3.py",
                           finish], [])
    recon = FakeBackend([], ["SASS-OK kernel=fa_fwd instructions=2615 opcodes=118\n"
                             "SCHED-OK loop=backedge hgmma_runs=5 depbars=5 ew_runs=4\n"
                             "LIVE-OK peak=232 guessed_frac=0.02"])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "probe3.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    # the truncated turn burned a budget round but no GPU job, and the probe it was
    # about to request still ran
    assert "did not end in a legal way" in teacher.prompts[1]
    assert ("gpu", "cmd:python3 notes/probe3.py") in recon.calls


def test_recon_seeds_sass_tools_and_persists_job_logs(tmp_path):
    """The toolkit rides the committed tree — delivery is ``git archive`` — so no
    probe hand-rolls a parser again; and every probe's fed-back output lands in
    ``notes/`` instead of living only in the job logs."""
    from forgekernel.loop.driver import run_recon_step
    def done(prompt):  # write-up happens this session; the SASS-OK log opened the gate
        (tmp_path / "recon-report.md").write_text("observations")  # deliberately no waiver
        (tmp_path / "candidates.md").write_text("A|B")
        return "done"

    teacher = FakeBackend(["RECON_REQUEST: notes/sass_probe.py", done], [])
    recon = FakeBackend([], ["SASS-OK kernel=fa_fwd instructions=2615 opcodes=118\n"
                             "SCHED-OK loop=backedge hgmma_runs=5 depbars=5 ew_runs=4\n"
                             "LIVE-OK peak=232 guessed_frac=0.02"])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "sass_probe.py").write_text("import sass_tools")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True                      # the SASS-OK log opened the gate
    seeded = tmp_path / "notes" / "sass_tools.py"
    assert seeded.exists() and "def analyze" in seeded.read_text()
    assert ("commit", "recon: vendo") in recon.calls    # seeded file was committed
    saved = tmp_path / "notes" / "sass_probe.run0.log"
    assert saved.read_text().startswith("SASS-OK")


def test_recon_finish_without_sass_evidence_is_bounced_to_a_choice(tmp_path):
    """Deliver or waive — never silence. A past run finished with the opcode histogram
    missing while its own report named the single round that would have closed
    four open gaps; nothing objected. Now the finish bounces until either a job
    log carries the tool's SASS-OK verdict or the report owns the absence."""
    from forgekernel.loop.driver import run_recon_step

    def fresh_writeup(prompt):  # artifacts written this session, but no evidence
        (tmp_path / "recon-report.md").write_text("observations")
        (tmp_path / "candidates.md").write_text("A|B")
        return "nothing further from me"

    def waive(prompt):
        assert "SASS evidence missing" in prompt and "SASS-WAIVED" in prompt
        report = tmp_path / "recon-report.md"
        report.write_text(report.read_text()
                          + "\nSASS-WAIVED: ncu permission denied, no cubin on the static channel"
                          + "\nSCHED-WAIVED: no SASS page to transcribe\nLIVE-WAIVED: no SASS page to analyze")
        return "waived, nothing further"

    teacher = FakeBackend([fresh_writeup, waive], [])
    recon = FakeBackend([], [])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    assert not [c for c in recon.calls if c[0] == "gpu"]


def test_sass_ok_regex_accepts_demangled_kernel_names_with_spaces():
    """Regression (sparse recon false-negative): the tool's genuine verdict line carries a
    demangled C++ template kernel name WITH SPACES, and `kernel=\\S+` stops at the
    first one so the full-field anchor never matches. The teacher delivered, the
    driver saw nothing, and the deadlock burned every write-up turn. The fixture
    is the real line from that run's job log, verbatim."""
    from forgekernel.loop.driver import _SASS_OK, _STALL_OK
    real = ("SASS-OK kernel=void sparse_attn_fwd_kernel<KernelTemplate<512, 1>, "
            "KernelTemplate<512, 1>::TmaP instructions=3520 opcodes=108")
    assert _SASS_OK.search(real) is not None
    assert _STALL_OK.search("STALL-OK kernel=void foo<Bar<1, 2>>::baz "
                            "samples=172541 rows=47 coverage=90.3") is not None
    # anti-cheat force retained: prose without the numeric tail stays shut out
    assert _SASS_OK.search("SASS-OK kernel=whatever, trust me") is None
    assert _SASS_OK.search("the run printed SASS-OK kernel=x "
                           "instructions=3 opcodes=1") is None  # not line-start


def test_verdict_near_miss_is_surfaced_and_mismatch_line_closes_the_gate(tmp_path):
    """A verdict-shaped line that fails the full-field pattern is an
    acceptance-format mismatch, not a missing delivery. The write-up prompt must
    show BOTH sides of the disagreement (the actual line, verbatim) and offer the
    third legal state — `SASS-DELIVERED-MISMATCH:` — which closes the gate the
    way a waiver would, while reading as a checker defect for the human audit."""
    from forgekernel.loop.driver import run_recon_step

    seen = {}

    def first_writeup(prompt):
        (tmp_path / "recon-report.md").write_text("observations")
        (tmp_path / "candidates.md").write_text("A|B")
        return "report written, nothing further"

    def own_mismatch(prompt):
        seen["prompt"] = prompt
        assert "SASS reconciliation" in prompt
        assert "SASS-OK kernel=futureformat!" in prompt      # the near-miss, verbatim
        assert "SASS-DELIVERED-MISMATCH" in prompt
        report = tmp_path / "recon-report.md"
        report.write_text(report.read_text()
                          + "\nSASS-DELIVERED-MISMATCH: SASS-OK kernel=futureformat! n=3520"
                          + "\nSCHED-WAIVED: not collected\nLIVE-WAIVED: not collected")
        return "owned, nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/p.py", first_writeup,
                           own_mismatch], [])
    # starts like a verdict, fails the full-field pattern (no numeric tail shape)
    recon = FakeBackend([], ["SASS-OK kernel=futureformat! n=3520"])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    assert "prompt" in seen, "the near-miss write-up turn never happened"


def test_recon_never_runs_through_the_student_tree(tmp_path):
    """Reconnaissance output is exactly what the student must not see, and the student's
    job receives its operator tree as a tarball. Submitting through that backend would
    put the analysis in the student's job."""
    from forgekernel.loop.driver import run_recon_step
    teacher = FakeBackend(["RECON_REQUEST: notes/x.py", "done"], [])
    recon = FakeBackend([], ["ok"])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    run_recon_step(ctx)
    assert not [c for c in teacher.calls if c[0] == "gpu"], "recon leaked into the student tree"


def test_recon_without_a_backend_degrades_rather_than_crashing(tmp_path):
    from forgekernel.loop.driver import run_recon_step
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_backend = None
    assert run_recon_step(ctx).finished is False


def test_blackbox_mode_gets_its_own_distill_brief(tmp_path):
    """The source-mode prompt is left untouched so a control run runs the instructions
    it would have run without this experiment existing."""
    from forgekernel.loop.driver import _distill_prompt_path
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.oracle_mode = "source"
    assert _distill_prompt_path(ctx).name == "distill_prompt.md"
    ctx.oracle_mode = "blackbox"
    assert _distill_prompt_path(ctx).name == "distill_blackbox_prompt.md"


# --- forced M0 audit (the gate verifies the student; the audit verifies the -----
# --- milestone against the hardware truth) --------------------------------------

def test_inner_loop_stops_at_first_green_when_auditing(tmp_path):
    be = FakeBackend([REC("D1", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    assert run_inner_loop(ctx, 0, initial_green=None, audit_on_green=True) == "first_green"


def test_inner_loop_target_outranks_the_audit_stop(tmp_path):
    # a kernel already at target has nothing left to audit for
    be = FakeBackend([REC("D1", 1.8)], [EVAL(1.8), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    assert run_inner_loop(ctx, 0, initial_green=None, audit_on_green=True) == "target"


def test_inner_loop_already_green_segment_never_audits(tmp_path):
    # audit_on_green only means anything for a segment that ENTERS non-green
    be = FakeBackend([REC("D1", 1.8)], [EVAL(1.8), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    assert run_inner_loop(ctx, 0, initial_green=True, audit_on_green=True) == "target"


def test_m0_audit_clean_resumes_without_spending_meta_budget(tmp_path):
    from forgekernel.meta_trajectory import load_meta_records
    agent = [REC("A", 2.66),                                # M0 gate lands -> first_green
             "cause_class: OK\npatch_direction: M0 signature reconciles with the reference",  # audit
             REC("B", 1.8)]                                 # resumed segment hits target
    gpu = [EVAL(2.66), NCU, EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.m0_audit = True
    summary = run_two_layer(ctx, max_meta=2)
    assert summary["stop"] == "target"
    assert summary["meta_steps"] == 0                       # a clean audit is free
    recs = load_meta_records(ctx.meta_status_path)
    assert [r.outcome for r in recs] == ["AUDIT_CLEAN"]
    # the audit consumed the m0_audit prompt, not the plateau-diagnose prompt
    assert any("Mandatory M0 finish audit" in p for p in be.prompts)


def test_m0_audit_defect_patches_and_the_rebuild_is_audited_again(tmp_path):
    from forgekernel.meta_trajectory import load_meta_records
    agent = [REC("A", 2.66),                                # M0 gate -> first_green
             "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: pin the atom",
             _patching(tmp_path, "\n### D1.1 atom pinned\nx\n",
                       "patch_class: structural\nchanged_entries: []\n"),
             REC("B", 2.0),                                 # rebuild goes green again
             "cause_class: OK\npatch_direction: reconciled",   # rebuild audited too
             REC("C", 1.8)]                                 # then hits target
    gpu = [EVAL(2.66), NCU, EVAL(2.0), NCU, EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.m0_audit = True
    summary = run_two_layer(ctx, max_meta=2)
    assert summary["stop"] == "target"
    assert summary["meta_steps"] == 1                       # only the patch costs budget
    outcomes = [r.outcome for r in load_meta_records(ctx.meta_status_path)]
    # the patch is judged at the target stop (1.8 beat 2.66), the audits stay terminal
    assert outcomes == ["KEEP_PATCH", "AUDIT_CLEAN"]
    assert any(c[0] == "reset" for c in be.calls)           # the patch reset happened


def test_m0_audit_fires_on_the_m0_gate_of_an_in_place_reopen(tmp_path):
    """A foundation patch under in-place reopening keeps the best (green)
    tree, so the green-flip trigger cannot fire; the rebuilt foundation's M0
    close is the segment's first M0-gate KEEP, and the audit keys on that row
    — one audit per reopen, then the segment resumes."""
    from forgekernel.meta_trajectory import load_meta_records
    dag = [Decision(id="D0.1"), Decision(id="D0.2", depends_on=["D0.1"]),
           Decision(id="D1.1", depends_on=["D0.2"])]
    agent = [REC("D0.1", 2.66),                             # M0 lands -> first_green
             "cause_class: R1\ndeficient_locus: D0.2\npatch_direction: pin the atom",
             _patching(tmp_path, "\n### D0.2 atom pinned\nx\n",
                       "patch_class: structural\nchanged_entries: [D0.2]\n"),
             REC("D0.2", 2.60),                             # redone on the best tree
             REC("M0-gate", 2.55),                          # the reopened M0's close
             "cause_class: OK\npatch_direction: reconciled",   # audited HERE
             REC("D1.1", 1.8)]                              # then hits target
    gpu = [EVAL(2.66), NCU, EVAL(2.60), NCU, EVAL(2.55), NCU, EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9, dag=dag)
    ctx.m0_audit = True
    summary = run_two_layer(ctx, max_meta=2)
    assert summary["stop"] == "target"
    assert summary["meta_steps"] == 1
    rows = load_records(ctx.status_path)
    assert [r.decision_id for r in rows] == ["D0.1", "REOPEN", "D0.2", "M0-gate", "D1.1"]
    outcomes = [r.outcome for r in load_meta_records(ctx.meta_status_path)]
    assert outcomes == ["KEEP_PATCH", "AUDIT_CLEAN"]
    # the second audit prompt was rendered after the M0-gate row, not at a plateau
    audits = [p for p in be.prompts if "Mandatory M0 finish audit" in p]
    assert len(audits) == 2


def test_reopen_above_m0_does_not_audit(tmp_path):
    """A reopen whose closure stays above the foundation leaves M0 as built:
    nothing for the M0 audit to read, so the segment runs to its next stop."""
    dag = [Decision(id="D0.1"), Decision(id="D1.1", depends_on=["D0.1"]),
           Decision(id="D2.1", depends_on=["D1.1"])]
    agent = [REC("D0.1", 2.66),
             "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: ring depth",
             _patching(tmp_path, "\n### D1.1 revised\nx\n",
                       "patch_class: structural\nchanged_entries: [D1.1]\n"),
             REC("D1.1", 2.5), REC("M0-gate", 2.45), REC("D2.1", 1.8)]
    gpu = [EVAL(2.66), NCU, EVAL(2.5), NCU, EVAL(2.45), NCU, EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9, dag=dag)
    ctx.m0_audit = True
    assert run_two_layer(ctx, max_meta=2)["stop"] == "target"
    assert len([p for p in be.prompts if "Mandatory M0 finish audit" in p]) == 1


def test_m0_audit_patch_is_not_judged_at_the_rebuilds_first_green(tmp_path):
    """The transaction must survive the audit interleave: at the rebuild's first green
    best_after (2.9) is still worse than best_before (2.66) — judging there would
    revert a correct foundation for running out of runway. The judgment belongs to the
    next actionable stop, after the rebuild's segment has actually run."""
    from forgekernel.meta_trajectory import load_meta_records
    agent = [REC("A", 2.66),
             "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: pin the atom",
             _patching(tmp_path, "\n### D1.1 atom pinned\nx\n",
                       "patch_class: structural\nchanged_entries: []\n"),
             REC("B", 2.9),                                 # rebuild green, WORSE than 2.66
             "cause_class: OK\npatch_direction: reconciled",
             REC("C", 1.8)]                                 # ...then beats target
    gpu = [EVAL(2.66), NCU, EVAL(2.9), NCU, EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.m0_audit = True
    summary = run_two_layer(ctx, max_meta=2)
    assert summary["stop"] == "target"
    outcomes = [r.outcome for r in load_meta_records(ctx.meta_status_path)]
    assert outcomes == ["KEEP_PATCH", "AUDIT_CLEAN"]        # NOT reverted at first green


def test_m0_audit_skipped_when_meta_budget_is_exhausted(tmp_path):
    from forgekernel.meta_trajectory import load_meta_records
    agent = [REC("A", 2.66), REC("B", 1.8)]                 # no audit reply queued
    gpu = [EVAL(2.66), NCU, EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.m0_audit = True
    summary = run_two_layer(ctx, max_meta=0)
    assert summary["stop"] == "target"
    assert load_meta_records(ctx.meta_status_path) == []


def test_recon_finish_without_sched_live_evidence_is_bounced_to_a_choice(tmp_path):
    """The schedule/liveness pair gets the same deliver-or-waive gate as SASS.
    Observed on a bwd run: every counting fact was delivered and the milestone still could
    not name which chain flies over which elementwise segment, or its register
    price — so a finish with SASS-OK but neither SCHED nor LIVE evidence must
    bounce, naming both waiver spellings."""
    from forgekernel.loop.driver import run_recon_step

    def fresh_writeup(prompt):  # artifacts written this session; SCHED/LIVE still silent
        (tmp_path / "recon-report.md").write_text("observations [MEASURED, job 1]")
        (tmp_path / "candidates.md").write_text("A|B")
        return "nothing further from me"

    def waive(prompt):
        assert "SCHED/LIVE evidence missing" in prompt
        assert "SCHED-WAIVED" in prompt and "LIVE-WAIVED" in prompt
        report = tmp_path / "recon-report.md"
        report.write_text(report.read_text()
                          + "\nSCHED-WAIVED: static page out of order, cannot locate the loop body"
                          + "\nLIVE-WAIVED: same as above")
        return "waived, nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/sass_probe.py",
                           fresh_writeup, waive], [])
    recon = FakeBackend([], ["SASS-OK kernel=fa_bwd instructions=2615 opcodes=118"])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "sass_probe.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True


# --- resume points inside a scaffolded lineage ------------------------------------
# Observed on a self-ref attn_bwd run: two M0 audits in a row. The first (R1 @ D0.1) rebuilt
# from the baseline; the second (G4 @ D0.10) joined the open transaction and, on that
# run's harness (no in-place reopen yet), prefix-reused back to D0.9's checkpoint — a
# SCAFFOLD-NEUTRAL commit inside the M0 construction. _green_at_ref said False, the
# closure was redone in scaffold mode, re-gated, and audited a third time. Reading that
# as "the segment lost the lineage's green" and forcing green from the lineage evaluated
# half-built sub-steps as regressions against an unverified anchor and silently dropped
# the rebuild's M0 audit (initial_green=True never fires audit_on_green). The tests here
# pin the contracts that reading broke, and the two things that were actually wrong.

_AUDIT_OK = "cause_class: OK\npatch_direction: reconciled"
_M0_DAG = [Decision(id="D0.1"), Decision(id="D0.2", depends_on=["D0.1"]),
           Decision(id="D0.3", depends_on=["D0.2"])]


def test_parse_list_strips_the_quotes_teachers_put_around_ids():
    from forgekernel.loop.driver import _parse_list
    out = _parse_list('patch_class: structural\nchanged_entries: ["D0.1", \'D0.3\', `D1.1`, D0.gate, ]\n',
                      "changed_entries")
    assert out == ["D0.1", "D0.3", "D1.1", "D0.gate"]


def test_quoted_changed_entries_still_plan_the_foundation_restart(tmp_path):
    """Observed (meta step 1): every id in the R1 patch was quoted, so none matched the DAG —
    the plan came out as "resume after the tip, nothing invalidated, nothing reopened",
    and the rebuild only restarted from the baseline because no record owned that tip."""
    dag = [Decision(id="D0.1"), Decision(id="D0.2", depends_on=["D0.1"])]
    patch = _patching(tmp_path, "\n### D0.1 pinned\nx\n",
                      'patch_class: structural\nchanged_entries: ["D0.1"]\n')
    be = FakeBackend(["cause_class: R1\ndeficient_locus: D0.1", patch], [])
    ctx = _ctx(tmp_path, be, dag=dag)
    res = run_meta_step(ctx)
    assert res.changed_entries == ["D0.1"]
    assert res.resume_after is None and res.invalidated == ["D0.1", "D0.2"]
    assert res.reopened == ["D0.1", "D0.2"]


def test_a_foundation_rebuild_after_a_plateau_patch_scaffolds_and_is_audited(tmp_path):
    """A regular-path foundation patch restarts from the baseline with an UNKNOWN
    lineage: the first FAIL enters scaffold mode (half-built sub-steps are committed,
    not evaluated, and nothing is reset), and the rebuild's first green is an M0 close,
    audited like the first one. The lineage having been green before the patch does not
    make the rebuilt tree green."""
    from forgekernel.meta_trajectory import load_meta_records
    agent = [REC("M0-gate", 2.66), _AUDIT_OK, "EXHAUSTED: no more decisions",
             "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: pin the atom",
             _patching(tmp_path, "\n### D1.1 atom pinned\nx\n",
                       "patch_class: structural\nchanged_entries: []\n"),
             REC("D0.1", None), REC("D0.2", None), REC("M0-gate", 2.0), _AUDIT_OK,
             REC("D1.1", 1.8)]
    gpu = [EVAL(2.66), NCU, EVAL(0.0, ok=False), NCU, EVAL(2.0), NCU, EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.m0_audit = True
    summary = run_two_layer(ctx, max_meta=3)
    assert summary["stop"] == "target"
    assert [r.verdict for r in load_records(ctx.status_path)] == \
        ["KEEP", "REVERT", "NEUTRAL", "KEEP", "KEEP"]
    assert sum("Mandatory M0 finish audit" in p for p in be.prompts) == 2
    assert [c[1] for c in be.calls if c[0] == "reset"] == ["sha0"]      # the restart only
    assert [m.outcome for m in load_meta_records(ctx.meta_status_path)] == \
        ["AUDIT_CLEAN", "KEEP_PATCH", "AUDIT_CLEAN"]


def _two_audits(tmp_path, tail):
    """scaffold ladder -> M0 gate -> audit R1 (rebuild from the baseline) -> ladder ->
    gate -> audit G4 @ D0.3 (structural; joins the open transaction) -> ``tail``"""
    return [REC("D0.1", None), REC("D0.2", None), REC("D0.3", None), REC("M0-gate", 2.66),
            "cause_class: R1\ndeficient_locus: D0.1\npatch_direction: pin the atom",
            _patching(tmp_path, "\n### D0.1 atom pinned\nx\n",
                      "patch_class: structural\nchanged_entries: []\n"),
            REC("D0.1", None), REC("D0.2", None), REC("D0.3", None), REC("M0-gate", 2.5),
            "cause_class: G4\ndeficient_locus: D0.3\npatch_direction: fix the flush",
            _patching(tmp_path, "\n### D0.3 flush fixed\nx\n",
                      "patch_class: structural\nchanged_entries: [D0.3]\n")] + tail


def test_a_joined_audit_patch_under_prefix_reuse_resumes_in_scaffold_and_is_regated(tmp_path):
    """reopen_in_place off: the resume point is D0.2's checkpoint — a NEUTRAL commit
    inside the construction that never passed the gate. Green there would be a lie:
    the closure is redone in scaffold mode and the M0 re-close is audited again."""
    agent = _two_audits(tmp_path, [REC("D0.3", None), REC("M0-gate", 2.4), _AUDIT_OK,
                                   REC("D1.1", 1.8)])
    gpu = [EVAL(0.0, ok=False), NCU, EVAL(2.66), NCU, EVAL(0.0, ok=False), NCU,
           EVAL(2.5), NCU, EVAL(2.4), NCU, EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9, dag=_M0_DAG)
    ctx.m0_audit = True
    ctx.reopen_in_place = False
    summary = run_two_layer(ctx, max_meta=3)
    assert summary["stop"] == "target"
    rows = load_records(ctx.status_path)
    assert [(r.decision_id, r.verdict) for r in rows[8:]] == \
        [("D0.3", "NEUTRAL"), ("M0-gate", "KEEP"), ("D1.1", "KEEP")]
    assert sum("Mandatory M0 finish audit" in p for p in be.prompts) == 3
    assert [c[1] for c in be.calls if c[0] == "reset"] == ["sha0", rows[5].sha]  # D0.2's checkpoint


def test_a_joined_audit_patch_reopens_in_place_and_its_m0_close_is_audited(tmp_path):
    """The same two audits on the shipped harness (reopen_in_place on): the G4 patch's
    closure is redone ON the green best tree — the segment enters green, the redo is
    evaluated, and the M0 re-close is audited through audit_on_gate. The run that showed
    the closure redone pre-green ran a harness from before in-place reopening."""
    from forgekernel.meta_trajectory import load_meta_records
    agent = _two_audits(tmp_path, [REC("D0.3", 2.45), REC("M0-gate", 2.4), _AUDIT_OK,
                                   REC("D1.1", 1.8)])
    gpu = [EVAL(0.0, ok=False), NCU, EVAL(2.66), NCU, EVAL(0.0, ok=False), NCU,
           EVAL(2.5), NCU, EVAL(2.45), NCU, EVAL(2.4), NCU, EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9, dag=_M0_DAG)
    ctx.m0_audit = True
    summary = run_two_layer(ctx, max_meta=3)
    assert summary["stop"] == "target"
    rows = load_records(ctx.status_path)
    assert [(r.decision_id, r.verdict) for r in rows[8:]] == \
        [("REOPEN", "NEUTRAL"), ("D0.3", "KEEP"), ("M0-gate", "KEEP"), ("D1.1", "KEEP")]
    assert sum("Mandatory M0 finish audit" in p for p in be.prompts) == 3
    assert [c[1] for c in be.calls if c[0] == "reset"] == ["sha0", rows[7].sha]  # the green best
    assert [m.outcome for m in load_meta_records(ctx.meta_status_path)][-1] == "AUDIT_CLEAN"


def test_recon_completion_marker_survives_a_restart(tmp_path):
    """A finished reconnaissance is a fact about the experiment, not the process.

    Observed live (bwd-mha): a watchdog restart re-entered reconnaissance, the
    freshness gate classified the previous session's finished report as stale
    residue, and all 12 rounds burned on re-affirmation turns before the WARNING
    path let distillation proceed anyway. The completion marker makes the second
    session skip straight to distillation without a single teacher turn."""
    from forgekernel.loop.driver import run_recon_step

    def write_up(prompt):
        (tmp_path / "recon-report.md").write_text(
            "observations\nSASS-WAIVED: no SASS page in this scenario\n"
            "SCHED-WAIVED: same\nLIVE-WAIVED: same")
        (tmp_path / "candidates.md").write_text("A|B")
        return "report written, nothing further"

    teacher = FakeBackend([write_up], [])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = FakeBackend([], [])
    ctx.recon_dir = tmp_path
    ctx.recon_path = tmp_path / "recon-report.md"
    assert run_recon_step(ctx).finished is True
    assert (tmp_path / "recon_complete.json").is_file()

    # restart: a teacher backend with no scripted replies would raise on any turn,
    # and a recon backend that is a bare object would raise on any use — reaching
    # either means the marker was not honored
    ctx2 = _ctx(tmp_path, FakeBackend([], []))
    ctx2.recon_backend = object()
    ctx2.recon_dir = tmp_path
    ctx2.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx2)
    assert result.finished is True and result.report == ctx2.recon_path


def test_recon_completion_marker_is_distrusted_when_artifacts_change(tmp_path):
    """The marker vouches for specific bytes, not for the phase in the abstract:
    artifacts missing, empty, older than the marker (swapped in from elsewhere),
    or a marker naming a foreign tree all mean reconnaissance runs again — the
    pre-fix behavior, never a wrongly-skipped phase."""
    import os

    from forgekernel.loop.driver import _load_recon_state, _save_recon_state

    report = tmp_path / "recon-report.md"
    candidates = tmp_path / "candidates.md"
    report.write_text("observations")
    candidates.write_text("A|B")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_path = report
    _save_recon_state(ctx, rounds=12)
    good = _load_recon_state(ctx)
    assert good is not None and good.rounds == 12 and good.finished is True

    # an artifact older than the marker recorded was swapped in underneath it
    old = os.path.getmtime(report) - 30
    os.utime(report, (old, old))
    assert _load_recon_state(ctx) is None
    _save_recon_state(ctx, rounds=12)  # re-affirm against the current bytes

    candidates.write_text("")  # empty artifact vouches for nothing
    assert _load_recon_state(ctx) is None
    candidates.write_text("A|B")
    _save_recon_state(ctx, rounds=12)

    marker = tmp_path / "recon_complete.json"
    state = marker.read_text().replace(str(report), str(tmp_path / "other.md"))
    marker.write_text(state)  # a foreign tree's marker
    assert _load_recon_state(ctx) is None

    marker.write_text("{not json")
    assert _load_recon_state(ctx) is None

    marker.unlink()
    report.unlink()
    assert _load_recon_state(ctx) is None


def test_recon_marker_saved_when_budget_exhausts_but_write_up_stands(tmp_path):
    """The exhausted path persists completion too — that is exactly the path the
    observed session took (12 runs, write-up landed on a guard turn), and the one the
    restart then re-litigated. Incompleteness (the WARNING branch) must NOT be
    persisted: a fresh session's turns are the only remaining chance to finish."""
    from forgekernel.loop.driver import run_recon_step

    def write_up(prompt):
        (tmp_path / "recon-report.md").write_text(
            "observations\nSASS-WAIVED: budget spent\n"
            "SCHED-WAIVED: same\nLIVE-WAIVED: same")
        (tmp_path / "candidates.md").write_text("A|B")
        return "written"

    teacher = FakeBackend(["RECON_REQUEST: notes/p.py", write_up], [])
    recon = FakeBackend([], ["probe output"])
    ctx = _ctx(tmp_path, teacher)
    ctx.max_recon_rounds = 1
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is False  # the budget did exhaust...
    assert (tmp_path / "recon_complete.json").is_file()  # ...but completion persisted


# --- kb-tripwire: expensive probe victories owe a KB write-up ----------------

def test_kb_tripwire_battle_accumulates_across_iterations(tmp_path):
    # the fight spans two iterations: 2 FAIL rounds in the first, the PASS in the
    # second. The bill is per probe *path*, so the battle reads 3 rounds.
    from forgekernel.loop.driver import kb_harvest_debt
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D2.3", 2.6),
         "PROBE_REQUEST: notes/p.py", REC("D2.4", 2.5)],
        ["PROBE FAIL max_diff=0.5", "PROBE FAIL max_diff=0.5", EVAL(2.6), NCU,
         "PROBE PASS max_diff=0.0", EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    ctx.kb_tripwire_rounds = 3
    run_inner_iteration(ctx, 0, metric_before=3.0)
    run_inner_iteration(ctx, 1, metric_before=2.6)
    battle = ctx.probe_battles["notes/p.py"]
    assert battle.rounds == 3 and battle.passed and battle.pass_iter == 1
    assert battle.verdict_line.startswith("PROBE PASS")
    assert kb_harvest_debt(ctx) == [battle]


def test_kb_tripwire_cheap_or_lost_fights_owe_nothing(tmp_path):
    from forgekernel.loop.driver import ProbeBattle, kb_harvest_debt
    be = FakeBackend(
        ["PROBE_REQUEST: notes/cheap.py", REC("D2.3", 2.6)],
        ["PROBE PASS max_diff=0.0", EVAL(2.6), NCU],
    )
    ctx = _ctx(tmp_path, be)
    ctx.kb_tripwire_rounds = 3
    run_inner_iteration(ctx, 0, metric_before=3.0)
    assert ctx.probe_battles["notes/cheap.py"].rounds == 1
    # a fight the student never won has no PASS to cite as evidence
    ctx.probe_battles["notes/lost.py"] = ProbeBattle(path="notes/lost.py", rounds=9)
    assert kb_harvest_debt(ctx) == []


def test_kb_tripwire_disabled_by_zero(tmp_path):
    from forgekernel.loop.driver import ProbeBattle, kb_harvest_debt, run_kb_harvest
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.kb_tripwire_rounds = 0
    ctx.probe_battles["notes/p.py"] = ProbeBattle(
        path="notes/p.py", rounds=12, passed=True, pass_iter=3)
    assert kb_harvest_debt(ctx) == []
    assert run_kb_harvest(ctx) == []
    assert be.calls == []  # no agent turn, no GPU


def test_kb_harvest_one_turn_no_gpu_parses_recipes(tmp_path):
    from forgekernel.loop.driver import ProbeBattle, run_kb_harvest
    be = FakeBackend(
        ["wrote two recipes.\nRECIPE_RECORDED: hw_kb/cutedsl/sm90/foo.md\n"
         "RECIPE_RECORDED: hw_kb/cutedsl/sm90/bar.md"],
        [],
    )
    ctx = _ctx(tmp_path, be)
    ctx.kb_tripwire_rounds = 6
    ctx.probe_battles["notes/hard.py"] = ProbeBattle(
        path="notes/hard.py", rounds=10, passed=True, pass_iter=17,
        verdict_line="PROBE PASS (max_diff=0.0)", log_tail="note: rung passed")
    ctx.probe_battles["notes/cheap.py"] = ProbeBattle(
        path="notes/cheap.py", rounds=1, passed=True, pass_iter=2)
    recipes = run_kb_harvest(ctx)
    assert recipes == ["hw_kb/cutedsl/sm90/foo.md", "hw_kb/cutedsl/sm90/bar.md"]
    assert [c for c in be.calls if c[0] == "gpu"] == []  # zero GPU jobs
    assert len([c for c in be.calls if c[0] == "agent"]) == 1  # one write-up turn
    prompt = be.prompts[0]
    assert "notes/hard.py" in prompt and "10 GPU rounds" in prompt
    assert "notes/cheap.py" not in prompt  # below threshold: not billed


def test_kb_harvest_agent_error_degrades_to_status_quo(tmp_path):
    from forgekernel.loop.driver import AgentError, ProbeBattle, run_kb_harvest

    class DeadAgentBackend(FakeBackend):
        def run_agent(self, prompt: str) -> str:
            raise AgentError("agent died")

    be = DeadAgentBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.kb_tripwire_rounds = 2
    ctx.probe_battles["notes/hard.py"] = ProbeBattle(
        path="notes/hard.py", rounds=8, passed=True, pass_iter=4)
    assert run_kb_harvest(ctx) == []  # lessons stay in the notes; no crash


def test_kb_tripwire_cheap_win_plus_reprobes_is_not_a_phantom_epic(tmp_path):
    # a 1-round win, then a later 2-round re-fight on the same path: the bill is
    # the dearest single fight (2), never the 3-round sum — the harvest prompt
    # must not report a battle that never happened.
    from forgekernel.loop.driver import kb_harvest_debt
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", REC("D2.3", 2.6),
         "PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D2.4", 2.5)],
        ["PROBE PASS max_diff=0.0", EVAL(2.6), NCU,
         "PROBE FAIL max_diff=0.5", "PROBE PASS max_diff=0.0", EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    ctx.kb_tripwire_rounds = 3
    run_inner_iteration(ctx, 0, metric_before=3.0)
    run_inner_iteration(ctx, 1, metric_before=2.6)
    battle = ctx.probe_battles["notes/p.py"]
    assert battle.rounds == 2  # dearest fight, not 1+2 summed
    assert kb_harvest_debt(ctx) == []  # 2 < 3: no debt


def test_kb_tripwire_rewon_fight_refreshes_the_evidence_whole(tmp_path):
    # the same path is fought and won twice at equal cost: the bill must cite the
    # LATER fight's verdict, iter and log tail — rounds and evidence travel
    # together, never a mix of two fights.
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D2.3", 2.6),
         "PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D2.4", 2.5)],
        ["PROBE FAIL max_diff=0.9", "PROBE PASS max_diff=0.25", EVAL(2.6), NCU,
         "PROBE FAIL max_diff=0.9", "PROBE PASS max_diff=0.125", EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    ctx.kb_tripwire_rounds = 2
    run_inner_iteration(ctx, 0, metric_before=3.0)
    run_inner_iteration(ctx, 1, metric_before=2.6)
    battle = ctx.probe_battles["notes/p.py"]
    assert battle.rounds == 2 and battle.pass_iter == 1
    assert "0.125" in battle.verdict_line and "0.25" not in battle.verdict_line


def test_kb_harvest_missing_template_degrades_not_raises(tmp_path):
    # version skew: the prompt template is absent. The promise is total — the
    # caller sits before the harvest report and publish_kb, so this must return
    # [] instead of raising FileNotFoundError.
    from forgekernel.loop.driver import ProbeBattle, run_kb_harvest
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.meta_dir = tmp_path / "no-such-meta"  # kb_harvest_prompt.md cannot exist
    ctx.kb_tripwire_rounds = 2
    ctx.probe_battles["notes/hard.py"] = ProbeBattle(
        path="notes/hard.py", rounds=8, passed=True, pass_iter=4)
    assert run_kb_harvest(ctx) == []
    assert be.calls == []  # died before the agent turn, quietly


# --- HOLD_PATCH: a metric-less segment must not refute the patch ------------

def test_a_metricless_segment_holds_the_patch(tmp_path):
    # the post-patch segment is pure pre-green construction: scaffold NEUTRALs and a
    # correctness-FAIL gate attempt, zero measured KEEPs. Rolling that back would
    # convert "ran out of runway" into "never happened" (a failure mode seen in
    # earlier runs), so the verdict is HOLD_PATCH: milestone kept, workspace kept.
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    from forgekernel.loop.driver import _judge_patch, _PendingPatch
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    ms_text = ctx.milestone.read_text()
    append_record(ctx.status_path, IR(iter=0, decision_id="D0.1", change="scaffold",
                                      correctness=None, verdict="NEUTRAL", sha="s1"))
    append_record(ctx.status_path, IR(iter=1, decision_id="M0-gate", change="gate",
                                      correctness=False, verdict="REVERT",
                                      error="correctness FAIL"))
    out = _judge_patch(ctx, _PendingPatch(best_before=None,
                                          milestone_before="pre-patch snapshot",
                                          good_ref="good", records_before=0))
    assert out == "HOLD_PATCH"
    assert not any(c[0] == "reset" for c in be.calls)   # workspace untouched
    assert ctx.milestone.read_text() == ms_text          # milestone untouched


def test_an_empty_segment_holds_the_patch(tmp_path):
    # zero rows after the patch (instant EXHAUSTED) is the degenerate metric-less
    # segment: unattributable, held — not refuted.
    from forgekernel.loop.driver import _judge_patch, _PendingPatch
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    out = _judge_patch(ctx, _PendingPatch(best_before=2.66,
                                          milestone_before="snap", good_ref="good",
                                          records_before=0))
    assert out == "HOLD_PATCH"
    assert not any(c[0] == "reset" for c in be.calls)


# --- HOLD_PATCH: a declared multi-iter package in flight is construction ----

def _pkg_ctx(tmp_path):
    from forgekernel.loop.driver import _judge_patch, _PendingPatch
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    pending = _PendingPatch(best_before=2.66, milestone_before="snap",
                            good_ref="good", records_before=0)
    return ctx, be, pending, _judge_patch


def _pkg_row(i, change, verdict="KEEP", ms=4.4):
    from forgekernel.inner.trajectory import IterationRecord as IR
    measured = verdict == "KEEP"
    return IR(iter=i, decision_id=f"D0.{i}", change=change,
              correctness=measured or None if verdict != "REVERT" else False,
              verdict=verdict,
              bwd_median_ms=ms if measured else None,
              device_median_ms=ms if measured else None,
              sha=f"s{i}" if verdict != "REVERT" else "")


def test_a_declared_package_in_flight_holds_the_patch(tmp_path):
    # the segment's latest delivery is a measured-but-slower KEEP declaring
    # "part 1 of 3": an unfinished companion-group step. Its reading is a
    # construction fact, not a refutation — HOLD, nothing rolled back. Trailing
    # failed attempts at the next part (within the slack) do not void the plea.
    from forgekernel.inner.trajectory import append_record
    ctx, be, pending, judge = _pkg_ctx(tmp_path)
    append_record(ctx.status_path, _pkg_row(0, "D0.5 smem access atoms, part 1 of 3. sdS write side"))
    append_record(ctx.status_path, _pkg_row(1, "part 2 of 3 attempt", verdict="REVERT"))
    assert judge(ctx, pending) == "HOLD_PATCH"
    assert not any(c[0] == "reset" for c in be.calls)


def test_a_completed_package_with_no_gain_is_reverted(tmp_path):
    # "part 3 of 3" means the package is delivered: from here the ordinary law
    # applies, and a finished package that moved nothing is a refuted patch.
    from forgekernel.inner.trajectory import append_record
    ctx, be, pending, judge = _pkg_ctx(tmp_path)
    for k in (1, 2, 3):
        append_record(ctx.status_path, _pkg_row(k, f"exchange face, part {k} of 3"))
    assert judge(ctx, pending) == "REVERT_PATCH"
    assert ("reset", "good") in be.calls


def test_a_later_undeclared_keep_forfeits_the_package_exemption(tmp_path):
    # the student landed part 1 of 3, then moved on to unrelated work without
    # declaring: the package is abandoned, the plea lapses with it.
    from forgekernel.inner.trajectory import append_record
    ctx, be, pending, judge = _pkg_ctx(tmp_path)
    append_record(ctx.status_path, _pkg_row(0, "atoms, part 1 of 3"))
    append_record(ctx.status_path, _pkg_row(1, "unrelated elementwise refit"))
    assert judge(ctx, pending) == "REVERT_PATCH"
    assert ("reset", "good") in be.calls


def test_package_keep_budget_bounds_the_exemption(tmp_path):
    # "part k of n" grants at most n declared KEEP rows of grace. A fourth
    # declared row against n=3 is the budget spent: back to normal pricing —
    # re-declaring step 1 forever must not extend the plea.
    from forgekernel.inner.trajectory import append_record
    ctx, be, pending, judge = _pkg_ctx(tmp_path)
    for i in range(4):
        append_record(ctx.status_path, _pkg_row(i, "atoms again, part 1 of 3"))
    assert judge(ctx, pending) == "REVERT_PATCH"
    assert ("reset", "good") in be.calls


def test_package_cap_rejects_oversized_declarations(tmp_path):
    # a coupled group wider than the cap should be re-cut, not pleaded: the
    # declaration is rejected outright and the segment is priced normally.
    from forgekernel.inner.trajectory import append_record
    ctx, be, pending, judge = _pkg_ctx(tmp_path)
    append_record(ctx.status_path, _pkg_row(0, "grand rewrite, part 1 of 9"))
    assert judge(ctx, pending) == "REVERT_PATCH"
    assert ("reset", "good") in be.calls


def test_trailing_failures_beyond_the_slack_lapse_the_package(tmp_path):
    # four failed attempts since the last delivered step is past what the split
    # discipline tolerates: the package has stalled, the plea goes stale.
    from forgekernel.inner.trajectory import append_record
    ctx, be, pending, judge = _pkg_ctx(tmp_path)
    append_record(ctx.status_path, _pkg_row(0, "atoms, part 1 of 3"))
    for i in range(1, 5):
        append_record(ctx.status_path, _pkg_row(i, "part 2 of 3 attempt", verdict="REVERT"))
    assert judge(ctx, pending) == "REVERT_PATCH"
    assert ("reset", "good") in be.calls


def test_a_mention_outside_the_opening_sentence_is_not_a_declaration(tmp_path):
    # the contract is "the change's first sentence": a KEEP whose prose merely mentions a
    # part-of split deeper in (here: describing the ref's tiling) is ordinary
    # work, and an ordinary measured KEEP with no gain stays a refutation.
    from forgekernel.inner.trajectory import append_record
    ctx, be, pending, judge = _pkg_ctx(tmp_path)
    append_record(ctx.status_path, _pkg_row(
        0, "Refit of the K-loop tiling. Mirrors the ref's part 2 of 4 split of K."))
    assert judge(ctx, pending) == "REVERT_PATCH"
    assert ("reset", "good") in be.calls


def test_a_package_step_that_sets_a_new_best_is_still_credited(tmp_path):
    # the exemption defers refutation, never credit: a package whose step already
    # beats the pre-patch best beyond the noise margin earns KEEP_PATCH as usual.
    from forgekernel.inner.trajectory import append_record
    ctx, be, pending, judge = _pkg_ctx(tmp_path)
    append_record(ctx.status_path, _pkg_row(0, "atoms, part 2 of 3", ms=2.20))
    assert judge(ctx, pending) == "KEEP_PATCH"
    assert not any(c[0] == "reset" for c in be.calls)


def test_hold_then_join_one_rollback_restores_the_state_before_both(tmp_path):
    # patch1's segment is metric-less (instant EXHAUSTED) -> HOLD_PATCH keeps the
    # transaction open. patch2 joins it. The joint segment runs measured KEEPs with
    # no gain -> ONE rollback restores the milestone from before BOTH patches.
    from forgekernel.meta_trajectory import load_meta_records
    ms = tmp_path / "m.md"
    ms.write_text(_MILESTONE, encoding="utf-8")
    bad1 = _MILESTONE + "\n<!-- PATCH1_MARKER -->\n"
    bad2 = _MILESTONE + "\n<!-- PATCH1_MARKER -->\n<!-- PATCH2_MARKER -->\n"
    be = _E2EFake(
        agent_outputs=[REC("A", 2.66), REC("B", 2.66),                     # seg1 -> plateau
                       "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: x",
                       "patch_class: additive\nchanged_entries: []\n",     # patch1
                       "EXHAUSTED: nothing actionable",                    # seg2: 0 rows
                       "cause_class: R2\ndeficient_locus: D2.3\npatch_direction: y",
                       "patch_class: additive\nchanged_entries: []\n",     # patch2 joins
                       REC("C", 2.70), REC("C2", 2.71)],                   # seg3: no gain
        gpu_logs=[EVAL(2.66), NCU, EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.71), NCU],
        milestone_path=ms, patch_writes=[bad1, bad2])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.milestone = ms
    ctx.plateau_window = 1
    summary = run_two_layer(ctx, max_meta=2)
    assert summary["meta_steps"] == 2
    final = ms.read_text()
    assert "PATCH1_MARKER" not in final and "PATCH2_MARKER" not in final
    recs = load_meta_records(ctx.meta_status_path)
    assert [r.outcome for r in recs] == ["HOLD_PATCH", "REVERT_PATCH"]


def test_a_malformed_retry_must_not_reset_a_held_workspace(tmp_path):
    # patch1 is held (metric-less segment: workspace tip = the preserved pre-green
    # construction, good_ref = a checkpoint from BEFORE the chain). If the next meta
    # patch is malformed and never lands, the retry rollback must restore the
    # milestone only — resetting the workspace to good_ref would wipe the held
    # construction over a patch that never even existed, and leave the milestone
    # (still carrying patch1) prescribing work the workspace no longer has.
    from forgekernel.meta_trajectory import load_meta_records
    ms = tmp_path / "m.md"
    ms.write_text(_MILESTONE, encoding="utf-8")
    held = _MILESTONE + "\n<!-- PATCH1_MARKER -->\n"
    be = _E2EFake(
        agent_outputs=["EXHAUSTED: nothing left",                       # seg1: 0 rows
                       "cause_class: R2\ndeficient_locus: M1\npatch_direction: add one",
                       "patch_class: additive\nchanged_entries: []\n",  # patch1 lands
                       "EXHAUSTED: still nothing",                      # seg2: 0 rows -> HOLD
                       "cause_class: R1\ndeficient_locus: M2\npatch_direction: another",
                       *["patch_class: additive\nchanged_entries: []\n"] * 3],  # never writes
        gpu_logs=[], milestone_path=ms, patch_writes=[held])            # patch2 unwritten
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.milestone = ms
    summary = run_two_layer(ctx, max_meta=2)
    assert summary["stop"] == "meta_budget" and summary["meta_steps"] == 2
    # the held patch survives its neighbour's failure: milestone back to the HELD
    # state (not pre-chain), and the only reset is patch1's own resume — the
    # malformed retry added none.
    assert ms.read_text(encoding="utf-8") == held
    assert [c for c in be.calls if c[0] == "reset"] == [("reset", be.head())]
    recs = load_meta_records(ctx.meta_status_path)
    assert [r.outcome for r in recs] == ["HOLD_PATCH", "REVERT_PATCH"]


# --- diagnose-step verdict probes -------------------------------------------


def test_meta_step_runs_verdict_probe_then_diagnoses(tmp_path):
    # The teacher asks for one GPU experiment; the driver runs it on the -metaprobe
    # branch and feeds the log readings back; the diagnosis that follows is recorded
    # with the probe accounted for.
    from forgekernel.meta_trajectory import load_meta_records
    probe_log = ("variant A/base = 1.060 (3 rounds: 1.068/1.046/1.069)\n"
                 "PROBE FAIL max_diff=n/a\n")
    diagnosis = ("cause_class: R1\ndeficient_locus: D1.4\n"
                 "patch_direction: register economy before issue order\n")
    patch = _patching(tmp_path, "\n### D1.4 revised\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend(["PROBE_REQUEST: notes/refsched_ab.py", diagnosis, patch],
                     [probe_log])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert res.patch_class == "structural"
    assert ("gpu", "cmd:python3 notes/refsched_ab.py") in be.calls
    assert ("push", "fs-demo-metaprobe") in be.calls
    # the second diagnose prompt carries the experiment's readings, not just a verdict
    assert "variant A/base = 1.060" in be.prompts[1]
    assert "Verdict experiment 1/4" in be.prompts[1]
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].probe_rounds == 1


def test_verdict_probe_budget_exhaustion_demands_a_diagnosis(tmp_path):
    # Once the budget is spent a further PROBE_REQUEST burns no GPU job; the teacher
    # is told to diagnose from the evidence in hand.
    diagnosis = "cause_class: G1\ndeficient_locus: D1.4\npatch_direction: x\n"
    patch = _patching(tmp_path, "\n### D1.4 guided\nx\n",
                      "patch_class: additive\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend(["PROBE_REQUEST: notes/a.py", "PROBE_REQUEST: notes/b.py",
                      diagnosis, patch],
                     ["PROBE PASS max_diff=0.1\n"])
    ctx = _ctx(tmp_path, be)
    ctx.max_verdict_probe_rounds = 1
    res = run_meta_step(ctx)
    assert res.patch_class == "additive"
    assert sum(1 for c in be.calls if c[0] == "gpu") == 1
    assert "Verdict-experiment budget exhausted" in be.prompts[2]


def test_verdict_probes_can_be_disabled(tmp_path):
    # max_verdict_probe_rounds=0 is the control-run switch: a probe request costs
    # nothing and the diagnose flow is otherwise unchanged.
    diagnosis = "cause_class: R2\ndeficient_locus: M1\npatch_direction: x\n"
    patch = _patching(tmp_path, "\n### D1.4 extended\nx\n",
                      "patch_class: additive\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend(["PROBE_REQUEST: notes/a.py", diagnosis, patch], [])
    ctx = _ctx(tmp_path, be)
    ctx.max_verdict_probe_rounds = 0
    run_meta_step(ctx)
    assert not any(c[0] == "gpu" for c in be.calls)


def test_verdict_probe_without_verdict_line_still_feeds_readings_back(tmp_path):
    # A probe that crashed after printing half its table has still measured
    # something: the tail flows back, flagged as missing its pre-registered verdict.
    diagnosis = "cause_class: R1\ndeficient_locus: D1.4\npatch_direction: x\n"
    patch = _patching(tmp_path, "\n### D1.4 revised\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend(["PROBE_REQUEST: notes/a.py", diagnosis, patch],
                     ["tensor_pct A=61.1 B=36.5\nTraceback: boom\n"])
    ctx = _ctx(tmp_path, be)
    run_meta_step(ctx)
    assert "tensor_pct A=61.1 B=36.5" in be.prompts[1]
    assert "PROBE PASS/FAIL" in be.prompts[1]  # the protocol reminder


def test_m0_audit_supports_verdict_probes(tmp_path):
    # The audit variant shares the machinery: probe first, then AUDIT_CLEAN.
    from forgekernel.meta_trajectory import load_meta_records
    be = FakeBackend(
        ["PROBE_REQUEST: notes/mech.py",
         "cause_class: OK\npatch_direction: signatures reconcile\n"],
        ["smem_wavefronts vs ref = 1.1\nPROBE PASS max_diff=n/a\n"])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx, audit=True)
    assert res.patch_class == "audit-clean"
    assert ("gpu", "cmd:python3 notes/mech.py") in be.calls
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].outcome == "AUDIT_CLEAN" and recs[0].probe_rounds == 1


def test_probe_commits_never_enter_the_student_lineage(tmp_path):
    # add -A sweeps whatever the diagnose agent touched — not just the probe script —
    # and every downstream branch resumes from head(); the reset is what keeps a stray
    # teacher edit out of the student line (the -metaprobe branch keeps the evidence).
    diagnosis = "cause_class: R1\ndeficient_locus: D1.4\npatch_direction: x\n"
    patch = _patching(tmp_path, "\n### D1.4 revised\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend(["PROBE_REQUEST: notes/a.py", diagnosis, patch],
                     ["PROBE FAIL max_diff=0.5\n"])
    ctx = _ctx(tmp_path, be)
    pre = be.head()
    run_meta_step(ctx)
    gpu_at = be.calls.index(("gpu", "cmd:python3 notes/a.py"))
    assert be.calls.index(("reset", pre)) > gpu_at


def test_no_probe_no_reset(tmp_path):
    # a probe-less diagnose must not move or reset the workspace at all
    diagnosis = "cause_class: R1\ndeficient_locus: D1.4\npatch_direction: x\n"
    patch = _patching(tmp_path, "\n### D1.4 revised\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend([diagnosis, patch], [])
    run_meta_step(_ctx(tmp_path, be))
    assert not any(c[0] == "reset" for c in be.calls)


def test_a_diagnosis_citing_the_marker_is_not_hijacked_into_a_probe(tmp_path):
    # the probe rules ask the final diagnosis to cite probe names in `evidence`; a
    # whole-text marker match must not burn a GPU job re-running the cited script
    diagnosis = ("cause_class: R1\ndeficient_locus: D1.4\n"
                 "evidence: verdict experiment PROBE_REQUEST: notes/refsched_ab.py ruled FAIL, the orchestration ported alone is negative\n"
                 "patch_direction: register economy first\n")
    patch = _patching(tmp_path, "\n### D1.4 revised\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend([diagnosis, patch], [])
    res = run_meta_step(_ctx(tmp_path, be))
    assert res.patch_class == "structural"
    assert not any(c[0] == "gpu" for c in be.calls)


def test_zero_budget_renders_the_pre_mechanism_prompt(tmp_path):
    # a control run must run the instructions it would have run before the
    # mechanism existed: no probe section, no protocol vocabulary in the prompt
    diagnosis = "cause_class: R1\ndeficient_locus: D1.4\npatch_direction: x\n"
    patch = _patching(tmp_path, "\n### D1.4 revised\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend([diagnosis, patch], [])
    ctx = _ctx(tmp_path, be)
    ctx.max_verdict_probe_rounds = 0
    run_meta_step(ctx)
    assert "verdict experiment" not in be.prompts[0].lower()
    assert "{VERDICT_PROBE_BRIEF}" not in be.prompts[0]  # token substituted, not leaked
    be2 = FakeBackend([diagnosis,
                       _patching(tmp_path, "\n### D1.4 again\nx\n",
                                 "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")],
                      [])
    run_meta_step(_ctx(tmp_path, be2))
    assert "verdict experiment" in be2.prompts[0].lower()


# --- per-SASS stall pricing --------------------------------------------------

_STALL_LOG_OK = (
    "[sanity] kernel=main_kernel\n"
    "=== STALL LEDGER BEGIN ===\n"
    "[ledger] keying=source-line (81% of samples line-resolved)\n"
    "   900 45.00% (cum  45.0%)  kernel.py:101  HMMA.16816.F32 | long_scoreboard:70%\n"
    "   500 25.00% (cum  70.0%)  kernel.py:230  MUFU.EX2       | barrier:60%\n"
    "   440 22.00% (cum  92.0%)  kernel.py:311  LDS.128        | mio_throttle:88%\n"
    "=== STALL LEDGER END ===\n"
    "STALL-OK kernel=main_kernel samples=2000 rows=3 coverage=92.0\n")

_DIAG = ("cause_class: R2\ndeficient_locus: D6\npatch_direction: add decision\n"
         "reachability: floor_opt=1.20 ms vs target=1.40 ms -> OPEN")


def _stall_ctx(tmp_path, be):
    ctx = _ctx(tmp_path, be)
    ctx.stall_pricing = True
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    return ctx


_PRICE_LOG_OK = (
    "PRICE variants: base +ring +lane all all-ring all-lane\n"
    "PRICE ROUND 0 base pass=True device=3.40 vs_ref_device=0.605\n"
    "LEVER ring add=+8.56% remove=-8.10% add_ms=3.135 remove_ms=3.21 class=independent "
    "spin_add=880319 spin_all_minus=996231 cubin_same_as_base=False\n"
    "LEVER lane add=+5.56% remove=-4.90% add_ms=3.211 remove_ms=3.135 class=independent "
    "spin_add=996231 spin_all_minus=880319 cubin_same_as_base=False\n"
    "LEVER nb_sym add=+0.10% remove=-3.20% add_ms=3.40 remove_ms=3.25 class=coupled "
    "spin_add=1120000 spin_all_minus=900000 cubin_same_as_base=False\n"
    "STACK all=+13.10% sum_singles=+14.22% coupling=-1.12%\n"
    "PRICE noise_floor=0.50% metric=vs_ref_device\n"
    "PRICE-OK levers=3 rounds=3 base_ms=3.4052 all_ms=2.96 spin_base=1125779 spin_all=700000 "
    "tensor_base=37.85 tensor_all=44.0\n")


def _lever_kernel(tmp_path):
    """The best tree's kernel file, with every `old` the fixtures' levers edit
    occurring exactly once (the driver applies the levers locally before it
    submits the pricing job)."""
    kf = tmp_path / "flash_attn_dsl" / "flash_bwd.py"
    kf.parent.mkdir(parents=True, exist_ok=True)
    kf.write_text("a\nc\ne\n")
    return kf


def _lever_design(tmp_path, n=3):
    """A design-turn agent that writes notes/levers.json the way a real one must."""
    def agent(prompt):
        assert "Pricing round" in prompt and "levers.json" in prompt
        (tmp_path / "notes").mkdir(exist_ok=True)
        _lever_kernel(tmp_path)
        (tmp_path / "notes" / "levers.json").write_text(json.dumps({"rounds": 3, "levers": [
            {"name": "ring", "edits": [{"old": "a", "new": "b"}]},
            {"name": "lane", "edits": [{"old": "c", "new": "d"}]},
            {"name": "nb_sym", "edits": [{"old": "e", "new": "f"}]}][:n]}))
        return f"LEVERS: {n}"
    return agent


def _lever_ctx(tmp_path, be):
    ctx = _stall_ctx(tmp_path, be)
    ctx.lever_pricing_rounds = 1
    ctx.evidence_dir = tmp_path / "evidence"
    return ctx


def test_lever_pricing_round_runs_before_the_diagnosis_and_prices_the_table(tmp_path):
    """Plateau path: a design turn writes levers.json, the driver checks out the
    best tree, seeds the tool, commits, runs ONE pricing job, and the diagnose
    prompt carries the measured table with its reading rules before any
    cause_class is asked for."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD
    be = FakeBackend([_lever_design(tmp_path), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, _PRICE_LOG_OK])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    gpu = [c for c in be.calls if c[0] == "gpu"]
    assert gpu[-1] == ("gpu", _LEVER_PRICING_CMD)
    # order: design turn -> pricing job -> diagnosis turn
    agent_idx = [i for i, c in enumerate(be.calls) if c[0] == "agent"]
    price_idx = be.calls.index(("gpu", _LEVER_PRICING_CMD))
    assert agent_idx[0] < price_idx < agent_idx[1]
    assert (tmp_path / "notes" / "lever_pricing_run.py").is_file()      # seeded
    diag = be.prompts[1]
    assert "## Pricing table (measured by the driver on the best tree" in diag and "`ring`" in diag and "+8.56%" in diag
    assert "**coupled**" in diag and "companion" in diag
    assert "all-on = +13.10%" in diag and "coupling = -1.12%" in diag
    # the table is banked on the meta record and in the evidence dir
    recs = [json.loads(l) for l in (tmp_path / "meta_status.jsonl").read_text().splitlines()]
    assert [lv["name"] for lv in recs[-1]["lever_prices"] if lv["kind"] == "lever"] == ["ring", "lane", "nb_sym"]
    banked = json.loads((tmp_path / "evidence" / "lever_prices.jsonl").read_text().splitlines()[-1])
    assert banked["best_ref"] and [lv["name"] for lv in banked["table"] if lv["kind"] == "lever"] == ["ring", "lane", "nb_sym"]


_PRICE_LOG_SUBSETS = (
    "PRICE variants: base +ring +lane +bad all all-ring all-lane all-bad ring_lane\n"
    "LEVER ring add=+8.56% remove=-8.10% add_ms=3.135 remove_ms=3.21 class=independent "
    "spin_add=880319 spin_all_minus=996231 cubin_same_as_base=False\n"
    "LEVER lane add=+0.20% remove=-4.90% add_ms=3.39 remove_ms=3.135 class=coupled "
    "spin_add=996231 spin_all_minus=880319 cubin_same_as_base=False\n"
    "LEVER bad add=-17.70% remove=+6.39% add_ms=4.10 remove_ms=3.05 class=harmful "
    "spin_add=1120000 spin_all_minus=900000 cubin_same_as_base=False\n"
    "VARIANT ring_lane levers=ring,lane delta=+12.40% ms=3.03 spin=700000 q=does ring need lane?\n"
    "VARIANT all-pos levers=ring,lane delta=+12.30% ms=3.04 spin=700000 q=auto: the stack without the levers that hurt alone (dropped bad)\n"
    "STACK all=-9.29% sum_singles=-8.94% coupling=-0.35% all_pos=+12.30% all_pos_levers=ring,lane dropped=bad\n"
    "PRICE noise_floor=0.50% metric=vs_ref_device\n"
    "PRICE-OK levers=3 rounds=3 base_ms=3.4052 all_ms=3.72 spin_base=1125779 spin_all=1300000 "
    "tensor_base=37.85 tensor_all=30.0\n")


def _lever_design_with_subsets(tmp_path):
    def agent(prompt):
        assert "variants" in prompt and "in_stack" in prompt and "all-pos" in prompt
        (tmp_path / "notes").mkdir(exist_ok=True)
        _lever_kernel(tmp_path)
        (tmp_path / "notes" / "levers.json").write_text(json.dumps({"rounds": 3, "levers": [
            {"name": "ring", "decisions": ["D1.1"], "edits": [{"old": "a", "new": "b"}]},
            {"name": "lane", "decisions": ["D2.3"], "edits": [{"old": "c", "new": "d"}]},
            {"name": "bad", "edits": [{"old": "e", "new": "f"}]}],
            "variants": [{"name": "ring lane", "levers": ["ring", "lane"], "question": "does ring need lane?"}]}))
        return "LEVERS: 3"
    return agent


def test_teacher_subsets_are_priced_rendered_and_citable(tmp_path):
    """The design turn may name its own combinations (`variants`) and keep a
    suspect lever out of the stack; the tool's VARIANT rows reach the diagnose
    prompt as a subset table, the auto all-pos reading replaces a stack dragged
    down by one harmful lever, a subset is a legal `lever:` reopen price,
    and a subset inherits its members' decisions."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD, _verify_reopen_price
    be = FakeBackend([_lever_design_with_subsets(tmp_path), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, _PRICE_LOG_SUBSETS])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert ("gpu", _LEVER_PRICING_CMD) in be.calls
    spec = json.loads((tmp_path / "notes" / "levers.json").read_text())
    assert spec["variants"] == [{"name": "ring_lane", "levers": ["ring", "lane"], "question": "does ring need lane?"}]
    diag = be.prompts[1]
    assert "| `ring_lane` | ring, lane | +12.40% |" in diag and "does ring need lane?" in diag
    assert "`all-pos` = +12.30%" in diag and "Harmful levers `bad` (negative add-one, and all-on does not depend on them) went into all-on" in diag
    assert "A negative add-one whose removal from all-on does lose is `coupled`, not harmful" in diag
    assert "when a subset and the class disagree, the subset wins" in diag
    recs = [json.loads(l) for l in (tmp_path / "meta_status.jsonl").read_text().splitlines()]
    table = recs[-1]["lever_prices"]
    sub = next(t for t in table if t["name"] == "ring_lane")
    assert sub["kind"] == "variant" and sub["decisions"] == ["D1.1", "D2.3"]
    stack = next(t for t in table if t["kind"] == "stack")
    assert stack["all_pos_pct"] == 12.3 and stack["dropped"] == ["bad"]
    # a subset prices a reopen
    assert _verify_reopen_price("reopen_price: +12.4% via lever:ring_lane", [], table, ctx) is None
    assert _verify_reopen_price("reopen_price: +12.4% via lever:nope", [], table, ctx)


def test_lever_pricing_round_is_skipped_when_the_teacher_has_no_levers(tmp_path):
    """`LEVERS: none` (or no file) costs no job and the diagnosis proceeds as before."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD
    be = FakeBackend(["LEVERS: none", _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert ("gpu", _LEVER_PRICING_CMD) not in be.calls
    assert "## Pricing table (measured by the driver on the best tree" not in be.prompts[1]


def test_lever_pricing_refused_edit_is_reprompted_then_dropped(tmp_path):
    """PRICE-FAIL (an `old` that is not unique) re-prompts the design once
    (_LEVER_PRICE_RETRIES); a second failure leaves the diagnosis without a table."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD
    def design(prompt):
        _lever_design(tmp_path, 1)(prompt)
        # the tree the job archives: the pre-check reads it and would refuse a
        # duplicated `old` itself — here the file is unreadable to it, so the
        # refusal comes back from the job as before
        (tmp_path / "flash_attn_dsl" / "flash_bwd.py").unlink()
        return "LEVERS: 1"
    be = FakeBackend([design, design, _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK,
                      "PRICE-FAIL reason=lever-edit (lever 'ring' edit #1: `old` occurs 2 times)\n",
                      "PRICE-FAIL reason=lever-edit (lever 'ring' edit #1: `old` occurs 2 times)\n"])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", _LEVER_PRICING_CMD)) == 2
    assert "The pricing job rejected the lever table" in be.prompts[1]
    assert "## Pricing table (measured by the driver on the best tree" not in be.prompts[2]


def test_lever_spec_is_checked_locally_before_the_job(tmp_path):
    """A subset that shadows a lever, or an `old` that does not apply on the
    best tree, is refused by the driver with the tool's own reason and the
    design turn re-prompted — no GPU job is spent to learn it."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD

    def bad_then_good(prompt):
        (tmp_path / "notes").mkdir(exist_ok=True)
        _lever_kernel(tmp_path)
        bad = "The pricing job rejected the lever table" not in prompt
        spec = {"rounds": 3, "levers": [
            {"name": "ring", "edits": [{"old": "a", "new": "b"}]},
            {"name": "lane", "edits": [{"old": "c", "new": "d"}]}],
            "variants": [{"name": "ring" if bad else "pair", "levers": ["ring", "lane"]}]}
        (tmp_path / "notes" / "levers.json").write_text(json.dumps(spec))
        return "LEVERS: 2"
    be = FakeBackend([bad_then_good, bad_then_good, _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, _PRICE_LOG_OK])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", _LEVER_PRICING_CMD)) == 1        # the bad spec cost no job
    assert "must not equal any lever's name" in be.prompts[1] and "is a lever's name" in be.prompts[1]
    spec = json.loads((tmp_path / "notes" / "levers.json").read_text())
    assert spec["variants"][0]["name"] == "pair"


def test_lever_edit_that_does_not_apply_is_refused_before_the_job(tmp_path):
    """An `old` that occurs twice on the best tree is the tool's classic
    PRICE-FAIL; the driver now reads it off the tree first — twice refused,
    the diagnosis runs without a table and no job was submitted."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD

    def dup(prompt):
        (tmp_path / "notes").mkdir(exist_ok=True)
        _lever_kernel(tmp_path).write_text("a\na\n")
        (tmp_path / "notes" / "levers.json").write_text(json.dumps({"rounds": 3, "levers": [
            {"name": "ring", "edits": [{"old": "a", "new": "b"}]}]}))
        return "LEVERS: 1"
    be = FakeBackend([dup, dup, _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", _LEVER_PRICING_CMD)) == 0
    assert "occurs 2 times" in be.prompts[1]

def test_lever_pricing_log_without_a_verdict_is_not_a_table(tmp_path):
    """A job log with no PRICE-OK / PRICE-FAIL / ABLATE verdict (archive read
    before the tool finished, or the tool never ran) is not a table: the same
    design is bought once more, and when that log has no verdict either the
    diagnosis runs without a table — told so, in place of the table, so the
    teacher does not cite priced_by= for levers it just designed — and nothing
    is banked. A `broken` lever (its variants did not pass) reaches the prompt
    with its own reading rule."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD, _lever_price_block
    cut = ("PRICE variants: base +ring all\nPRICE ROUND 0 base pass=True device=3.41\n"
           "$ python3 notes/stall_price_run.py --pm")
    be = FakeBackend([_lever_design(tmp_path, 1), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, cut, cut])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", _LEVER_PRICING_CMD)) == 2  # bought once more, no design turn
    assert be.prompts.count(be.prompts[0]) == 1               # (one design prompt)
    assert "## Pricing table (measured by the driver on the best tree" not in be.prompts[1]
    assert "## The pricing job produced no table" in be.prompts[1] and "never `priced_by=`" in be.prompts[1]
    assert ctx.lever_price_lost == ""                          # said once
    assert not (tmp_path / "evidence" / "lever_prices.jsonl").exists()
    block = _lever_price_block([{"kind": "lever", "name": "idx", "add_pct": None, "remove_pct": None,
                                 "cls": "broken", "spin_add_ns": None, "spin_all_minus_ns": None}])
    assert "**broken**" in block and "`broken` = the variants containing it failed" in block


def test_lever_pricing_timing_copy_without_the_closing_line_is_not_a_table(tmp_path):
    """The tool prints the table twice (PRICE-TIMED, then PRICE-OK with the role
    columns). An archive cut between the two copies holds LEVER rows and no
    verdict: NOT a table (the rows would be banked and reused on every later
    entry with the role columns None). The re-bought job's whole log is."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD
    timed_cut = _PRICE_LOG_OK.split("PRICE-OK")[0] + "PRICE-TIMED levers=3 rounds=1\n"
    be = FakeBackend([_lever_design(tmp_path), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, timed_cut, _PRICE_LOG_OK + "[gpu_entry.sh] task=cmd finished\n"])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", _LEVER_PRICING_CMD)) == 2
    assert "## Pricing table (measured by the driver on the best tree" in be.prompts[1]
    assert (tmp_path / "evidence" / "lever_prices.jsonl").exists()


def test_lever_pricing_crashed_tool_is_not_resubmitted(tmp_path):
    """A traceback in a verdict-less log is the tool crashing, which a second
    job would repeat: no resubmit, no table, the reason names the crash."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD
    be = FakeBackend([_lever_design(tmp_path, 1), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, "PRICE variants: base +ring all\nTraceback (most recent call last):\n"
                                     "  File notes/lever_pricing_run.py\nKeyError: 'x'\n"])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", _LEVER_PRICING_CMD)) == 1
    assert "## The pricing job produced no table" in be.prompts[1] and "the pricing tool crashed" in be.prompts[1]


def test_price_log_verdict_reads_the_tool_closing_lines_not_the_rows():
    """ABLATE-FAIL / ABLATE-OK are verdicts (an ablation-only spec prints no
    PRICE-* line by design); PRICE-TIMED is one only on a whole archive; LEVER
    rows alone are not; the closing entry line counts only as the last line and
    for the outer task."""
    from forgekernel.loop.driver import _price_log_settled, _price_log_verdict, archive_is_whole
    assert _price_log_verdict("ABLATE ROUND 0\nABLATE-FAIL reason=base-hung (x)\n") == "ABLATE-FAIL"
    assert _price_log_verdict("ABLATE no_bar delta=+9.20%\nABLATE-OK ablations=1 rounds=3 base_ncu_ns=1\n") == "ABLATE-OK"
    assert _price_log_verdict("PRICE-FAIL reason=no-levers\n") == "PRICE-FAIL"
    assert _price_log_verdict(_PRICE_LOG_OK) == "PRICE-OK"
    timed = "LEVER ring add=+8.56%\nPRICE-TIMED levers=1 rounds=1\n"
    assert _price_log_verdict(timed) is None                          # cut between the copies
    assert _price_log_verdict(timed + "[gpu_entry] task=cmd finished\n") == "PRICE-TIMED"
    assert _price_log_verdict("LEVER ring add=+8.56% remove=-8.10% add_ms=3.1 remove_ms=3.2 "
                              "class=independent spin_add=1 spin_all_minus=2\n") is None
    assert _price_log_verdict("") is None
    # the closing line: last line, outer task
    assert archive_is_whole("x\n[gpu_entry.sh] task=cmd finished\n\n", "cmd")
    assert not archive_is_whole("x\n[gpu_entry.sh] task=ncu finished\n", "cmd")       # nested entry
    assert not archive_is_whole("x\n[gpu_entry.sh] task=ncu finished\nORCH-CENSUS a=1\n")
    assert archive_is_whole("[gpu_entry] task=ncu finished")

    # the settled read: a backend that can re-read hands back the longer log
    class _Be:
        last_job_id = "7"
        def reread_log(self, job_id):
            assert job_id == "7"
            return timed + "PRICE-OK levers=1 rounds=1 base_ms=1 all_ms=1\n[gpu_entry] task=cmd finished\n"
    assert "PRICE-OK" in _price_log_settled(_Be(), timed, "t")
    assert _price_log_settled(_Be(), _PRICE_LOG_OK, "t") == _PRICE_LOG_OK   # already a verdict: no re-read
    assert _price_log_settled(object(), timed, "t") == timed                # a fake cannot re-read


def test_ablation_only_fail_is_its_own_verdict_not_a_cut_archive(tmp_path, capsys):
    """An ablation-only spec whose base hangs prints ABLATE-FAIL and no PRICE-*
    line by design: that is a verdict, not an incomplete archive — one job,
    no resubmit, the ABLATE-FAIL reason in the log, no 'produced no table' block."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD

    def agent(prompt):
        (tmp_path / "notes").mkdir(exist_ok=True)
        _lever_kernel(tmp_path)
        (tmp_path / "notes" / "levers.json").write_text(json.dumps({"ablations": [
            {"name": "no_bar", "hypothesis": "h", "edits": [{"old": "c", "new": "C"}]}]}))
        return "LEVERS: none"
    be = FakeBackend([agent, _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, "PRICE kernel_file=k variants: base | ablations: ~no_bar\n"
                                     "ABLATE-FAIL reason=base-hung (the unablated tree did not return under ncu)\n"
                                     "[gpu_entry.sh] task=cmd finished\n"])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", _LEVER_PRICING_CMD)) == 1
    assert "## The pricing job produced no table" not in be.prompts[1]
    out = capsys.readouterr().out
    assert "ABLATE-FAIL base-hung" in out and "no PRICE-OK / PRICE-FAIL / ABLATE verdict" not in out


def test_json_probe_cut_archive_is_not_scored_as_no_ablate_line(tmp_path):
    """A verdict probe whose job log has no verdict (archive read short) is
    bought once more; when that is cut too the round says the reading was not
    read — not 'no ABLATE line' (the probe did run) — and no ablation row is
    banked off the cut log."""
    def probe(prompt):
        _lever_kernel(tmp_path)
        (tmp_path / "notes").mkdir(exist_ok=True)
        (tmp_path / "notes" / "abl.json").write_text(json.dumps({
            "ablations": [{"name": "no_bar", "hypothesis": "h", "edits": [{"old": "c", "new": "C"}]}]}))
        return "PROBE_REQUEST: notes/abl.json"
    cut = "PRICE kernel_file=k variants: base | ablations: ~no_bar\nABLATE ROUND 0 base ncu_ns=1\n"
    be = FakeBackend([_R1, probe, _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [cut, cut])
    ctx = _priced_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", "cmd:python3 notes/lever_pricing_run.py notes/abl.json")) == 2
    fb = be.prompts[2]
    assert "carries no PRICE/ABLATE verdict" in fb and "no ABLATE line" not in fb


def test_lever_pricing_runs_at_the_audit_too(tmp_path):
    """The M0 audit is where structural reopens are decided, so it prices too:
    design turn (on the audit's own prompt) -> one pricing job -> the audit
    verdict reads the measured table."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD
    be = FakeBackend([_lever_design(tmp_path),
                      "cause_class: OK\npatch_direction: signatures reconcile\n"],
                     [_STALL_LOG_OK, _PRICE_LOG_OK])
    ctx = _lever_ctx(tmp_path, be)
    res = run_meta_step(ctx, audit=True)
    assert res.patch_class == "audit-clean"
    assert ("gpu", _LEVER_PRICING_CMD) in be.calls
    assert "Mandatory M0 finish audit" in be.prompts[0]          # the design turn reads the audit's question
    assert "## Pricing table (measured by the driver on the best tree" in be.prompts[1]


def test_lever_pricing_is_off_when_disabled(tmp_path):
    from forgekernel.loop.driver import _LEVER_PRICING_CMD
    be2 = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                        "patch_class: additive\nchanged_entries: []\n")],
                      [_STALL_LOG_OK])
    ctx2 = _stall_ctx(tmp_path, be2)
    ctx2.lever_pricing_rounds = 0
    run_meta_step(ctx2)
    assert ("gpu", _LEVER_PRICING_CMD) not in be2.calls

# --- phase exhaustion: the price table closes the phase it priced ----------------
#
# A milestone with a two-decision M1 and an M2 behind it; the student has closed
# M0 (an M0-gate KEEP) and delivered nothing of M1 yet. The table prices two M1
# levers null: the phase's remaining walk would only re-measure the table.

_MILESTONE_PHASED = _MILESTONE.replace(
    "**D2.3 next**",
    "**D1.2 wait depth**\n- Action: a\n- Purpose: b\n- Trigger condition: c\n- Cost: d\n- Depends on: D1.1\n\n**D2.3 next**")

_PRICE_LOG_NULL = (
    "PRICE variants: base +ring +lane all all-ring all-lane\n"
    "PRICE ROUND 0 base pass=True device=2.23 vs_ref_device=0.935\n"
    "LEVER ring add=+0.45% remove=+0.21% add_ms=2.214 remove_ms=2.464 class=null "
    "spin_add=n/a spin_all_minus=n/a cubin_same_as_base=False\n"
    "LEVER lane add=-0.64% remove=+1.42% add_ms=2.244 remove_ms=2.434 class=null "
    "spin_add=n/a spin_all_minus=n/a cubin_same_as_base=False\n"
    "STACK all=-7.03% sum_singles=-0.19% coupling=-6.84%\n"
    "PRICE noise_floor=1.50% metric=vs_ref_device\n"
    "PRICE-OK levers=2 rounds=3 base_ms=2.2273 all_ms=2.384\n")


def _lever_design_attributed(tmp_path, decisions=("D1.1", "D1.1")):
    """A design turn that names the milestone decision each lever moves."""
    def agent(prompt):
        assert "`decisions`" in prompt
        (tmp_path / "notes").mkdir(exist_ok=True)
        (tmp_path / "notes" / "levers.json").write_text(json.dumps({"rounds": 3, "levers": [
            {"name": "ring", "decisions": [decisions[0]], "edits": [{"old": "a", "new": "b"}]},
            {"name": "lane", "decisions": [decisions[1]], "edits": [{"old": "c", "new": "d"}]}]}))
        return "LEVERS: 2"
    return agent


def _phased_ctx(tmp_path, be):
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    ctx = _lever_ctx(tmp_path, be)
    ctx.milestone.write_text(_MILESTONE_PHASED, encoding="utf-8")
    ctx.decisions_dag = []          # parse the phased milestone, not the fixture DAG
    append_record(ctx.status_path, IR(iter=0, decision_id="M0-gate", change="x",
                                      correctness=True, verdict="KEEP",
                                      sha="s0", bwd_median_ms=2.3, device_median_ms=2.3))
    return ctx


_AUDIT_OK = "cause_class: OK\npatch_direction: signatures reconcile\n"


def test_phase_exhaustion_reads_the_current_phase_off_the_table():
    from forgekernel.loop.driver import _phase_exhaustion, _current_phase
    from forgekernel.inner.trajectory import IterationRecord as IR

    class _C:  # just enough context for _milestone_dag
        decisions_dag = [Decision("D1.1", frozenset()), Decision("D1.2", frozenset({"D1.1"})),
                         Decision("D2.3", frozenset({"D1.2"}))]
        milestone = "/nonexistent"
    gate = IR(iter=0, decision_id="M0-gate", change="", correctness=True, verdict="KEEP")
    table = [{"kind": "lever", "name": "ring", "cls": "null", "decisions": ["D1.1"]},
             {"kind": "lever", "name": "lane", "cls": "noop", "decisions": ["D1.1"]},
             {"kind": "lever", "name": "free", "cls": "null", "decisions": []},
             {"kind": "stack"}]
    ex = _phase_exhaustion(_C(), [gate], table)
    assert ex == {"phase": "M1", "gate": "M1-gate", "levers": ["ring", "lane"],
                  "priced": ["D1.1"], "unpriced": ["D1.2"], "delivered": []}
    # a positive lever in the phase IS the way forward: not exhausted
    live = table[:1] + [{"kind": "lever", "name": "lane", "cls": "independent",
                         "add_pct": 4.0, "decisions": ["D1.2"]}]
    assert _phase_exhaustion(_C(), [gate], live) is None
    # a table that priced only another phase's levers says nothing about this one
    assert _phase_exhaustion(_C(), [gate], [dict(table[0], decisions=["D2.3"])]) is None
    # the phase is the lowest declared one whose gate is not closed
    assert _current_phase([], {1, 2}) == 1 and _current_phase([gate], {1, 2}) == 1
    m1 = IR(iter=1, decision_id="M1-gate", change="", correctness=True, verdict="KEEP")
    assert _current_phase([gate, m1], {1, 2}) == 2
    m2 = IR(iter=2, decision_id="M2-gate", change="", correctness=True, verdict="KEEP")
    assert _current_phase([gate, m1, m2], {1, 2}) is None
    # a delivered decision is neither priced nor unpriced
    done = IR(iter=1, decision_id="D1.2+D1.1", change="", correctness=True, verdict="KEEP")
    assert _phase_exhaustion(_C(), [gate, done], table)["unpriced"] == []


def test_phase_verdict_is_demanded_once_then_close_is_the_default(tmp_path):
    """Audit path: the design turn attributes both levers to
    D1.1, the job prices them null, and the audit's OK verdict must carry a
    phase_verdict. Missing once -> bounce with the walkable decisions named;
    missing twice -> the driver closes M1 itself and the student is told to
    submit M1-gate."""
    be = FakeBackend([_lever_design_attributed(tmp_path), _AUDIT_OK, _AUDIT_OK],
                     [_STALL_LOG_OK, _STALL_LOG_OK, _PRICE_LOG_NULL])
    ctx = _phased_ctx(tmp_path, be)
    res = run_meta_step(ctx, audit=True)
    assert res.patch_class == "audit-clean"
    assert "## Phase levers exhausted" in be.prompts[1]   # best + head stall jobs, then the price job
    assert "`ring`, `lane`" in be.prompts[1] and "`D1.2`" in be.prompts[1]   # unpriced, walkable
    assert "## Phase ruling (binding)" in be.prompts[2] and "D1.2" in be.prompts[2]
    assert res.phase_close and res.phase_close["gate"] == "M1-gate"
    assert res.phase_close["verdict"] == "close (default: unanswered)"
    assert "submit the `M1-gate` row directly" in res.phase_close["note"]
    rec = json.loads((tmp_path / "meta_status.jsonl").read_text().splitlines()[-1])
    assert rec["phase_verdict"] == "close (default: unanswered)"
    assert [lv["decisions"] for lv in rec["lever_prices"] if lv["kind"] == "lever"] == [["D1.1"], ["D1.1"]]


def test_phase_verdict_continue_must_name_an_unpriced_decision(tmp_path):
    """`continue:<id>` is accepted only for a phase decision no lever priced and
    no KEEP delivered; the student note then narrows the walk to that one."""
    be = FakeBackend([_lever_design_attributed(tmp_path),
                      _AUDIT_OK + "phase_verdict: continue:D1.1\n",     # priced null -> bounced
                      _AUDIT_OK + "phase_verdict: continue:D1.2\n"],
                     [_STALL_LOG_OK, _STALL_LOG_OK, _PRICE_LOG_NULL])
    ctx = _phased_ctx(tmp_path, be)
    res = run_meta_step(ctx, audit=True)
    assert "already priced null" in be.prompts[2]
    assert res.phase_close["verdict"] == "continue:D1.2"
    assert "only `D1.2` is still worth doing" in res.phase_close["note"]
    assert "submit the `M1-gate` row directly next round" in res.phase_close["note"]


def test_phase_close_rides_student_turns_until_the_gate_row(tmp_path):
    """run_two_layer carries the verdict into ctx.phase_close; the student
    prompt shows it; a KEEP gate row retires it; a FAILED gate row reduces it
    to the "do not re-walk the priced knobs" half (the phase is open again but
    the table still describes this best tree) until a KEEP moves the tree."""
    from forgekernel.loop.driver import _apply_phase_close, _student_prompt
    ctx = _ctx(tmp_path, FakeBackend([], []))
    meta = MetaResult("", "audit-clean", [], None, [],
                      phase_close={"gate": "M1-gate", "phase": "M1", "verdict": "close",
                                   "levers": ["ring_depth"],
                                   "note": "- **Phase close** submit M1-gate\n"})
    _apply_phase_close(ctx, meta)
    assert ctx.phase_close["gate"] == "M1-gate"
    assert "submit M1-gate" in _student_prompt(ctx, 5)
    # a structural patch redoes the phase's decisions: the premise is gone
    _apply_phase_close(ctx, MetaResult("", "structural", ["D1.1"], None, []))
    assert ctx.phase_close is None
    # a FAILED gate row: the next student turn is told not to re-walk the null
    # knobs; the KEEP after it (the tree moved) retires the notice
    ctx.phase_close = dict(meta.phase_close)
    be = FakeBackend([REC("M1-gate", 2.0), REC("D2.1", 1.95), "EXHAUSTED: done"],
                     [EVAL(2.0, ok=False), NCU, EVAL(1.95), NCU])
    ctx.backend = be
    ctx.max_consecutive_reverts = 3
    run_inner_loop(ctx, start_iter=0)
    recs = load_records(ctx.status_path)
    assert recs[0].decision_id == "M1-gate" and recs[0].verdict == "REVERT"
    assert "do not go back to point-by-point A/B" in be.prompts[1] and "`ring_depth`" in be.prompts[1]
    assert "submit M1-gate" not in be.prompts[1]
    assert recs[1].verdict == "KEEP" and ctx.phase_close is None
    assert "Phase levers already priced" not in be.prompts[2]
    # a KEEP gate row retires the notice outright
    ctx.phase_close = dict(meta.phase_close)
    be = FakeBackend([REC("M1-gate", 1.8)], [EVAL(1.8), NCU])
    ctx.backend = be
    run_inner_loop(ctx, start_iter=len(recs))
    assert ctx.phase_close is None
    assert load_records(ctx.status_path)[-1].decision_id == "M1-gate"


def test_phase_exhaustion_is_not_asked_twice_for_a_closed_phase(tmp_path):
    """A standing notice for the same gate means the question was answered;
    the next meta step (same best, table reused) does not ask again."""
    be = FakeBackend([_lever_design_attributed(tmp_path), _AUDIT_OK],
                     [_STALL_LOG_OK, _STALL_LOG_OK, _PRICE_LOG_NULL])
    ctx = _phased_ctx(tmp_path, be)
    ctx.phase_close = {"gate": "M1-gate", "phase": "M1", "verdict": "close", "note": "n"}
    res = run_meta_step(ctx, audit=True)
    assert res.patch_class == "audit-clean" and res.phase_close is None
    assert "## Phase levers exhausted" not in be.prompts[1]
    assert json.loads((tmp_path / "meta_status.jsonl").read_text().splitlines()[-1])["phase_verdict"] == ""


def test_lever_decisions_outside_the_milestone_are_dropped_not_bounced(tmp_path):
    be = FakeBackend([_lever_design_attributed(tmp_path, ("D1.1", "D9.9")), _AUDIT_OK, _AUDIT_OK],
                     [_STALL_LOG_OK, _STALL_LOG_OK, _PRICE_LOG_NULL])
    ctx = _phased_ctx(tmp_path, be)
    res = run_meta_step(ctx, audit=True)
    rec = json.loads((tmp_path / "meta_status.jsonl").read_text().splitlines()[-1])
    assert [lv["decisions"] for lv in rec["lever_prices"] if lv["kind"] == "lever"] == [["D1.1"], []]
    # the one attributed lever still exhausts the phase; the unattributed one is not counted
    assert res.phase_close["gate"] == "M1-gate" and res.phase_close["levers"] == ["ring"]


# --- no reopen without a price ------------------------------------------------

_R1 = "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: flip the stage split axis\n"


def _priced_ctx(tmp_path, be):
    ctx = _ctx(tmp_path, be)
    ctx.reopen_requires_price = True
    return ctx


def test_r1_without_a_measured_price_is_bounced_then_refused(tmp_path):
    """An R1 that cites no driver-measured price is bounced once with the three
    routes (verdict probe / price table / certified KB); a second unpriced R1 is
    refused: no patch turn, milestone untouched, the step recorded UNPRICED_REOPEN."""
    from forgekernel.meta_trajectory import load_meta_records
    be = FakeBackend([_R1, _R1], [])
    ctx = _priced_ctx(tmp_path, be)
    before = ctx.milestone.read_text(encoding="utf-8")
    res = run_meta_step(ctx)
    assert res.patch_class == "unpriced-reopen" and res.changed_entries == []
    assert "## A reopen must be priced first" in be.prompts[1] and "reopen_price:" in be.prompts[1]
    assert len(be.prompts) == 2                                   # no patch turn
    assert ctx.milestone.read_text(encoding="utf-8") == before
    rec = load_meta_records(ctx.meta_status_path)[-1]
    assert (rec.outcome, rec.cause_class, rec.deficient_locus, rec.patch_class) == \
        ("UNPRICED_REOPEN", "R1", "D1.1", "unpriced-reopen")


def test_r1_priced_by_a_verdict_probe_reopens(tmp_path):
    """The bounce invites a verdict probe; an R1 that cites that probe's reading
    (`reopen_price: … via probe:<path>#<n>`) passes the gate and patches."""
    from forgekernel.meta_trajectory import load_meta_records
    priced = _R1 + "reopen_price: +6.1% via probe:notes/axis.py#1\n"
    be = FakeBackend([_R1, "PROBE_REQUEST: notes/axis.py", priced,
                      _patching(tmp_path, "\n### D1.1 axis flipped\nx\n",
                                "patch_class: structural\nchanged_entries: [D1.1]\n")],
                     ["PROBE PASS max_diff=1.061 — key-split variant / base, same job\n"])
    ctx = _priced_ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert res.patch_class == "structural" and res.changed_entries == ["D1.1"]
    rec = load_meta_records(ctx.meta_status_path)[-1]
    assert rec.outcome == "PENDING" and "probe:notes/axis.py#1" in rec.reopen_price
    assert rec.probe_rounds == 1


def test_r1_citing_a_probe_that_printed_no_verdict_is_still_unpriced(tmp_path):
    """A probe that ran but reported nothing is not a price — an audit's
    spill probe once came back NO VERDICT and the reopen went ahead regardless."""
    cite = _R1 + "reopen_price: -3% via probe:notes/spill.py\n"
    be = FakeBackend(["PROBE_REQUEST: notes/spill.py", cite, cite],
                     ["the probe crashed before printing a verdict\n"])
    ctx = _priced_ctx(tmp_path, be)
    assert run_meta_step(ctx).patch_class == "unpriced-reopen"
    assert "printed no PROBE PASS/FAIL line" in be.prompts[2]


def test_reopen_price_sources_are_machine_checked(tmp_path):
    """Only the SOURCE is checked — it must be something the driver measured this
    step (a probe contract with a reading, a lever on the table) or a hw_kb entry
    that exists AND is GPU-certified; the delta is the teacher's reading and is
    not re-derived. Teachers gloss the name in fullwidth parens and vary the
    case / suffix; none of that is a reason to refuse a real source."""
    from forgekernel.loop.driver import _verify_reopen_price
    ctx = _priced_ctx(tmp_path, FakeBackend([], []))
    table = [{"kind": "lever", "name": "ring", "add_pct": 8.56, "remove_pct": -8.1, "cls": "independent"},
             {"kind": "stack", "all_pct": 13.1}]
    readings = [{"path": "notes/axis.py", "round": 1, "contract": 1, "verdict": "PASS",
                 "reading": "1.061", "note": ""},
                {"path": "notes/axis.py", "round": 1, "contract": 2, "verdict": "FAIL",
                 "reading": "unreported", "note": ""}]
    (tmp_path / "sm90").mkdir(exist_ok=True)
    (tmp_path / "sm90" / "gather_producer_lead_stack.md").write_text(
        "# kb\n## Sources\n- ✅ Verified: probe notes/p.py, job 870533, `PROBE PASS ...`\n",
        encoding="utf-8")
    (tmp_path / "sm90" / "transcribed.md").write_text("# kb\n## Sources\n- PTX ISA\n", encoding="utf-8")
    (tmp_path / "sm90" / "index.md").write_text("| x | ✅ | job 870533 |\n", encoding="utf-8")

    def why(c):
        return _verify_reopen_price(c, readings, table, ctx, ["notes/axis.py", "notes/dead.py"])
    assert why("+8.6% via lever:ring") is None
    assert why("+8.6% via lever:ring（independent）") is None       # fullwidth gloss
    assert why("+6.1% via probe:notes/axis.py#1") is None
    assert why("+6.1% via probe:./notes/axis.py") is None          # any contract with a reading
    assert why("+16.2% via KB:gather_producer_lead_stack") is None
    assert why("+16.2% via kb:gather_producer_lead_stack.md（certified）") is None
    assert "no reopen_price line" in why("")
    assert "names no probe:/lever:/KB:" in why("+5% (estimated from the SASS)")
    assert "not on the price table" in why("+5% via lever:lane")
    assert "printed no reading" in why("via probe:notes/axis.py#2")
    assert "printed no PROBE PASS/FAIL line" in why("via probe:notes/dead.py")
    assert "was not run this step" in why("via probe:notes/other.py")
    assert "not found" in why("via KB:nonexistent_entry")
    assert "no GPU certification" in why("+3% via KB:transcribed")
    assert "not a recipe entry" in why("+3% via KB:index")


def test_structural_patch_off_a_non_r1_diagnosis_needs_a_price_too(tmp_path):
    """The rule is on the reopen, not the enum: a G1 diagnosis whose patch declares
    structural without a cited price is withdrawn — milestone restored, recorded
    UNPRICED_REOPEN — while the same patch with a KB-cited price lands."""
    from forgekernel.meta_trajectory import load_meta_records
    g1 = "cause_class: G1\ndeficient_locus: D1.1\npatch_direction: re-tile\n"
    be = FakeBackend([g1, _patching(tmp_path, "\n### D1.1 re-tiled\nx\n",
                                    "patch_class: structural\nchanged_entries: [D1.1]\n")], [])
    ctx = _priced_ctx(tmp_path, be)
    before = ctx.milestone.read_text(encoding="utf-8")
    res = run_meta_step(ctx)
    assert res.patch_class == "unpriced-reopen"
    assert ctx.milestone.read_text(encoding="utf-8") == before      # the landed patch was withdrawn
    assert load_meta_records(ctx.meta_status_path)[-1].outcome == "UNPRICED_REOPEN"
    (tmp_path / "sm90" / "retile_recipe.md").write_text("# kb\n- ✅ Verified: job 870533 `PROBE PASS`\n", encoding="utf-8")
    be2 = FakeBackend([g1 + "reopen_price: +4% via KB:retile_recipe\n",
                       _patching(tmp_path, "\n### D1.1 re-tiled\nx\n",
                                 "patch_class: structural\nchanged_entries: [D1.1]\n")], [])
    ctx2 = _priced_ctx(tmp_path, be2)
    assert run_meta_step(ctx2).patch_class == "structural"
    assert load_meta_records(ctx.meta_status_path)[-1].reopen_price == "+4% via KB:retile_recipe"


def test_reopen_price_gate_stands_down_without_an_in_step_way_to_measure(tmp_path):
    """A control run with neither verdict probes nor a pricing round cannot
    measure a price in-step, so the rule does not fire; the loop.toml opt-out
    (the fixture default here) turns it off outright."""
    structural = _patching(tmp_path, "\n### D1.1 flipped\nx\n",
                           "patch_class: structural\nchanged_entries: [D1.1]\n")
    be = FakeBackend([_R1, structural], [])
    ctx = _priced_ctx(tmp_path, be)
    ctx.max_verdict_probe_rounds = 0
    ctx.lever_pricing_rounds = 0
    assert run_meta_step(ctx).patch_class == "structural"
    be2 = FakeBackend([_R1, structural], [])
    assert run_meta_step(_ctx(tmp_path, be2)).patch_class == "structural"


def test_m0_audit_refuses_an_unpriced_reopen_and_resumes_the_segment(tmp_path):
    """Observed shape: first green -> audit -> R1 on an inferred masking window.
    Unpriced twice, the reopen is refused: nothing is redone (no REOPEN row), the
    segment resumes on the built foundation, the refusal is in the meta memory."""
    from forgekernel.meta_trajectory import load_meta_records
    dag = [Decision(id="D0.1"), Decision(id="D0.2", depends_on=["D0.1"]),
           Decision(id="D1.1", depends_on=["D0.2"])]
    r1 = "cause_class: R1\ndeficient_locus: D0.2\npatch_direction: flip the split axis\n"
    agent = [REC("D0.1", 2.66), r1, r1, REC("D1.1", 1.8)]
    gpu = [EVAL(2.66), NCU, EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9, dag=dag)
    ctx.m0_audit = True
    ctx.reopen_requires_price = True
    summary = run_two_layer(ctx, max_meta=2)
    assert summary["stop"] == "target" and summary["meta_steps"] == 1
    assert [r.decision_id for r in load_records(ctx.status_path)] == ["D0.1", "D1.1"]
    assert [r.outcome for r in load_meta_records(ctx.meta_status_path)] == ["UNPRICED_REOPEN"]
    assert "## A reopen must be priced first" in be.prompts[2]


def test_plateau_unpriced_reopen_retries_the_meta_step_not_the_segment(tmp_path):
    """Plateau path: the refusal costs a meta step and goes straight back to
    diagnose — no student turn and no restart in between — and that diagnosis
    reads the refused reopen in its reflection."""
    from forgekernel.meta_trajectory import load_meta_records
    dag = [Decision(id="A", depends_on=[])]
    r1 = "cause_class: R1\ndeficient_locus: A\npatch_direction: flip\n"
    agent = [REC("A", 2.66), REC("B", 2.70), REC("C", 2.68), REC("D", 2.69),   # plateau
             r1, r1,                                                            # refused
             "cause_class: R2\ndeficient_locus: A\npatch_direction: add one",
             _patching(tmp_path, "\n### patched\nx\n",
                       "patch_class: additive\nchanged_entries: []\n"),
             REC("E", 1.8)]
    gpu = [EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.68), NCU, EVAL(2.69), NCU,
           EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.9, dag=dag)
    ctx.reopen_requires_price = True
    summary = run_two_layer(ctx, max_meta=3)
    assert summary["stop"] == "target" and summary["meta_steps"] == 2
    recs = load_meta_records(ctx.meta_status_path)
    assert len(recs) == 2 and recs[0].outcome == "UNPRICED_REOPEN" and recs[1].cause_class == "R2"
    # prompts: [0..3] students, [4] R1, [5] the bounce, [6] the retried diagnosis
    assert "[unpriced reopen refused] step0 R1 @ A" in be.prompts[6]


def test_stall_pricing_ledger_reaches_the_diagnose_prompt(tmp_path):
    """Plateau diagnosis prices the kernel first: the pricing job runs before the
    diagnose agent, and its ledger rows land verbatim in the diagnose prompt."""
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    ctx = _stall_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert ("gpu", "cmd:python3 notes/stall_price_run.py --pm") in be.calls
    diagnose_prompt = be.prompts[0]
    assert "Per-instruction stall ledger" in diagnose_prompt
    assert "kernel.py:101" in diagnose_prompt  # the rows themselves, not a summary
    # the pricing job precedes the first agent call
    assert be.calls.index(("gpu", "cmd:python3 notes/stall_price_run.py --pm")) \
        < be.calls.index(("agent",))


def test_stall_pricing_selfcheck_failure_degrades_the_ledger(tmp_path):
    # markers but no STALL-OK: the ledger is served with an explicit downgrade note
    log = _STALL_LOG_OK.replace("STALL-OK kernel=main_kernel samples=2000 rows=3 "
                                "coverage=92.0", "STALL-FAIL reason=coverage (88.0%)")
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [log])
    ctx = _stall_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert "ledger self-check failed" in be.prompts[0]


def test_stall_pricing_job_failure_never_blocks_the_diagnosis(tmp_path):
    from forgekernel.loop.driver import BackendError

    class _Failing(FakeBackend):
        def submit_gpu(self, branch, mode):
            raise BackendError("queue died")

    be = _Failing([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                    "patch_class: additive\nchanged_entries: []\n")], [])
    ctx = _stall_ctx(tmp_path, be)
    run_meta_step(ctx)  # must not raise
    assert "Ledger unavailable" in be.prompts[0]


_DIAG_NO_REACH = "cause_class: R2\ndeficient_locus: D6\npatch_direction: add decision"


def test_missing_reachability_line_is_reprompted_once(tmp_path):
    # Plateau diagnosis with the ledger channel present must carry the §1c floor
    # arithmetic; a diagnosis without the line costs exactly one re-prompt turn.
    be = FakeBackend([_DIAG_NO_REACH, _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    res = run_meta_step(_stall_ctx(tmp_path, be))
    assert res.patch_class == "additive"
    assert "reachability" in be.prompts[1]  # the re-prompt names the missing line


def test_candidates_table_path_reaches_the_diagnose_prompt(tmp_path):
    # A BLOCKED reachability verdict sends the diagnosis to the recon candidates
    # table (the surviving readings of the reference); the path binding must land
    # in the rendered prompt whenever a recon report exists.
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [])
    ctx = _ctx(tmp_path, be)
    recon_dir = tmp_path / "_recon"
    recon_dir.mkdir()
    (recon_dir / "candidates.md").write_text("## Candidate A\n", encoding="utf-8")
    ctx.recon_path = recon_dir / "report.md"
    run_meta_step(ctx)
    assert str(recon_dir / "candidates.md") in be.prompts[0]
    assert "{CANDIDATES_PATH}" not in be.prompts[0]  # substituted, not leaked


def test_fullwidth_colon_reachability_line_passes_the_gate(tmp_path):
    # The prompt is Chinese; `reachability：` (full-width colon) must count as
    # present, not burn the one re-prompt.
    diag_cn = ("cause_class: R2\ndeficient_locus: D6\npatch_direction: add decision\n"
               "reachability：floor_opt=1.20 ms vs target=1.40 ms -> OPEN")
    be = FakeBackend([diag_cn, _patching(tmp_path, "\n### D6 added\nx\n",
                                         "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    res = run_meta_step(_stall_ctx(tmp_path, be))
    assert res.patch_class == "additive"
    assert sum(1 for c in be.calls if c == ("agent",)) == 2  # diagnose + patch, no re-prompt


def test_stubborn_missing_reachability_is_accepted_after_one_demand(tmp_path):
    # A second omission is taken as-is: the gate may never cost more than one turn.
    be = FakeBackend([_DIAG_NO_REACH, _DIAG_NO_REACH,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    res = run_meta_step(_stall_ctx(tmp_path, be))
    assert res.patch_class == "additive"


def test_stall_pricing_targets_the_best_sha_when_the_backend_can_push_refs(tmp_path):
    from forgekernel.inner.trajectory import IterationRecord, append_record

    class _RefPushing(FakeBackend):
        def push_ref(self, ref, branch):
            self.calls.append(("push_ref", ref, branch))

    # two logs: the best tree (ledger source) is priced first, then the tree
    # under judgment (HEAD = "sha0" in the fake), which is a different commit
    be = _RefPushing([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, _STALL_LOG_OK])
    ctx = _stall_ctx(tmp_path, be)
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="a", correctness=True, verdict="KEEP",
        bwd_median_ms=3.0, device_median_ms=3.0, sha="shaworse"))
    append_record(ctx.status_path, IterationRecord(
        iter=1, decision_id="D2", change="b", correctness=True, verdict="KEEP",
        bwd_median_ms=2.0, device_median_ms=2.0, sha="shabest"))
    append_record(ctx.status_path, IterationRecord(
        iter=2, decision_id="D3", change="c", correctness=True, verdict="KEEP",
        bwd_median_ms=2.5, device_median_ms=2.5, sha="shalater"))
    run_meta_step(ctx)
    pushes = [c for c in be.calls if c[0] == "push_ref"]
    assert pushes == [("push_ref", "shabest", "fs-demo-stallprice"),
                      ("push_ref", "sha0", "fs-demo-stallprice")]
    assert not any(c == ("push", "fs-demo-stallprice") for c in be.calls)
    assert len([c for c in be.calls if c[0] == "gpu"]) == 2


def test_seed_ws_stall_tools_vendors_once_and_commits(tmp_path):
    from forgekernel.loop.driver import _seed_ws_stall_tools

    be = FakeBackend([], [])
    ctx = _stall_ctx(tmp_path, be)
    _seed_ws_stall_tools(ctx)
    assert (tmp_path / "notes" / "stall_tools.py").is_file()
    assert (tmp_path / "notes" / "stall_price_run.py").is_file()
    seeds = [c for c in be.calls if c[0] == "commit"]
    assert len(seeds) == 1
    _seed_ws_stall_tools(ctx)  # unchanged content -> no second commit
    assert len([c for c in be.calls if c[0] == "commit"]) == 1
    ctx.stall_pricing = False
    (tmp_path / "notes" / "stall_tools.py").unlink()
    _seed_ws_stall_tools(ctx)  # disabled run -> not reseeded
    assert not (tmp_path / "notes" / "stall_tools.py").exists()


def test_recon_finish_without_stall_evidence_is_bounced_to_a_choice(tmp_path):
    """The fourth evidence channel gets the same deliver-or-waive gate: a finish
    with the three static verdicts but no measured pricing must bounce, naming
    the STALL-WAIVED spelling."""
    from forgekernel.loop.driver import run_recon_step

    def fresh_writeup(prompt):
        (tmp_path / "recon-report.md").write_text("observations [MEASURED, job 1]")
        (tmp_path / "candidates.md").write_text("A|B")
        return "nothing further from me"

    def waive(prompt):
        assert "STALL evidence missing" in prompt and "STALL-WAIVED" in prompt
        # the overlap channels bounce in the same turn, naming their own waivers
        assert "overlap ledger is missing (ROLE/PMTL)" in prompt and "ROLE/PMTL-WAIVED" in prompt
        report = tmp_path / "recon-report.md"
        report.write_text(report.read_text()
                          + "\nSTALL-WAIVED: pcsamp metrics all empty on this image, verified by measurement"
                          + "\nROLE-WAIVED: same as above, no PM sampling\nPMTL-WAIVED: same as above")
        return "waived, nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/sass_probe.py",
                           fresh_writeup, waive], [])
    recon = FakeBackend([], ["SASS-OK kernel=fa instructions=2615 opcodes=118\n"
                             "SCHED-OK loop=backedge hgmma_runs=5 depbars=4\n"
                             "LIVE-OK peak=232 guessed_frac=0.08\n"])
    ctx = _stall_ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "sass_probe.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True


_ORCH_LOG_OK = (
    "=== ORCH DOSSIER BEGIN ===\n"
    "[parse] 3520 instructions, unparsed=21\n"
    "=== mbarrier objects (2, 2 resolved) ===\n"
    "  [CONST] smem+0x34980: ARRIVE@r1 x1 INIT@r0 x1 WAIT@r6 x1\n"
    "=== ORCH DOSSIER END ===\n"
    "ORCH-OK regions=42 loops=40 nbar_ids=8 mbar_objs=25 resolved=25 "
    "edges=6 itc_candidates=2\n")


def test_recon_orch_delivery_persists_the_ref_dossier(tmp_path):
    """A job log carrying ORCH-OK banks the dossier body at the fixed recon
    path (same contract as the ref stall ledger) — and the channel stays
    advisory: the run below finishes with every gated channel waived and NO
    orch waiver line, which must not block the finish."""
    from forgekernel.loop.driver import run_recon_step

    def writeup(prompt):
        (tmp_path / "recon-report.md").write_text(
            "obs [MEASURED, job 1]\nSASS-WAIVED: x\nSCHED-WAIVED: x\n"
            "LIVE-WAIVED: x\nSTALL-WAIVED: x\nROLE-WAIVED: x\nPMTL-WAIVED: x")
        (tmp_path / "candidates.md").write_text("A")
        return "nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/p.py", writeup], [])
    recon = FakeBackend([], [_ORCH_LOG_OK])
    ctx = _stall_ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    dossier = (tmp_path / "ref_orch_dossier.txt").read_text()
    assert "smem+0x34980" in dossier
    assert "ORCH-OK" not in dossier          # body only, verdict line outside
    assert "ORCH DOSSIER BEGIN" not in dossier


def test_recon_orch_verdict_without_brackets_banks_nothing(tmp_path):
    """A hand-typed ORCH-OK line with no dossier brackets must not write an
    empty/garbage file — the bank requires the bracketed body."""
    from forgekernel.loop.driver import run_recon_step

    def writeup(prompt):
        (tmp_path / "recon-report.md").write_text(
            "obs [MEASURED, job 1]\nSASS-WAIVED: x\nSCHED-WAIVED: x\n"
            "LIVE-WAIVED: x\nSTALL-WAIVED: x\nROLE-WAIVED: x\nPMTL-WAIVED: x")
        (tmp_path / "candidates.md").write_text("A")
        return "nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/p.py", writeup], [])
    recon = FakeBackend([], ["ORCH-OK regions=1 loops=1 nbar_ids=0 "
                             "mbar_objs=0 resolved=0 edges=0 "
                             "itc_candidates=0\n"])
    ctx = _stall_ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    assert not (tmp_path / "ref_orch_dossier.txt").exists()


def _dossier_log(kernel: str, instrs: int, verdict: str = "ORCH-OK") -> str:
    return (f"[stall_price] dominant launch index=0 name={kernel}\n"
            "=== ORCH DOSSIER BEGIN ===\n"
            f"[parse] {instrs} instructions, unparsed=1\n"
            "=== mbarrier objects (1, 1 resolved) ===\n"
            f"  [CONST] smem+0x{instrs:x}: ARRIVE@r1 x1 INIT@r0 x1 WAIT@r6 x1\n"
            "=== ORCH DOSSIER END ===\n"
            f"{verdict} regions=4 loops=2 nbar_ids=1 mbar_objs=1 resolved=1 "
            "edges=1 itc_candidates=0\n")


def test_banked_ref_dossier_is_not_overwritten_by_a_smaller_kernels_log(tmp_path):
    """Recon probes arrive one log at a time. A gdn_prefill_bwd run
    banked the main kernel (9112 instrs) off one probe, then the next probe —
    which only priced the auxiliary kernels — replaced it with prepare_h
    (6232 instrs): a 20%-of-runtime side kernel would have been the
    blueprint for the whole run. Across logs the same preference holds as
    within one: a banked OK dossier of a bigger kernel stays; an equal or
    bigger one supersedes."""
    from forgekernel.loop.driver import _save_ref_orch_dossier

    ctx = _stall_ctx(tmp_path, FakeBackend([], []))
    ctx.recon_path = tmp_path / "recon-report.md"
    dst = tmp_path / "ref_orch_dossier.txt"

    _save_ref_orch_dossier(ctx, _dossier_log("main_kernel", 9112))
    assert dst.read_text().startswith(
        "# banked: block 1/1 kernel=main_kernel instrs=9112 verdict=OK\n")

    _save_ref_orch_dossier(ctx, _dossier_log("prepare_h_kernel", 6232))
    assert "kernel=main_kernel instrs=9112" in dst.read_text()   # kept
    assert "prepare_h_kernel" not in dst.read_text()

    _save_ref_orch_dossier(ctx, _dossier_log("main_kernel_v2", 9500))
    assert "kernel=main_kernel_v2 instrs=9500" in dst.read_text()  # superseded


def test_recon_stall_delivery_persists_the_ref_ledger_for_diagnosis(tmp_path):
    """A delivered STALL-OK saves the ref's ledger where the diagnose step can
    re-read it, and the next plateau diagnosis carries it."""
    from forgekernel.loop.driver import run_recon_step

    def writeup(prompt):
        (tmp_path / "recon-report.md").write_text("obs [MEASURED, job 1]\n"
                                                  "SASS-WAIVED: x\nSCHED-WAIVED: x\n"
                                                  "LIVE-WAIVED: x\nROLE-WAIVED: x\n"
                                                  "PMTL-WAIVED: x")
        (tmp_path / "candidates.md").write_text("A")
        return "nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/p.py", writeup], [])
    recon = FakeBackend([], [_STALL_LOG_OK])
    ctx = _stall_ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    ledger = (tmp_path / "ref_stall_ledger.txt").read_text()
    assert "kernel.py:101" in ledger and "STALL-OK" not in ledger

    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    ctx2 = _stall_ctx(tmp_path, be)
    ctx2.recon_path = tmp_path / "recon-report.md"
    run_meta_step(ctx2)
    assert "Reference (collected at recon time)" in be.prompts[0]
    assert "kernel.py:230" in be.prompts[0]


def test_stall_ledger_truncation_keeps_the_top_rows(tmp_path):
    """Ledger rows print most-expensive first, and the diagnose hard rule holds the
    locus against the top rows — so a ledger longer than the prompt slice (the
    flat-distribution case this mechanism exists for: MAX_ROWS=80 rows > 9KB) must
    keep its HEAD when truncated. A tail slice drops exactly the rows that matter."""
    filler = "\n".join(
        f"     10  0.10% (cum 90.{i % 100:02d}%)  kernel.py:{2000 + i}  LDS.128 | mio:90%"
        for i in range(300))
    log = ("=== STALL LEDGER BEGIN ===\n"
           "[ledger] keying=source-line (81% of samples line-resolved)\n"
           "   9000 45.00% (cum  45.0%)  kernel.py:101  HMMA.16816.F32 | long_scoreboard:70%\n"
           + filler +
           "\n=== STALL LEDGER END ===\n"
           "STALL-OK kernel=main_kernel samples=99999 rows=80 coverage=90.1\n")
    assert len(log) > 9000  # the truncation path is actually exercised
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [log])
    ctx = _stall_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert "kernel.py:101" in be.prompts[0]  # the top row survived


# --- best-sha confirmation (post-stop) --------------------------------------
# One eval job is one node draw; the run's final ratio claim must not rest on a
# single draw (a fwd run's 1.0418x single-job reading re-measured at 0.969x).

_EVAL_CONFIRM = lambda dev, ref_dev, ok=True: (  # noqa: E731
    '=== JSON ===\n{"correctness": {"pass": %s}, "performance": '
    '{"bwd_median_ms": %r, "device_median_ms": %r, "ref_device_ms": %r, '
    '"vs_ref_device": %r}}'
    % ("true" if ok else "false", dev * 1.2, dev, ref_dev, ref_dev / dev)
)


class _ConfirmBackend(FakeBackend):
    def push_ref(self, ref, branch):
        self.calls.append(("push_ref", ref, branch))


def _confirm_ctx(tmp_path, be):
    from forgekernel.inner.trajectory import IterationRecord, append_record
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="a", correctness=True, verdict="KEEP",
        bwd_median_ms=3.0, device_median_ms=3.0, sha="shaworse"))
    append_record(ctx.status_path, IterationRecord(
        iter=1, decision_id="D2", change="b", correctness=True, verdict="KEEP",
        bwd_median_ms=2.0, device_median_ms=2.0, sha="shabest"))
    return ctx


def test_best_confirmation_reports_the_median_over_n_jobs(tmp_path):
    be = _ConfirmBackend([], [_EVAL_CONFIRM(2.0, 2.1),   # 1.05
                              _EVAL_CONFIRM(2.0, 1.94),  # 0.97
                              _EVAL_CONFIRM(2.0, 2.02)])  # 1.01
    out = run_best_confirmation(_confirm_ctx(tmp_path, be), jobs=3)
    assert ("push_ref", "shabest", "fs-demo-confirm") in be.calls
    assert out["sha"] == "shabest" and out["window"] == "device"
    assert len(out["jobs"]) == 3
    assert abs(out["median_ratio"] - 1.01) < 1e-9


def test_best_confirmation_skips_without_a_pushable_backend(tmp_path):
    # HEAD is not necessarily the best sha; measuring it would confirm the wrong thing
    be = FakeBackend([], [_EVAL_CONFIRM(2.0, 2.1)])
    assert run_best_confirmation(_confirm_ctx(tmp_path, be), jobs=3) is None
    assert not any(c[0] == "gpu" for c in be.calls)


def test_best_confirmation_needs_a_green_record_and_a_positive_budget(tmp_path):
    be = _ConfirmBackend([], [])
    ctx = _ctx(tmp_path, be)  # empty status.jsonl
    ctx.gate_window = "device"
    assert run_best_confirmation(ctx, jobs=3) is None
    assert run_best_confirmation(_confirm_ctx(tmp_path, be), jobs=0) is None
    assert not any(c[0] == "gpu" for c in be.calls)


def test_best_confirmation_spends_extra_attempts_on_unusable_samples(tmp_path):
    # a correctness break or a missing window reading is a wasted draw, not a sample
    be = _ConfirmBackend([], [_EVAL_CONFIRM(2.0, 2.1, ok=False),
                              _EVAL_CONFIRM(2.0, 2.1),
                              _EVAL_CONFIRM(2.0, 1.94),
                              _EVAL_CONFIRM(2.0, 2.02)])
    out = run_best_confirmation(_confirm_ctx(tmp_path, be), jobs=3, max_extra=2)
    assert len(out["jobs"]) == 3
    assert abs(out["median_ratio"] - 1.01) < 1e-9


def test_best_confirmation_short_run_is_reported_not_padded(tmp_path):
    # attempts run out with 2/3 samples: report the 2-job median rather than nothing
    be = _ConfirmBackend([], [_EVAL_CONFIRM(2.0, 2.1, ok=False),
                              _EVAL_CONFIRM(2.0, 2.1, ok=False),
                              _EVAL_CONFIRM(2.0, 2.1, ok=False),
                              _EVAL_CONFIRM(2.0, 1.94),
                              _EVAL_CONFIRM(2.0, 2.02)])
    out = run_best_confirmation(_confirm_ctx(tmp_path, be), jobs=3, max_extra=2)
    assert len(out["jobs"]) == 2 and out["requested"] == 3


def test_best_confirmation_gives_up_after_consecutive_backend_errors(tmp_path):
    from forgekernel.loop.driver import BackendError

    class _Down(_ConfirmBackend):
        def submit_gpu(self, branch, mode):
            self.calls.append(("gpu", mode))
            raise BackendError("queue died")

    be = _Down([], [])
    assert run_best_confirmation(_confirm_ctx(tmp_path, be), jobs=3) is None
    assert len([c for c in be.calls if c[0] == "gpu"]) == 2  # not jobs + max_extra


def test_single_ref_contract_drops_in_job_readings_on_out_of_job_ref_arms(tmp_path):
    from forgekernel.loop.driver import _enforce_single_ref
    from forgekernel.loop.parse import parse_eval

    log = _EVAL_CONFIRM(2.0, 2.1)  # bench smuggled an in-job comparator
    ctx = _confirm_ctx(tmp_path, _ConfirmBackend([], []))
    ctx.ref_rides_job = False
    ev = _enforce_single_ref(ctx, parse_eval(log))
    assert ev.ref_device_ms is None and ev.vs_ref_device is None
    assert ev.ref_median_ms is None and ev.vs_ref is None
    assert ev.device_median_ms == 2.0  # the student reading itself survives

    ctx2 = _confirm_ctx(tmp_path, _ConfirmBackend([], []))  # rides_job (default)
    ev2 = _enforce_single_ref(ctx2, parse_eval(log))
    assert ev2.vs_ref_device is not None  # the ref's own in-job reading is kept


def test_best_confirmation_absolute_mode_when_the_ref_cannot_ride_the_job(tmp_path):
    # blackbox run: bench times only the student; confirmation reports the absolute
    # median and leaves the ratio to the caller's ref re-anchor (median-vs-median)
    STUDENT_ONLY = lambda dev: (  # noqa: E731
        '=== JSON ===\n{"correctness": {"pass": true}, "performance": '
        '{"bwd_median_ms": %r, "device_median_ms": %r}}' % (dev * 1.2, dev))
    be = _ConfirmBackend([], [STUDENT_ONLY(2.1), STUDENT_ONLY(1.9), STUDENT_ONLY(2.0)])
    ctx = _confirm_ctx(tmp_path, be)
    ctx.ref_rides_job = False
    out = run_best_confirmation(ctx, jobs=3)
    assert out["median_ms"] == 2.0 and "median_ratio" not in out
    assert all(s["ratio"] is None for s in out["jobs"])


def test_stall_seed_lands_below_the_foundation_reset_floor(tmp_path):
    # The seed is a workspace commit; foundation patches reset to the baseline_ref
    # captured at run start. Seeded AFTER that capture, the first such reset drops
    # the toolkit and every later pricing job dies file-not-found (an earlier blindness bug,
    # reintroduced by the very feature built to prevent it). The ordering contract:
    # the seed commit must land before baseline_ref is read.
    class _HeadLogging(FakeBackend):
        def head(self):
            self.calls.append(("head",))
            return super().head()

    be = _HeadLogging([], [])
    ctx = _stall_ctx(tmp_path, be)
    ctx.deadline_epoch = -1.0  # inner loop stops on budget immediately
    run_two_layer(ctx, max_meta=0)
    seed_i = next(i for i, c in enumerate(be.calls) if c[0] == "commit")
    head_i = next(i for i, c in enumerate(be.calls) if c == ("head",))
    assert seed_i < head_i


def test_best_confirmation_degrades_to_absolute_when_no_ref_reading_comes_back(tmp_path):
    # ref_rides_job misjudged True (or the bench is defective): no sample carries a
    # ratio. Rejecting them would burn the whole job budget and return None — the
    # absolute median must survive instead, with every sample accepted.
    STUDENT_ONLY = lambda dev: (  # noqa: E731
        '=== JSON ===\n{"correctness": {"pass": true}, "performance": '
        '{"bwd_median_ms": %r, "device_median_ms": %r}}' % (dev * 1.2, dev))
    be = _ConfirmBackend([], [STUDENT_ONLY(2.1), STUDENT_ONLY(1.9), STUDENT_ONLY(2.0)])
    ctx = _confirm_ctx(tmp_path, be)
    assert ctx.ref_rides_job is True
    out = run_best_confirmation(ctx, jobs=3)
    assert out["median_ms"] == 2.0 and "median_ratio" not in out
    assert len(out["jobs"]) == 3
    assert len([c for c in be.calls if c[0] == "gpu"]) == 3  # no burned extra attempts


# --- suspicious-watermark confirmation ---------------------------------------
# A single-job outlier that takes the watermark poisons every later meta
# transaction (observed on a fwd run: 0.2585 vs re-measured 0.282 rolled back two rebuilds).

def _wm_ctx(tmp_path, be, margin=0.03):
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.watermark_margin = margin
    return ctx


def test_watermark_jump_confirmed_by_agreement_records_the_median_of_both(tmp_path):
    # 2.0 beats best 3.0 by 33%; one confirmation agrees (within margin) -> the jump
    # is real, and the watermark takes the median of the two draws, not the faster
    # first one (a draw 1.5% fast used to stand as the bar every later patch met)
    be = FakeBackend([REC("D1", 2.0)], [EVAL_DW(2.4, 2.0), NCU, EVAL_DW(2.4, 2.02)])
    rec = run_inner_iteration(_wm_ctx(tmp_path, be), 0, metric_before=3.0)
    assert rec.device_median_ms == pytest.approx(2.01)
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 2


def test_watermark_jump_not_confirmed_records_the_median(tmp_path):
    # first draw 2.0, confirmation 2.4 (outlier suspected), tie-break 2.35 -> median 2.35
    be = FakeBackend([REC("D1", 2.0)],
                     [EVAL_DW(2.4, 2.0), NCU, EVAL_DW(2.6, 2.4), EVAL_DW(2.5, 2.35)])
    rec = run_inner_iteration(_wm_ctx(tmp_path, be), 0, metric_before=3.0)
    assert rec.device_median_ms == 2.35
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 3


def test_watermark_within_noise_improvement_costs_nothing(tmp_path):
    # 2.97 vs best 3.0 is inside the 3% margin -> no confirmation job
    be = FakeBackend([REC("D1", 2.97)], [EVAL_DW(3.1, 2.97), NCU])  # no extra log needed
    rec = run_inner_iteration(_wm_ctx(tmp_path, be), 0, metric_before=3.0)
    assert rec.device_median_ms == 2.97
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 1


def test_watermark_confirmation_infra_failure_keeps_the_reading(tmp_path):
    from forgekernel.loop.driver import BackendError

    class _FlakyConfirm(FakeBackend):
        def submit_gpu(self, branch, mode):
            if len([c for c in self.calls if c == ("gpu", "all")]) == 1 and mode == "all":
                self.calls.append(("gpu", mode))
                raise BackendError("queue died")
            return super().submit_gpu(branch, mode)

    be = _FlakyConfirm([REC("D1", 2.0)], [EVAL_DW(2.4, 2.0), NCU])
    rec = run_inner_iteration(_wm_ctx(tmp_path, be), 0, metric_before=3.0)
    assert rec.device_median_ms == 2.0  # a flaky queue must not veto a measured KEEP


def _seed_best_keep(ctx, ms=2.0, t_ns=280000):
    from forgekernel.inner.trajectory import IterationRecord, append_record
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="a", correctness=True, verdict="KEEP",
        bwd_median_ms=ms, device_median_ms=ms, profiler={"time_ns": t_ns}, sha="shabest"))


def test_watermark_ncu_regression_triggers_confirmation_below_the_margin(tmp_path):
    # A worse kernel on a fast node: device reading improves 1.5% (under the 3%
    # margin) while its clock-locked NCU time regressed 10.7% vs the sitting best.
    # The free cross-signal must trigger the confirmation the margin alone misses.
    from forgekernel.loop.driver import _confirm_suspicious_watermark
    from forgekernel.loop.parse import parse_eval
    be = FakeBackend([], [EVAL_DW(2.4, 2.05), EVAL_DW(2.4, 2.06)])
    ctx = _wm_ctx(tmp_path, be)
    _seed_best_keep(ctx, ms=2.0, t_ns=280000)
    ev = parse_eval(EVAL_DW(2.36, 1.97))
    out = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0,
                                        profiler={"time_ns": 310000})
    assert out.device_median_ms == 2.05  # median of 1.97 / 2.05 / 2.06 — fake best rejected
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 2


def test_watermark_ncu_consistent_small_improvement_costs_nothing(tmp_path):
    from forgekernel.loop.driver import _confirm_suspicious_watermark
    from forgekernel.loop.parse import parse_eval
    be = FakeBackend([], [])
    ctx = _wm_ctx(tmp_path, be)
    _seed_best_keep(ctx, ms=2.0, t_ns=280000)
    ev = parse_eval(EVAL_DW(2.36, 1.97))
    out = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0,
                                        profiler={"time_ns": 279000})
    assert out.device_median_ms == 1.97
    assert not be.calls  # in-band improvement, NCU consistent — free


def test_watermark_tiebreak_failure_does_not_reinstate_the_outlier(tmp_path):
    """"A flaky queue must not veto a KEEP" covers the FIRST confirmation only.
    Once a dissenting confirmation measured the jump as unconfirmed, a dead
    tie-break job settles on the median of what was measured — returning the
    original reading would let the very outlier the dissent disproved take the
    watermark after all."""
    from forgekernel.loop.driver import BackendError, _confirm_suspicious_watermark
    from forgekernel.loop.parse import parse_eval

    class _DiesOnTieBreak(FakeBackend):
        def submit_gpu(self, branch, mode):
            if mode == "all" and len([c for c in self.calls if c == ("gpu", "all")]) == 1:
                self.calls.append(("gpu", mode))
                raise BackendError("queue died")
            return super().submit_gpu(branch, mode)

    be = _DiesOnTieBreak([], [EVAL_DW(2.6, 2.4)])  # confirm disagrees, tie-break dies
    ctx = _wm_ctx(tmp_path, be)
    ev = parse_eval(EVAL_DW(2.4, 2.0))  # suspect jump vs best 3.0
    out = _confirm_suspicious_watermark(ctx, be, "b", ev, 3.0, profiler={})
    assert out.device_median_ms == pytest.approx(2.2)  # median of 2.0 / 2.4


def test_watermark_ncu_trigger_is_not_cleared_by_ms_agreement_alone(tmp_path, monkeypatch):
    """Bimodal queue: the confirmation job lands in the same fast-node pool as
    the suspect, so band-consistent ms readings are near coin-flip likely even
    for a genuinely worse kernel. The alarm came from the kernel clock — the
    confirmation must re-profile, and an uncleared regression tie-breaks."""
    from forgekernel.loop import driver
    from forgekernel.loop.driver import _confirm_suspicious_watermark
    from forgekernel.loop.parse import parse_eval
    monkeypatch.setattr(driver, "_ncu_profiler",
                        lambda ctx, branch, correctness: {"time_ns": 315000})
    be = FakeBackend([], [EVAL_DW(2.36, 1.98), EVAL_DW(2.4, 2.05)])
    ctx = _wm_ctx(tmp_path, be)
    _seed_best_keep(ctx, ms=2.0, t_ns=280000)
    ev = parse_eval(EVAL_DW(2.36, 1.97))  # in-band ms, NCU time regressed 10.7%
    out = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0,
                                        profiler={"time_ns": 310000})
    assert out.device_median_ms == pytest.approx(1.98)  # median of 1.97/1.98/2.05
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 2


def test_watermark_ncu_trigger_cleared_by_the_reprofile_confirms(tmp_path, monkeypatch):
    # Same setup, but the re-profiled kernel clock recovered — the jump is real
    # and the median of the two draws stands, with no tie-break job spent.
    from forgekernel.loop import driver
    from forgekernel.loop.driver import _confirm_suspicious_watermark
    from forgekernel.loop.parse import parse_eval
    monkeypatch.setattr(driver, "_ncu_profiler",
                        lambda ctx, branch, correctness: {"time_ns": 282000})
    be = FakeBackend([], [EVAL_DW(2.36, 1.98)])
    ctx = _wm_ctx(tmp_path, be)
    _seed_best_keep(ctx, ms=2.0, t_ns=280000)
    ev = parse_eval(EVAL_DW(2.36, 1.97))
    out = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0,
                                        profiler={"time_ns": 310000})
    assert out.device_median_ms == pytest.approx(1.975)
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 1


# --- segment ledger: aligned same-position comparison -------------------------

_SEG_BLOCK_STUDENT = (
    "=== SEGMENT LEDGER BEGIN ===\n"
    "[join] 200 sampled PCs vs 900 parsed addresses; 98.0% of 2000 stall samples resolved\n"
    "SEG idx=0 kind=SYNCS depth=none pct=6.0 stalls=long_sb:80,barrier:12\n"
    "SEG idx=1 kind=HG depth=none pct=4.0 stalls=dispatch:55\n"
    "SEG idx=2 kind=DEPBAR depth=none imm=0x1 pct=7.0 stalls=barrier:90\n"
    "SEG idx=3 kind=EW depth=0x1 pct=35.4 stalls=wait:51,barrier:20\n"
    "SEG idx=4 kind=DEPBAR depth=0x1 imm=0x0 pct=5.0 stalls=barrier:88\n"
    "SEG idx=5 kind=EW depth=0x0 pct=20.0 stalls=wait:44\n"
    "SEG idx=6 kind=BRA depth=0x0 pct=12.0 stalls=long_sb:88\n"
    "JOIN-OK resolved=98.0% body_samples=1960 segments=7 hgmma_runs=1 depbars=2\n"
    "=== SEGMENT LEDGER END ===\n")

_SEG_BLOCK_REF = (
    "SEG idx=0 kind=SYNCS depth=none pct=5.0 stalls=long_sb:70\n"
    "SEG idx=1 kind=HG depth=none pct=6.0 stalls=dispatch:60\n"
    "SEG idx=2 kind=DEPBAR depth=none imm=0x1 pct=3.0 stalls=barrier:85\n"
    "SEG idx=3 kind=EW depth=0x1 pct=23.5 stalls=dispatch:47,selected:22\n"
    "SEG idx=4 kind=DEPBAR depth=0x1 imm=0x0 pct=4.0 stalls=barrier:80\n"
    "SEG idx=5 kind=EW depth=0x0 pct=22.0 stalls=dispatch:40\n"
    "SEG idx=6 kind=BRA depth=0x0 pct=18.0 stalls=long_sb:90\n"
    "JOIN-OK resolved=97.0% body_samples=1900 segments=7 hgmma_runs=1 depbars=2\n")


def test_align_segment_table_puts_same_anchor_side_by_side():
    from forgekernel.loop.driver import align_segment_table
    table = align_segment_table(_SEG_BLOCK_REF, _SEG_BLOCK_STUDENT)
    ew_row = next(l for l in table.splitlines() if l.startswith("EW@0x1"))
    # the reading that needed manual alignment before: 23.5 vs 35.4, dispatch vs wait
    assert "23.5" in ew_row and "35.4" in ew_row
    assert "dispatch:47" in ew_row and "wait:51" in ew_row
    assert any(l.startswith("DEPBAR 0x1") and "3.0" in l and "7.0" in l
               for l in table.splitlines())
    assert align_segment_table("no seg lines here", _SEG_BLOCK_STUDENT) is None


def test_stall_pricing_injects_the_aligned_table_when_both_sides_have_segments(tmp_path):
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                            "patch_class: additive\nchanged_entries: []\n")],
                          [_STALL_LOG_OK + _SEG_BLOCK_STUDENT])
    ctx = _stall_ctx(tmp_path, be)
    recon_dir = tmp_path / "recon"
    recon_dir.mkdir()
    ctx.recon_path = recon_dir / "report.md"
    (recon_dir / "ref_segment_ledger.txt").write_text(_SEG_BLOCK_REF)
    (recon_dir / "ref_stall_ledger.txt").write_text("ref row ledger\n")
    run_meta_step(ctx)
    prompt = be.prompts[0]
    assert "same-position segment comparison" in prompt
    assert "23.5" in prompt and "35.4" in prompt
    assert "ref row ledger" not in prompt  # raw ledgers replaced by the aligned table


def test_stall_pricing_falls_back_to_raw_ledgers_without_a_ref_segment_account(tmp_path):
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                            "patch_class: additive\nchanged_entries: []\n")],
                          [_STALL_LOG_OK + _SEG_BLOCK_STUDENT])
    ctx = _stall_ctx(tmp_path, be)
    run_meta_step(ctx)  # no recon_path -> no ref segment file
    assert "kernel.py:101" in be.prompts[0]  # the raw row ledger still lands


def test_stall_pricing_segment_join_verdict_reads_the_full_block(tmp_path):
    """JOIN-OK is segment_ledger's LAST line and SEG rows are unbounded, so a
    many-segment kernel's block overruns the head-slice. Slicing before the
    verdict check read every successful join as unsafe — disabling the aligned
    table on exactly the big kernels it exists for."""
    from forgekernel.loop.driver import _run_stall_pricing
    rows = "".join(
        f"SEG idx={i} kind=EW depth=0x0 pct=0.10 stalls=wait:40,barrier:20\n"
        for i in range(150))  # ~8.5 KB of SEG rows: JOIN-OK sits past the 6000-char slice
    big = ("=== SEGMENT LEDGER BEGIN ===\n"
           "[join] 200 sampled PCs vs 900 parsed addresses\n" + rows +
           "JOIN-OK resolved=98.0% body_samples=1960 segments=150 hgmma_runs=1 depbars=2\n"
           "=== SEGMENT LEDGER END ===\n")
    be = FakeBackend([], [_STALL_LOG_OK + big])
    ctx = _stall_ctx(tmp_path, be)
    body, seg, _, _ = _run_stall_pricing(ctx)
    assert body is not None
    assert seg is not None  # the join DID succeed; the verdict must see the full block
    assert len(seg) <= 6000  # the prompt still gets only the head-slice
    # and a genuinely unsafe join is still dropped, however small the block
    be2 = FakeBackend([], [_STALL_LOG_OK +
                           "=== SEGMENT LEDGER BEGIN ===\nSEG idx=0 kind=EW depth=0x0 "
                           "pct=1.0 stalls=wait:40\n=== SEGMENT LEDGER END ===\n"])
    ctx2 = _stall_ctx(tmp_path, be2)
    _, seg2, _, _ = _run_stall_pricing(ctx2)
    assert seg2 is None


_ORCH_SUMMARY_REF = ('ORCH-SUMMARY-JSON: {"commit": ["r1: DEPBAR {LE 0x0 x2, '
                     'LE 0x1 x1}"], "mbar_objs": 22}')
_ORCH_SUMMARY_STU = ('ORCH-SUMMARY-JSON: {"commit": ["r1: DEPBAR {LE 0x0 x2}"],'
                     ' "mbar_objs": 9}')
_ORCH_PRICING_BLOCK = (
    "=== ORCH DOSSIER BEGIN ===\n"
    "[parse] 2487 instructions, unparsed=3\n"
    + _ORCH_SUMMARY_STU + "\n"
    "=== ORCH DOSSIER END ===\n"
    "ORCH-OK regions=3 loops=1 nbar_ids=8 mbar_objs=9 resolved=8 edges=5 "
    "itc_candidates=0\n")


def _seed_ref_orch(ctx, tmp_path):
    recon_dir = tmp_path / "recon"
    recon_dir.mkdir(exist_ok=True)
    ctx.recon_path = recon_dir / "report.md"
    (recon_dir / "ref_orch_dossier.txt").write_text(
        "[parse] 3541 instructions\n" + _ORCH_SUMMARY_REF + "\n")
    return recon_dir


def test_orch_dossier_diff_reaches_the_diagnose_prompt(tmp_path):
    """The pricing job's student dossier is banked beside the ref's and the
    driver-side diff (summary lines, no SASS needed) lands as a diagnose
    evidence block — the 'blueprint says prefetched, never built' class
    becomes a `!` row before any timing argument."""
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK + _ORCH_PRICING_BLOCK])
    ctx = _stall_ctx(tmp_path, be)
    recon_dir = _seed_ref_orch(ctx, tmp_path)
    run_meta_step(ctx)
    prompt = be.prompts[0]
    assert "Orchestration dossier diff" in prompt
    assert "! mbar_objs" in prompt and "ref:     22" in prompt
    assert "LE 0x1" in prompt
    # the student dossier body is banked for later audits
    banked = (recon_dir / "student_orch_dossier.txt").read_text()
    assert "2487 instructions" in banked


def test_orch_dossier_diff_names_the_gap_without_a_student_summary(tmp_path):
    """A pricing log with no dossier must not render a half-sided diff — and
    must not stay silent either: the missing face is named so the teacher
    does not read the ref side as if the student had been compared."""
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    ctx = _stall_ctx(tmp_path, be)
    _seed_ref_orch(ctx, tmp_path)
    run_meta_step(ctx)
    prompt = be.prompts[0]
    assert "Faces absent this time" in prompt and "dossier unavailable" in prompt
    assert "ref (recon time) vs" not in prompt     # no face rendered one-sided


def test_distill_prompt_carries_the_ref_orch_dossier(tmp_path):
    """The banked recon-time dossier travels INTO the distill prompt so the
    [SYNC:]/[GATHER:]/[PHASE:] adjudications cite machine readings instead of
    guessing; absent the file, the prompt is byte-identical to the status quo
    (control-run principle)."""
    from forgekernel.loop.driver import run_distill_step

    def author(prompt):
        (tmp_path / "milestone.md").write_text(_MILESTONE, encoding="utf-8")
        return "authored"

    be = FakeBackend([author], [])
    ctx = _ctx(tmp_path, be)
    _seed_ref_orch(ctx, tmp_path)
    run_distill_step(ctx)
    assert "Reference orchestration dossier" in be.prompts[0]
    assert "3541 instructions" in be.prompts[0]

    be2 = FakeBackend([author], [])
    ctx2 = _ctx(tmp_path, be2)
    run_distill_step(ctx2)  # no recon_path -> no dossier block
    assert "Reference orchestration dossier" not in be2.prompts[0]


def test_m0_audit_prompt_carries_the_ref_orch_dossier(tmp_path):
    """The M0 audit gets the ref dossier for the paper-level contract check,
    AND prices the tree under judgment so the built structure is read too:
    the audit is a teacher entry like any other."""
    be = FakeBackend([
        "cause_class: OK\ndeficient_locus: -\npatch_direction: -\n"
        "evidence: checked\n"], [_STALL_LOG_OK + _ORCH_PRICING_BLOCK])
    ctx = _stall_ctx(tmp_path, be)
    recon_dir = _seed_ref_orch(ctx, tmp_path)
    run_meta_step(ctx, audit=True)
    prompt = be.prompts[0]
    assert "Reference orchestration dossier" in prompt
    assert "for the audit's ruling on the blueprint" in prompt
    # the judged tree was priced (one job: no KEEP yet, HEAD is the only tree)
    assert len([c for c in be.calls if c[0] == "gpu"]) == 1
    assert "Orchestration dossier diff" in prompt and "vs HEAD" in prompt
    assert "! mbar_objs" in prompt and "student: 9" in prompt
    assert "student best" not in prompt  # no second tree, no second face
    assert "Routing rule" in prompt
    assert (recon_dir / "student_orch_dossier.txt").is_file()
    assert not (recon_dir / "student_orch_dossier_best.txt").exists()
    # the audit prompt has no ledger rules: the ledger blocks stay diagnose-only
    assert "Per-instruction stall ledger" not in prompt


_ORCH_SUMMARY_HEAD = ('ORCH-SUMMARY-JSON: {"commit": ["r1: DEPBAR {LE 0x0 x2, '
                      'LE 0x1 x1}"], "mbar_objs": 14}')
_ORCH_PRICING_BLOCK_HEAD = (
    "=== ORCH DOSSIER BEGIN ===\n"
    "[parse] 2600 instructions, unparsed=1\n"
    + _ORCH_SUMMARY_HEAD + "\n"
    "=== ORCH DOSSIER END ===\n"
    "ORCH-OK regions=3 loops=1 nbar_ids=8 mbar_objs=14 resolved=8 edges=5 "
    "itc_candidates=0\n")


def _two_tree_ctx(tmp_path, agent_outputs):
    """A segment whose best KEEP ("shabest") is not the tree under judgment
    (the fake's HEAD is "sha0"): the shape of every M0 rebuild after the first,
    and of a plateau whose last KEEPs were slower than the best."""
    from forgekernel.inner.trajectory import IterationRecord, append_record

    class _RefPushing(FakeBackend):
        def push_ref(self, ref, branch):
            self.calls.append(("push_ref", ref, branch))

    be = _RefPushing(agent_outputs,
                     [_STALL_LOG_OK + _ORCH_PRICING_BLOCK,       # best: mbar 9
                      _STALL_LOG_OK + _ORCH_PRICING_BLOCK_HEAD])  # HEAD: mbar 14
    ctx = _stall_ctx(tmp_path, be)
    recon_dir = _seed_ref_orch(ctx, tmp_path)
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="a", correctness=True, verdict="KEEP",
        bwd_median_ms=2.0, device_median_ms=2.0, sha="shabest"))
    append_record(ctx.status_path, IterationRecord(
        iter=1, decision_id="M0-gate", change="rebuild", correctness=True,
        verdict="KEEP", bwd_median_ms=2.4, device_median_ms=2.4, sha="sha0"))
    return be, ctx, recon_dir


def _assert_three_faces(be, recon_dir):
    prompt = be.prompts[0]
    pushes = [c for c in be.calls if c[0] == "push_ref"]
    assert pushes == [("push_ref", "shabest", "fs-demo-stallprice"),
                      ("push_ref", "sha0", "fs-demo-stallprice")]
    assert "### ref (recon time) vs HEAD sha0" in prompt
    assert "### ref (recon time) vs student best shabest" in prompt
    assert "### student best shabest → HEAD sha0" in prompt
    # the HEAD face reads the HEAD dossier, the best face the best dossier
    head_face = prompt.split("### ref (recon time) vs HEAD sha0")[1].split("### ")[0]
    best_face = prompt.split("### ref (recon time) vs student best")[1].split("### ")[0]
    move_face = prompt.split("### student best shabest → HEAD sha0")[1]
    assert "student: 14" in head_face and "student: 9" not in head_face
    assert "student: 9" in best_face
    assert "(best vs HEAD)" in move_face
    assert "best: 9" in move_face and "HEAD: 14" in move_face
    # both dossiers are banked under their own names
    assert "2600 instructions" in (recon_dir / "student_orch_dossier.txt").read_text()
    assert "2487 instructions" in (recon_dir / "student_orch_dossier_best.txt").read_text()


def test_m0_audit_prices_head_and_best_and_diffs_all_three_faces(tmp_path):
    """An M0 rebuild's audit reads the rebuilt tree (HEAD), the best-so-far
    tree, and what the rebuild changed between them — the reading that says
    whether the previous foundation patch was built, not just re-worded."""
    be, ctx, recon_dir = _two_tree_ctx(tmp_path, [
        "cause_class: OK\ndeficient_locus: -\npatch_direction: -\nevidence: checked\n"])
    run_meta_step(ctx, audit=True)
    _assert_three_faces(be, recon_dir)


def test_diagnose_prompt_carries_the_best_to_head_face(tmp_path):
    """The plateau path gets the same three faces; its ledger still comes from
    the best tree (the diagnose ledger rules speak of 'student best')."""
    be, ctx, recon_dir = _two_tree_ctx(tmp_path, [
        _DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                         "patch_class: additive\nchanged_entries: []\n")])
    run_meta_step(ctx)
    _assert_three_faces(be, recon_dir)
    assert "Per-instruction stall ledger" in be.prompts[0]


_ORCH_FAIL_BLOCK = (
    "=== ORCH DOSSIER BEGIN ===\n"
    "[parse] 2487 instructions, unparsed=3\n"
    'ORCH-SUMMARY-JSON: {"commit": [], "mbar_objs": 0, "nbar_verdicts": [], '
    '"verdict": "ORCH-FAIL no-sync-sites"}\n'
    "=== ORCH DOSSIER END ===\n"
    "[orch] raw samples for debugging:\n"
    "    0010  LDC R1, c[0x0][0x28]\n"
    "ORCH-FAIL no-sync-sites\n")


def test_orch_fail_dossier_is_banked_and_flagged(tmp_path):
    """No ORCH-OK line, but a parsable summary: the dossier is banked, the
    diff renders (mbar_objs 22 vs 0 is the whole story), and the face opens
    with the verdict — the 'declared, never built' case must not vanish."""
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK + _ORCH_FAIL_BLOCK])
    ctx = _stall_ctx(tmp_path, be)
    recon_dir = _seed_ref_orch(ctx, tmp_path)
    run_meta_step(ctx)
    prompt = be.prompts[0]
    assert "## Orchestration dossier diff" in prompt
    assert "HEAD page verdict `ORCH-FAIL no-sync-sites`" in prompt
    assert "! mbar_objs" in prompt and "student: 0" in prompt
    assert "ORCH-FAIL no-sync-sites" in (recon_dir / "student_orch_dossier.txt").read_text()


def test_meta_prices_once_when_head_is_the_best(tmp_path):
    """HEAD == best: one job, its dossier banked under both names, and a
    single ref-vs-HEAD face (no best face, no move face)."""
    from forgekernel.inner.trajectory import IterationRecord, append_record

    be = FakeBackend([
        "cause_class: OK\ndeficient_locus: -\npatch_direction: -\nevidence: checked\n"],
        [_STALL_LOG_OK + _ORCH_PRICING_BLOCK])
    ctx = _stall_ctx(tmp_path, be)
    recon_dir = _seed_ref_orch(ctx, tmp_path)
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="M0-gate", change="a", correctness=True, verdict="KEEP",
        bwd_median_ms=2.0, device_median_ms=2.0, sha="sha0"))
    run_meta_step(ctx, audit=True)
    assert len([c for c in be.calls if c[0] == "gpu"]) == 1
    prompt = be.prompts[0]
    assert "### ref (recon time) vs HEAD sha0" in prompt
    assert "student best" not in prompt
    assert (recon_dir / "student_orch_dossier.txt").read_text() == \
        (recon_dir / "student_orch_dossier_best.txt").read_text()


def test_truncated_recon_turns_are_retried_without_charging_rounds(tmp_path, monkeypatch):
    """A gateway stream that dies early yields a tiny, traceless turn with rc=0 —
    charged as ordinary rounds, one bad hour spent 8 of 12 GPU rounds on nothing
    (observed on bwd-mha). Tiny AND traceless is retried free; a short turn that emits a
    request or writes artifacts is legal and never retried."""
    from forgekernel.loop import driver
    from forgekernel.loop.driver import run_recon_step

    monkeypatch.setattr(driver, "TRUNCATION_BACKOFF_S", (0.0, 0.0, 0.0))

    def write_up(prompt):
        (tmp_path / "recon-report.md").write_text(
            "observations\nSASS-WAIVED: x\nSCHED-WAIVED: x\nLIVE-WAIVED: x")
        (tmp_path / "candidates.md").write_text("A|B")
        return "written"

    teacher = FakeBackend(
        ["Next I will measure the exchange face.",       # truncated announce-then-yield -> free retry
         "Next I will measure the exchange face.",       # truncated again -> second free retry
         "RECON_REQUEST: notes/p.py",   # the turn that actually happened
         write_up], [])
    recon = FakeBackend([], ["probe output"])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    # 4 teacher turns happened, but only 1 probing round was charged: the two
    # truncated turns cost retries, not measurement opportunities
    assert result.rounds == 1
    assert len([c for c in teacher.calls if c[0] == "agent"]) == 4


def test_truncation_storm_stops_the_run_as_agent_failed(tmp_path, monkeypatch):
    """Past the storm budget the channel is systematically sick — the run must stop
    with a diagnosis (AgentError -> stop=agent_failed, the watchdog-restartable
    class), not spend the whole recon budget on re-affirmation theater."""
    from forgekernel.loop import driver
    from forgekernel.loop.driver import AgentError, run_recon_step

    monkeypatch.setattr(driver, "TRUNCATION_BACKOFF_S", (0.0, 0.0, 0.0))
    monkeypatch.setattr(driver, "TRUNCATION_STORM_BUDGET", 1)
    teacher = FakeBackend(["OK.", "OK."], [])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = FakeBackend([], [])
    ctx.recon_dir = tmp_path
    ctx.recon_path = tmp_path / "recon-report.md"
    with pytest.raises(AgentError, match="truncation storm"):
        run_recon_step(ctx)


def test_truncated_distill_turn_does_not_burn_a_patch_round(tmp_path, monkeypatch):
    """A distill turn whose stream died leaves the milestone byte-identical and a
    tiny output. Re-prompting it as a schema round would burn one of only
    max_patch_rounds chances (a bwd-mha run lost v1 to exactly this); it is
    retried free instead, and the next attempt's write goes through."""
    from forgekernel.loop import driver
    from forgekernel.loop.driver import run_distill_step

    monkeypatch.setattr(driver, "TRUNCATION_BACKOFF_S", (0.0, 0.0, 0.0))

    def write_v1(prompt):
        (tmp_path / "m.md").write_text(_MILESTONE + "\n<!-- v1 -->", encoding="utf-8")
        return "milestone written"

    teacher = FakeBackend(["I will start writing the milestone now.", write_v1], [])
    ctx = _ctx(tmp_path, teacher)
    result = run_distill_step(ctx)
    assert result.milestone_violations == []
    assert "<!-- v1 -->" in (tmp_path / "m.md").read_text(encoding="utf-8")
    # both teacher calls belong to ONE schema round (truncation retry, then write)
    assert len([c for c in teacher.calls if c[0] == "agent"]) == 2


def test_ncu_profiler_merges_census_line(tmp_path):
    """A SASS-OPERANDS line in the ncu job log lands in the digest; without one the
    digest is unchanged (census is opt-in per operator)."""
    from types import SimpleNamespace
    from pathlib import Path as _P
    from forgekernel.loop.driver import _ncu_profiler
    csv = (_P(__file__).parent / "fixtures" / "ncu_sample.csv").read_text()
    with_census = csv + "\nSASS-OPERANDS kernel=BwdMainKernel mma_total=40 a_reg=8 a_desc=32 a_other=0\n"
    ctx = SimpleNamespace(backend=FakeBackend([], [with_census, csv]), ncu_retries=0)
    prof = _ncu_profiler(ctx, "b1", correctness=True)
    assert prof["mma_a_reg"] == 8 and prof["mma_a_desc"] == 32
    assert prof["mma_census_kernel"] == "BwdMainKernel"
    prof2 = _ncu_profiler(ctx, "b2", correctness=True)
    assert "mma_a_reg" not in prof2 and prof2["regs_per_thread"] == 168


# --- host-bound audit on an evaluated round -----------------------------------
#
# The device window equals device time only while the host keeps the pipeline fed.
# The two sides of a comparison need OPPOSITE handling when it does not: a host-bound
# STUDENT has a real defect worth telling it about, a host-bound REFERENCE has a
# measurement problem the student must never be told about (a kernel writer reads
# "the reference's python is thick" as "a thin python path wins").

EVAL_HB = lambda ms, dev, host, ref_ratio=None: (  # noqa: E731
    '=== JSON ===\n{"correctness": {"pass": true}, "performance": '
    '{"bwd_median_ms": %r, "device_median_ms": %r, "host_enqueue_ms": %r, '
    '"host_bound_ratio": %r, "bwd_mfu_pct": 32.0%s}}'
    % (ms, dev, host, host / dev,
       "" if ref_ratio is None else ', "cublas_host_bound_ratio": %r' % ref_ratio)
)


def test_the_enqueue_clock_lands_in_the_trajectory(tmp_path):
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL_HB(0.52, 0.4085, 0.004), NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.host_enqueue_ms == 0.004
    assert rec.host_bound_ratio == pytest.approx(0.0098, rel=1e-2)
    # and survives the round-trip through status.jsonl, which is what the meta agent
    # and the digest actually read
    assert load_records(ctx.status_path)[0].host_bound_ratio == rec.host_bound_ratio


def test_a_host_bound_student_round_is_diagnosed_to_the_student(tmp_path):
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL_HB(0.52, 0.40, 0.36), NCU])
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 0, metric_before=3.0)
    assert ctx.host_bound_note
    assert "host" in ctx.host_bound_note


def test_a_gpu_bound_student_round_leaves_no_note(tmp_path):
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL_HB(0.52, 0.4085, 0.004), NCU])
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 0, metric_before=3.0)
    assert ctx.host_bound_note == ""


def test_a_host_bound_reference_never_becomes_student_feedback(tmp_path):
    # Student thin (0.004 / 0.4085), in-job reference host-paced (0.964). The round
    # must produce NO student note: the reference's wrapper is a measurement fact.
    be = FakeBackend([REC("D2.3", 2.6)],
                     [EVAL_HB(0.52, 0.4085, 0.004, ref_ratio=0.964), NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.ref_host_bound_ratio == 0.964
    assert ctx.host_bound_note == ""


EVAL_SPAN = (  # dsa_sparse, MEASURED: the card's span-basis JSON
    '=== JSON ===\n{"correctness": {"pass": true}, "performance": '
    '{"bwd_median_ms": 0.08398, "device_median_ms": 0.01522, '
    '"device_window_basis": "cupti_kernel_span", "host_enqueue_ms": 0.06463, '
    '"host_bound_ratio": null, "host_bound_ratio_plain": 4.246, '
    '"ref_device_ms": 0.00957, "ref_host_enqueue_ms": 0.01911, '
    '"ref_host_bound_ratio": null, "bwd_mfu_pct": 0.088}}')


def test_a_span_basis_round_is_not_audited_as_host_bound(tmp_path, capsys):
    # The raw timings divide to 4.25; on a span basis that is not a verdict on
    # the gate reading. No WARNING, no "inflated" diagnosis — a deployment NOTE
    # for the operator and a "gate is clean" note for the student instead.
    be = FakeBackend([REC("D0.4", 0.05)], [EVAL_SPAN, NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=0.06)
    out = capsys.readouterr().out
    assert "HOST-BOUND STUDENT" not in out
    assert "inflated" not in out
    note = next(l for l in out.splitlines() if "NOTE HOST PATH:" in l)
    assert "NOT affected" in note and "3.4x thinner" in note
    assert rec.host_bound_ratio is None
    assert rec.device_window_basis == "cupti_kernel_span"
    assert load_records(ctx.status_path)[0].device_window_basis == "cupti_kernel_span"
    assert "not part of the gate" in ctx.host_bound_note and "not inflated by the host" in ctx.host_bound_note
    assert "The pipeline is not kept full" not in ctx.host_bound_note


def test_a_plain_basis_round_with_the_same_timings_is_still_audited(tmp_path, capsys):
    plain = EVAL_SPAN.replace('"cupti_kernel_span"', '"plain_batch"').replace(
        '"host_bound_ratio": null', '"host_bound_ratio": 4.246')
    be = FakeBackend([REC("D0.4", 0.05)], [plain, NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=0.06)
    assert "HOST-BOUND STUDENT" in capsys.readouterr().out
    assert rec.host_bound_ratio == pytest.approx(4.246)
    assert "The pipeline is not kept full" in ctx.host_bound_note


def test_the_note_is_delivered_once_then_cleared(tmp_path):
    from forgekernel.loop.driver import _student_prompt
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.host_bound_note = "- Timing diagnosis: host-bound\n"
    first = _student_prompt(ctx, 1)
    assert "Timing diagnosis" in first
    assert ctx.host_bound_note == ""
    assert "Timing diagnosis" not in _student_prompt(ctx, 2)


def test_mismatch_line_without_observed_near_miss_does_not_close_the_gate(tmp_path):
    """SASS-DELIVERED-MISMATCH only counts when the driver itself recorded a
    verdict-shaped near-miss for the channel — otherwise the line is a free
    bypass weaker than a waiver, and the human audit would read a missing
    delivery as a checker defect."""
    from forgekernel.loop.driver import run_recon_step

    bounced = {}

    def writeup_with_bare_mismatch(prompt):
        (tmp_path / "recon-report.md").write_text(
            "observations\nSASS-DELIVERED-MISMATCH: whatever, trust me"
            "\nSCHED-WAIVED: not collected\nLIVE-WAIVED: not collected")
        (tmp_path / "candidates.md").write_text("A|B")
        return "done, nothing further"

    def waive(prompt):
        bounced["prompt"] = prompt
        assert "SASS evidence missing" in prompt
        report = tmp_path / "recon-report.md"
        report.write_text(report.read_text()
                          + "\nSASS-WAIVED: the channel really produced no verdict; waived")
        return "waived, nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/p.py",
                           writeup_with_bare_mismatch, waive], [])
    recon = FakeBackend([], ["job ran, no verdict lines at all"])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    assert "prompt" in bounced, "the bare mismatch line closed the gate for free"


def test_verdict_near_miss_keeps_the_latest_line():
    """A round-1 garbage line (partial echo) must not shadow the real tool
    verdict a later round prints — the gate note quotes the stored line
    verbatim, and preserving the wrong side of the disagreement instructs the
    teacher to enshrine garbage in the DELIVERED-MISMATCH record."""
    from forgekernel.loop.driver import _collect_verdict_near_misses

    near = {}
    _collect_verdict_near_misses("noise\nSASS-OK kern", near)
    assert near["SASS"] == "SASS-OK kern"
    _collect_verdict_near_misses(
        "SASS-OK kernel=real_tool_output n=3520 fmt=v2", near)
    assert near["SASS"] == "SASS-OK kernel=real_tool_output n=3520 fmt=v2"


def test_validate_milestone_file_has_a_strict_authoring_mode(tmp_path):
    """Resume/patch paths must not read a pre-tag milestone as 'not distilled
    yet'; the freshly-authored path must still enforce the tags."""
    from forgekernel.loop.driver import _validate_milestone_file
    from test_milestone import VALID
    legacy = VALID.replace(
        "- Handoff protocol: [SYNC: none] (single-role demo, no cross-role supply)\n", "").replace(
        "- Feed form: [GATHER: none] (pure affine loads)\n", "").replace(
        "- Consumer phase: [PHASE: none] (single consumer WG demo)\n", "")
    p = tmp_path / "milestone.md"
    p.write_text(legacy, encoding="utf-8")
    assert _validate_milestone_file(p) == []
    assert any("[SYNC:]" in x
               for x in _validate_milestone_file(p, authoring=True))



# --- role exposure ledger: the three-tool join reaches the teacher --------------

_ROLE_CORE_STU = ('ROLE-JSON: {"duration_ns": 874880, "resolved_pct": 97.2, '
                  '"roles": {"copy-producer": {"exposed_ns": 22309, "exposed_pct": 2.6, '
                  '"regions": 1, "share_pct": 15.0, "stalls": {"long_sb": 88.0}}, '
                  '"mma-consumer": {"exposed_ns": 118984, "exposed_pct": 13.6, '
                  '"regions": 3, "share_pct": 80.0, "stalls": {"wait": 51.0}}}, '
                  '"tensor_active": 83.0, "tensor_source": "pm-plateau"}')
_ROLE_CORE_REF = ('ROLE-JSON: {"duration_ns": 843424, "resolved_pct": 99.5, '
                  '"roles": {"copy-producer": {"exposed_ns": 20242, "exposed_pct": 2.4, '
                  '"regions": 1, "share_pct": 20.0, "stalls": {"long_sb": 90.0}}, '
                  '"mma-consumer": {"exposed_ns": 70847, "exposed_pct": 8.4, '
                  '"regions": 3, "share_pct": 70.0, "stalls": {"dispatch": 47.0}}}, '
                  '"tensor_active": 88.0, "tensor_source": "pm-plateau"}')


def _role_block(core):
    return ("=== ROLE EXPOSURE BEGIN ===\n"
            "[join] 900 sampled PCs onto 12 regions; 97.2% of 20000 stall samples resolved\n"
            "[tensor] active=83.0% source=pm-plateau duration_ns=874880 exposed_frac=0.170\n"
            "ROLE label=mma-consumer regions=3 samples=16000 share=80.0 exposed_pct=13.6 "
            "exposed_ns=118984 stalls=wait:51\n"
            + core + "\n"
            "=== ROLE EXPOSURE END ===\n"
            "ROLE-OK roles=2 samples=20000 resolved=97.2 tensor=83.0\n")


_ROLE_CORE_PARTIAL = ('ROLE-JSON: {"duration_ns": 843424, "resolved_pct": 99.5, '
                      '"roles": {"copy-producer": {"exposed_ns": null, "exposed_pct": null, '
                      '"regions": 1, "share_pct": 20.0, "stalls": {"long_sb": 90.0}}}, '
                      '"tensor_active": null, "tensor_source": "unavailable"}')


def _role_block_partial():
    """What a bare --set source collection yields: shares, no tensor term."""
    return ("=== ROLE EXPOSURE BEGIN ===\n"
            "[join] 900 sampled PCs onto 12 regions; 99.5% of 20000 stall samples resolved\n"
            "[tensor] active=unknown source=unavailable duration_ns=843424\n"
            "ROLE label=copy-producer regions=1 samples=4000 share=20.0 stalls=long_sb:90\n"
            + _ROLE_CORE_PARTIAL + "\n"
            "=== ROLE EXPOSURE END ===\n"
            "ROLE-PARTIAL roles=1 samples=20000 resolved=99.5 tensor=none\n")


_PMTL_LOG_OK = ("=== PM TIMELINE BEGIN ===\n"
                "[phase] 0-30% producer-bound, 30-95% tensor plateau 88%, 95-100% drain\n"
                "=== PM TIMELINE END ===\n"
                "PMTL-OK kernel=main_kernel series=6 stall_series=4 stall_samples=1800\n")


def test_pricing_job_collects_with_pm_sections(tmp_path):
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    ctx = _stall_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert ("gpu", "cmd:python3 notes/stall_price_run.py --pm") in be.calls


def test_role_exposure_table_reaches_the_diagnose_prompt_aligned(tmp_path):
    """Both sides banked -> one same-role table in the prompt: the consumer's
    +48137 ns exposure next to its stall class flip (dispatch -> wait) is the
    reading the locus rule routes on."""
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    ctx = _stall_ctx(tmp_path, be)
    recon_dir = tmp_path / "recon"
    recon_dir.mkdir()
    ctx.recon_path = recon_dir / "report.md"
    (recon_dir / "ref_role_exposure.txt").write_text(_role_block(_ROLE_CORE_REF))
    run_meta_step(ctx)
    prompt = be.prompts[0]
    assert "Role exposure ledger" in prompt
    row = next(l for l in prompt.splitlines() if l.startswith("mma-consumer"))
    assert "+48137" in row and "dispatch:47" in row and "wait:51" in row
    assert "88.0% (pm-plateau)" in prompt and "83.0% (pm-plateau)" in prompt
    # the student's block is banked beside the dossiers for later audits
    assert (recon_dir / "student_role_exposure.txt").read_text().startswith("[join]")


def test_role_exposure_without_a_ref_side_still_lands_the_student_account(tmp_path):
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    ctx = _stall_ctx(tmp_path, be)
    run_meta_step(ctx)
    prompt = be.prompts[0]
    assert "Only the student side has a ledger" in prompt
    assert "mma-consumer" in prompt


def test_role_block_without_role_ok_is_not_banked(tmp_path):
    block = _role_block(_ROLE_CORE_STU).replace("ROLE-OK roles=2 samples=20000 resolved=97.2 tensor=83.0\n",
                                                "ROLE-FAIL reason=address-space-mismatch\n")
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK + block])
    ctx = _stall_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert "## Role exposure ledger (" not in be.prompts[0]


def test_recon_role_delivery_persists_the_ref_account(tmp_path):
    """A recon job log carrying ROLE-OK banks the block at the fixed path the
    diagnose step reads — same contract as the ref ledger and dossier."""
    from forgekernel.loop.driver import _save_ref_role_exposure
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    recon_dir = tmp_path / "recon"
    recon_dir.mkdir()
    ctx.recon_path = recon_dir / "report.md"
    _save_ref_role_exposure(ctx, "noise\n" + _role_block(_ROLE_CORE_REF) + "tail\n")
    banked = (recon_dir / "ref_role_exposure.txt").read_text()
    assert banked.startswith("[join]") and "ROLE-JSON" in banked
    assert "ROLE EXPOSURE BEGIN" not in banked


def test_m0_audit_also_carries_the_role_table(tmp_path):
    """Every teacher entry prices — the audit included — so the role table
    is in the audit prompt too (an M0 rebuild whose consumer exposure grew is
    a reading the audit must see before it re-legislates wording)."""
    from forgekernel.loop.driver import run_meta_step as rms
    be = FakeBackend(["cause_class: OK\npatch_direction: signatures reconcile\n"],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    ctx = _stall_ctx(tmp_path, be)
    recon_dir = tmp_path / "recon"
    recon_dir.mkdir()
    ctx.recon_path = recon_dir / "report.md"
    (recon_dir / "ref_role_exposure.txt").write_text(_role_block(_ROLE_CORE_REF))
    res = rms(ctx, audit=True)
    assert res.patch_class == "audit-clean"
    assert "Role exposure ledger" in be.prompts[0] and "+48137" in be.prompts[0]


def test_verdict_probe_feedback_lists_every_contract_with_its_reading(tmp_path):
    # observed on grouped-gemm: the decisive contract PASSed with the
    # exposure readings, two controls could not run. The loop log said
    # "PROBE FAIL (max_diff=unreported)"; the teacher must see the three answers.
    probe_log = ("PROBE FAIL max_diff=n/a  # c1 (control): a segment variant returned no reading\n"
                 "PROBE PASS max_diff=ref_exposed=106.3,student_exposed=159.6  # c3 (decisive)\n"
                 "PROBE FAIL max_diff=n/a  # c2: a segment variant returned no reading\n")
    diagnosis = "cause_class: R1\ndeficient_locus: D1.4\npatch_direction: x\n"
    patch = _patching(tmp_path, "\n### D1.4 revised\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend(["PROBE_REQUEST: notes/exposure.py", diagnosis, patch], [probe_log])
    ctx = _ctx(tmp_path, be)
    run_meta_step(ctx)
    fb = be.prompts[1]
    assert "Per-contract verdicts" in fb
    assert "#2 PASS max_diff=ref_exposed=106.3,student_exposed=159.6 — c3 (decisive)" in fb
    assert "#1 FAIL max_diff=n/a — c1 (control)" in fb
    assert "unreported" not in fb


def test_verdict_probe_pass_with_a_reading_table_is_not_downgraded(tmp_path, capsys):
    # the certification audit rejected a PASS whose max_diff was a reading string;
    # in the diagnose channel that read as the hypothesis being falsified.
    diagnosis = "cause_class: R1\ndeficient_locus: D1.4\npatch_direction: x\n"
    patch = _patching(tmp_path, "\n### D1.4 revised\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend(["PROBE_REQUEST: notes/a.py", diagnosis, patch],
                     ["PROBE PASS max_diff=samejob_ncu_ratio=1.0373,trace_ratio=1.0321\n"])
    ctx = _ctx(tmp_path, be)
    run_meta_step(ctx)
    assert "PROBE PASS (max_diff=samejob_ncu_ratio=1.0373,trace_ratio=1.0321)" in be.prompts[1]


def test_student_probe_feedback_names_the_failing_contract(tmp_path):
    # certification semantics unchanged (any FAIL loses); the feedback says which.
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D2.4", 2.5)],
        ["PROBE PASS max_diff=0.0  # c1 smoke\nPROBE FAIL max_diff=n/a  # c2 swizzle census\n",
         "PROBE PASS max_diff=0.0  # c1 smoke\nPROBE PASS max_diff=n/a  # c2 swizzle census\n",
         EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 6, metric_before=2.66)
    assert rec.decision_id == "D2.4"
    fb = be.prompts[1]
    assert "Per-contract verdicts:" in fb
    assert "#2 FAIL max_diff=n/a — c2 swizzle census" in fb


def test_student_probe_log_keeps_the_audit_reason_when_every_line_passed(tmp_path, capsys):
    # two PASS lines, one with an 8e10 "diff": the audit rejects the probe. The
    # per-contract tally reads "2/2 PASS", which next to the FAIL label explained
    # nothing once it had displaced the reason from the loop-log line.
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D2.4", 2.5)],
        ["PROBE PASS max_diff=0.0  # c1\nPROBE PASS max_diff=80000000000.0  # c2\n",
         "PROBE PASS max_diff=0.0  # c1\nPROBE PASS max_diff=0.0  # c2\n",
         EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 6, metric_before=2.66)
    lines = [l for l in capsys.readouterr().out.splitlines() if "probe notes/p.py ->" in l]
    assert "PROBE FAIL (max_diff=unreported) — 2/2 PASS — reported PASS with max_diff=80000000000.0, above the" in lines[0]
    assert lines[1].endswith("PROBE PASS (max_diff=0.0) — 2/2 PASS")


def test_student_probe_log_does_not_repeat_the_tally_as_a_reason(tmp_path, capsys):
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D2.4", 2.5)],
        ["PROBE PASS max_diff=0.0  # c1\nPROBE FAIL max_diff=n/a  # c2\n",
         "PROBE PASS max_diff=0.0  # c1\nPROBE PASS max_diff=n/a  # c2\n",
         EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 6, metric_before=2.66)
    line = next(l for l in capsys.readouterr().out.splitlines() if "probe notes/p.py ->" in l)
    assert line.endswith("PROBE FAIL (max_diff=n/a) — 1/2 PASS; FAIL #2 FAIL max_diff=n/a — c2")
    assert "PROBE lines reported FAIL" not in line


# --- phantom-gap re-anchor ---------------------------------------------------
# The stop target derives from a ref measured once at launch; a gap-driven
# diagnosis whose residual sits inside the node-noise band first re-anchors the
# ref same-period (observed on bwd-GQA: a +3.6% residual, entirely the stale anchor,
# was chased through a 20h structural rebuild).

def _reanchor_ctx(tmp_path, *, target_ms=2.0, band=0.06, samples=None, calls=None,
                  cluster="", anchor_ms=2.0, anchor_cluster=""):
    ctx = _ctx(tmp_path, FakeBackend([], []), target_ms=target_ms)
    ctx.gate_window = "device"
    ctx.reanchor_band = band
    ctx.target_anchor_ms = anchor_ms          # launch fastest-of-N, ref channel
    ctx.target_anchor_cluster = anchor_cluster
    if samples is not None:
        def channel():
            if calls is not None:
                calls.append(1)
            return samples, cluster
        ctx.reanchor_ref = channel
    return ctx


def _stub_confirmation(monkeypatch, median):
    """Stand in for the same-period student re-measure the stop path requires."""
    import forgekernel.loop.driver as drv
    out = None if median is None else \
        {"median_ms": median, "jobs": [{"job": "c1"}, {"job": "c2"}]}
    monkeypatch.setattr(drv, "run_best_confirmation",
                        lambda ctx, jobs=2, max_extra=1: out)


def test_reanchor_stop_needs_both_sides_same_period(tmp_path, monkeypatch):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    # best 2.07 vs stale target 2.0 = +3.5%. Fresh ref min 2.15 (drift x1.075):
    # transported target 2.15 — met on paper, but the run only stops after the
    # STUDENT is re-measured in the same period and agrees.
    ctx = _reanchor_ctx(tmp_path, samples=[2.16, 2.15, 2.15])
    _stub_confirmation(monkeypatch, 2.05)
    out = _phantom_gap_reanchor(ctx, 2.07)
    assert out is not None and out["verdict"] == "target"
    assert out["fresh_target_ms"] == pytest.approx(2.15)
    assert out["student_fresh_ms"] == 2.05
    assert ctx.reanchor_note is None  # a stop needs no prompt note


def test_reanchor_old_best_reading_can_be_the_lucky_draw(tmp_path, monkeypatch):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    # The transported target is met by the archived best reading, but the
    # same-period student re-measure disagrees: no stop — the note carries the
    # honest (fresh) student number instead.
    ctx = _reanchor_ctx(tmp_path, samples=[2.16, 2.15, 2.15])
    _stub_confirmation(monkeypatch, 2.30)
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    assert ctx.reanchor_note is not None and "student same-period re-measure 2.3" in ctx.reanchor_note


def test_reanchor_never_stops_without_the_student_side(tmp_path, monkeypatch):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    # confirm_jobs=0-style runs included: when the student cannot be re-measured
    # the check must not stop the run on half a claim.
    ctx = _reanchor_ctx(tmp_path, samples=[2.16, 2.15, 2.15])
    _stub_confirmation(monkeypatch, None)
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    assert ctx.reanchor_note is not None


def test_reanchor_estimator_is_min_vs_min(tmp_path):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    # Zero drift: fresh fastest == launch anchor (the archive keeps the fastest).
    # A median estimator would read the median-vs-min offset (2.04/2.00 = +2%)
    # as drift and hand a genuinely-2%-slow student a free stop=target.
    ctx = _reanchor_ctx(tmp_path, samples=[2.00, 2.04, 2.06])
    assert _phantom_gap_reanchor(ctx, 2.04) is None       # NOT a stop
    assert ctx.reanchor_note is not None                  # honest residual: +2%
    assert "+2.0%" in ctx.reanchor_note


def test_reanchor_confirmed_gap_stashes_note_and_persists(tmp_path):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    import json as _json
    ctx = _reanchor_ctx(tmp_path, samples=[2.0, 2.0, 2.0])
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    assert ctx.reanchor_note is not None and "+3.5%" in ctx.reanchor_note
    assert "must not" in ctx.reanchor_note        # no writing the transported target to §0
    rec = _json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[0])
    assert rec["verdict"] == "confirmed_gap"
    assert rec["estimator"] == "min-vs-min"


def test_reanchor_memo_does_not_rebuy_jobs_for_an_unchanged_best(tmp_path):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    calls: list[int] = []
    ctx = _reanchor_ctx(tmp_path, samples=[2.0, 2.0, 2.0], calls=calls)
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    ctx.reanchor_note = None                  # consumed by the meta entry
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    assert calls == [1]                       # second check spent nothing
    assert ctx.reanchor_note is not None      # but the note is re-issued


def test_reanchor_admission_needs_two_samples_and_bounded_spread(tmp_path):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    # an interrupted pair leaving one slow sample must not rewrite the bar
    ctx = _reanchor_ctx(tmp_path, samples=[2.4])
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    assert ctx.reanchor_note is None
    # samples straddling the node modes represent no single period
    ctx2 = _reanchor_ctx(tmp_path, samples=[2.00, 2.31, 2.33])
    assert _phantom_gap_reanchor(ctx2, 2.07) is None
    assert ctx2.reanchor_note is None


def test_reanchor_refuses_cross_cluster_transport(tmp_path, capsys):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    # the run moved machines: fresh samples come from a machine the
    # launch anchor (and the best) never ran on.
    ctx = _reanchor_ctx(tmp_path, samples=[2.15, 2.15, 2.16],
                        cluster="machine_b", anchor_cluster="machine_a")
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    assert ctx.reanchor_note is None
    assert "do not transport across machines" in capsys.readouterr().out


def test_reanchor_refuses_a_rebaselined_target(tmp_path, capsys):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    # meta re-baselined §0: that target already reflects its own measurement
    # period — multiplying it by drift-since-launch would count the drift twice.
    calls: list[int] = []
    ctx = _reanchor_ctx(tmp_path, samples=[2.2, 2.2, 2.2], calls=calls)
    ctx.milestone.write_text("## 0. t\ntarget_ms: 1.9\ntarget_window: device\n",
                             encoding="utf-8")
    assert _phantom_gap_reanchor(ctx, 1.97) is None   # +3.7% vs §0's 1.9
    assert calls == []                                # refused before measuring
    assert "count the drift twice" in capsys.readouterr().out


def test_reanchor_refuses_an_unanchored_target(tmp_path, capsys):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    # --target-ms / toml constants were never node-sampled: no launch-period
    # conditions exist to transport drift from.
    calls: list[int] = []
    ctx = _reanchor_ctx(tmp_path, samples=[2.2, 2.2, 2.2], calls=calls,
                        anchor_ms=None)
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    assert calls == []
    assert "not derived from a measured baseline" in capsys.readouterr().out


def test_reanchor_only_fires_inside_the_band(tmp_path):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    calls: list[int] = []
    ctx = _reanchor_ctx(tmp_path, samples=[2.2, 2.2], calls=calls)
    assert _phantom_gap_reanchor(ctx, 2.5) is None   # +25%: drift can't explain it
    assert _phantom_gap_reanchor(ctx, 1.9) is None   # already met
    assert _phantom_gap_reanchor(ctx, None) is None  # no best yet
    assert calls == []                               # no GPU spend outside the band


def test_reanchor_disabled_or_unwired_is_inert(tmp_path):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    calls: list[int] = []
    ctx = _reanchor_ctx(tmp_path, band=0.0, samples=[2.2, 2.2], calls=calls)
    assert _phantom_gap_reanchor(ctx, 2.07) is None and calls == []
    ctx2 = _reanchor_ctx(tmp_path)  # target-ms-only run: no channel wired
    assert _phantom_gap_reanchor(ctx2, 2.07) is None


def test_reanchor_measurement_failure_degrades_loudly_to_the_old_behavior(tmp_path, capsys):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    ctx = _reanchor_ctx(tmp_path)

    def boom() -> tuple[list[float], str]:
        raise RuntimeError("pool starved")

    ctx.reanchor_ref = boom
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    assert "diagnosing against the stale anchor" in capsys.readouterr().out
    assert ctx.reanchor_note is None


def test_two_layer_phantom_gap_stops_as_target_without_spending_meta(tmp_path):
    # Inner plateaus 2.3% over target; the fresh anchor explains the whole
    # residual and the same-period student re-measure agrees -> the run stops as
    # "target" with the meta budget unspent (the agent script has no meta
    # replies to consume). max_meta=0 also proves the check runs before the
    # meta_budget return: a genuinely-met target must never be recorded as
    # meta_budget for want of budget to check the residual with.
    dag = [Decision(id="A", depends_on=[])]
    agent = [REC("A", 2.66), REC("B", 2.70), REC("C", 2.68), REC("D", 2.69)]
    gpu = [EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.68), NCU, EVAL(2.69), NCU,
           EVAL(2.60), EVAL(2.61)]  # the two same-period student confirm jobs
    be = _ConfirmBackend(agent, gpu)  # push_ref-capable: the stop re-measures
    ctx = _ctx(tmp_path, be, target_ms=2.6, dag=dag)
    ctx.gate_window = "device"
    ctx.target_anchor_ms = 2.0
    ctx.reanchor_ref = lambda: ([2.7, 2.7, 2.7], "")  # drift x1.35 -> target 3.51
    summary = run_two_layer(ctx, max_meta=0)
    assert summary["stop"] == "target"
    assert summary["meta_steps"] == 0
    assert summary["reanchor"]["verdict"] == "target"
    assert summary["reanchor"]["student_fresh_ms"] == pytest.approx(2.605)
    assert any(c[0] == "push_ref" for c in be.calls)


def test_two_layer_confirmed_gap_still_routes_to_meta(tmp_path):
    # Same plateau, but the fresh ref agrees with the launch anchor: the gap is
    # real, meta runs as before, and its diagnose prompt carries the re-anchor
    # note.
    dag = [Decision(id="A", depends_on=[])]
    agent = [REC("A", 2.66), REC("B", 2.70), REC("C", 2.68), REC("D", 2.69),
             "cause_class: R1",
             _patching(tmp_path, "\n### patched\nx\n",
                       "patch_class: structural\nchanged_entries: []\n"),
             REC("E", 1.8)]
    gpu = [EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.68), NCU, EVAL(2.69), NCU,
           EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=2.6, dag=dag)
    ctx.gate_window = "device"
    ctx.target_anchor_ms = 2.0
    ctx.reanchor_ref = lambda: ([2.0, 2.0, 2.0], "")
    summary = run_two_layer(ctx, max_meta=2)
    assert summary["meta_steps"] == 1
    diagnose_prompt = next(p for p in be.prompts if "Contemporaneous ref re-anchor" in p)
    assert "true residual" in diagnose_prompt
    assert ctx.reanchor_note is None  # handed to the meta entry, then cleared


def test_two_layer_reanchor_note_survives_a_malformed_patch_retry(tmp_path):
    # The first patch never lands (canned reply, file untouched) -> run_meta_step
    # returns malformed and run_two_layer retries the WHOLE meta step. The retry
    # re-runs the diagnosis, which must still see the re-anchored residual — not
    # fall back to the stale one and open an R1 against a number the re-anchor
    # already corrected.
    dag = [Decision(id="A", depends_on=[])]
    agent = [REC("A", 2.66), REC("B", 2.70), REC("C", 2.68), REC("D", 2.69),
             "cause_class: R1",
             "patch_class: structural\nchanged_entries: []\n",   # never lands
             "cause_class: R1",
             _patching(tmp_path, "\n### patched\nx\n",
                       "patch_class: structural\nchanged_entries: []\n"),
             REC("E", 1.8)]
    gpu = [EVAL(2.66), NCU, EVAL(2.70), NCU, EVAL(2.68), NCU, EVAL(2.69), NCU,
           EVAL(1.8), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=2.6, dag=dag)
    ctx.gate_window = "device"
    ctx.target_anchor_ms = 2.0
    ctx.max_patch_rounds = 1
    ctx.reanchor_ref = lambda: ([2.0, 2.0, 2.0], "")
    run_two_layer(ctx, max_meta=3)
    noted = [p for p in be.prompts if "Contemporaneous ref re-anchor" in p]
    assert len(noted) == 2  # the original diagnosis AND the retry's

def test_verdict_probe_readings_persist_and_reach_the_next_diagnosis(tmp_path):
    """The contract lines a verdict probe printed are recorded on the meta
    step's ledger row and surface as "measured" sub-rows in the NEXT step's memory —
    the teacher stops re-spending a probe to re-measure a known quantity."""
    from forgekernel.meta_trajectory import load_meta_records
    probe_log = ("PROBE FAIL max_diff=n/a  # c1 (control): a segment variant returned no reading\n"
                 "PROBE PASS max_diff=ref_exposed=106.3,student_exposed=159.6  # c3 (decisive)\n")
    diagnosis = "cause_class: R1\ndeficient_locus: D1.4\npatch_direction: x\n"
    patch = _patching(tmp_path, "\n### D1.4 revised\nx\n",
                      "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n")
    be = FakeBackend(["PROBE_REQUEST: notes/exposure.py", diagnosis, patch,
                      diagnosis, patch],
                     [probe_log])
    ctx = _ctx(tmp_path, be)
    run_meta_step(ctx)
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].probe_rounds == 1
    assert [(d["contract"], d["verdict"]) for d in recs[0].probe_readings] == \
        [(1, "FAIL"), (2, "PASS")]
    assert recs[0].probe_readings[1]["reading"] == "ref_exposed=106.3,student_exposed=159.6"
    assert recs[0].probe_readings[1]["note"] == "c3 (decisive)"
    # second meta step: no probe this time; its diagnose prompt carries step 0's readings
    run_meta_step(ctx)
    prompt = be.prompts[3]
    assert "measured notes/exposure.py #2 PASS max_diff=ref_exposed=106.3,student_exposed=159.6 — c3 (decisive)" in prompt
    assert "measured notes/exposure.py #1 FAIL max_diff=n/a" in prompt



# --- same-stage gate early stop ----------------------------------------------------

def _seed_keep(ctx, did, ms, sha="prev"):
    from forgekernel.inner.trajectory import IterationRecord, append_record
    append_record(ctx.status_path, IterationRecord(
        iter=len(load_records(ctx.status_path)), decision_id=did, change="c",
        correctness=True, verdict="KEEP", bwd_median_ms=ms, device_median_ms=ms,
        sha=sha, profiler={}))


def test_parse_gate_contract_reads_per_gate_bands():
    from forgekernel.loop.driver import _parse_gate_contract
    out = _parse_gate_contract("patch_class: structural\n"
                               "gate_contract: M0-gate ≤ +15%; M1-gate <= 0%; M2-gate ≤ -3%\n")
    assert out == {"M0-gate": 0.15, "M1-gate": 0.0, "M2-gate": -0.03}
    assert _parse_gate_contract("patch_class: additive\n") == {}


def test_gate_regression_stops_the_rebuild_at_the_same_stage(tmp_path):
    """ixbwd step-6 replay: the previous segment closed M0 at 12.31; the
    rebuild's M0-gate reads 14.43 (+17%). Same-stage parity fails -> the
    segment stops right there (the live run walked M1 for four more iters)."""
    be = FakeBackend([REC("M0-gate", 14.43)], [EVAL(14.43), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.0)
    _seed_keep(ctx, "D0.9", 14.31)
    _seed_keep(ctx, "M0-gate", 12.31)
    before = len(load_records(ctx.status_path))
    reason = run_inner_loop(ctx, before, initial_green=True,
                            segment_records_before=before)
    assert reason == "gate_regression"


def test_gate_regression_honours_a_declared_band(tmp_path):
    # the teacher declared M0 may lose 20% (payoff lands at M1): 14.43 vs 12.31 is
    # +17%, inside the band -> the segment continues to its next stop
    be = FakeBackend([REC("M0-gate", 14.43), "EXHAUSTED: done"], [EVAL(14.43), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.0)
    _seed_keep(ctx, "M0-gate", 12.31)
    before = len(load_records(ctx.status_path))
    reason = run_inner_loop(ctx, before, initial_green=True,
                            segment_records_before=before,
                            gate_contract={"M0-gate": 0.20})
    assert reason == "exhausted"


def test_gate_at_parity_within_noise_is_not_a_regression(tmp_path):
    be = FakeBackend([REC("M0-gate", 12.45), "EXHAUSTED: done"], [EVAL(12.45), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.0)
    _seed_keep(ctx, "M0-gate", 12.31)   # +1.1% < 1.5% tolerance
    before = len(load_records(ctx.status_path))
    assert run_inner_loop(ctx, before, initial_green=True,
                          segment_records_before=before) == "exhausted"


def test_gate_baseline_skips_a_reverted_segment(tmp_path):
    """Segment A closed M0 at 12.31; patch 1's rebuild read 14.43, was caught
    and REVERTED (tree back on A). Patch 2's rebuild reads 14.0: against the
    standing tree that is +14%, and the reverted segment's 14.43 must not be
    the reading it is held to — that would re-admit the ixbwd case one
    revert later."""
    from forgekernel.inner.trajectory import gate_ms
    from forgekernel.loop.driver import _gate_regression, _previous_gate_record
    be = FakeBackend([REC("M0-gate", 14.0)], [EVAL(14.0), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.0)
    _seed_keep(ctx, "M0-gate", 12.31, sha="shaA")
    reverted_lo = len(load_records(ctx.status_path))
    _seed_keep(ctx, "D0.3", 14.5, sha="shaB")
    _seed_keep(ctx, "M0-gate", 14.43, sha="shaB2")
    reverted_hi = len(load_records(ctx.status_path))
    recs = load_records(ctx.status_path)
    assert gate_ms(_previous_gate_record(recs, "M0-gate", reverted_hi, ctx.gate_window),
                   ctx.gate_window) == 14.43
    assert gate_ms(_previous_gate_record(recs, "M0-gate", reverted_hi, ctx.gate_window,
                                         voided=[(reverted_lo, reverted_hi)]),
                   ctx.gate_window) == 12.31
    ctx.voided_ranges.append((reverted_lo, reverted_hi))
    reason = run_inner_loop(ctx, reverted_hi, initial_green=True,
                            segment_records_before=reverted_hi)
    assert reason == "gate_regression"
    # the message names the standing tree's reading, not the reverted one
    rec = load_records(ctx.status_path)[-1]
    msg = _gate_regression(load_records(ctx.status_path), rec, reverted_hi, {},
                           ctx.gate_window, voided=ctx.voided_ranges)
    assert "12.3100 ms" in msg and "14.4300" not in msg


def test_revert_patch_voids_its_segment_for_the_gate_baseline(tmp_path):
    from forgekernel.loop.driver import _judge_patch, _PendingPatch
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.meta_status_path = None
    ctx.milestone.write_text("OLD")
    _seed_keep(ctx, "M0-gate", 12.31, sha="shaA")
    n = len(load_records(ctx.status_path))
    _seed_keep(ctx, "M0-gate", 14.43, sha="shaB")
    pending = _PendingPatch(best_before=12.31, milestone_before="OLD", good_ref="shaA",
                            records_before=n)
    assert _judge_patch(ctx, pending) == "REVERT_PATCH"
    assert ctx.voided_ranges == [(n, n + 1)]


def test_gate_check_is_off_outside_a_post_patch_segment(tmp_path):
    # no pending patch -> no segment_records_before -> a slower gate is just a KEEP
    be = FakeBackend([REC("M0-gate", 14.43), "EXHAUSTED: done"], [EVAL(14.43), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.0)
    _seed_keep(ctx, "M0-gate", 12.31)
    assert run_inner_loop(ctx, 1, initial_green=True) == "exhausted"


def test_two_layer_gate_regression_judges_the_patch_and_records_the_contract(tmp_path):
    """End to end: seg1 plateaus after its M0-gate; the structural patch
    declares no band; the rebuild's M0-gate loses 17% -> gate_regression ->
    REVERT_PATCH at once, and the meta ledger row carries the (empty) contract."""
    from forgekernel.meta_trajectory import load_meta_records
    dag = [Decision(id="A", depends_on=[])]
    agent = [REC("A", 13.0), REC("M0-gate", 12.31), REC("B", 12.30), REC("C", 12.31),
             REC("D", 12.30),                                       # seg1 plateau
             "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: x",
             _patching(tmp_path, "\n### patched\nx\n",
                       "patch_class: structural\nchanged_entries: []\n"),
             REC("M0-gate", 14.43),                                  # rebuild loses
             "EXHAUSTED: never reached"]
    gpu = [EVAL(13.0), NCU, EVAL(12.31), NCU, EVAL(12.30), NCU, EVAL(12.31), NCU,
           EVAL(12.30), NCU, EVAL(14.43), NCU]
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=1.0, dag=dag)
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["stop"] == "meta_budget"
    rec = load_meta_records(ctx.meta_status_path)[0]
    assert rec.outcome == "REVERT_PATCH" and rec.gate_contract == {}
    # the rebuild segment spent exactly one evaluated row before the judgment
    rows = load_records(ctx.status_path)
    assert [r.decision_id for r in rows][-1] == "M0-gate"
    assert len(rows) == 6



# --- source-mode runs: evidence dir + reference pricing -------------------------

def test_source_run_banks_student_evidence_without_a_recon_tree(tmp_path):
    """recon_path is None on every source-mode run; the evidence dir is where the
    student's role account and dossier land — and where a ref side placed
    there is read from (an offline check found the table one-sided)."""
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    ctx = _stall_ctx(tmp_path, be)
    assert ctx.recon_path is None
    ctx.evidence_dir = tmp_path / "evidence"          # not created yet
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "ref_role_exposure.txt").write_text(_role_block(_ROLE_CORE_REF))
    run_meta_step(ctx)
    prompt = be.prompts[0]
    row = next(l for l in prompt.splitlines() if l.startswith("mma-consumer"))
    assert "+48137" in row
    assert (tmp_path / "evidence" / "student_role_exposure.txt").read_text().startswith("[join]")


def _ref_tree(tmp_path, with_launch_target=True):
    d = tmp_path / "_ref"
    (d / "eval").mkdir(parents=True)
    if with_launch_target:
        (d / "eval" / "ncu_run.py").write_text("pass\n")
    return d


def test_reference_pricing_runs_once_and_banks_the_ref_side(tmp_path):
    """[loop] ref_price_op: the driver seeds the toolkit into the ref tree, runs
    the same pricing recipe there ONCE, and the next teacher entry reads a
    two-sided role table. A second entry does not resubmit."""
    from forgekernel.loop.driver import _STALL_PRICE_CMD
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n"),
                      "cause_class: OK\npatch_direction: signatures reconcile\n"],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU),
                      _STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    rbe = FakeBackend([], [_STALL_LOG_OK + _role_block(_ROLE_CORE_REF)])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path)
    ctx.ref_price_backend = rbe
    run_meta_step(ctx)
    assert ("gpu", _STALL_PRICE_CMD) in rbe.calls
    assert (ctx.ref_price_dir / "notes" / "stall_price_run.py").is_file()
    assert any(c[0] == "commit" for c in rbe.calls)           # seeded AND committed
    assert (tmp_path / "evidence" / "ref_role_exposure.txt").read_text().startswith("[join]")
    assert (tmp_path / "evidence" / "ref_stall_ledger.txt").is_file()
    assert "+48137" in be.prompts[0]
    n = len(rbe.calls)
    run_meta_step(ctx, audit=True)
    assert len(rbe.calls) == n                                 # once per run


def test_reference_pricing_retries_only_when_the_tree_changed(tmp_path):
    """A job that came back without the account would come back the same on
    the same tree (no LINEINFO, PM sections crashing, the sample floor are
    deterministic), so a retry runs only when the head / launch target / app
    spelling changed — and at most three attempts per run."""
    from forgekernel.loop.driver import _STALL_PRICE_CMD
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")]
                     + ["cause_class: OK\npatch_direction: signatures reconcile\n"] * 3,
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)] * 4)
    rbe = FakeBackend([], ["ROLE-FAIL reason=too-few-samples (120 < 500)\n"] * 3)
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path)
    ctx.ref_price_backend = rbe
    run_meta_step(ctx)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 1
    assert (tmp_path / "evidence" / "ref_price_job.log").read_text().startswith("[ref-price] key=")
    # the pricing job archives the branch: the tree is committed before submit
    gpu_at = rbe.calls.index(("gpu", _STALL_PRICE_CMD))
    assert any(c[0] == "commit" for c in rbe.calls[:gpu_at])
    run_meta_step(ctx, audit=True)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 1      # unchanged tree: no blind retry
    (ctx.ref_price_dir / "eval" / "ncu_run.py").write_text("pass  # window fixed\n")
    run_meta_step(ctx, audit=True)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 2      # changed launch target: retry
    assert (tmp_path / "evidence" / "ref_price_job.2.log").is_file()
    run_meta_step(ctx, audit=True)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 2


def test_reference_pricing_is_not_run_behind_a_waiver(tmp_path):
    """The teacher owned the absence (ROLE-WAIVED / PMTL-WAIVED in the report):
    no job is bought behind its back."""
    from forgekernel.loop.driver import _STALL_PRICE_CMD
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    rbe = FakeBackend([], [])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path)
    ctx.ref_price_backend = rbe
    ctx.recon_path = tmp_path / "recon-report.md"
    ctx.recon_path.write_text("obs\nROLE-WAIVED: no PM on this node\nPMTL-WAIVED: same\n")
    run_meta_step(ctx)
    assert rbe.calls == []


def test_reference_pricing_retries_a_partial_table_and_keeps_the_full_one(tmp_path):
    """A first job that came back ROLE-PARTIAL (no tensor term) banks the
    shares; once the launch target changes it is retried and the retry's
    ROLE-OK replaces it; a later partial reading can never downgrade the
    banked account."""
    from forgekernel.loop.driver import (_STALL_PRICE_CMD, _ref_overlap_banked,
                                         _save_ref_role_exposure)
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n"),
                      "cause_class: OK\npatch_direction: signatures reconcile\n",
                      "cause_class: OK\npatch_direction: signatures reconcile\n"],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)] * 3)
    rbe = FakeBackend([], [_STALL_LOG_OK + _role_block_partial(),
                           _STALL_LOG_OK + _role_block(_ROLE_CORE_REF) + _PMTL_LOG_OK])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path)
    ctx.ref_price_backend = rbe
    run_meta_step(ctx)
    banked = (tmp_path / "evidence" / "ref_role_exposure.txt").read_text()
    assert '"tensor_active": null' in banked and not _ref_overlap_banked(ctx)
    assert "no tensor duty cycle" in be.prompts[0]          # the teacher is told
    (ctx.ref_price_dir / "eval" / "ncu_run.py").write_text("pass  # PM window fixed\n")
    run_meta_step(ctx, audit=True)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 2
    assert _ref_overlap_banked(ctx)
    assert (tmp_path / "evidence" / "ref_pm_timeline.txt").read_text().startswith("[phase]")
    assert "+48137" not in be.prompts[1] or "ref" in be.prompts[1]
    _save_ref_role_exposure(ctx, _role_block_partial())
    assert _ref_overlap_banked(ctx)                        # partial never overwrites full
    run_meta_step(ctx, audit=True)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 2  # account banked: no job


def test_reference_pricing_uses_the_configured_launch_target(tmp_path):
    """A recon tree whose profiler window is not eval/ncu_run.py (the
    dsa_sparse tree: python3 dsa_ref.py ncu) is priced with the same frozen
    recipe through loop.toml ref_price_app."""
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n"),
                      "cause_class: OK\npatch_direction: signatures reconcile\n"],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)] * 2)
    rbe = FakeBackend([], [_STALL_LOG_OK + _role_block(_ROLE_CORE_REF) + _PMTL_LOG_OK])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path, with_launch_target=False)
    ctx.ref_price_backend = rbe
    ctx.ref_price_app = ["python3", "dsa_ref.py", "ncu"]
    run_meta_step(ctx)
    assert rbe.calls == []                          # the app's entry is not in the tree
    (ctx.ref_price_dir / "dsa_ref.py").write_text("pass\n")
    run_meta_step(ctx, audit=True)
    assert ("gpu", "cmd:python3 notes/stall_price_run.py --pm --app python3 dsa_ref.py ncu") \
        in rbe.calls
    assert (tmp_path / "evidence" / "ref_role_exposure.txt").is_file()


def test_recon_finish_prices_the_ref_itself_before_holding_the_teacher(tmp_path):
    """The overlap channels are the driver's to deliver: at the recon finish
    it runs the frozen pricing recipe on the ref tree, banks the role table
    with tensor + the phase timeline, and the teacher is NOT bounced for
    ROLE/PMTL. The account then reaches the distill prompt, and the first
    teacher entry does not price the ref again."""
    from forgekernel.loop.driver import (_STALL_PRICE_CMD,
                                         _ref_overlap_block_for_distill,
                                         run_recon_step)

    def writeup(prompt):
        assert "overlap ledger is missing" not in prompt
        (tmp_path / "recon-report.md").write_text(
            "obs [MEASURED, job 1]\nSASS-WAIVED: x\nSCHED-WAIVED: x\n"
            "LIVE-WAIVED: x\nSTALL-WAIVED: x")
        (tmp_path / "candidates.md").write_text("A")
        return "nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/p.py", writeup], [])
    recon = FakeBackend([], [_ORCH_LOG_OK,
                             _STALL_LOG_OK + _role_block(_ROLE_CORE_REF) + _PMTL_LOG_OK])
    ctx = _stall_ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('probe')")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "ncu_run.py").write_text("pass\n")
    ctx.recon_path = tmp_path / "recon-report.md"
    ctx.ref_price_dir = tmp_path          # a blackbox run: the recon tree itself
    ctx.ref_price_backend = recon
    result = run_recon_step(ctx)
    assert result.finished is True
    assert recon.calls.count(("gpu", _STALL_PRICE_CMD)) == 1
    # The ledgers the job just paid for are committed where they were written —
    # not left for a human before the next preflight (observed on mla_paged_decode: the
    # snapshot commit ran BEFORE the job, the four ledgers landed after it).
    gpu_at = recon.calls.index(("gpu", _STALL_PRICE_CMD))
    assert ("commit", "recon: bank ") in recon.calls[gpu_at:], "ref-price ledgers left uncommitted"
    assert '"tensor_active": 88.0' in (tmp_path / "ref_role_exposure.txt").read_text()
    assert (tmp_path / "ref_pm_timeline.txt").read_text().startswith("[phase]")
    block = _ref_overlap_block_for_distill(ctx)
    assert "Reference overlap ledger (measured by the recon-time pricing job" in block and "tensor plateau 88%" in block
    # a later teacher entry reads the banked account instead of pricing again
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    ctx2 = _stall_ctx(tmp_path, be)
    ctx2.recon_path = tmp_path / "recon-report.md"
    ctx2.recon_dir = tmp_path
    ctx2.ref_price_dir = tmp_path
    ctx2.ref_price_backend = recon
    run_meta_step(ctx2)
    assert recon.calls.count(("gpu", _STALL_PRICE_CMD)) == 1
    assert "+48137" in be.prompts[0]


def test_transient_ref_pricing_failures_do_not_spend_the_attempts(tmp_path):
    """A submit/queue failure priced nothing: it must not count as one of the
    two attempts (two queue hiccups would retire the ref's account for the
    run). Bounded per session, and the gate note names the failure."""
    from forgekernel.loop.driver import (BackendError, _STALL_PRICE_CMD,
                                         _evidence_gate_msg)

    class Flaky(FakeBackend):
        def submit_gpu(self, branch, mode):
            self.calls.append(("gpu", mode))
            if not self.gpu_logs:
                raise BackendError("queue starved")
            return self.gpu_logs.pop(0)

    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n"),
                      "cause_class: OK\npatch_direction: signatures reconcile\n",
                      "cause_class: OK\npatch_direction: signatures reconcile\n"],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)] * 3)
    rbe = Flaky([], [])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path)
    ctx.ref_price_backend = rbe
    run_meta_step(ctx)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 1
    assert not (tmp_path / "evidence" / "ref_price_job.log").exists()   # not an attempt
    assert "queue starved" in _evidence_gate_msg(["ROLE"], None, ctx)
    rbe.gpu_logs = [_STALL_LOG_OK + _role_block(_ROLE_CORE_REF) + _PMTL_LOG_OK]
    run_meta_step(ctx, audit=True)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 2
    assert (tmp_path / "evidence" / "ref_price_job.log").is_file()       # the first real attempt
    assert '"tensor_active": 88.0' in (tmp_path / "evidence" / "ref_role_exposure.txt").read_text()


def test_a_job_that_never_ran_after_launch_is_not_a_ref_pricing_attempt(tmp_path):
    """The lost-runner verdict (every resubmit lost its runner) arrives with the job id
    moved — every resubmit was a submit — and a message that never says the job ran.
    Read as a job that ran and failed, it would write the attempt marker and (same
    tree, same key) never price the reference again. The verdict is typed NoRunError;
    that is what the accounting reads."""
    from forgekernel.loop.cli import RunnerLostError as LeaseQueuePatienceExhaustedError
    from forgekernel.loop.driver import _STALL_PRICE_CMD

    class Queued(FakeBackend):
        last_job_id = ""

        def submit_gpu(self, branch, mode):
            self.calls.append(("gpu", mode))
            if not self.gpu_logs:
                self.last_job_id = "1789777383"      # the re-adoptions were submits
                raise LeaseQueuePatienceExhaustedError(
                    "job lost its runner 4 times in a row (last: 1789777383); nothing "
                    "ran to completion")
            self.last_job_id = "1789790000"
            return self.gpu_logs.pop(0)

    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n"),
                      "cause_class: OK\npatch_direction: signatures reconcile\n",
                      "cause_class: OK\npatch_direction: signatures reconcile\n"],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)] * 3)
    rbe = Queued([], [])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path)
    ctx.ref_price_backend = rbe
    run_meta_step(ctx)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 1
    assert not (tmp_path / "evidence" / "ref_price_job.log").exists()   # not an attempt
    assert ctx.ref_price_submit_failures == 1
    assert "nothing ran to completion" in ctx.ref_price_note
    # the next teacher entry prices the same tree again — the wait was the queue's
    rbe.gpu_logs = [_STALL_LOG_OK + _role_block(_ROLE_CORE_REF) + _PMTL_LOG_OK]
    run_meta_step(ctx, audit=True)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 2
    assert (tmp_path / "evidence" / "ref_price_job.log").is_file()       # the first real attempt
    assert '"tensor_active": 88.0' in (tmp_path / "evidence" / "ref_role_exposure.txt").read_text()


def test_a_job_that_launched_and_produced_nothing_is_still_an_attempt(tmp_path):
    """The other side of the boundary the type draws: a plain BackendError
    after the job id moved (poll timeout on a hung reference kernel, log lost
    after success) IS an attempt — the same tree would hang again."""
    from forgekernel.loop.driver import BackendError, _STALL_PRICE_CMD

    class Hung(FakeBackend):
        last_job_id = ""

        def submit_gpu(self, branch, mode):
            self.calls.append(("gpu", mode))
            self.last_job_id = "1789777383"
            raise BackendError("job tasks/1789777383 did not reach a terminal state "
                               "within 14400s (last status 'Running'); giving up to "
                               "avoid an unbounded poll")

    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n"),
                      "cause_class: OK\npatch_direction: signatures reconcile\n"],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)] * 2)
    rbe = Hung([], [])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path)
    ctx.ref_price_backend = rbe
    run_meta_step(ctx)
    assert rbe.calls.count(("gpu", _STALL_PRICE_CMD)) == 1
    marker = tmp_path / "evidence" / "ref_price_job.log"
    assert marker.is_file() and "job failed after launch" in marker.read_text()
    assert ctx.ref_price_submit_failures == 0


def test_role_block_is_banked_by_its_own_verdict_not_a_later_one(tmp_path):
    """A recon probe may print a rig's partial table and then the real one;
    the LAST block is banked and judged by the verdict line after it."""
    from forgekernel.loop.driver import _ref_overlap_banked, _save_ref_role_exposure
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    recon_dir = tmp_path / "recon"
    recon_dir.mkdir()
    ctx.recon_path = recon_dir / "report.md"
    # full block first, partial block last -> the partial one is what the job
    # ended on, but it must not overwrite a banked full table
    _save_ref_role_exposure(ctx, _role_block(_ROLE_CORE_REF))
    assert _ref_overlap_banked(ctx)
    _save_ref_role_exposure(ctx, _role_block(_ROLE_CORE_REF) + "noise\n" + _role_block_partial())
    assert _ref_overlap_banked(ctx)
    # partial first, full last (the observed shape: rig round-trip, then the table)
    (recon_dir / "ref_role_exposure.txt").unlink()
    _save_ref_role_exposure(ctx, _role_block_partial() + "noise\n" + _role_block(_ROLE_CORE_REF))
    assert _ref_overlap_banked(ctx)


def test_recon_gate_credits_the_ledger_the_drivers_pricing_job_banked(tmp_path):
    """The driver's own pricing job prints STALL-OK too; the recon loop's
    per-log flag never sees that job, so the gate must read the banked file
    rather than force a STALL-WAIVED the teacher knows is false."""
    from forgekernel.loop.driver import _STALL_PRICE_CMD, run_recon_step

    def writeup(prompt):
        assert "STALL evidence missing" not in prompt
        (tmp_path / "recon-report.md").write_text(
            "obs [MEASURED, job 1]\nSASS-WAIVED: x\nSCHED-WAIVED: x\nLIVE-WAIVED: x")
        (tmp_path / "candidates.md").write_text("A")
        return "nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/p.py", writeup], [])
    recon = FakeBackend([], [_ORCH_LOG_OK,
                             _STALL_LOG_OK + _role_block(_ROLE_CORE_REF) + _PMTL_LOG_OK])
    ctx = _stall_ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('probe')")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "ncu_run.py").write_text("pass\n")
    ctx.recon_path = tmp_path / "recon-report.md"
    ctx.ref_price_dir = tmp_path
    ctx.ref_price_backend = recon
    assert run_recon_step(ctx).finished is True
    assert recon.calls.count(("gpu", _STALL_PRICE_CMD)) == 1
    assert (tmp_path / "ref_stall_ledger.txt").is_file()
    # the seeded toolkit in the recon tree includes the pricing recipe itself
    assert (tmp_path / "notes" / "stall_price_run.py").is_file()
    assert (tmp_path / "notes" / "role_exposure_tools.py").is_file()


def test_recon_gate_option_one_reprices_after_the_teacher_adds_the_launch_target(tmp_path):
    """No launch target at the first finish -> the gate names that reason;
    the teacher adds eval/ncu_run.py and ends the turn again -> priced."""
    from forgekernel.loop.driver import _STALL_PRICE_CMD, run_recon_step

    def writeup(prompt):
        (tmp_path / "recon-report.md").write_text(
            "obs [MEASURED, job 1]\nSASS-WAIVED: x\nSCHED-WAIVED: x\nLIVE-WAIVED: x\nSTALL-WAIVED: x")
        (tmp_path / "candidates.md").write_text("A")
        return "nothing further"

    def add_target(prompt):
        assert "has no `eval/ncu_run.py`" in prompt and "simply end the turn again" in prompt
        (tmp_path / "eval").mkdir()
        (tmp_path / "eval" / "ncu_run.py").write_text("pass\n")
        return "added the launch target, nothing further"

    teacher = FakeBackend(["RECON_REQUEST: notes/p.py", writeup, add_target], [])
    recon = FakeBackend([], [_ORCH_LOG_OK,
                             _STALL_LOG_OK + _role_block(_ROLE_CORE_REF) + _PMTL_LOG_OK])
    ctx = _stall_ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    ctx.ref_price_dir = tmp_path
    ctx.ref_price_backend = recon
    assert run_recon_step(ctx).finished is True
    assert recon.calls.count(("gpu", _STALL_PRICE_CMD)) == 1
    assert (tmp_path / "ref_pm_timeline.txt").is_file()
    # the file the teacher wrote reaches the job: committed before submit
    gpu_at = recon.calls.index(("gpu", _STALL_PRICE_CMD))
    assert any(c[0] == "commit" for c in recon.calls[:gpu_at])


def test_overlap_gate_status_is_derived_from_the_markers(tmp_path):
    """The gate's Status/Deliver text comes from disk (attempt markers + banked
    files), so it is right after a resume, after a job that banked the role
    table but not the timeline, and after the attempts are spent."""
    from forgekernel.loop.driver import (_STALL_PRICE_CMD, _ref_price_status,
                                         _ref_pricing_key, _save_ref_role_exposure)
    be = FakeBackend([], [])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.evidence_dir.mkdir()
    ctx.ref_price_dir = _ref_tree(tmp_path)
    ctx.ref_price_backend = FakeBackend([], [])
    note, opt = _ref_price_status(ctx, ["ROLE", "PMTL"])
    assert "has not run pricing yet" in note
    (ctx.evidence_dir / "ref_price_job.log").write_text(
        f"[ref-price] key={_ref_pricing_key(ctx)} app=None\nSTALL-OK kernel=k samples=9 rows=1 coverage=95\n"
        "PMTL-FAIL reason=no-pm-sections\n" + _role_block(_ROLE_CORE_REF))
    _save_ref_role_exposure(ctx, _role_block(_ROLE_CORE_REF))
    note, opt = _ref_price_status(ctx, ["PMTL"])
    assert "PMTL-FAIL reason=no-pm-sections" in note and "PMTL-WAIVED" in opt
    note, opt = _ref_price_status(ctx, ["ROLE"])
    assert "after changing the entry file" in opt                      # unchanged key: no blind retry promised
    ctx.ref_price_app = ["python3", "missing.py", "ncu"]
    note, opt = _ref_price_status(ctx, ["ROLE"])
    assert "does **not** fall back" in opt and "ref_price_app" in note


def test_kept_probe_survives_a_meta_step_between_iterations(tmp_path):
    """A meta step's verdict-probe commit (git add -A) + reset would sweep the
    untracked kept probe away; the next student iteration restores it."""
    from forgekernel.loop.driver import _restore_kept_probes
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('rig v3')\n")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", "garbage"],
                     ["PROBE FAIL max_diff=9", "PROBE PASS max_diff=0.0"])
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev")
    assert ctx.kept_probes == {"notes/p.py": "print('rig v3')\n"}
    (tmp_path / "notes" / "p.py").unlink()               # what the meta step's reset does
    _restore_kept_probes(ctx)
    assert (tmp_path / "notes" / "p.py").read_text() == "print('rig v3')\n"
    # the delivery turn carried the probe's reading, not just "it passed"
    assert "PROBE PASS" in be.prompts[2] and "can only deliver" in be.prompts[2]


def test_distill_prompt_names_a_missing_ref_overlap_account(tmp_path):
    """No banked account (or a partial one) -> the distill prompt says so in
    words the milestone rules bind to: overlap adjudications stay [ESTIMATE]
    candidates, never M0 hard lines."""
    from forgekernel.loop.driver import (_ref_overlap_block_for_distill,
                                         _save_ref_role_exposure)
    be = FakeBackend([], [])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    # a source-mode run that neither reconnoitres nor prices its ref: no note at all
    assert _ref_overlap_block_for_distill(ctx) == ""
    ctx.oracle_mode = "blackbox"
    block = _ref_overlap_block_for_distill(ctx)
    assert "Reference overlap ledger: **absent**" in block and "[ESTIMATE]" in block
    _save_ref_role_exposure(ctx, _role_block_partial())
    assert "Reference overlap ledger: **absent**" in _ref_overlap_block_for_distill(ctx)


def test_reference_pricing_needs_the_launch_target(tmp_path):
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    rbe = FakeBackend([], [])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path, with_launch_target=False)
    ctx.ref_price_backend = rbe
    run_meta_step(ctx)
    assert rbe.calls == []
    assert "Only the student side has a ledger" in be.prompts[0]


def test_reference_pricing_reads_a_head_clipped_batch_log(tmp_path):
    """The ref tree's log is read back capped: only the last ~300 lines, and
    the ledger at the head is the first thing to go. The bundle at the tail
    restores it (stall_price_run.expand_bundle)."""
    from forgekernel.skills.stall_price_run import emit_bundle
    full = _STALL_LOG_OK + _role_block(_ROLE_CORE_REF)
    lines = []
    emit_bundle(full, out=lines.append)
    clipped = _role_block(_ROLE_CORE_REF) + "\n".join(lines) + "\n"
    assert "STALL LEDGER BEGIN" not in clipped
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    rbe = FakeBackend([], [clipped])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path)
    ctx.ref_price_backend = rbe
    run_meta_step(ctx)
    assert (tmp_path / "evidence" / "ref_stall_ledger.txt").is_file()


def test_ref_segment_account_banks_on_its_own_join_verdict(tmp_path):
    """A library kernel without LINEINFO fails the row ledger's coverage floor
    (deep_gemm ref) while its PC-keyed segment account is clean; the
    segment table must not be lost with the ledger."""
    from forgekernel.loop.driver import _save_ref_stall_ledger
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    log = ("=== STALL LEDGER BEGIN ===\nrow\n=== STALL LEDGER END ===\n"
           "STALL-FAIL reason=coverage (85.6% < 90%)\n"
           "=== SEGMENT LEDGER BEGIN ===\nSEG idx=1 kind=EW\nJOIN-OK resolved=100.0%\n"
           "=== SEGMENT LEDGER END ===\n")
    _save_ref_stall_ledger(ctx, log)
    assert not (ctx.evidence_dir / "ref_stall_ledger.txt").exists()
    assert (ctx.evidence_dir / "ref_segment_ledger.txt").read_text().startswith("SEG idx=1")
    _save_ref_stall_ledger(ctx, log.replace("JOIN-OK resolved=100.0%", "JOIN-UNSAFE"))
    # an unsafe join is not evidence: the earlier clean copy is left in place
    assert "JOIN-OK" in (ctx.evidence_dir / "ref_segment_ledger.txt").read_text()



# --- source-mode runs may reconnoitre ---------------------------------------------
# The blackbox/source split assumed the oracle answers every question a probe
# would, at no cost. For templated references it does not: a grouped-GEMM
# source-mode run guessed a runtime split test and a ping-pong phase the SASS never
# had (-30% at M0) while a blackbox run on the same card measured both.
# Source-mode runs now reconnoitre too (default tree _recon; recon_op = "" declares
# a source-only control run), and the control renders exactly what it always did.

def _authoring(tmp_path):
    """A distill agent that writes v1 the way a real one must."""
    def _run(_prompt):
        (tmp_path / "m.md").write_text(_MILESTONE, encoding="utf-8")
        return "authored"
    return _run


def _recon_write_up(tmp_path):
    def _run(_prompt):
        (tmp_path / "recon-report.md").write_text(
            "observations\nSASS-WAIVED: x\nSCHED-WAIVED: x\nLIVE-WAIVED: x\n")
        (tmp_path / "candidates.md").write_text("A")
        return "done"
    return _run


def test_source_run_recon_brief_says_the_source_is_in_hand(tmp_path):
    """Same brief, same gates, plus the block that tells the teacher what
    reconnaissance is FOR when the source is readable: measuring what the
    source states only as a template. The blackbox rendering stays
    byte-identical."""
    from forgekernel.loop.driver import _bindings, _recon_prompt, render_prompt
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.oracle_mode = "blackbox"
    ctx.oracle_path = "/ref/gemm.cu"
    plain = render_prompt(ctx.meta_dir / "recon_prompt.md", _bindings(ctx))
    assert _recon_prompt(ctx) == plain
    assert "/ref/gemm.cu" not in plain
    ctx.oracle_mode = "source"
    p = _recon_prompt(ctx, extra="\n\nTAIL")
    assert p.startswith(render_prompt(ctx.meta_dir / "recon_prompt.md", _bindings(ctx)))
    assert "Source-mode addendum" in p and "/ref/gemm.cu" in p
    assert "the binary's reading is the fact" in p
    assert p.endswith("TAIL")
    # a source-mode run with no oracle path configured gets no addendum either
    ctx.oracle_path = None
    assert "Source-mode addendum" not in _recon_prompt(ctx)


def test_source_run_runs_recon_before_distilling(tmp_path):
    """With a recon backend wired, a source-mode run reconnoitres first and its
    distill brief carries the report; the probe never touches the student tree."""
    teacher = FakeBackend(["RECON_REQUEST: notes/launch_cfg.py",
                           _recon_write_up(tmp_path),
                           _authoring(tmp_path),
                           "EXHAUSTED: nothing to do"],
                          [])
    recon = FakeBackend([], ["Block Size (384,1,1)  Grid Size (132,1,1)"])
    ctx = _ctx(tmp_path, teacher)
    ctx.oracle_mode = "source"
    ctx.oracle_path = "/ref/gemm.cu"
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "launch_cfg.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    ctx.milestone.unlink()
    run_two_layer(ctx, max_meta=0)
    assert ("gpu", "cmd:python3 notes/launch_cfg.py") in recon.calls
    assert not [c for c in teacher.calls if c[0] == "gpu"]
    assert "Source-mode addendum" in teacher.prompts[0]          # the recon brief
    distill = teacher.prompts[2]
    assert "Recon report (supplementary input in source mode)" in distill
    assert str(ctx.recon_path) in distill and "/ref/gemm.cu" in distill
    assert "transcribe the measurement" in distill
    # the addendum rides the SOURCE brief, not the blackbox one
    assert "read the original operator's open-source implementation (ORACLE)" in distill


def test_source_run_without_a_recon_tree_is_the_unchanged_control(tmp_path):
    """recon_op = "" (or no tree on disk): no recon turn, no GPU job, and the
    distill brief is exactly the source brief with nothing appended."""
    from forgekernel.loop.driver import _bindings, render_prompt
    teacher = FakeBackend([_authoring(tmp_path), "EXHAUSTED: nothing to do"], [])
    ctx = _ctx(tmp_path, teacher)
    ctx.oracle_mode = "source"
    ctx.oracle_path = "/ref/gemm.cu"
    ctx.milestone.unlink()
    run_two_layer(ctx, max_meta=0)
    assert teacher.prompts[0] == render_prompt(ctx.meta_dir / "distill_prompt.md",
                                               _bindings(ctx))
    assert len(teacher.prompts) == 2                      # distill, then the student


def test_source_recon_addendum_needs_a_report_on_disk(tmp_path):
    """A recon path that was wired but never produced a report (recon skipped,
    budget spent without a write-up) adds nothing: the brief must not point the
    teacher at a file that is not there."""
    from forgekernel.loop.driver import _source_recon_distill_addendum
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.oracle_mode = "source"
    ctx.recon_path = tmp_path / "recon-report.md"
    assert _source_recon_distill_addendum(ctx) == ""
    ctx.recon_path.write_text("findings\n")
    assert str(ctx.recon_path) in _source_recon_distill_addendum(ctx)
    ctx.oracle_mode = "blackbox"                          # blackbox has its own brief
    assert _source_recon_distill_addendum(ctx) == ""


def test_source_recon_addendum_overrides_the_no_report_clauses(tmp_path):
    """The source brief's own §3b / §3c clauses say a source-mode run has no report
    and must derive residency / the inter-kernel ledger itself; with a report on
    disk the addendum must name both so a multi-kernel run transcribes the
    measured "who zeroes" column rather than the templated source's."""
    from forgekernel.loop.driver import _source_recon_distill_addendum
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.oracle_mode = "source"
    ctx.recon_path = tmp_path / "recon-report.md"
    ctx.recon_path.write_text("findings\n")
    add = _source_recon_distill_addendum(ctx)
    assert "§3b" in add and "residency" in add
    assert "§3c" in add and "who zeroes" in add
    # and the brief itself no longer asserts the report is absent on a source-mode run
    brief = (ctx.meta_dir / "distill_prompt.md").read_text(encoding="utf-8")
    assert "source mode has no recon report" not in brief.lower()
    assert "if a recon report is attached, its §3c" in brief


def test_candidates_table_is_briefed_for_every_reconnoitred_run(tmp_path):
    """diagnose / patch used to label the candidates table blackbox-only; a
    reconnoitred source-mode run now has one, and the BLOCKED / fork-flip rules that
    read it must apply there too."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    for name in ("diagnose_prompt.md", "patch_prompt.md"):
        text = (ctx.meta_dir / name).read_text(encoding="utf-8")
        assert "`CANDIDATES` (blackbox runs)" not in text, name
        assert "source runs with a recon tree attached" in text, name
    assert "a source-distilled run without recon has no candidates table" in \
        (ctx.meta_dir / "diagnose_prompt.md").read_text(encoding="utf-8")


# --- resume_note: the meta reset's visible half -------------------------------

def _meta(patch_class, resume_after="D0.13", invalidated=("D0.14", "D0.15")):
    return MetaResult("diag", patch_class, ["D0.14"], resume_after, list(invalidated))


def test_structural_resume_sets_a_standing_note_naming_the_rollback(tmp_path):
    """After a foundation patch the student's only delivery ledger is the trajectory,
    whose old records survive the reset — a gdn run read them as deliveries
    and reported EXHAUSTED into four 0-row segments. The note is the reset's
    visible half, and unlike host_bound_note it stands for every turn of the
    segment."""
    from forgekernel.loop.driver import _set_resume_note, _student_prompt
    ctx = _ctx(tmp_path, FakeBackend([], []))
    _set_resume_note(ctx, _meta("structural"), resume_ref="abc123", baseline_ref="base0")
    assert "D0.13" in ctx.resume_note
    assert "D0.14, D0.15" in ctx.resume_note
    assert "EXHAUSTED" in ctx.resume_note
    assert "baseline (stub kernel)" not in ctx.resume_note
    first = _student_prompt(ctx, 1)
    assert "D0.14, D0.15" in first
    # standing, not one-shot
    assert "D0.14, D0.15" in _student_prompt(ctx, 2)


def test_resume_to_baseline_says_so_and_asks_for_every_decision(tmp_path):
    from forgekernel.loop.driver import _set_resume_note
    ctx = _ctx(tmp_path, FakeBackend([], []))
    _set_resume_note(ctx, _meta("structural", resume_after=None),
                     resume_ref="base0", baseline_ref="base0")
    assert "baseline (stub kernel)" in ctx.resume_note
    assert "Every decision" in ctx.resume_note
    assert "Start from the earliest decision" in ctx.resume_note


@pytest.mark.parametrize("patch_class", ["additive", "kb-example"])
def test_a_patch_that_resets_nothing_leaves_the_standing_note_alone(tmp_path, patch_class):
    """additive / kb-example resume from head(): an earlier structural rollback is
    still in force, so its note still describes the tree. Clearing it here is the
    original bug in miniature — a structural rollback the student exhausted out of,
    then an additive step, and the survived old records read as deliveries again."""
    from forgekernel.loop.driver import _set_resume_note
    ctx = _ctx(tmp_path, FakeBackend([], []))
    _set_resume_note(ctx, _meta("structural"), resume_ref="abc123", baseline_ref="base0")
    standing = ctx.resume_note
    assert "D0.14, D0.15" in standing
    _set_resume_note(ctx, _meta(patch_class), resume_ref="head0", baseline_ref="base0")
    assert ctx.resume_note == standing


def test_an_unmatched_patch_does_not_manufacture_a_rollback(tmp_path):
    """compute_resume returns ResumePlan(tip, []) when the patch touches no
    decision — nothing invalidated, the reset is a no-op. Claiming "everything
    after the resume point" there would spend the segment re-delivering rows that
    are already delivered."""
    from forgekernel.loop.driver import _set_resume_note
    ctx = _ctx(tmp_path, FakeBackend([], []))
    _set_resume_note(ctx, _meta("structural", resume_after="D0.19", invalidated=()),
                     resume_ref="tipsha", baseline_ref="base0")
    assert ctx.resume_note == ""


def test_a_resume_point_with_no_recorded_sha_reads_as_the_stub_it_is(tmp_path):
    """_resolve_resume falls back to baseline when the checkpoint sha is not on
    record. Naming D0.13 there, with a list that omits D0.1..D0.13, would tell the
    student those are still delivered and send it to write D0.14 onto a stub."""
    from forgekernel.loop.driver import _set_resume_note
    ctx = _ctx(tmp_path, FakeBackend([], []))
    _set_resume_note(ctx, _meta("structural", resume_after="D0.13"),
                     resume_ref="base0", baseline_ref="base0")
    assert "baseline (stub kernel)" in ctx.resume_note
    assert "Every decision" in ctx.resume_note
    assert "D0.13" not in ctx.resume_note  # no checkpoint the tree is not at
    assert "D0.14, D0.15" not in ctx.resume_note  # no partial list on a stub


def test_a_foundation_patch_with_an_unparsed_dag_still_reads_as_the_stub(tmp_path):
    """_compute_resume_plan returns (None, []) when the milestone DAG does not
    parse; that is a real full restart, so the empty list must not be read as
    "nothing invalidated"."""
    from forgekernel.loop.driver import _set_resume_note
    ctx = _ctx(tmp_path, FakeBackend([], []))
    _set_resume_note(ctx, _meta("structural", resume_after=None, invalidated=()),
                     resume_ref="base0", baseline_ref="base0")
    assert "baseline (stub kernel)" in ctx.resume_note
    assert "Every decision" in ctx.resume_note


def test_revert_patch_clears_the_note_with_the_workspace(tmp_path):
    """A REVERT_PATCH puts the workspace back on the pre-patch good tree, where the
    'invalidated' decisions are delivered again — a surviving note would send the
    student off to rebuild them."""
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    from forgekernel.loop.driver import _judge_patch, _PendingPatch
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    ctx.resume_note = "- The outer meta loop rolled the workspace back\n"
    append_record(ctx.status_path, IR(iter=0, decision_id="D", change="x",
                                      correctness=True, verdict="KEEP",
                                      bwd_median_ms=2.655, sha="s0"))
    out = _judge_patch(ctx, _PendingPatch(best_before=2.66,
                                          milestone_before="snapshot", good_ref="good"))
    assert out == "REVERT_PATCH"
    assert ("reset", "good") in be.calls
    assert ctx.resume_note == ""


# --- same-job parity (ref_in_job) --------------------------------------------------
# Observed on a grouped-GEMM run: bench.py timed deep_gemm inside every student job,
# the single-ref contract dropped the reading (ref declared via a separate
# tree), and the loop chased a +2.3% "gap" against anchors from other nodes while
# the same-card ratio read parity in every job. ref_in_job keeps the reading
# and judges parity by it — repeated before it ends the run.

def test_ref_in_job_keeps_the_in_job_reference_reading(tmp_path):
    from forgekernel.loop.driver import _enforce_single_ref
    from forgekernel.loop.parse import parse_eval
    ctx = _confirm_ctx(tmp_path, _ConfirmBackend([], []))
    ctx.ref_rides_job = False
    ctx.ref_in_job = True
    ev = _enforce_single_ref(ctx, parse_eval(_EVAL_CONFIRM(2.0, 2.1)))
    assert ev.ref_device_ms == 2.1 and abs(ev.vs_ref_device - 1.05) < 1e-9


def test_paired_parity_stops_only_after_a_same_card_re_measure(tmp_path):
    """One job reads the student 1.4% faster than the same-job ref: parity on
    paper. The run stops only when two more jobs of that sha agree (median
    ratio >= equivalence); the account lands in paired_verdict.jsonl."""
    import json as _json
    be = _ConfirmBackend([REC("D1", 2.0)],
                         [_EVAL_CONFIRM(2.0, 2.028), NCU,          # the iteration
                          _EVAL_CONFIRM(2.0, 2.01), _EVAL_CONFIRM(2.0, 1.99)])  # re-measure
    ctx = _ctx(tmp_path, be, target_ms=1.5)                     # absolute target NOT met
    ctx.gate_window = "device"
    ctx.ref_rides_job = False
    ctx.ref_in_job = True
    assert run_inner_loop(ctx, 0, initial_green=True) == "target"
    assert ("push_ref", be.head(), "fs-demo-confirm") in be.calls \
        or any(c[0] == "push_ref" for c in be.calls)
    rec = _json.loads((tmp_path / "paired_verdict.jsonl").read_text().splitlines()[-1])
    assert rec["verdict"] == "target" and rec["median_ratio"] == pytest.approx(1.0)
    # the record itself kept the same-job reading
    row = load_records(ctx.status_path)[-1]
    assert row.ref_device_ms == 2.028 and row.vs_ref_device == pytest.approx(1.014)


def test_paired_parity_unconfirmed_is_memoised_and_the_run_continues(tmp_path):
    """The re-measure disagrees (median 0.97): a lucky draw. No stop, the sha is
    memoised so the next KEEP of it costs nothing, and the account says so."""
    import json as _json
    be = _ConfirmBackend([REC("D1", 2.0), "EXHAUSTED: done"],
                         [_EVAL_CONFIRM(2.0, 2.05), NCU,
                          _EVAL_CONFIRM(2.0, 1.94), _EVAL_CONFIRM(2.0, 1.90)])
    ctx = _ctx(tmp_path, be, target_ms=1.5)
    ctx.gate_window = "device"
    ctx.ref_rides_job = False
    ctx.ref_in_job = True
    assert run_inner_loop(ctx, 0, initial_green=True) == "exhausted"
    rec = _json.loads((tmp_path / "paired_verdict.jsonl").read_text().splitlines()[-1])
    assert rec["verdict"] == "unconfirmed" and rec["median_ratio"] < 1.0
    assert rec["sha"] in ctx.paired_memo
    assert be.gpu_logs == []                                    # exactly two extra jobs


def test_paired_parity_is_inert_without_the_flag_or_below_equivalence(tmp_path):
    from forgekernel.loop.driver import _paired_parity_stop
    from forgekernel.inner.trajectory import IterationRecord
    be = _ConfirmBackend([], [_EVAL_CONFIRM(2.0, 2.1)])
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    row = IterationRecord(iter=0, decision_id="D1", change="c", correctness=True,
                          verdict="KEEP", device_median_ms=2.0, ref_device_ms=2.1,
                          vs_ref_device=1.05, sha="s1")
    assert _paired_parity_stop(ctx, row, 0) is False           # flag off
    ctx.ref_in_job = True
    ctx.equivalence_ratio = 1.10
    assert _paired_parity_stop(ctx, row, 0) is False           # 1.05 < 1.10
    assert not any(c[0] == "gpu" for c in be.calls)            # nothing bought


def test_paired_reanchor_replaces_the_cross_channel_drift_transport(tmp_path, monkeypatch):
    """ref_in_job run at a plateau with an absolute residual (+3.5%): the
    phantom-gap check re-measures the best same-card instead of buying fresh
    ref jobs. Parity -> stop as target; below parity -> the note carries the
    PAIRED residual."""
    import json as _json
    from forgekernel.loop.driver import _phantom_gap_reanchor
    calls: list[int] = []
    ctx = _reanchor_ctx(tmp_path, samples=[2.0, 2.0, 2.0], calls=calls)
    ctx.ref_in_job = True
    ctx.backend = _ConfirmBackend([], [_EVAL_CONFIRM(2.07, 2.09), _EVAL_CONFIRM(2.07, 2.07)])
    from forgekernel.inner.trajectory import IterationRecord, append_record
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=2.07, sha="shabest"))
    out = _phantom_gap_reanchor(ctx, 2.07)
    assert out is not None and out["verdict"] == "target"
    assert out["estimator"] == "paired-in-job"
    assert calls == []                                          # no fresh ref channel
    # below parity: paired residual in the note, memoised per best
    ctx2 = _reanchor_ctx(tmp_path, samples=[2.0, 2.0, 2.0], calls=calls)
    ctx2.ref_in_job = True
    ctx2.backend = _ConfirmBackend([], [_EVAL_CONFIRM(2.07, 2.03), _EVAL_CONFIRM(2.07, 2.01)])
    append_record(ctx2.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=2.07, sha="shabest"))
    assert _phantom_gap_reanchor(ctx2, 2.07) is None
    assert ctx2.reanchor_note and "Same-card paired re-measure" in ctx2.reanchor_note
    assert "+2.5%" in ctx2.reanchor_note                       # 1/0.9758 - 1
    rec = _json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[-1])
    assert rec["verdict"] == "confirmed_gap" and rec["estimator"] == "paired-in-job"
    assert calls == []
    # unchanged best: the verdict is reused, nothing re-bought
    ctx2.reanchor_note = None
    assert _phantom_gap_reanchor(ctx2, 2.07) is None
    assert ctx2.reanchor_note is not None and ctx2.backend.gpu_logs == []


def test_best_confirmation_can_target_a_named_sha(tmp_path):
    be = _ConfirmBackend([], [_EVAL_CONFIRM(3.0, 3.0), _EVAL_CONFIRM(3.0, 3.0)])
    out = run_best_confirmation(_confirm_ctx(tmp_path, be), jobs=2, ref="shaworse")
    assert ("push_ref", "shaworse", "fs-demo-confirm") in be.calls
    assert out["sha"] == "shaworse"


def test_paired_re_measure_needs_two_agreeing_samples(tmp_path):
    """One usable sample, or two that disagree beyond the same-job floor, is no
    verdict: no stop, no memo (a later plateau may try again)."""
    import json as _json
    from forgekernel.loop.driver import _admit_paired
    # a lone sample (second job failed correctness) -> inadmissible
    be = _ConfirmBackend([REC("D1", 2.0), "EXHAUSTED: done"],
                         [_EVAL_CONFIRM(2.0, 2.05), NCU,
                          _EVAL_CONFIRM(2.0, 2.05), _EVAL_CONFIRM(2.0, 2.05, ok=False),
                          _EVAL_CONFIRM(2.0, 2.05, ok=False)])
    ctx = _ctx(tmp_path, be, target_ms=1.5)
    ctx.gate_window = "device"
    ctx.ref_rides_job = False
    ctx.ref_in_job = True
    assert run_inner_loop(ctx, 0, initial_green=True) == "exhausted"
    rec = _json.loads((tmp_path / "paired_verdict.jsonl").read_text().splitlines()[-1])
    assert rec["verdict"] == "remeasure_unavailable" and "need >= 2" in rec["why"]
    assert ctx.paired_memo == {} and ctx.paired_stop is None
    # two samples straddling the floor: median would pass, admission refuses
    med, why = _admit_paired({"jobs": [{"ratio": 1.04}, {"ratio": 0.98}]})
    assert med is None and "spread" in why
    med, why = _admit_paired({"jobs": [{"ratio": 1.01}, {"ratio": 0.995}]})
    assert why is None and med == pytest.approx(1.0025)


def test_paired_stop_hands_its_sha_and_samples_to_the_finish_confirmation(tmp_path):
    """The parity stop records which sha stopped the run; the finish confirmation
    confirms that sha and counts the samples already bought."""
    be = _ConfirmBackend([REC("D1", 2.0)],
                         [_EVAL_CONFIRM(2.0, 2.028), NCU,
                          _EVAL_CONFIRM(2.0, 2.01), _EVAL_CONFIRM(2.0, 1.99),   # parity re-measure
                          _EVAL_CONFIRM(2.0, 2.0)])                             # the ONE extra confirm job
    ctx = _ctx(tmp_path, be, target_ms=1.5)
    ctx.gate_window = "device"
    ctx.ref_rides_job = False
    ctx.ref_in_job = True
    assert run_inner_loop(ctx, 0, initial_green=True) == "target"
    assert ctx.paired_stop and len(ctx.paired_stop["samples"]) == 2
    sha = ctx.paired_stop["sha"]
    out = run_best_confirmation(ctx, jobs=3, ref=sha, seed_samples=ctx.paired_stop["samples"])
    assert out["sha"] == sha and len(out["jobs"]) == 3
    assert be.gpu_logs == []                                    # bought exactly one more
    assert out["median_ratio"] == pytest.approx(1.0)


def test_paired_reanchor_reuses_an_unconfirmed_re_measure_of_the_same_sha(tmp_path):
    """iter N read parity on paper, the re-measure said 0.985 (memoised); a
    later plateau whose best is that sha reuses the memo instead of buying the
    same two jobs, and the diagnosis gets the paired residual from it."""
    import json as _json
    from forgekernel.loop.driver import _phantom_gap_reanchor
    from forgekernel.inner.trajectory import IterationRecord, append_record
    ctx = _reanchor_ctx(tmp_path, samples=[2.0, 2.0, 2.0])
    ctx.ref_in_job = True
    ctx.backend = _ConfirmBackend([], [])                       # nothing to buy with
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=2.07, sha="shabest"))
    ctx.paired_memo["shabest"] = {"iter": 0, "cluster": "", "median_ratio": 0.985,
                                  "jobs": ["j1", "j2"], "verdict": "unconfirmed"}
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    assert not any(c[0] == "gpu" for c in ctx.backend.calls)
    rec = _json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[-1])
    assert rec["reused_from_iter"] == 0 and rec["verdict"] == "confirmed_gap"
    assert rec["student_jobs"] == ["j1", "j2"]
    assert "+1.5%" in ctx.reanchor_note                         # 1/0.985 - 1
    # a memo from another cluster does not travel: measure afresh (and fail
    # admission here, since the backend has nothing to give)
    ctx2 = _reanchor_ctx(tmp_path, samples=[2.0, 2.0, 2.0])
    ctx2.ref_in_job = True
    ctx2.backend = _ConfirmBackend([], [])
    ctx2.home_cluster = "machine_a"
    append_record(ctx2.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=2.07, sha="shabest"))
    ctx2.paired_memo["shabest"] = {"iter": 0, "cluster": "machine_b", "median_ratio": 0.985,
                                   "jobs": ["j1", "j2"], "verdict": "unconfirmed"}
    assert _phantom_gap_reanchor(ctx2, 2.07) is None
    assert ctx2.reanchor_note is None                           # no verdict at all


def test_parse_lever_prices_keeps_the_last_copy_of_the_table():
    """The run prints the table twice (timing-only, then with roles); a job
    killed in the role pass leaves only the first. The parser keeps the last."""
    from forgekernel.loop.driver import parse_lever_prices
    log = ("LEVER ring add=+8.56% remove=-8.10% add_ms=3.135 remove_ms=3.21 class=independent "
           "spin_add=None spin_all_minus=None cubin_same_as_base=unknown\n"
           "STACK all=+8.56% sum_singles=+8.56% coupling=+0.00%\n"
           "PRICE-TIMED levers=1 rounds=3\n")
    levers, stack, fail = parse_lever_prices(log)
    assert fail is None and len(levers) == 1 and levers[0]["spin_add_ns"] is None
    assert stack["all_pct"] == 8.56 and stack.get("base_ms") is None
    full = log + ("LEVER ring add=+8.56% remove=-8.10% add_ms=3.135 remove_ms=3.21 class=independent "
                  "spin_add=880319 spin_all_minus=996231 cubin_same_as_base=False\n"
                  "STACK all=+8.60% sum_singles=+8.56% coupling=+0.04%\n"
                  "PRICE-OK levers=1 rounds=3 base_ms=3.40 all_ms=3.13 spin_base=1 spin_all=2 "
                  "tensor_base=37.0 tensor_all=44.0\n")
    levers, stack, _ = parse_lever_prices(full)
    assert len(levers) == 1 and levers[0]["spin_add_ns"] == 880319
    assert stack["all_pct"] == 8.60 and stack["base_ms"] == 3.40


def test_levers_priced_at_an_earlier_step_still_verify(tmp_path):
    """The reflection lists earlier steps' levers as priced and the design turn
    is told not to re-list them, so a citation of one must verify against the
    same readings — the current step's table wins on a name clash."""
    from forgekernel.loop.driver import _merge_lever_tables, _verify_reopen_price
    from forgekernel.meta_trajectory import MetaRecord
    earlier = MetaRecord(meta_step=0, cause_class="R1", deficient_locus="D1", patch_class="unpriced-reopen",
                         changed_entries=[], patch_direction="flip", outcome="UNPRICED_REOPEN",
                         lever_prices=[{"kind": "lever", "name": "ring", "add_pct": 8.56, "remove_pct": -8.1,
                                        "cls": "independent"},
                                       {"kind": "stack", "all_pct": 8.56}])
    merged = _merge_lever_tables([earlier], [])
    assert [lv["name"] for lv in merged] == ["ring"] and merged[0]["step"] == 0
    ctx = _priced_ctx(tmp_path, FakeBackend([], []))
    assert _verify_reopen_price("+8.6% via lever:ring", [], merged, ctx, []) is None
    current = [{"kind": "lever", "name": "ring", "add_pct": 2.0, "remove_pct": None, "cls": "dependent"},
               {"kind": "stack", "all_pct": 2.0}]
    merged = _merge_lever_tables([earlier], current)
    assert [(lv["name"], lv.get("add_pct")) for lv in merged if lv["kind"] == "lever"] == [("ring", 2.0)]
    assert [lv for lv in merged if lv["kind"] == "stack"] == [current[1]]


def test_lever_pricing_reuses_the_table_priced_on_the_same_best_tree(tmp_path):
    """A retried meta step (or an audit followed by a plateau on the same first
    green) prices the same best tree: the banked table is reused — no design
    turn, no job — and the diagnosis still reads it."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD
    be = FakeBackend([_lever_design(tmp_path), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, _PRICE_LOG_OK])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", _LEVER_PRICING_CMD)) == 1
    be2 = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                        "patch_class: additive\nchanged_entries: []\n")],
                      [_STALL_LOG_OK])
    ctx2 = _lever_ctx(tmp_path, be2)
    run_meta_step(ctx2)
    assert ("gpu", _LEVER_PRICING_CMD) not in be2.calls
    assert len([c for c in be2.calls if c[0] == "agent"]) == 2          # diagnose + patch only
    assert "## Pricing table (measured by the driver on the best tree" in be2.prompts[0] and "`ring`" in be2.prompts[0]


def test_reopen_gate_needs_a_priced_tree_for_the_pricing_route(tmp_path):
    """With stall pricing off the pricing round never runs (it prices the best
    tree the stall pricing found), so it is not an in-step route."""
    from forgekernel.loop.driver import _reopen_gate_on
    ctx = _priced_ctx(tmp_path, FakeBackend([], []))
    ctx.max_verdict_probe_rounds = 0
    ctx.lever_pricing_rounds = 3
    ctx.stall_pricing = False
    assert not _reopen_gate_on(ctx)
    ctx.stall_pricing = True
    assert _reopen_gate_on(ctx)
    ctx.lever_pricing_rounds = 0
    assert not _reopen_gate_on(ctx)
    ctx.max_verdict_probe_rounds = 1
    assert _reopen_gate_on(ctx)


def test_kb_harvest_lints_the_write_up_and_reprompts_once(tmp_path):
    """What the harvest writes into hw_kb is linted the way the loop start and
    publish lint it; findings come back once, with the offending lines."""
    from forgekernel.loop.driver import ProbeBattle, run_kb_harvest
    (tmp_path / "sm90").mkdir(exist_ok=True)
    idx = tmp_path / "sm90" / "index.md"

    def dirty(prompt):
        idx.write_text("| foo | ✅ Verified | (iter 12 KEEP, axis cache 0→1, gate 3.4→3.1, job 870533) |\n",
                       encoding="utf-8")
        return "RECIPE_RECORDED: hw_kb/cutedsl/sm90/foo.md"

    def clean(prompt):
        assert "failed kb_lint" in prompt and "iteration number" in prompt
        idx.write_text("| foo | ✅ Verified | (KEEP in job 870533, axis cache 0→1, gate 3.4→3.1) |\n",
                       encoding="utf-8")
        return "RECIPE_RECORDED: hw_kb/cutedsl/sm90/foo.md"

    be = FakeBackend([dirty, clean], [])
    ctx = _ctx(tmp_path, be)
    ctx.kb_tripwire_rounds = 6
    ctx.probe_battles["notes/hard.py"] = ProbeBattle(
        path="notes/hard.py", rounds=10, passed=True, pass_iter=17,
        verdict_line="PROBE PASS (max_diff=0.0)", log_tail="note: rung passed")
    assert run_kb_harvest(ctx) == ["hw_kb/cutedsl/sm90/foo.md"]
    assert len([c for c in be.calls if c[0] == "agent"]) == 2
    assert "iter 12" not in idx.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Fourth-round review fixes (phase reading on live rows, gate FAIL notice,     #
# phase_verdict spelling, same-stage gate check, PM interval, KB certify       #
# lineage, kept probes, ROLE near-miss)                                        #
# --------------------------------------------------------------------------- #

def test_phase_verdict_accepts_spaced_quoted_and_glossed_spellings():
    from forgekernel.loop.driver import _parse_phase_verdict, _verify_phase_verdict
    ex = {"phase": "M1", "gate": "M1-gate", "priced": ["D1.1"],
          "unpriced": ["D1.2"], "delivered": []}
    for line in ("phase_verdict: continue: D1.2",
                 "phase_verdict: continue:`D1.2`",
                 "phase_verdict: continue: `D1.2`",
                 "phase_verdict: `continue: D1.2`（the only unpriced one）",
                 "phase_verdict: continue:D1.2  # the one unpriced decision"):
        v = _parse_phase_verdict(line)
        assert v == "continue:D1.2", (line, v)
        assert _verify_phase_verdict(v, ex) is None
    assert _parse_phase_verdict("phase_verdict: close (M1 fully measured)") == "close"
    assert _parse_phase_verdict("phase_verdict: **close**") == "close"
    assert _parse_phase_verdict("phase_verdict:\ncause_class: R1") == ""


def test_live_records_void_reopened_keeps_and_reverted_segments():
    from forgekernel.loop.driver import _live_records
    from forgekernel.inner.trajectory import IterationRecord as IR

    def keep(did, i):
        return IR(iter=i, decision_id=did, change="", correctness=True, verdict="KEEP")
    rows = [keep("D1.1", 0), keep("M0-gate", 1), keep("M1-gate", 2),
            IR(iter=3, decision_id="REOPEN", correctness=None, verdict="NEUTRAL",
               change="REOPEN on abc: milestone patch changed D1.1; the dependency "
                      "closure [D1.1, D1.2] is reopened — earlier KEEP rows for these "
                      "decisions are void, redo them. Kept as built: [D0.3]."),
            keep("D1.1", 4)]
    live = [r.decision_id for r in _live_records(rows) if r.verdict == "KEEP"]
    # the pre-reopen D1.1 and the M1-gate premised on it are void; the redo stands
    assert live == ["M0-gate", "D1.1"]
    # a REOPEN-REVERT restores what the REOPEN voided
    rows.append(IR(iter=5, decision_id="REOPEN-REVERT", change="reverted",
                   correctness=None, verdict="NEUTRAL"))
    assert [r.decision_id for r in _live_records(rows) if r.verdict == "KEEP"] == \
        ["D1.1", "M0-gate", "M1-gate", "D1.1"]
    # a reverted segment's rows are void by index range
    assert [r.decision_id for r in _live_records(rows, voided=[(4, 5)]) if r.verdict == "KEEP"] == \
        ["D1.1", "M0-gate", "M1-gate"]


def test_phase_exhaustion_reads_the_reopened_phase_not_the_next_one():
    from forgekernel.loop.driver import _phase_exhaustion, _live_records
    from forgekernel.inner.trajectory import IterationRecord as IR

    class _C:
        decisions_dag = [Decision("D1.1", frozenset()), Decision("D1.2", frozenset({"D1.1"})),
                         Decision("D2.1", frozenset({"D1.2"}))]
        milestone = "/nonexistent"
    rows = [IR(iter=0, decision_id="M0-gate", change="", correctness=True, verdict="KEEP"),
            IR(iter=1, decision_id="D1.1", change="", correctness=True, verdict="KEEP"),
            IR(iter=2, decision_id="M1-gate", change="", correctness=True, verdict="KEEP"),
            IR(iter=3, decision_id="REOPEN", correctness=None, verdict="NEUTRAL",
               change="the dependency closure [D1.1, D1.2] is reopened — earlier KEEP rows")]
    table = [{"kind": "lever", "name": "a", "cls": "null", "decisions": ["D1.1"]}]
    # on the raw rows M1 looks closed (its gate KEEP is on file) -> the reading
    # would target M2 and find no lever there
    assert _phase_exhaustion(_C(), rows, table) is None
    ex = _phase_exhaustion(_C(), _live_records(rows), table)
    assert ex is not None and ex["phase"] == "M1"
    assert ex["priced"] == ["D1.1"] and ex["unpriced"] == ["D1.2"] and ex["delivered"] == []


def test_a_paying_subset_keeps_the_phase_open():
    """Every lever of the phase reads null / harmful, but a subset (the
    teacher's, or the tool's all-pos) that moves the phase's decisions reads
    above the noise floor: the combination is the way forward, the phase is
    not exhausted — as the reading rule handed to the teacher says (the subset
    wins over the class column)."""
    from forgekernel.loop.driver import _phase_exhaustion, _table_noise_pct
    from forgekernel.inner.trajectory import IterationRecord as IR

    class _C:
        decisions_dag = [Decision("D1.1", frozenset()), Decision("D1.2", frozenset({"D1.1"})),
                         Decision("D1.3", frozenset()), Decision("D2.1", frozenset({"D1.2"}))]
        milestone = "/nonexistent"
    gate = IR(iter=0, decision_id="M0-gate", change="", correctness=True, verdict="KEEP")
    table = [{"kind": "lever", "name": "a", "cls": "null", "add_pct": 0.1, "decisions": ["D1.1"]},
             {"kind": "lever", "name": "b", "cls": "null", "add_pct": 0.2, "decisions": ["D1.2"]},
             {"kind": "lever", "name": "c", "cls": "harmful", "add_pct": -18.0, "decisions": ["D1.3"]},
             {"kind": "variant", "name": "all-pos", "levers": ["a", "b"], "delta_pct": 4.0,
              "decisions": ["D1.1", "D1.2"]},
             {"kind": "stack", "all_pct": -10.0, "all_pos_pct": 4.0, "noise_pct": 0.9}]
    assert _table_noise_pct(table) == 0.9
    assert _phase_exhaustion(_C(), [gate], table) is None
    # inside the noise band the subset is no way forward: exhausted as before
    table[3]["delta_pct"] = 0.6
    assert _phase_exhaustion(_C(), [gate], table)["phase"] == "M1"
    # a subset of another phase's decisions does not keep THIS phase open
    table[3].update({"delta_pct": 4.0, "decisions": ["D2.1"]})
    assert _phase_exhaustion(_C(), [gate], table)["phase"] == "M1"
    # no noise line banked (an older table): the tool's default floor
    assert _table_noise_pct([{"kind": "stack"}]) == 0.5


def test_harmful_lever_counts_as_exhausted_not_as_a_way_forward():
    from forgekernel.loop.driver import _phase_exhaustion
    from forgekernel.inner.trajectory import IterationRecord as IR

    class _C:
        decisions_dag = [Decision("D1.1", frozenset()), Decision("D1.2", frozenset({"D1.1"}))]
        milestone = "/nonexistent"
    gate = IR(iter=0, decision_id="M0-gate", change="", correctness=True, verdict="KEEP")
    table = [{"kind": "lever", "name": "a", "cls": "harmful", "add_pct": -8.0, "decisions": ["D1.1"]},
             {"kind": "lever", "name": "b", "cls": "null", "decisions": ["D1.2"]}]
    ex = _phase_exhaustion(_C(), [gate], table)
    assert ex is not None and ex["levers"] == ["a", "b"] and ex["unpriced"] == []
    # an unstacked lever (positive single reading, stack broke) IS a way forward
    table[1] = {"kind": "lever", "name": "b", "cls": "unstacked", "add_pct": 3.0, "decisions": ["D1.2"]}
    assert _phase_exhaustion(_C(), [gate], table) is None


def test_gate_regression_reads_the_same_job_ratio_and_same_cluster_only():
    from forgekernel.loop.driver import _gate_regression
    from forgekernel.inner.trajectory import IterationRecord as IR

    def keep(ms, ratio=None, cluster="", i=0):
        return IR(iter=i, decision_id="M0-gate", change="", correctness=True, verdict="KEEP",
                  bwd_median_ms=ms, device_median_ms=ms, vs_ref=ratio, vs_ref_device=ratio,
                  cluster=cluster)
    # both rows carry a same-job ratio: the ratio decides, absolute ms do not
    prev, cur = keep(12.31, 1.10, i=0), keep(14.43, 1.09, i=1)     # +17% ms on a slow node
    assert _gate_regression([prev, cur], cur, 1, {}, "device") is None
    cur = keep(14.43, 0.95, i=1)                                     # ratio fell 14%
    msg = _gate_regression([prev, cur], cur, 1, {}, "device")
    assert msg and "same-job ratio" in msg
    # absolute readings across clusters are not a comparison
    prev, cur = keep(12.31, cluster="machine_a", i=0), keep(14.43, cluster="machine_b", i=1)
    assert _gate_regression([prev, cur], cur, 1, {}, "device") is None
    # same cluster: the cross-job band (6%) applies, not the same-job 1.5%
    prev, cur = keep(12.31, cluster="machine_a", i=0), keep(12.31 * 1.05, cluster="machine_a", i=1)
    assert _gate_regression([prev, cur], cur, 1, {}, "device") is None
    cur = keep(14.43, cluster="machine_a", i=1)
    msg = _gate_regression([prev, cur], cur, 1, {}, "device")
    assert msg and "cross-job noise" in msg


def test_pm_interval_follows_the_target_and_the_knob(tmp_path):
    from forgekernel.loop.driver import _pm_interval, _stall_price_cmd, _STALL_PRICE_CMD
    ctx = _ctx(tmp_path, FakeBackend([], []), target_ms=1.9)
    assert _pm_interval(ctx) == 8192 and _stall_price_cmd(ctx) == _STALL_PRICE_CMD
    ctx.target_ms = 0.25                       # fp8_gemm / attn-fwd class reference
    assert _pm_interval(ctx) == 2048
    assert _stall_price_cmd(ctx) == _STALL_PRICE_CMD + " --pm-interval 2048"
    ctx.ref_price_pm_interval = 4096           # the knob wins over the reading
    assert _pm_interval(ctx) == 4096


def test_reference_pricing_retries_a_pmtl_only_miss_when_the_interval_changed(tmp_path):
    """Only the timeline is missing: no retry on the same interval (the same
    collection would starve again) — one retry once the run's interval moved."""
    from forgekernel.loop.driver import (_ensure_ref_pricing, _ref_pricing_key,
                                         _save_ref_role_exposure, _STALL_PRICE_CMD)
    be = FakeBackend([], [])
    rbe = FakeBackend([], [_STALL_LOG_OK + _role_block(_ROLE_CORE_REF)] * 3)
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    ctx.ref_price_dir = _ref_tree(tmp_path)
    ctx.ref_price_backend = rbe
    ctx.target_ms = 1.9
    _save_ref_role_exposure(ctx, _role_block(_ROLE_CORE_REF))   # ROLE banked, PMTL missing
    (tmp_path / "evidence" / "ref_price_job.log").write_text(
        f"[ref-price] key={_ref_pricing_key(ctx)} app=None pm_interval=8192\nPMTL-FAIL\n")
    _ensure_ref_pricing(ctx)
    assert not [c for c in rbe.calls if c[0] == "gpu"]              # same interval: no job
    ctx.ref_price_pm_interval = 2048
    _ensure_ref_pricing(ctx)
    cmds = [c[1] for c in rbe.calls if c[0] == "gpu"]
    assert cmds == [_STALL_PRICE_CMD + " --pm-interval 2048"]
    assert "pm_interval=2048" in (tmp_path / "evidence" / "ref_price_job.2.log").read_text()


def test_kb_certify_leaves_the_student_lineage_teacher_free(tmp_path):
    """The kb-probe commits stay on the -kbprobe branch; the workspace goes back
    to the pre-certification head and the probe file is re-materialised, so the
    student's next KEEP pairs with a censused ancestor the records know."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "kb.py").write_text("print('rig v1')\n")
    be = FakeBackend(["PROBE_REQUEST: notes/kb.py", "RECIPE_CERTIFIED: hw_kb/x.md"],
                     ["PROBE PASS max_diff=0.0"])
    ctx = _ctx(tmp_path, be)
    pre = be.head()
    res = run_kb_certify(ctx, diagnosis="cause_class: G3")
    assert res.certified is True and res.probe == "notes/kb.py"
    assert ("reset", pre) in be.calls and be.calls.index(("reset", pre)) > \
        be.calls.index(("push", "fs-demo-kbprobe"))
    assert (tmp_path / "notes" / "kb.py").read_text() == "print('rig v1')\n"
    # no probe round (teacher finalized in prose): nothing to reset
    be2 = FakeBackend(["RECIPE_CERTIFIED: hw_kb/y.md"], [])
    run_kb_certify(_ctx(tmp_path, be2), diagnosis="cause_class: G3")
    assert not [c for c in be2.calls if c[0] == "reset"]


def test_kept_probes_end_with_the_landed_decision_and_yield_to_the_student(tmp_path):
    from forgekernel.loop.driver import _restore_kept_probes
    ctx = _ctx(tmp_path, FakeBackend([], []))
    (tmp_path / "notes").mkdir()
    ctx.kept_probes = {"notes/p.py": "print('rig v3')\n", "notes/q.py": "print('q')\n"}
    # the student edited p.py: the workspace copy wins and the keep is dropped
    (tmp_path / "notes" / "p.py").write_text("print('rig v4 by the student')\n")
    _restore_kept_probes(ctx)
    assert (tmp_path / "notes" / "p.py").read_text() == "print('rig v4 by the student')\n"
    assert "notes/p.py" not in ctx.kept_probes
    assert (tmp_path / "notes" / "q.py").read_text() == "print('q')\n"     # erased one restored
    # a landed decision (delivered without re-probing) ends every kept fight
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx.backend = be
    run_inner_iteration(ctx, 1, metric_before=3.0)
    assert ctx.kept_probes == {}


def test_role_ok_without_tensor_is_not_a_near_miss_and_mismatch_cannot_close_role(tmp_path):
    from forgekernel.loop.driver import _collect_verdict_near_misses, _missing_recon_evidence
    near = {}
    _collect_verdict_near_misses("ROLE-OK roles=3 samples=8000 resolved=97.1\nSASS-OK kern", near)
    assert "SASS" in near and "ROLE" not in near and "PMTL" not in near
    ctx = _stall_ctx(tmp_path, FakeBackend([], []))
    ctx.recon_dir = tmp_path
    ctx.recon_path = tmp_path / "report.md"
    ctx.recon_path.write_text("ROLE-DELIVERED-MISMATCH: ROLE-OK roles=3 samples=8000\n"
                              "PMTL-DELIVERED-MISMATCH: PMTL-OK kernel=k\n")
    missing = _missing_recon_evidence(ctx, True, True, True, True,
                                      near_miss={"ROLE": "x", "PMTL": "y"})
    assert "ROLE" in missing and "PMTL" in missing


# --- ablations: conviction on the same pricing job / a JSON verdict probe ----

_ABLATE_LOG = (
    "ABLATE no_bar delta=+9.20% ncu_ns=910000.0 base_ncu_ns=1002000.0 correctness=fail "
    "device_delta=n/a kernel=main_kernel status=ok h=the per-stage barrier is the exposed wait\n"
    "ABLATE serial_kv delta=+0.30% ncu_ns=999000.0 base_ncu_ns=1002000.0 correctness=pass "
    "device_delta=+0.10% kernel=main_kernel status=ok h=KV supply is not the wait\n"
    "ABLATE helps delta=-5.00% ncu_ns=1054000.0 base_ncu_ns=1002000.0 correctness=fail "
    "device_delta=n/a kernel=main_kernel status=ok h=the prefetch is dead weight\n"
    "PRICE ablation_noise_floor=1.20% metric=ncu_ns\n"
    "ABLATE-OK ablations=2 rounds=2 base_ncu_ns=1002000.0\n")


def _lever_design_with_ablations(tmp_path):
    def agent(prompt):
        assert "ablation" in prompt and "ablations" in prompt and "ablation:<name>" in prompt
        (tmp_path / "notes").mkdir(exist_ok=True)
        _lever_kernel(tmp_path)
        (tmp_path / "notes" / "levers.json").write_text(json.dumps({"rounds": 3, "levers": [
            {"name": "ring", "decisions": ["D1.1"], "edits": [{"old": "a", "new": "b"}]}],
            "ablations": [
                {"name": "no bar", "decisions": ["D1.1", "D9.9"], "hypothesis": "the per-stage\nbarrier is the exposed wait",
                 "edits": [{"old": "c", "new": "C"}]},
                {"name": "serial_kv", "edits": [{"old": "e", "new": "E"}]}]}))
        return "LEVERS: 1"
    return agent


def test_design_turn_ablations_ride_the_pricing_job_and_are_citable(tmp_path):
    """`ablations` in levers.json are normalised (one-token name, unknown
    decision ids dropped), priced in the SAME job after the lever table, rendered
    as their own table with the reading rules, banked on the record, and a
    measured one is a legal `reopen_price ... via ablation:<name>` — while an
    n/a one and an unknown one are not. Ablation rows are not levers: they
    never make a phase read exhausted."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD, _phase_exhaustion, _verify_reopen_price
    be = FakeBackend([_lever_design_with_ablations(tmp_path), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, _PRICE_LOG_OK + _ABLATE_LOG])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert ("gpu", _LEVER_PRICING_CMD) in be.calls
    spec = json.loads((tmp_path / "notes" / "levers.json").read_text())
    assert [a["name"] for a in spec["ablations"]] == ["no_bar", "serial_kv"]
    assert spec["ablations"][0]["decisions"] == ["D1.1"]            # D9.9 dropped, not bounced
    assert spec["ablations"][0]["hypothesis"] == "the per-stage barrier is the exposed wait"
    diag = be.prompts[1]
    assert "Ablation table" in diag and "| `no_bar` | +9.20% | ok | fail | n/a | D1.1 | the per-stage barrier is the exposed wait |" in diag
    assert "noise floor 1.20%" in diag and "not the gain of any particular fix" in diag and "this mechanism is helping" in diag
    recs = [json.loads(l) for l in (tmp_path / "meta_status.jsonl").read_text().splitlines()]
    table = recs[-1]["lever_prices"]
    abl = [t for t in table if t["kind"] == "ablation"]
    assert [a["name"] for a in abl] == ["no_bar", "serial_kv", "helps"] and abl[0]["decisions"] == ["D1.1"]
    # only a POSITIVE delta above the pass's noise floor (1.20%) is a price
    assert _verify_reopen_price("+9.2% via ablation:no_bar", [], table, ctx) is None
    assert "within the 1.20% noise floor" in _verify_reopen_price("+0.3% via ablation:serial_kv", [], table, ctx)
    assert "SLOWER" in _verify_reopen_price("+5% via ablation:helps", [], table, ctx)
    assert "not on the price table" in _verify_reopen_price("+9.2% via ablation:nope", [], table, ctx)
    dead = table + [{"kind": "ablation", "name": "hung", "delta_pct": None, "status": "hung"}]
    assert "no measured delta (n/a: status=hung)" in _verify_reopen_price("+9.2% via ablation:hung", [], dead, ctx)
    # a phase whose only rows are ablations is not "exhausted": there is no lever to read
    only_abl = [t for t in table if t["kind"] == "ablation"]
    assert _phase_exhaustion(ctx, load_records(ctx.status_path), only_abl) is None


def test_ablation_only_design_turn_still_buys_the_job(tmp_path):
    """No candidate lever but ablations to run: the teacher writes only
    `ablations` (and may still say LEVERS: none); the job runs, the prompt
    says there is no lever table, the ablation table is there."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD

    def agent(prompt):
        (tmp_path / "notes").mkdir(exist_ok=True)
        _lever_kernel(tmp_path)
        (tmp_path / "notes" / "levers.json").write_text(json.dumps({"ablations": [
            {"name": "no_bar", "hypothesis": "h", "edits": [{"old": "c", "new": "C"}]}]}))
        return "LEVERS: none"
    be = FakeBackend([agent, _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, "PRICE kernel_file=k variants: base | ablations: ~no_bar\n" + _ABLATE_LOG])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert ("gpu", _LEVER_PRICING_CMD) in be.calls
    diag = be.prompts[1]
    assert "No candidate levers this step" in diag and "| `no_bar` | +9.20% |" in diag


def test_json_probe_request_runs_the_frozen_ablation_recipe(tmp_path):
    """A verdict probe written as `notes/<x>.json` runs lever_pricing_run on the
    current tree instead of a hand-written script: toolkit seeded, spec
    normalised (levers dropped, kernel_file set), one GPU job, the ablation
    table fed back, its rows on the step's price table and in probe_readings,
    and the R1 that cites `ablation:<name>` passes the price gate."""
    from forgekernel.meta_trajectory import load_meta_records

    def probe(prompt):
        _lever_kernel(tmp_path)
        (tmp_path / "notes").mkdir(exist_ok=True)
        (tmp_path / "notes" / "abl.json").write_text(json.dumps({
            "levers": [{"name": "ignored", "edits": [{"old": "a", "new": "b"}]}],
            "ablations": [{"name": "no bar", "decisions": ["D1.1"], "hypothesis": "h",
                           "edits": [{"old": "c", "new": "C"}]},
                          {"name": "serial_kv", "edits": [{"old": "e", "new": "E"}]}]}))
        return "PROBE_REQUEST: notes/abl.json"
    priced = _R1 + "reopen_price: +9.2% via ablation:no_bar\n"
    be = FakeBackend([_R1, probe, priced,
                      _patching(tmp_path, "\n### D1.1 axis flipped\nx\n",
                                "patch_class: structural\nchanged_entries: [D1.1]\n")],
                     [_ABLATE_LOG])
    ctx = _priced_ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert res.patch_class == "structural"
    assert ("gpu", "cmd:python3 notes/lever_pricing_run.py notes/abl.json") in be.calls
    assert (tmp_path / "notes" / "lever_pricing_run.py").is_file()
    spec = json.loads((tmp_path / "notes" / "abl.json").read_text())
    assert spec["levers"] == [] and spec["roles"] is False
    assert spec["kernel_file"] == "flash_attn_dsl/flash_bwd.py"
    assert [a["name"] for a in spec["ablations"]] == ["no_bar", "serial_kv"]
    fb = be.prompts[2]
    assert "Verdict experiment 1/" in fb and "ablation, driver-frozen recipe" in fb
    assert "| `no_bar` | +9.20% | ok | fail | n/a | D1.1 | the per-stage barrier is the exposed wait |" in fb
    assert "via ablation:<name>" in fb and "via probe:notes/abl.json#<name>" in fb
    rec = load_meta_records(ctx.meta_status_path)[-1]
    assert rec.reopen_price == "+9.2% via ablation:no_bar" and rec.probe_rounds == 1
    verdicts = {r["contract"]: (r["verdict"], r["reading"]) for r in rec.probe_readings}
    # PASS only above the floor AND positive: within-noise and "removing it is slower" both falsify
    assert verdicts == {"no_bar": ("PASS", "+9.20%"), "serial_kv": ("FAIL", "+0.30%"),
                        "helps": ("FAIL", "-5.00%")}
    assert "SLOWER: the mechanism helps" in next(r["note"] for r in rec.probe_readings if r["contract"] == "helps")
    assert [t["name"] for t in rec.lever_prices if t["kind"] == "ablation"] == ["no_bar", "serial_kv", "helps"]


def test_json_probe_with_no_ablations_feeds_back_without_a_job(tmp_path):
    """A JSON probe that names no ablations (or does not parse) costs no GPU
    job: the feedback says so and points at the format; the budget round is
    still spent (it was a turn)."""
    def probe(prompt):
        _lever_kernel(tmp_path)
        (tmp_path / "notes").mkdir(exist_ok=True)
        (tmp_path / "notes" / "abl.json").write_text(json.dumps({"levers": []}))
        return "PROBE_REQUEST: notes/abl.json"
    diagnosis = "cause_class: G1\ndeficient_locus: D1.1\npatch_direction: x\n"
    be = FakeBackend([probe, diagnosis,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")], [])
    ctx = _ctx(tmp_path, be)
    run_meta_step(ctx)
    assert not [c for c in be.calls if c[0] == "gpu"]
    assert "No ablation reading: the spec names no `ablations`" in be.prompts[1]


# --- confirmation vs the KEEP reading (node hygiene) --------------------------

def test_best_confirmation_without_a_lease_keeps_a_deviating_sample(tmp_path):
    # the deviation is reported, and the reading stands
    be = _ConfirmBackend([], [_EVAL_CONFIRM(4.6, 2.0), _EVAL_CONFIRM(4.7, 2.0),
                              _EVAL_CONFIRM(4.65, 2.0)])
    out = run_best_confirmation(_confirm_ctx(tmp_path, be), jobs=3)
    assert len(out["jobs"]) == 3 and out["keep_disputed"] is True
    assert all(s["off_keep_pct"] > 100 for s in out["jobs"])


def test_lever_pricing_reruns_once_on_a_dirty_node_without_a_design_turn(tmp_path):
    """The pricing job's GPU held an orphan (PRICE-FAIL node-dirty): the SAME design
    is re-priced once — one more job, no second design turn, and the table comes from
    the rerun."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD
    dirty = ("PRICE kernel_file=kernel.py variants: base +ring +lane +nb_sym all\n"
             "PRICE ROUND 0 base pass=True device=3.4 vs_ref_device=None pass_tag=main\n"
             "NODE-DIRTY after `python3 eval/bench.py --mode all` timed out (2400s): "
             "pid(s) [77] still hold the GPU 30s after SIGKILL\n"
             "PRICE-FAIL reason=node-dirty (after `python3 eval/bench.py --mode all` timed out "
             "(2400s): pid(s) [77] still hold the GPU 30s after SIGKILL)\n")
    be = FakeBackend([_lever_design(tmp_path), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, dirty, _PRICE_LOG_OK])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert [c for c in be.calls if c == ("gpu", _LEVER_PRICING_CMD)] == [("gpu", _LEVER_PRICING_CMD)] * 2
    assert len([c for c in be.calls if c[0] == "agent"]) == 3     # design, diagnose, patch
    diag = be.prompts[1]
    assert "## Pricing table (measured by the driver on the best tree" in diag and "`ring`" in diag and "+8.56%" in diag


# --- placeholder ledger: a conservative form is a debt the gate row cannot cross ---

from forgekernel.loop.driver import _placeholder_ledger, _student_prompt  # noqa: E402
from forgekernel.inner.trajectory import IterationRecord  # noqa: E402

REC_C = lambda did, change, kf="flash_attn_dsl/flash_bwd.py": (  # noqa: E731
    f"decision_id: {did}\nchange: {change}\nfiles_touched: {kf}\n"
)


def test_gate_row_is_refused_while_a_placeholder_is_open(tmp_path):
    """A row that lands a stand-in declares it; the gate row is not committed or
    evaluated until the [RESTORED:] row lands; then the gate evaluates as usual."""
    be = FakeBackend([
        REC_C("D1.1", "[PLACEHOLDER: D1.1 → fp16 block-doubling solve] f32 scalar solve"),
        REC_C("M1-gate", "gate row, tree as D1.1 left it"),          # refused
        REC_C("D1.1", "[RESTORED: D1.1] fp16 block-doubling solve on the T slot"),
        REC_C("M1-gate", "gate row"),                                 # accepted
    ], [EVAL(2.5), NCU])
    ctx = _ctx(tmp_path, be)
    r0 = run_inner_iteration(ctx, 0, metric_before=None, green=False)
    assert r0.verdict == "NEUTRAL"
    assert _placeholder_ledger(load_records(ctx.status_path)) == {"D1.1": "fp16 block-doubling solve"}
    r1 = run_inner_iteration(ctx, 1, metric_before=None, green=False)
    # the refusal re-prompted within the iteration; the restore row landed instead
    assert r1.decision_id == "D1.1" and "[RESTORED: D1.1]" in r1.change and r1.verdict == "NEUTRAL"
    assert "placeholder ledger not cleared" in be.prompts[2].lower() and "M1-gate" in be.prompts[2]
    assert "[RESTORED: D1.1]" in be.prompts[2]
    assert _placeholder_ledger(load_records(ctx.status_path)) == {}
    r2 = run_inner_iteration(ctx, 2, metric_before=None, green=False)
    assert r2.decision_id == "M1-gate" and r2.verdict == "KEEP"
    # exactly one full eval was paid — the accepted gate row's — and the refused
    # gate row was never committed
    assert [c for c in be.calls if c == ("gpu", "all")] == [("gpu", "all")]
    landed = [c for c in be.calls if c[0] == "commit" and c[1].startswith("M1-gate")]
    assert len(landed) == 1


def test_untagged_placeholder_prose_is_bounced_for_the_tag(tmp_path):
    be = FakeBackend([
        REC_C("D1.1", "f32 scalar forward substitution as the sanctioned first-green form; "
                      "the fp16 block-doubling form replaces it once correctness is green"),
        REC_C("D1.1", "[PLACEHOLDER: D1.1 → fp16 block-doubling] f32 scalar forward substitution"),
    ], [])
    ctx = _ctx(tmp_path, be)
    r = run_inner_iteration(ctx, 0, metric_before=None, green=False)
    assert r.verdict == "NEUTRAL" and r.change.startswith("[PLACEHOLDER: D1.1")
    assert "[PLACEHOLDER: D<x>.<y>" in be.prompts[1]
    assert _placeholder_ledger(load_records(ctx.status_path)) == {"D1.1": "fp16 block-doubling"}
    assert not [c for c in be.calls if c[0] == "gpu"]


def test_gate_row_refused_every_round_is_recorded_as_such(tmp_path):
    be = FakeBackend([REC_C("D1.1", "[PLACEHOLDER: D1.1 → real form] stub")]
                     + [REC_C("M1-gate", "gate row")] * 12, [])
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 0, metric_before=None, green=False)
    r = run_inner_iteration(ctx, 1, metric_before=None, green=False)
    assert r.decision_id == "gate-refused" and r.verdict == "REVERT"
    assert "D1.1" in r.error and "RESTORED" in r.error
    assert not [c for c in be.calls if c[0] == "gpu"]
    # the debt is still open for the next iteration
    assert _placeholder_ledger(load_records(ctx.status_path)) == {"D1.1": "real form"}


def test_a_reverted_placeholder_row_leaves_nothing_open(tmp_path):
    be = FakeBackend([REC_C("D1.1", "[PLACEHOLDER: D1.1 → real form] stub")],
                     [EVAL(0.0, ok=False), NCU])
    ctx = _ctx(tmp_path, be)
    r = run_inner_iteration(ctx, 0, metric_before=None, green=True)
    assert r.verdict == "REVERT"
    assert _placeholder_ledger(load_records(ctx.status_path)) == {}


def test_open_placeholders_ride_every_student_turn(tmp_path):
    be = FakeBackend([REC_C("D1.1", "[PLACEHOLDER: D1.1 → real form] stub")], [])
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 0, metric_before=None, green=False)
    note = _student_prompt(ctx, 1)
    assert "Placeholder ledger not cleared" in note and "D1.1 (→ real form)" in note and "[RESTORED: D1.1]" in note


def test_placeholder_ledger_replays_tags_and_falls_back_to_the_row_members():
    rows = [
        IterationRecord(iter=0, decision_id="D1.1+D2.3", change="[PLACEHOLDER: → naive] both",
                        correctness=None, verdict="NEUTRAL"),
        IterationRecord(iter=1, decision_id="D2.3", change="[RESTORED: D2.3] real",
                        correctness=None, verdict="NEUTRAL"),
        IterationRecord(iter=2, decision_id="D1.1", change="[RESTORED: D1.1] real but broken",
                        correctness=False, verdict="REVERT"),
    ]
    assert _placeholder_ledger(rows) == {"D1.1": "naive"}
    assert _placeholder_ledger(rows[:1]) == {"D1.1": "naive", "D2.3": "naive"}
    # an ordinary change text opens nothing
    plain = [IterationRecord(iter=0, decision_id="D1.1", change="did D1.1",
                             correctness=True, verdict="KEEP")]
    assert _placeholder_ledger(plain) == {}


def test_placeholder_ledger_follows_the_lineage_of_the_given_head():
    """A workspace reset rewinds the lineage: a restore landed past the reset
    point no longer closes the debt, and a head no row owns reads every row."""
    rows = [
        IterationRecord(iter=0, decision_id="D1.1", change="[PLACEHOLDER: D1.1 → real] stub",
                        correctness=None, verdict="NEUTRAL", sha="sha1", base_sha="sha0"),
        IterationRecord(iter=1, decision_id="D2.3", change="did D2.3",
                        correctness=None, verdict="NEUTRAL", sha="sha2", base_sha="sha1"),
        IterationRecord(iter=2, decision_id="D1.1", change="[RESTORED: D1.1] real",
                        correctness=None, verdict="NEUTRAL", sha="sha3", base_sha="sha2"),
    ]
    assert _placeholder_ledger(rows, "sha3") == {}
    assert _placeholder_ledger(rows, "sha2") == {"D1.1": "real"}   # reset to before the restore
    assert _placeholder_ledger(rows, "sha0") == {}                 # unknown head: every row replays
    assert _placeholder_ledger(rows, "") == {}


# --------------------------------------------------------------------- review fixes
from forgekernel.loop.driver import (  # noqa: E402
    REF_STRUCT_PROMPT_CHARS, _lineage_rows, _sass_serial_status)
from forgekernel.inner.trajectory import append_record  # noqa: E402


def test_serialization_note_reads_the_previous_keep_not_the_row_just_written(tmp_path):
    """The KEEP's own record is on status.jsonl before the note is written; read
    back as 'the previous KEEP' it made prev == cur and every verdict a standing
    state — a change that introduced serialization never read as a regression."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    prev = {"wgmma_serial_ratio": 0.16, "wgmma_total": 56, "wgmma_depbar0": 9, "wgmma_tight": 9}
    cur = {"wgmma_serial_ratio": 1.0, "wgmma_total": 56, "wgmma_depbar0": 56, "wgmma_tight": 56}
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1.1", change="x", correctness=True, verdict="KEEP", profiler=prev))
    append_record(ctx.status_path, IterationRecord(
        iter=1, decision_id="D1.2", change="y", correctness=True, verdict="KEEP", profiler=cur))
    note = _sass_serial_status(ctx, cur, exclude_iter=1)
    assert "serialization regression" in note and "0.16" in note
    # the trap, kept as a regression guard: without the exclusion the row reads itself
    assert "(persistent)" in _sass_serial_status(ctx, cur)
    # a first KEEP has no predecessor: standing state, not a regression
    ctx.status_path.write_text("")
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1.1", change="x", correctness=True, verdict="KEEP", profiler=cur))
    assert "(persistent)" in _sass_serial_status(ctx, cur, exclude_iter=0)


def test_the_construction_table_travels_whole_into_the_distill_prompt():
    """A real ref_struct_table.txt is ~9.1k chars / 115 lines; a 4k cut fell inside
    the decisive region's first hand-off line."""
    real_table_chars = 9135
    assert REF_STRUCT_PROMPT_CHARS >= real_table_chars * 1.25


def test_a_broken_lineage_replays_every_row_instead_of_dropping_the_debt():
    """A row built on a sha no row owns that is NOT the run's root (a driver-side
    commit that was not reset) breaks the chain; the rows above the break may be
    ancestors, so they replay — the ledger errs towards refusing."""
    opened = IterationRecord(iter=0, decision_id="D1.1", change="[PLACEHOLDER: D1.1 → real] stub",
                             correctness=None, verdict="NEUTRAL", sha="sha1", base_sha="sha0")
    on_break = IterationRecord(iter=1, decision_id="D2.3", change="did D2.3",
                               correctness=None, verdict="NEUTRAL", sha="sha2", base_sha="shaX")
    assert [r.iter for r in _lineage_rows([opened, on_break], "sha2")] == [0, 1]
    assert _placeholder_ledger([opened, on_break], "sha2") == {"D1.1": "real"}
    # the run's root is a legitimate end of the chain, not a break
    assert _placeholder_ledger([opened], "sha1") == {"D1.1": "real"}
    # a reset back to the baseline really did drop the placeholder with the tree
    on_root = IterationRecord(iter=1, decision_id="D2.3", change="did D2.3",
                              correctness=None, verdict="NEUTRAL", sha="sha2", base_sha="sha0")
    assert _placeholder_ledger([opened, on_root], "sha2") == {}


def test_a_gate_row_that_declares_a_placeholder_itself_is_refused(tmp_path):
    """Observed shape: the stand-in lands WITH the gate row. The ledger saw no open
    debt (nothing had landed yet) and let it through; now the row's own tag
    refuses it, and untagged placeholder prose on a gate row is bounced for the
    tag like any other row."""
    be = FakeBackend([
        REC_C("M1-gate", "[PLACEHOLDER: D1.1 → fp16 block-doubling] f32 scalar solve, gate row"),
        REC_C("M1-gate", "f32 scalar solve as the sanctioned first-green form, swapped back after green"),
        REC_C("D1.1", "[PLACEHOLDER: D1.1 → fp16 block-doubling] f32 scalar solve"),
    ], [])
    ctx = _ctx(tmp_path, be)
    r = run_inner_iteration(ctx, 0, metric_before=None, green=False)
    assert r.decision_id == "D1.1" and r.verdict == "NEUTRAL"
    assert "Gate row `M1-gate` not accepted" in be.prompts[1] and "itself declares a placeholder" in be.prompts[1]
    assert "[PLACEHOLDER: D<x>.<y>" in be.prompts[2]
    assert not [c for c in be.calls if c[0] == "gpu"]
    assert _placeholder_ledger(load_records(ctx.status_path)) == {"D1.1": "fp16 block-doubling"}


def test_mentioning_the_ledger_or_denying_a_placeholder_is_not_a_declaration(tmp_path):
    """A gate row that says the ledger is clear, or a row that says it is NOT a
    stand-in, carries the word but declares nothing — no bounce."""
    be = FakeBackend([
        REC_C("D1.1", "the real fp16 solve; no placeholder anywhere in this row"),
        REC_C("M1-gate", "placeholder ledger cleared, submitting the gate row"),
    ], [EVAL(2.5), NCU])
    ctx = _ctx(tmp_path, be)
    r0 = run_inner_iteration(ctx, 0, metric_before=None, green=False)
    assert r0.decision_id == "D1.1" and r0.verdict == "NEUTRAL" and len(be.prompts) == 1
    r1 = run_inner_iteration(ctx, 1, metric_before=None, green=False)
    assert r1.decision_id == "M1-gate" and r1.verdict == "KEEP" and len(be.prompts) == 2
# --- teacher-role view ------------------------------------------------------------

def test_teacher_view_passes_the_role_only_to_backends_that_take_it():
    from forgekernel.loop.driver import _teacher, AGENT_ROLE_TEACHER

    class Old:                       # predates the split: prompt only
        def run_agent(self, prompt):
            return "old:" + prompt

    class New:
        def __init__(self):
            self.roles = []
            self.last_job_id = "j1"

        def run_agent(self, prompt, *, role="student"):
            self.roles.append(role)
            return "new:" + prompt

    assert _teacher(Old()).run_agent("p") == "old:p"
    new = New()
    view = _teacher(new)
    assert view.run_agent("p") == "new:p"
    assert new.roles == [AGENT_ROLE_TEACHER]
    assert view.last_job_id == "j1"          # everything else passes through
    assert _teacher(view) is view            # idempotent


def test_meta_step_and_student_turns_use_their_own_roles(monkeypatch):
    """The student iteration asks for the student role; the meta step (diagnose)
    asks for the teacher role — the routing exists so that two models can share
    one run."""
    from forgekernel.loop import driver as drv
    roles = []

    class Be:
        def run_agent(self, prompt, *, role="student"):
            roles.append(role)
            return ""

    be = Be()
    drv._teacher(be).run_agent("x")
    be.run_agent("y")
    assert roles == ["teacher", "student"]


def test_a_long_demangled_kernel_name_does_not_blind_the_banked_header():
    """The header's kernel name is ncu's fully demangled signature. A CUTLASS or
    Triton one runs to hundreds of characters, and a fixed character window pushed
    instrs=/verdict= out of range: both sides parsed as None, and the guard against
    overwriting a bigger kernel's dossier degraded, silently, to overwriting
    unconditionally — the very failure it was added for."""
    from forgekernel.loop.driver import _banked_dossier_header_text

    name = ("void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<"
            + "cute::tuple<cute::C<128>, cute::C<256>, cute::C<64> >, " * 6
            + "cutlass::gemm::KernelTmaWarpSpecializedCooperative> >(...)")
    assert len(name) > 400
    text = (f"# banked: block 2/2 kernel={name} instrs=9112 verdict=OK\n"
            "[parse] 9112 instructions\nbody\n")
    got = _banked_dossier_header_text(text)
    assert got is not None, "the header was unreadable, so the guard falls back to overwriting"
    assert got[1:] == (9112, "OK")
    assert got[0] == name


def test_a_banked_header_is_read_from_the_first_line_only():
    """A `# banked:` string quoted deeper in a dossier body is not the header."""
    from forgekernel.loop.driver import _banked_dossier_header_text

    body = ("[parse] 40 instructions\n"
            "# banked: block 9/9 kernel=impostor instrs=99999 verdict=OK\n")
    assert _banked_dossier_header_text(body) is None
    real = "# banked: block 1/1 kernel=real instrs=40 verdict=OK\n" + body
    assert _banked_dossier_header_text(real) == ("real", 40, "OK")


def test_probe_pass_feedback_carries_the_log_tail(tmp_path):
    # observed on an fp4 run: the relaunch sweep put its answer (which TMEM alloc form
    # survives a second launch) on the FACT channel and PASSed its contracts, as
    # gpu_probe_ladder tells a probe to. The PASS branch handed back one line and
    # the student re-derived that answer eight hours later.
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", REC("D3.1", 2.5)],
        ["[[F]] rA shipped-148: exit=timeout after 55s; last progress: launch 2/3 begin\n"
         "[[F]] rG tmem-percta-148: relaunch 3/3 survived\n"
         "PROBE PASS max_diff=n/a  # S1-smoke", EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 9, metric_before=2.66)
    feedback = be.prompts[1]
    assert "PROBE PASS" in feedback and "Implement the full decision" in feedback
    assert "rG tmem-percta-148: relaunch 3/3 survived" in feedback  # the FACTS reach the student
    assert "exit=timeout after 55s" in feedback


def test_probe_pass_from_a_job_killed_afterwards_is_not_delivered(tmp_path):
    # observed on an attn-bwd run: the probe printed PASS for its trace rung, then its
    # launch rung hung until the task timeout killed the job (rc=137). That log
    # certified the tree; the student delivered a hanging pipeline as verified.
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D3.1", 2.5)],
        ["[gpu_entry.sh] timeout=1800s\n"
         "PROBE PASS max_diff=n/a  # T-trace\n"
         "S-smoke: launching the three kernels ...\n"
         "gpu_entry.sh: line 61:  3072 Killed                  timeout --signal=KILL\n"
         "[direct] gpu_entry cmd:python3 notes/p.py exited rc=137\n",
         "[gpu_entry.sh] timeout=1800s\n"
         "PROBE PASS max_diff=n/a  # T-trace\n"
         "PROBE PASS max_diff=n/a  # S-smoke\n"
         "[gpu_entry.sh] task=cmd finished\n",
         EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 10, metric_before=2.66)
    first = be.prompts[1]
    assert "Implement the full decision" not in first
    assert "did NOT certify" in first and "exit 137" in first
    assert "S-smoke: launching" in first  # the tail shows where the job died
    assert "Implement the full decision" in be.prompts[2]  # the whole re-run certifies
    assert rec is not None and rec.decision_id == "D3.1"


def test_probe_pass_from_a_cut_archive_is_no_verdict_not_a_rejection(tmp_path):
    # cli._fetch_log hands back a settled-but-unclosed archive on purpose (a capped
    # archiver lags the job). That must not read as the driver rejecting the probe:
    # the student is told there is NO verdict and to re-run, not to fix anything.
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D3.1", 2.5)],
        ["[gpu_entry.sh] timeout=1800s\nPROBE PASS max_diff=n/a  # T-trace\n",
         "[gpu_entry.sh] timeout=1800s\nPROBE PASS max_diff=n/a  # T-trace\n\n"
         "[gpu_entry.sh] task=cmd finished\n",
         EVAL(2.5), NCU],
    )
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 11, metric_before=2.66)
    first = be.prompts[1]
    assert "produced NO verdict" in first and "archive is cut" in first
    assert "did NOT certify" not in first and "Fix this" not in first
    assert "Implement the full decision" in be.prompts[2]
    assert rec is not None and rec.decision_id == "D3.1"


# --- the loop banks what it writes into the reconnaissance tree ---------------
# The driver writes ref ledgers, dossiers, timelines and lever prices there and
# nothing committed them. Two consequences, both paid for in production: a GPU
# job ships a COMMIT, so the next job did not carry the evidence the teacher had
# just bought; and every run's preflight refuses to start on a dirty reference
# tree, so the loop wedged its own restart until a human committed.

def test_the_last_recon_round_banks_its_ledgers(tmp_path):
    """The last round has no next round to carry its evidence into a commit."""
    from forgekernel.loop.driver import run_recon_step

    def write_up(prompt):
        (tmp_path / "recon-report.md").write_text(
            "observations\nSASS-WAIVED: launch-config probe only\n"
            "SCHED-WAIVED: no SASS page\nLIVE-WAIVED: no SASS page")
        (tmp_path / "candidates.md").write_text("A|B")
        return "report written"

    teacher = FakeBackend(["RECON_REQUEST: notes/p1.py", write_up], [])
    recon = FakeBackend([], ["Block Size (384,1,1)  Grid Size (2,1,66)"])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    ctx.evidence_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p1.py").write_text("print('probe')")
    ctx.recon_path = tmp_path / "recon-report.md"
    run_recon_step(ctx)
    banked = [c for c in recon.calls
              if c[0] == "commit" and "bank" in str(c[1])]
    assert banked, "the round's evidence was left uncommitted"


def test_banking_skips_a_clean_tree_and_never_raises(tmp_path):
    from forgekernel.loop.driver import _bank_evidence

    class _Clean(FakeBackend):
        def dirty(self):
            return False

    class _Broken(FakeBackend):
        def dirty(self):
            return True

        def commit(self, message):
            raise RuntimeError("git is on fire")

    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path
    clean = _Clean([], [])
    ctx.recon_backend = clean
    assert _bank_evidence(ctx, "ledgers") == "nothing to bank"
    assert not [c for c in clean.calls if c[0] == "commit"]

    ctx.recon_backend = _Broken([], [])
    assert "git is on fire" in _bank_evidence(ctx, "ledgers")   # logged, not raised


def test_no_reconnaissance_tree_is_not_an_error(tmp_path):
    from forgekernel.loop.driver import _bank_evidence
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_backend = None
    assert _bank_evidence(ctx, "ledgers") == "no reconnaissance tree"


def test_stall_pricing_banks_the_student_dossier_it_writes(tmp_path):
    """Observed: student_role_exposure.txt / student_orch_dossier*.txt were the
    files that held the restart at the dirty-tree preflight — written by the
    student pricing job into the reconnaissance tree, committed by nobody."""
    from forgekernel.loop.driver import _run_stall_pricing
    be = FakeBackend([], [_STALL_LOG_OK + _role_block(_ROLE_CORE_STU)])
    ctx = _stall_ctx(tmp_path, be)
    recon = FakeBackend([], [])
    ctx.recon_backend = recon
    ctx.evidence_dir = tmp_path
    body, _, _, roles = _run_stall_pricing(ctx)
    assert body is not None and roles is not None
    assert (tmp_path / "student_role_exposure.txt").is_file()
    assert ("commit", "recon: bank ") in recon.calls, "student dossier left uncommitted"


_NOTAPPLICABLE_LOG = ("SASS-OK kernel=gdn_core instructions=5872 opcodes=140\n"
                      "[isa] MMA census: wgmma=0 tcgen05=8 mma_sync=0 (UTCHMMAx8)\n"
                      "SCHED-NOTAPPLICABLE isa=tcgen05 tcgen05_mma=8 wgmma=0 mma_sync=0 "
                      "reason=accumulator-in-tmem-no-depbar\n"
                      "LIVE-OK peak=145 guessed_frac=0.17\n")


def test_sched_notapplicable_in_a_job_log_opens_the_gate_like_a_waiver(tmp_path):
    """sched_live_tools models Hopper warpgroup MMA; on a tcgen05 (or mma.sync)
    kernel it refuses with its own verdict line rather than reporting zeros. That
    refusal is a delivered answer about the instrument, so the SCHED channel must
    not be held open against the teacher — a gdn_prefill B200 run spent its
    entire 12-round recon budget re-probing a channel no tool on the node could
    satisfy. Driven through run_recon_step's own log scan: the report carries NO
    SCHED waiver and the finish must still stand."""
    from forgekernel.loop.driver import run_recon_step

    def done(prompt):
        (tmp_path / "recon-report.md").write_text("observations [MEASURED, job 1]")
        (tmp_path / "candidates.md").write_text("A|B")
        return "done"

    teacher = FakeBackend(["RECON_REQUEST: notes/sched_probe.py", done], [])
    recon = FakeBackend([], [_NOTAPPLICABLE_LOG])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "sched_probe.py").write_text("import sched_live_tools")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    # probe turn + one write-up turn; the instrument's own verdict must not be
    # re-prompted as missing evidence (compare the bounce in the next test)
    assert len(teacher.prompts) == 2
    assert "SCHED evidence missing" not in teacher.prompts[1]


def test_sched_notapplicable_narrated_by_the_teacher_does_not_open_it(tmp_path):
    """Same trust model as the other verdict lines: only the tool's full-field
    line, off a job log, opens the gate — never prose about it, and never a
    truncated echo of it. Here the job printed only prose, so the finish bounces
    to the deliver-or-waive choice exactly as if nothing had been said."""
    from forgekernel.loop.driver import _SCHED_NOT_APPLICABLE, run_recon_step
    assert _SCHED_NOT_APPLICABLE.search(
        "the tool says SCHED-NOTAPPLICABLE because the ISA is tcgen05") is None
    assert _SCHED_NOT_APPLICABLE.search(
        "SCHED-NOTAPPLICABLE isa=tcgen05 tcgen05_mma=8 wgmma=0\n") is None

    def writeup(prompt):
        (tmp_path / "recon-report.md").write_text("observations [MEASURED, job 1]")
        (tmp_path / "candidates.md").write_text("A|B")
        return "done"

    def waive(prompt):
        assert "SCHED" in prompt and "SCHED-WAIVED" in prompt
        (tmp_path / "recon-report.md").write_text(
            "observations\nSCHED-WAIVED: tcgen05 mbarrier topology transcribed by hand")
        return "waived"

    teacher = FakeBackend(["RECON_REQUEST: notes/sched_probe.py", writeup, waive], [])
    recon = FakeBackend([], ["SASS-OK kernel=k instructions=5872 opcodes=140\n"
                             "note: the tool says SCHED-NOTAPPLICABLE isa=tcgen05 here\n"
                             "LIVE-OK peak=145 guessed_frac=0.17\n"])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "sched_probe.py").write_text("import sched_live_tools")
    ctx.recon_path = tmp_path / "recon-report.md"
    result = run_recon_step(ctx)
    assert result.finished is True
    assert len(teacher.prompts) == 3, "prose must have bounced the finish once"


def test_blackbox_recon_seeds_the_guard_beside_the_tools(tmp_path):
    """The node only sees `git archive` of the reference tree, so the shared
    tripwire has to ride in it the way sass_tools does — otherwise every run is
    back to hand-copying the guard's body into gpu_entry.sh. Blackbox runs
    only: a source-mode run's reference tree has nothing to guard."""
    from forgekernel.loop.driver import BLACKBOX_GUARD_NAME, run_recon_step
    from forgekernel.loop.driver import _blackbox_guard_source

    def done(prompt):
        (tmp_path / "recon-report.md").write_text("observations [MEASURED, job 1]\n"
                                                 "STRUCT-WAIVED: no named barriers")
        (tmp_path / "candidates.md").write_text("A|B")
        return "done"

    for mode, expected in (("blackbox", True), ("source", False)):
        root = tmp_path / mode
        root.mkdir()
        teacher = FakeBackend(["RECON_REQUEST: notes/p.py", done], [])
        recon = FakeBackend([], [_NOTAPPLICABLE_LOG])
        ctx = _ctx(root, teacher)
        ctx.oracle_mode = mode
        ctx.recon_backend = recon
        ctx.recon_dir = root
        (root / "notes").mkdir()
        (root / "notes" / "p.py").write_text("import sass_tools")
        ctx.recon_path = tmp_path / "recon-report.md"
        run_recon_step(ctx)
        seeded = root / "notes" / BLACKBOX_GUARD_NAME
        assert seeded.exists() is expected, mode
        if expected:
            assert seeded.read_text() == _blackbox_guard_source().read_text()
            assert "blackbox_guard()" in seeded.read_text()
            vendor = [c for c in recon.calls if c[0] == "commit" and c[1].startswith("recon: vendo")]
            assert vendor, "the seeded guard must be committed with the other tools"


# the tail of a GDN pricing job: the timing copy is whole, the role
# pass ran one --pm, gpu_entry's SIGKILL took the second at TASK_TIMEOUT=1800
_PRICE_LOG_TIMED_KILLED = (
    "[gpu_entry.sh] timeout=1800s\n"
    "PRICE kernel_file=kernel.py variants: base +ring +lane all all-ring all-lane\n"
    "$ python3 eval/bench.py --mode all\n"
    "PRICE ROUND 0 base pass=True device=0.43649 vs_ref_device=0.2917 pass_tag=main\n"
    + _PRICE_LOG_OK.split("PRICE-OK")[0].replace("spin_add=880319", "spin_add=None")
    + "PRICE-TIMED levers=3 rounds=3\n"
    "\n$ python3 notes/stall_price_run.py --pm\n"
    "PRICE ROLES base tensor=1.53 spin_ns=72983 producer_ns=603664 consumer_ns=3882\n"
    "\n$ python3 notes/stall_price_run.py --pm\n"
    "gpu_entry.sh: line 61: 10644 Killed                  timeout --signal=KILL \"$TASK_TIMEOUT\" \"$@\"\n"
    "[direct] gpu_entry cmd:python3 notes/lever_pricing_run.py notes/levers.json exited rc=137\n")


def test_price_log_verdict_keeps_the_timing_copy_of_a_job_killed_in_the_role_pass():
    """Observed on B200: the per-variant --pm role pass outlived
    TASK_TIMEOUT, `timeout --signal=KILL` took the job at rc=137, and a killed
    entry script prints no closing line — so PRICE-TIMED, gated on a whole
    archive, was never a verdict, the complete timing table was refused, bought
    again, killed again, and the audit diagnosed without a table. The kill
    evidence is the other way a log says how the job ended: the table stands.
    A log that merely stops after PRICE-TIMED (no closing line, no kill) is
    still the archiver mid-read."""
    from forgekernel.loop.driver import (_price_log_verdict, _price_role_pass_killed,
                                         parse_lever_prices)
    assert _price_log_verdict(_PRICE_LOG_TIMED_KILLED) == "PRICE-TIMED"
    assert _price_role_pass_killed(_PRICE_LOG_TIMED_KILLED) == "the job was killed (exit 137)"
    # the same log cut before the kill lines: an archive read between the copies
    cut = _PRICE_LOG_TIMED_KILLED.split("gpu_entry.sh: line 61")[0]
    assert _price_log_verdict(cut) is None and _price_role_pass_killed(cut) is None
    # the runner's closing line alone (no bash Killed line) is kill evidence too
    runner_only = cut + "[direct] gpu_entry cmd:python3 notes/lever_pricing_run.py x exited rc=137\n"
    assert _price_log_verdict(runner_only) == "PRICE-TIMED"
    # kill evidence BEFORE the timing copy is not a table (the copy never printed)
    assert _price_log_verdict("[gpu_entry.sh] timeout=1800s\nPRICE ROUND 0 base pass=True\n"
                              "gpu_entry.sh: line 61: 1 Killed x\n") is None
    # a killed ablation-only run has no PRICE-TIMED: still no verdict
    assert _price_role_pass_killed("[gpu_entry.sh] timeout=1800s\nABLATE ROUND 0 base\n"
                                   "[direct] gpu_entry cmd:x exited rc=137\n") is None
    # the timing copy parses with the role columns None
    levers, stack, fail = parse_lever_prices(_PRICE_LOG_TIMED_KILLED)
    assert fail is None and [l["name"] for l in levers] == ["ring", "lane", "nb_sym"]
    assert levers[0]["spin_add_ns"] is None and levers[0]["add_pct"] == 8.56
    assert stack["all_pct"] == 13.10 and "base_ms" not in stack


def test_lever_pricing_job_killed_after_the_timing_copy_is_priced_not_rebought(tmp_path, capsys):
    """The meta step banks the timing copy of a job killed in the role pass:
    one job (no resubmit of a design that would be killed the same way), the
    table reaches the diagnosis prompt and the evidence file, and loop.log
    says the role pass was cut by the job timeout."""
    from forgekernel.loop.driver import _LEVER_PRICING_CMD
    be = FakeBackend([_lever_design(tmp_path), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, _PRICE_LOG_TIMED_KILLED])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", _LEVER_PRICING_CMD)) == 1
    assert "## Pricing table (measured by the driver on the best tree" in be.prompts[1]
    assert "## The pricing job produced no table" not in be.prompts[1]
    assert (tmp_path / "evidence" / "lever_prices.jsonl").exists()
    out = capsys.readouterr().out
    assert any("the job was killed (exit 137) after the timing table (PRICE-TIMED)" in l
               and "pricing from the timing copy" in l for l in out.splitlines())
    assert not any("carries no PRICE" in l for l in out.splitlines())


# --- SCHED verdicts of the two Blackwell-era grammars; §1 cluster contradiction ----

_MMA_SYNC_LOG = ("SASS-OK kernel=dsa_kernel instructions=3160 opcodes=140\n"
                 "[isa] MMA census: wgmma=0 tcgen05=0 mma_sync=296 (HMMAx296)\n"
                 "SCHED-OK isa=mma_sync loop=backedge mma_runs=296 barriers=12 ldsm=87 "
                 "depbars=0 cp_async_groups=0 mbar_waits=8 ew_runs=27 "
                 "handoff_before_each_ew=['mbarrier.trywait [UR67+0x2d850]']\n"
                 "LIVE-OK peak=183 guessed_frac=0.00\n")
_TCGEN05_LOG = ("SASS-OK kernel=gdn_core instructions=5872 opcodes=140\n"
                "[isa] MMA census: wgmma=0 tcgen05=88 mma_sync=22 (UTCHMMAx88, HMMAx22)\n"
                "SCHED-OK isa=tcgen05 loop=backedge utcmma_runs=44 commits=15 mbar_waits=16 "
                "mbar_arrives=9 tmem_ld=6 tmem_st=8 mma_sync_runs=22 barriers=5 ew_runs=31 "
                "handoff_before_each_ew=['tcgen05.commit -> [UR41]']\n"
                "LIVE-OK peak=145 guessed_frac=0.02\n")


def _recon_run(tmp_path, teacher_outputs, logs):
    from forgekernel.loop.driver import run_recon_step
    teacher = FakeBackend(teacher_outputs, [])
    recon = FakeBackend([], logs)
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / "probe.py").write_text("import sched_live_tools")
    ctx.recon_path = tmp_path / "recon-report.md"
    return run_recon_step(ctx), teacher


def test_mma_sync_and_tcgen05_sched_verdicts_open_the_gate_as_delivered(tmp_path):
    """A dsa_sparse B200 teacher hand-wrote an HMMA transcription that
    printed `SCHED-OK ... mma_runs=296 barriers=22 ldsm=87`; the HGMMA-only regex
    refused it and the report had to double-write a DELIVERED-MISMATCH line to
    close a channel that WAS delivered. Both Blackwell-era forms are verdicts."""
    for log in (_MMA_SYNC_LOG, _TCGEN05_LOG):
        d = tmp_path / log[15:25].strip().replace("=", "_")
        d.mkdir()

        def done(prompt, d=d):
            (d / "recon-report.md").write_text("observations [MEASURED, job 1]")
            (d / "candidates.md").write_text("A|B")
            return "done"

        result, teacher = _recon_run(d, ["RECON_REQUEST: notes/probe.py", done], [log])
        assert result.finished is True
        assert len(teacher.prompts) == 2
        assert "SCHED evidence missing" not in teacher.prompts[1]
        assert "SCHED-DELIVERED-MISMATCH" not in teacher.prompts[1]


_CLUSTER_RAW_LOG = ("SASS-OK kernel=dsa_kernel instructions=3160 opcodes=140\n"
                    "SCHED-OK isa=mma_sync loop=backedge mma_runs=296 barriers=12 ldsm=87\n"
                    "LIVE-OK peak=183 guessed_frac=0.00\n"
                    "  launch__block_dim_x                    256\n"
                    "  launch__cluster_dim_x                  16\n"
                    "  launch__cluster_dim_y                  1\n"
                    "  launch__cluster_dim_z                  1\n"
                    "  launch__cluster_max_active             7\n"
                    "  launch__grid_dim_x                     128\n")

_NO_CLUSTER_REPORT = ("## 1. Launch Configuration\n"
                      "- **Grid**: (128, 1, 1) = 128 CTAs\n"
                      "- **Cluster**: Grid shape is 1D, no evidence of 2D cluster structure\n"
                      "observations [MEASURED, job 1]\n")
_NO_CLUSTER_CANDIDATES = ("**Cluster**:\n"
                          "- `[VERIFY: ncu-direct]` **cluster_shape** = (1, 1, 1) (no cluster)\n"
                          "| **Cluster** | none | none | ncu-direct |\n")


def test_write_up_denying_a_cluster_the_ncu_metric_read_is_bounced(tmp_path):
    """Observed on dsa_sparse: the same p3 log that carried launch__cluster_dim_x=16
    fed a report whose §1 said "no cluster" and a candidates table with
    cluster=(1,1,1); the first milestone copied it and the student's first kernel
    was 16x slower. The driver now reads the cluster off the job log and holds
    the write-up against it — and lets the teacher fix it while rounds remain."""
    seen = []

    def wrong(prompt):
        (tmp_path / "recon-report.md").write_text(_NO_CLUSTER_REPORT)
        (tmp_path / "candidates.md").write_text(_NO_CLUSTER_CANDIDATES)
        return "written, nothing further"

    def fixed(prompt):
        seen.append(prompt)
        assert "RECON-CONTRADICTION cluster" in prompt
        assert "cluster=(16,1,1)" in prompt
        assert "candidates.md:2" in prompt and "candidates.md:3" in prompt
        (tmp_path / "recon-report.md").write_text(
            "## 1. Launch Configuration\n- Grid (128,1,1); LAUNCH cluster=(16,1,1) "
            "cluster_max_active=7 [MEASURED, job 1]\nobservations\n")
        (tmp_path / "candidates.md").write_text(
            "cluster=(16,1,1), 128 CTAs = 8 clusters of 16 | A | B\n")
        return "rewritten, nothing further"

    result, teacher = _recon_run(tmp_path, ["RECON_REQUEST: notes/probe.py", wrong, fixed],
                                 [_CLUSTER_RAW_LOG])
    assert result.finished is True
    assert len(seen) == 1, "exactly one contradiction bounce"
    assert len(teacher.prompts) == 3


def test_launch_ok_line_is_the_preferred_cluster_source(tmp_path):
    from forgekernel.loop.driver import _launch_cluster_from_log
    assert _launch_cluster_from_log(
        "LAUNCH-OK kernel=k grid=(128,1,1) block=(256,1,1) ctas=128 cluster=(16,1,1) "
        "cluster_max_active=7 regs=244 smem_dyn=227572\n"
        "  launch__cluster_dim_x   2\n") == (16, 1, 1)
    assert _launch_cluster_from_log(_CLUSTER_RAW_LOG) == (16, 1, 1)
    assert _launch_cluster_from_log("  launch__cluster_dim_x   4\n") == (4, 1, 1)
    assert _launch_cluster_from_log('"launch__cluster_dim_x","cluster","16"\n') == (16, 1, 1)
    assert _launch_cluster_from_log("nothing about launches\n") is None
    # the tool's PARTIAL line (no cluster metric collected) is not a reading
    assert _launch_cluster_from_log(
        "LAUNCH-PARTIAL reason=no-cluster-metrics kernel=k grid=(128,1,1) block=(256,1,1) "
        "ctas=128 cluster=n/a cluster_max_active=n/a regs=n/a smem_dyn=n/a\n") is None


def test_a_multi_kernel_log_is_read_from_the_last_kernel(tmp_path):
    """ncu without `-k` profiles the warmup / RNG / torch elementwise kernels too.
    Reading the FIRST metric in the text labelled a helper's launch shape (grid 64,
    cluster 0) as the reference's — the "no cluster" write-up then carried a
    tool-certified LAUNCH-OK line and the contradiction gate never fired."""
    from forgekernel.loop.driver import _launch_cluster_from_log
    from forgekernel.skills import launch_tools as LT
    log = ("kernel: helper_kernel\n"
           "  launch__grid_dim_x    64\n"
           "  launch__block_dim_x   128\n"
           "  launch__cluster_dim_x 0\n"
           "kernel: dsa_kernel\n"
           "  launch__grid_dim_x    128\n"
           "  launch__block_dim_x   256\n"
           "  launch__cluster_dim_x 16\n")
    assert _launch_cluster_from_log(log) == (16, 1, 1)      # the last, not the helper
    f = LT.parse_launch(log, kernel="dsa_kernel")
    assert (f.grid, f.cluster, f.kernel) == ((128, 1, 1), (16, 1, 1), "dsa_kernel")
    # naming a kernel the log does not carry must not fall back onto another's numbers
    empty = LT.parse_launch(log, kernel="not_here")
    assert empty.grid is None and empty.cluster is None
    assert LT.launch_line(empty).startswith("LAUNCH-PARTIAL reason=no-launch-metrics")
    assert LT.launch_line(f).startswith("LAUNCH-OK kernel=dsa_kernel grid=(128,1,1)")


def test_a_cluster_of_one_or_a_disputed_reading_does_not_bounce(tmp_path):
    from forgekernel.loop.driver import (_launch_contradictions,
                                         _launch_contradictions_open)
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_dir = tmp_path
    ctx.recon_path = tmp_path / "recon-report.md"
    ctx.recon_path.write_text(_NO_CLUSTER_REPORT)
    (tmp_path / "candidates.md").write_text(_NO_CLUSTER_CANDIDATES)
    # a measured (1,1,1) agrees with "no cluster"
    assert _launch_contradictions(ctx, {"cluster": (1, 1, 1)}) == []
    assert _launch_contradictions(ctx, {}) == []
    found = _launch_contradictions(ctx, {"cluster": (16, 1, 1)})
    # the candidates table declares one CTA and "no cluster" — both lines; the
    # report's §1 says "no evidence of 2D cluster structure", a statement about the
    # grid's shape rather than a declaration of the launch's cluster, so it stands
    # (the gate reads declarations: see test_the_contradiction_gate_reads_declarations)
    assert [f.split(":")[0] + ":" + f.split(":")[1] for f in found] == [
        "candidates.md:2", "candidates.md:3"]
    # the teacher may distrust the metric on the record — the line closes it
    ctx.recon_path.write_text(_NO_CLUSTER_REPORT
                              + "LAUNCH-DISPUTED: the profiled launch was the RNG kernel "
                                "(p3.run3.log:12), not the reference\n")
    assert _launch_contradictions_open(ctx, {"cluster": (16, 1, 1)}) == []


def test_the_contradiction_gate_reads_declarations_not_words(tmp_path):
    """A sentence containing the word "cluster" is not a denial of the measured
    launch. The old line-wide pattern bounced all of these, burning a write-up
    turn each — including the correction the bounce message itself asks for, the
    brief's own ≥2-candidate alternative, and the tool's LAUNCH-PARTIAL warning
    printed into §1 as the brief instructs."""
    from forgekernel.loop.driver import _launch_contradictions
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_dir = tmp_path
    ctx.recon_path = tmp_path / "recon-report.md"
    ok = ["the mainloop has no cluster barrier, only mbarrier",
          "no cluster barrier is used",
          "## Candidate B — no-cluster variant (each CTA its own tile)",
          "§1 correction: the earlier 'no cluster' reading was wrong; launch__cluster_dim_x=16",
          "[launch] no launch__cluster_dim_* metric on the page — absence is not evidence "
          "of 'no cluster'",
          "- **Cluster**: (16,1,1) = 8 groups × 16 CTAs",
          "launch__cluster_dim_x=16, cluster=(16,1,1)"]
    # each of these denies the measured cluster and must still be caught
    bad = ["- **Cluster**: (1, 1, 1) (no cluster)",
           "| **Cluster** | none | none | ncu-direct |",
           "- cluster shape: none",
           "cluster shape: 1",
           "cluster=1 (one tile per CTA)",
           "no cluster and no TMA multicast"]
    for i, line in enumerate(ok + bad):
        ctx.recon_path.write_text(line + "\n")
        found = _launch_contradictions(ctx, {"cluster": (16, 1, 1)})
        if i < len(ok):
            assert found == [], f"false bounce on: {line}"
        else:
            assert found, f"missed a denial: {line}"


def test_contradiction_holds_the_exhausted_write_up_and_is_not_persisted(tmp_path):
    """On the budget-exhausted path the write-up turns must carry the
    contradiction, and the completion marker must not be saved while the
    report still denies the measured cluster."""
    from forgekernel.loop.driver import _RECON_STATE_NAME
    prompts = []

    def stubborn(prompt):
        prompts.append(prompt)
        (tmp_path / "recon-report.md").write_text(_NO_CLUSTER_REPORT)
        (tmp_path / "candidates.md").write_text(_NO_CLUSTER_CANDIDATES)
        return "written"

    teacher = FakeBackend(["RECON_REQUEST: notes/probe.py"] + [stubborn] * 6, [])
    recon = FakeBackend([], [_CLUSTER_RAW_LOG])
    ctx = _ctx(tmp_path, teacher)
    ctx.recon_backend = recon
    ctx.recon_dir = tmp_path
    ctx.max_recon_rounds = 1
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "probe.py").write_text("import sched_live_tools")
    ctx.recon_path = tmp_path / "recon-report.md"
    from forgekernel.loop.driver import run_recon_step
    result = run_recon_step(ctx)
    assert result.finished is False
    assert sum("RECON-CONTRADICTION cluster" in p for p in prompts) == 3
    assert not (tmp_path / _RECON_STATE_NAME).exists() and \
        not list(tmp_path.glob(f"**/{_RECON_STATE_NAME}"))
# --- pre-green over an unknown lineage: declared construction, and resume ---------

_PART = lambda did, k, n: (  # noqa: E731  # the prompt's staged-build package declaration opens the change
    f"decision_id: {did}\nchange: part {k} of {n} of build step — did {did}\nhypothesis: h\n"
    f"binding_constraint: bc\nfiles_touched: flash_attn_dsl/flash_bwd.py\n"
)


def test_a_declared_construction_row_is_scaffolded_while_the_lineage_is_unknown(tmp_path, capsys):
    """green None (fresh workspace): a delivery whose change opens with `part k
    of n` (k < n) or `[PLACEHOLDER: …]` has declared the tree mid-build, so it is
    committed NEUTRAL without a GPU eval; the finishing part (k == n) and the gate
    row evaluate as before. A GDN run paid two full evals of such rows
    ("non-finite output" REVERTs) — one of them right after a budget resume."""
    agent = [_PART("D0.12", 1, 2),
             "decision_id: D0.9\nchange: [PLACEHOLDER: D0.9 → TMEM ring] scalar stand-in\n"
             "hypothesis: h\nbinding_constraint: bc\nfiles_touched: flash_attn_dsl/flash_bwd.py\n",
             _PART("D0.12", 2, 2)]
    gpu = [EVAL(0.0, ok=False), NCU]   # part 2 of 2: the package is finished -> evaluated
    be = FakeBackend(agent, gpu)
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 3
    assert run_inner_loop(ctx, start_iter=0, initial_green=None) == "max_iters"
    recs = load_records(ctx.status_path)
    assert [r.verdict for r in recs] == ["NEUTRAL", "NEUTRAL", "REVERT"]
    assert recs[0].correctness is None and recs[0].sha
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 1
    # (the gate row's evaluate-always rule over a not-green lineage is covered by
    # test_scaffold_mode_skips_eval_until_the_gate_row_goes_green)
    out = capsys.readouterr().out
    assert "D0.12 SCAFFOLD (declared construction: part 1 of 2; lineage state unknown" in out
    assert "D0.9 SCAFFOLD (declared construction: [PLACEHOLDER: D0.9 → TMEM ring]" in out


def test_a_declaration_deeper_in_the_change_or_on_a_gate_row_does_not_scaffold(tmp_path):
    from forgekernel.loop.driver import _declared_construction
    assert _declared_construction("part 1 of 2 of build step (6) — D0.12: x") == "part 1 of 2"
    assert _declared_construction("part 2 of 2 — done") is None            # finished package
    assert _declared_construction("mirrors the ref. part 1 of 2 split") is None  # not the opening
    assert _declared_construction("[PLACEHOLDER: D2.1 → TMA+mbarrier 3-stage] scalar gather") \
        == "[PLACEHOLDER: D2.1 → TMA+mbarrier 3-stage]"
    assert _declared_construction("D0.1 skeleton; kernel does not write output yet") is None
    # a gate row evaluates whatever it declares (green unknown)
    agent = ["decision_id: M0-gate\nchange: part 1 of 2 gate attempt\nhypothesis: h\n"
             "binding_constraint: bc\nfiles_touched: flash_attn_dsl/flash_bwd.py\n"]
    be = FakeBackend(agent, [EVAL(0.0, ok=False), NCU])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 1
    run_inner_loop(ctx, start_iter=0, initial_green=None)
    assert [r.verdict for r in load_records(ctx.status_path)] == ["REVERT"]
    # once green, a declared package step is a measured delivery (unchanged)
    be = FakeBackend([_PART("D1.1", 1, 2)], [EVAL(4.5), NCU])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 1
    run_inner_loop(ctx, start_iter=0, initial_green=True)
    assert [r.verdict for r in load_records(ctx.status_path)] == ["KEEP"]


def test_green_on_resume_reads_the_trajectory_when_no_record_owns_head():
    from forgekernel.inner.trajectory import IterationRecord as IR
    from forgekernel.loop.driver import _green_on_resume, _resume_anchor
    revert0 = IR(iter=0, decision_id="D0.1", change="c", correctness=False, verdict="REVERT",
                 error="non-finite", sha="")            # an evaluated REVERT owns no sha
    scaffold = [IR(iter=i, decision_id=f"D0.{i}", change="c", correctness=None,
                   verdict="NEUTRAL", sha=f"s{i}") for i in range(1, 6)]
    keep = IR(iter=6, decision_id="M0-gate", change="c", correctness=True, verdict="KEEP",
              bwd_median_ms=4.8, sha="s6")
    # the checkpoint's own record decides first
    assert _green_on_resume([revert0] + scaffold, "s5") is False
    assert _green_on_resume([revert0] + scaffold + [keep], "s6") is True
    # HEAD unowned (a driver seed / kept probe on top): a lineage never green is
    # under construction — the observed GDN case
    assert _green_on_resume([revert0] + scaffold, "seed9") is False
    assert _green_on_resume(scaffold, "seed9") is False
    # a FAIL after a KEEP is a post-green break: reset target unknown here -> None
    post = IR(iter=7, decision_id="D1.1", change="c", correctness=False, verdict="REVERT", sha="")
    assert _green_on_resume([revert0] + scaffold + [keep, post], "seed9") is None
    assert _green_on_resume([revert0] + scaffold + [keep], "seed9") is True
    assert _green_on_resume([], "sha0") is None
    # ...and a green resume whose HEAD no record owns anchors on the last KEEP, never
    # on that HEAD: the delivery HEAD carries was never evaluated, so resetting onto
    # it on a later correctness break would restore the very tree that broke
    assert _resume_anchor([revert0] + scaffold + [keep], "s6") == "s6"
    assert _resume_anchor([revert0] + scaffold + [keep], "seed9") == "s6"
    assert _resume_anchor([revert0] + scaffold + [keep], "s3") is None   # scaffold HEAD
    assert _resume_anchor([revert0] + scaffold, "seed9") is None         # never green


def test_a_budget_resume_over_an_evaluated_delivery_anchors_on_the_last_keep(tmp_path, capsys):
    """A process killed mid-eval leaves HEAD carrying a delivery no record owns. The
    resume reads green (the KEEP below it), so the segment enters on HEAD — but
    run_inner_loop's anchor is that KEEP: a delivery that then fails correctness
    resets to the verified tree, not to the unchecked HEAD it built on (resetting
    there restores the break and resets there again on every later one)."""
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    from forgekernel.loop.driver import _green_on_resume, _resume_anchor, run_inner_loop
    be = FakeBackend([REC("D0.9", 2.4)], [EVAL(2.4, ok=False), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.8)
    append_record(ctx.status_path, IR(
        iter=0, decision_id="M0-gate", change="c", correctness=True, verdict="KEEP",
        bwd_median_ms=2.0, sha="keep0"))
    be._sha = 8                                  # HEAD sha8: unowned (the killed eval)
    assert (_green_on_resume(load_records(ctx.status_path), be.head()),
            _resume_anchor(load_records(ctx.status_path), be.head())) == (True, "keep0")
    ctx.max_iters = 1
    run_inner_loop(ctx, start_iter=1, initial_green=True, initial_good_ref="keep0")
    assert ("reset", "keep0") in be.calls
    assert ("reset", "sha8") not in be.calls     # the unverified HEAD is never the anchor

def test_a_budget_resume_over_a_scaffold_ladder_reenters_scaffold_mode(tmp_path, capsys):
    """The process restarted (budget / certificate) over an iteration-0 REVERT + scaffold
    rows: the first post-resume construction row is committed NEUTRAL, not
    evaluated; the gate row evaluates and reaches target."""
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    be = FakeBackend([REC("D0.5", None), REC("M0-gate", 1.8)], [EVAL(1.8), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.9)     # (_ctx starts an empty status file)
    ctx.m0_audit = False
    append_record(ctx.status_path, IR(
        iter=0, decision_id="D0.1+D0.2", change="skeleton", correctness=False,
        verdict="REVERT", error="correctness FAIL: non-finite values in output", sha=""))
    for i in (1, 2):
        append_record(ctx.status_path, IR(
            iter=i, decision_id=f"D0.{i + 2}", change="c", correctness=None,
            verdict="NEUTRAL", sha="sha0" if i == 2 else "shaX"))   # HEAD = sha0 (FakeBackend)
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["stop"] == "target"
    assert [r.verdict for r in load_records(ctx.status_path)] == \
        ["REVERT", "NEUTRAL", "NEUTRAL", "NEUTRAL", "KEEP"]
    assert len([c for c in be.calls if c == ("gpu", "all")]) == 1
    out = capsys.readouterr().out
    assert "resume: lineage at sha0 reads under construction from the 3 recorded iteration(s) " \
           "-> scaffold mode until the gate row" in out
    assert "D0.5 SCAFFOLD (pre-green" in out


def test_a_green_resume_over_an_unowned_head_resets_to_the_last_keep(tmp_path, capsys):
    """The same resume through the production entry point: run_two_layer names the
    divergence in the log and hands the KEEP to the segment, so the first failing
    delivery after the restart rolls back to the verified tree."""
    from forgekernel.inner.trajectory import IterationRecord as IR, append_record
    be = FakeBackend([REC("D0.9", 2.4), REC("D0.9", 2.4)],
                     [EVAL(2.4, ok=False), NCU, EVAL(2.4, ok=False), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.8)
    ctx.max_iters = 2
    append_record(ctx.status_path, IR(
        iter=0, decision_id="M0-gate", change="c", correctness=True, verdict="KEEP",
        bwd_median_ms=2.0, sha="keep0"))
    be._sha = 7                                  # HEAD sha7: unowned (the killed eval)
    run_two_layer(ctx, max_meta=0)
    out = capsys.readouterr().out
    assert "resume: lineage at sha7 reads green from the 1 recorded iteration(s); " \
           "rollback anchor = last KEEP keep0 (HEAD carries commits no record verified)" in out
    assert ("reset", "keep0") in be.calls
    assert ("reset", "sha7") not in be.calls


# --- paired residual to the stop line; INFEASIBLE is a real stop ----------------

def test_paired_reanchor_residual_is_measured_to_the_stop_line_not_parity(tmp_path):
    """A 1.05x run at a same-job ratio of 0.985 stands +6.6% short of the line
    it stops on, not +1.5%: the note says so and drops the "inside the floor,
    prefer parity" advice that parity-bar runs get."""
    import json as _json
    from forgekernel.loop.driver import _phantom_gap_reanchor
    from forgekernel.inner.trajectory import IterationRecord as IR
    ctx = _reanchor_ctx(tmp_path, samples=[2.0, 2.0, 2.0])
    ctx.ref_in_job = True
    ctx.equivalence_ratio = 1.05
    ctx.backend = _ConfirmBackend([], [])
    append_record(ctx.status_path, IR(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=2.07, sha="shabest"))
    ctx.paired_memo["shabest"] = {"iter": 0, "cluster": "", "median_ratio": 0.985,
                                  "jobs": ["j1", "j2"], "verdict": "unconfirmed"}
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    note = ctx.reanchor_note
    assert "+6.6%" in note and "+1.5%" not in note                # 1.05/0.985 - 1
    assert "stop line is 1.05×" in note and "prefer calling it parity" not in note
    rec = _json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[-1])
    assert rec["paired_residual"] == pytest.approx(0.066, abs=1e-4)
    assert rec["parity_residual"] == pytest.approx(0.0152, abs=1e-4)
    assert ctx.paired_residual_memo[1] == pytest.approx(1.05 / 0.985 - 1)


def test_parse_cause_class_takes_infeasible_on_the_plateau_path_only():
    from forgekernel.loop.driver import _parse_cause_class
    assert _parse_cause_class("cause_class: INFEASIBLE\n") == "INFEASIBLE"
    # the M0 audit has no stop request: distance to target is expected there
    assert _parse_cause_class("cause_class: INFEASIBLE\n", allow_ok=True) == ""


_INFEASIBLE = ("reachability:      floor_opt=2.40 ms vs target=1.9 ms -> BLOCKED\n"
               "cause_class: INFEASIBLE\ndeficient_locus: none\n"
               "evidence: ref itself reads 0.93 ms; every in-tree lever priced null\n"
               "patch_direction: stop — re-distil a new candidate reading or re-set the target\n")
# the same request on a failed disproof: the floor sits inside the target
_INFEASIBLE_OPEN = _INFEASIBLE.replace("floor_opt=2.40 ms vs target=1.9 ms -> BLOCKED",
                                       "floor_opt=1.20 ms vs target=1.9 ms -> OPEN")


def test_infeasible_diagnosis_stops_without_a_patch_and_writes_the_notice(tmp_path):
    from forgekernel.meta_trajectory import load_meta_records
    be = FakeBackend([_INFEASIBLE], [])
    ctx = _ctx(tmp_path, be)
    before = ctx.milestone.read_text(encoding="utf-8")
    res = run_meta_step(ctx, best_before=0.978)
    assert res.patch_class == "infeasible-stop" and res.changed_entries == []
    assert len(be.prompts) == 1                                   # no patch turn
    assert ctx.milestone.read_text(encoding="utf-8") == before
    rec = load_meta_records(ctx.meta_status_path)[-1]
    assert (rec.outcome, rec.cause_class, rec.patch_class) == \
        ("INFEASIBLE_STOP", "INFEASIBLE", "infeasible-stop")
    notice = (tmp_path / "infeasible_stop.md").read_text(encoding="utf-8")
    assert "the teacher declared INFEASIBLE" in notice
    assert "floor_opt=2.40 ms vs target=1.9 ms -> BLOCKED" in notice   # the reachability line
    assert "every in-tree lever priced null" in notice                  # the diagnosis verbatim
    assert "0.978" in notice and "stop=infeasible" in notice
    # the stop is the run's verdict: the notice names the legitimate routes and
    # never invites a human to overrule it with a new direction on this run
    assert "the end of this run" in notice and "human-guided" in notice and "test it on a new run" in notice
    for invite in ("await a human ruling", "write the new direction into the milestone",
                   "then restart", "relaunch as usual"):
        assert invite not in notice.lower(), invite


# --- INFEASIBLE gate: a stop request needs a BLOCKED reachability disproof -----

def _replay_cases():
    import json as _json
    return _json.loads((Path(__file__).parent / "fixtures" / "infeasible_replay.json")
                       .read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _replay_cases(), ids=lambda c: c["id"])
def test_infeasible_gate_replays_real_stop_requests(case):
    """Nine real stop requests: the five BLOCKED ones (floor above target x 1.20)
    pass, the four on an OPEN line are refused."""
    from forgekernel.loop.driver import _infeasible_refusal, _plateau_diagnoses
    from forgekernel.meta_trajectory import MetaRecord
    prior = [MetaRecord(meta_step=i, cause_class="x", deficient_locus="", patch_class="",
                        changed_entries=[], patch_direction="", outcome=o)
             for i, o in enumerate(case["prior"])]
    text = f"reachability:      {case['reachability']}\ncause_class: INFEASIBLE\n"
    why = _infeasible_refusal(text, diagnoses=_plateau_diagnoses(prior) + 1,
                              min_diagnoses=1, target_ms=case["target_ms"])
    if case["expect"] == "accept":
        assert why == ""
    else:
        assert "OPEN" in why


def test_infeasible_gate_min_diagnoses_counts_plateau_diagnoses_only():
    """The distillation and a clean M0 audit are not diagnoses that could have
    asked to stop; a step-count floor of 2 holds a first plateau request back."""
    from forgekernel.loop.driver import _infeasible_refusal, _plateau_diagnoses
    from forgekernel.meta_trajectory import MetaRecord
    mk = lambda o: MetaRecord(meta_step=0, cause_class="x", deficient_locus="", patch_class="",
                              changed_entries=[], patch_direction="", outcome=o)
    assert _plateau_diagnoses([mk("AUTHORED"), mk("AUDIT_CLEAN")]) == 0
    assert _plateau_diagnoses([mk("AUTHORED"), mk("UNPRICED_REOPEN"), mk("KEEP_PATCH")]) == 2
    line = "reachability: floor_opt=2.40 ms vs target=1.9 ms -> BLOCKED\n"
    assert "at least 2 diagnosis steps" in _infeasible_refusal(line, diagnoses=1, min_diagnoses=2,
                                                 target_ms=1.9)
    assert _infeasible_refusal(line, diagnoses=2, min_diagnoses=2, target_ms=1.9) == ""


def test_infeasible_gate_checks_the_numbers_not_just_the_word():
    from forgekernel.loop.driver import _infeasible_refusal
    chk = lambda line, t=1.9: _infeasible_refusal(f"reachability: {line}\n", diagnoses=1,
                                                  min_diagnoses=1, target_ms=t)
    # BLOCKED claimed, but the floor does not clear the 20% margin
    assert "does not exceed" in chk("floor_opt=2.0 ms vs target=1.9 ms -> BLOCKED")
    # a target written below the driver's stop line does not inflate the claim
    assert "driver's stop threshold" in chk("floor_opt=1.5 ms vs target=1.0 ms -> BLOCKED", t=1.9)
    # no numbers to check
    assert "machine-readable" in chk("floor_opt=see evidence vs target=1.9 ms -> BLOCKED")
    assert "missing the `reachability:` line" in _infeasible_refusal("cause_class: INFEASIBLE\n", diagnoses=1,
                                                         min_diagnoses=1, target_ms=1.9)
    assert "unparseable" in chk("floor_opt=2.4 ms vs target=1.9 ms (BLOCKED or OPEN?)")
    # µs and a restated verdict are read; the last arrowed verdict wins
    assert chk("floor_opt=2400 µs vs target=1.9 ms -> OPEN -> BLOCKED") == ""
    # unknown driver line: the stated target alone is the bar
    assert chk("floor_opt=2.40 ms vs target=1.9 ms -> BLOCKED", t=None) == ""


def test_infeasible_on_open_is_bounced_and_the_class_that_answers_it_patches(tmp_path):
    """An OPEN stop request is put back once with the conditions; the teacher's
    re-diagnosis patches as usual and the record keeps the refusal."""
    from forgekernel.meta_trajectory import load_meta_records
    r2 = ("reachability: floor_opt=1.20 ms vs target=1.9 ms -> OPEN\n"
          "cause_class: R2\ndeficient_locus: D6\nevidence: ledger\npatch_direction: add decision\n")
    be = FakeBackend([_INFEASIBLE_OPEN, r2,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")], [])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx, best_before=0.978)
    assert res.patch_class == "additive"
    assert "## INFEASIBLE stop request not accepted" in be.prompts[1]
    assert "## INFEASIBLE has only one channel" not in "".join(be.prompts)   # no second, contradictory bounce
    assert not (tmp_path / "infeasible_stop.md").exists()
    rec = load_meta_records(ctx.meta_status_path)[-1]
    assert len(rec.infeasible_refusals) == 1 and "OPEN" in rec.infeasible_refusals[0]


def test_infeasible_on_open_twice_is_refused_without_a_stop(tmp_path):
    from forgekernel.meta_trajectory import load_meta_records, reflect_summary
    be = FakeBackend([_INFEASIBLE_OPEN, _INFEASIBLE_OPEN], [])
    ctx = _ctx(tmp_path, be)
    before = ctx.milestone.read_text(encoding="utf-8")
    res = run_meta_step(ctx, best_before=0.978)
    assert res.patch_class == "infeasible-refused" and res.changed_entries == []
    assert len(be.prompts) == 2                                   # one bounce, no patch turn
    assert ctx.milestone.read_text(encoding="utf-8") == before
    assert not (tmp_path / "infeasible_stop.md").exists()
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[-1].outcome == "INFEASIBLE_REFUSED"
    assert "stop request refused" in reflect_summary(recs)


def test_two_layer_run_goes_on_after_a_refused_stop_request(tmp_path):
    """End to end: an OPEN stop request (twice) is refused and the meta step is
    retried on the same state; a BLOCKED one then stops the run."""
    ms = tmp_path / "m.md"
    ms.write_text(_MILESTONE, encoding="utf-8")
    be = _E2EFake(agent_outputs=["EXHAUSTED: nothing left", _INFEASIBLE_OPEN, _INFEASIBLE_OPEN,
                                 _INFEASIBLE, "EXHAUSTED: must not run"],
                  gpu_logs=[], milestone_path=ms, patch_writes=[])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.milestone = ms
    summary = run_two_layer(ctx, max_meta=3)
    assert summary["stop"] == "infeasible" and summary["meta_steps"] == 2
    assert be.agent_outputs == ["EXHAUSTED: must not run"]


def test_infeasible_texts_never_hand_the_decision_to_a_human():
    """Every surface that tells the teacher (or the milestone author) what an
    INFEASIBLE stop does must describe it as the run's end — not as a request
    for a human to rule on and send the run on with a new direction."""
    from forgekernel.loop.driver import _INFEASIBLE_MENTION_BLOCK
    root = Path(__file__).resolve().parent.parent
    surfaces = {
        "mention bounce": _INFEASIBLE_MENTION_BLOCK,
        "diagnose prompt": (root / "meta" / "diagnose_prompt.md").read_text(encoding="utf-8"),
        "milestone template": (root / "docs" / "milestones-template.md").read_text(encoding="utf-8"),
    }
    for name, text in surfaces.items():
        for phrase in ("the user decides", "hand it to the user", "hand off to the user",
                       "decide whether to continue"):
            assert phrase not in text.lower(), (name, phrase)
    assert "the end of this run" in _INFEASIBLE_MENTION_BLOCK


def test_infeasible_mentioned_under_another_class_is_bounced_once(tmp_path):
    """The spin shape: the verdict sits in the milestone / evidence while the
    class is a patchable one. One bounce names the only channel; the stop
    request that answers it ends the step."""
    spin = ("cause_class: G4\ndeficient_locus: none\n"
            "evidence: 1.05x line INFEASIBLE (closure record 1), stop stands\n"
            "patch_direction: none\n")
    be = FakeBackend([spin, _INFEASIBLE], [])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx, best_before=0.978)
    assert "## INFEASIBLE has only one channel" in be.prompts[1]
    assert res.patch_class == "infeasible-stop"


def test_infeasible_mention_bounce_keeps_the_class_the_teacher_confirms(tmp_path):
    """A quote or a negation is not a stop request: after the one bounce the
    teacher's class stands and the step patches as usual."""
    negated = ("cause_class: R2\ndeficient_locus: D6\n"
               "evidence: the ref reaches this line, so it is no INFEASIBLE basis\n"
               "patch_direction: add decision\n")
    be = FakeBackend([negated, negated,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")], [])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert res.patch_class == "additive"
    assert sum("## INFEASIBLE has only one channel" in p for p in be.prompts) == 1
    assert not (tmp_path / "infeasible_stop.md").exists()


def test_two_layer_run_ends_on_infeasible_with_the_budget_unspent(tmp_path):
    """End to end: the student runs out, the teacher asks to stop, and the run
    stops — no further student turn, meta budget left over, stop=infeasible
    (a verdict, not a crash to restart) and the notice path."""
    ms = tmp_path / "m.md"
    ms.write_text(_MILESTONE, encoding="utf-8")
    be = _E2EFake(agent_outputs=["EXHAUSTED: nothing left", _INFEASIBLE,
                                 "EXHAUSTED: must not run"],
                  gpu_logs=[], milestone_path=ms, patch_writes=[])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.milestone = ms
    summary = run_two_layer(ctx, max_meta=3)
    assert summary["stop"] == "infeasible" and summary["meta_steps"] == 1
    assert summary["notice"].endswith("infeasible_stop.md")
    assert Path(summary["notice"]).exists()
    assert be.agent_outputs == ["EXHAUSTED: must not run"]
    assert ms.read_text(encoding="utf-8") == _MILESTONE


def test_lever_price_block_says_a_group_card_was_priced_on_its_representative_only():
    from forgekernel.loop.driver import _lever_price_block
    lever = {"kind": "lever", "name": "idx", "add_pct": 1.0, "remove_pct": -1.0, "cls": "independent",
             "spin_add_ns": None, "spin_all_minus_ns": None}
    stack = {"kind": "stack", "all_pct": 1.0, "sum_singles_pct": 1.0, "coupling_pp": 0.0,
             "workload_scope": {"workloads": "representative", "rep": "wl2", "of": "8"}}
    block = _lever_price_block([lever, stack])
    assert "Multi-workload tree: this table is timed only on the representative workload `wl2`" in block and "of 8 in the group" in block
    assert "The KEEP gate is still judged on the group geomean" in block
    stack.pop("workload_scope")
    assert "Multi-workload tree: this table is timed only on the representative" not in _lever_price_block([lever, stack])
