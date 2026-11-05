"""Rows that buy no information, and reopens that keep coming back empty.

Three driver mechanisms, each off-switchable in loop.toml:

* reuse_unchanged_reading — a commit that changes nothing outside notes/ on top
  of an evaluated KEEP (a gate row submitted alone "with no code change", a
  reopened decision re-filed as built) is archived on that KEEP's reading: no
  eval / NCU job, and no second noise draw of the same tree for best / patch
  judgment to select on.
* declared_gate_close — a delivery carrying `[CLOSES: Mk-gate]` that KEEPs on its
  own full eval gets the Mk-gate row archived right behind it (same tree, same
  reading), instead of a separate student turn that re-reads the same tree.
* reopen_max_fruitless — after N reopen transactions in a row closed
  REVERT_PATCH (KEEP_PATCH of any class resets), structural / split patches are
  refused (REOPEN_CAPPED) instead of sending the student back to re-deliver.
"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx, _patching  # noqa: E402

from forgekernel.inner.trajectory import IterationRecord as IR  # noqa: E402
from forgekernel.inner.trajectory import append_record, load_records  # noqa: E402
from forgekernel.loop.driver import (  # noqa: E402
    _fruitless_reopen_streak, run_inner_iteration, run_inner_loop, run_meta_step,
    run_two_layer,
)
from forgekernel.meta_trajectory import (  # noqa: E402
    MetaRecord, append_meta_record, load_meta_records, reflect_summary,
)
from forgekernel.resume import Decision  # noqa: E402


class TreeBackend(FakeBackend):
    """A FakeBackend that can list what a commit changes (``changes``: one answer
    per commit_paths call; a list of paths, or None for "cannot say")."""

    def __init__(self, agent_outputs, gpu_logs, changes=()):
        super().__init__(agent_outputs, gpu_logs)
        self.changes = list(changes)

    def commit_paths(self, base, head):
        self.calls.append(("diff", base, head))
        return self.changes.pop(0)


def _row(did, change, files="kernel.cu"):
    return (f"decision_id: {did}\nchange: {change}\nhypothesis: h\n"
            f"binding_constraint: bc\nfiles_touched: {files}\n")


def _keep(iter_, sha, ms=2.5, did="D1.1", change="built", base="", cluster=""):
    return IR(iter=iter_, decision_id=did, change=change, correctness=True, verdict="KEEP",
              bwd_median_ms=ms, device_median_ms=ms, vs_ref=0.9, vs_ref_device=0.9,
              sha=sha, base_sha=base, cluster=cluster, profiler={"tensor_pct": 50.0})


def _gpu_calls(be):
    return [c for c in be.calls if c[0] == "gpu"]


# --- reuse_unchanged_reading --------------------------------------------------

def test_a_gate_row_with_no_code_change_is_archived_on_the_evaluated_reading(tmp_path):
    be = TreeBackend([REC("M1-gate", 0)], [], changes=[["notes/smoke.py"]])
    ctx = _ctx(tmp_path, be)
    append_record(ctx.status_path, _keep(0, "sha0", ms=2.5))
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert _gpu_calls(be) == [] and ("push", "fs-demo-i1") not in be.calls   # no eval, no NCU
    assert ("diff", "sha0", "sha1") in be.calls
    assert (rec.decision_id, rec.verdict, rec.correctness) == ("M1-gate", "KEEP", True)
    assert rec.device_median_ms == 2.5 and rec.vs_ref_device == 0.9
    assert rec.profiler == {"tensor_pct": 50.0}
    assert (rec.sha, rec.base_sha, rec.reading_of, rec.iter) == ("sha1", "sha0", 0, 1)
    on_disk = load_records(ctx.status_path)[-1]
    assert on_disk.reading_of == 0 and on_disk.decision_id == "M1-gate"


def test_a_reused_reading_points_at_the_row_that_measured_it(tmp_path):
    """A no-change row on top of a no-change row carries the ORIGINAL job's iter."""
    be = TreeBackend([REC("M2-gate", 0)], [], changes=[[]])
    ctx = _ctx(tmp_path, be)
    append_record(ctx.status_path, _keep(0, "sha0", ms=2.5))
    reused = _keep(1, "sha1", ms=2.5, did="D2.1", base="sha0")
    reused.reading_of = 0
    append_record(ctx.status_path, reused)
    be._sha = 1
    rec = run_inner_iteration(ctx, 2, metric_before=2.5, last_good_ref="sha1", green=True)
    assert rec.reading_of == 0 and _gpu_calls(be) == []


def test_a_code_change_is_evaluated(tmp_path):
    be = TreeBackend([REC("M1-gate", 0)], [EVAL(2.4), NCU], changes=[["kernel.cu", "notes/x.py"]])
    ctx = _ctx(tmp_path, be)
    append_record(ctx.status_path, _keep(0, "sha0", ms=2.5))
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert ("gpu", "all") in be.calls and rec.reading_of is None and rec.device_median_ms == 2.4


@pytest.mark.parametrize("why", ["no_keep", "revert_base", "other_cluster", "cannot_say", "off"])
def test_reuse_needs_an_evaluated_keep_of_that_tree_here(tmp_path, why):
    """No KEEP for the base tree, a base that failed, a reading from another
    cluster, a backend that cannot list the change, or the switch off: evaluate."""
    changes = [None] if why == "cannot_say" else [[]]
    be = TreeBackend([REC("M1-gate", 0)], [EVAL(2.4), NCU], changes=changes)
    ctx = _ctx(tmp_path, be)
    if why == "revert_base":
        append_record(ctx.status_path, IR(iter=0, decision_id="D1.1", change="x",
                                          correctness=False, verdict="REVERT", sha="sha0"))
    elif why == "other_cluster":
        append_record(ctx.status_path, _keep(0, "sha0", cluster="far"))
    elif why != "no_keep":
        append_record(ctx.status_path, _keep(0, "sha0"))
    if why == "off":
        ctx.reuse_unchanged_reading = "off"
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert ("gpu", "all") in be.calls and rec.reading_of is None


def test_gate_mode_reuses_gate_rows_only(tmp_path):
    be = TreeBackend([REC("D1.2", 0), REC("M1-gate", 0)], [EVAL(2.45), NCU],
                     changes=[[], []])
    ctx = _ctx(tmp_path, be)
    ctx.reuse_unchanged_reading = "gate"
    append_record(ctx.status_path, _keep(0, "sha0", ms=2.5))
    dec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert dec.reading_of is None and be.calls.count(("gpu", "all")) == 1   # re-measured
    gate = run_inner_iteration(ctx, 2, metric_before=2.45, last_good_ref="sha1", green=True)
    assert gate.reading_of == 1 and be.calls.count(("gpu", "all")) == 1     # not re-measured


def test_all_mode_reuses_a_decision_refiled_as_built(tmp_path):
    be = TreeBackend([REC("D0.4+D0.8", 0)], [], changes=[[]])
    ctx = _ctx(tmp_path, be)
    append_record(ctx.status_path, _keep(0, "sha0", ms=2.5))
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.reading_of == 0 and _gpu_calls(be) == []


def test_a_reused_tree_does_not_rebuy_a_refused_parity_check(tmp_path):
    be = TreeBackend([REC("M1-gate", 0)], [], changes=[[]])
    ctx = _ctx(tmp_path, be)
    ctx.paired_memo["sha0"] = {"verdict": "not_confirmed"}
    append_record(ctx.status_path, _keep(0, "sha0"))
    run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert ctx.paired_memo["sha1"] == {"verdict": "not_confirmed"}


def test_a_reused_gate_row_still_closes_the_phase_and_cannot_set_a_lucky_best(tmp_path):
    """In the loop: the gate row retires the phase-close notice as before, and
    the best stays the evaluated reading (no second draw to pick a minimum of)."""
    be = TreeBackend([REC("M1-gate", 0)], [], changes=[[]])
    ctx = _ctx(tmp_path, be)
    ctx.max_iters = 1
    ctx.phase_close = {"gate": "M1-gate", "phase": "M1", "note": "- n\n"}
    append_record(ctx.status_path, _keep(0, "sha0", ms=2.5))
    assert run_inner_loop(ctx, start_iter=1) == "max_iters"
    assert ctx.phase_close is None
    recs = load_records(ctx.status_path)
    assert [r.device_median_ms for r in recs] == [2.5, 2.5]


# --- declared_gate_close ------------------------------------------------------

def test_a_delivery_that_closes_its_phase_files_the_gate_row_itself(tmp_path):
    be = TreeBackend([_row("D2.3", "the epilogue lever [CLOSES: M2-gate]")], [EVAL(2.3), NCU],
                     changes=[["kernel.cu"]])
    ctx = _ctx(tmp_path, be)
    append_record(ctx.status_path, _keep(0, "sha0", ms=2.5))
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    rows = load_records(ctx.status_path)
    assert [r.decision_id for r in rows] == ["D1.1", "D2.3", "M2-gate"]
    assert [r.iter for r in rows] == [0, 1, 2]
    gate = rows[-1]
    assert rec == gate and len(be.prompts) == 1 and len(_gpu_calls(be)) == 2   # one eval + NCU
    assert (gate.verdict, gate.sha, gate.base_sha, gate.reading_of) == ("KEEP", "sha1", "sha1", 1)
    assert gate.device_median_ms == 2.3 and "[CLOSES: M2-gate]" in gate.change
    assert rows[1].reading_of is None


@pytest.mark.parametrize("spelling", ["[CLOSES: M2 gate]", "[closes: m2-Gate]", "[CLOSES:M2-gate]"])
def test_the_closes_tag_reads_the_usual_spellings(tmp_path, spelling):
    be = TreeBackend([_row("D2.3", f"lever {spelling}")], [EVAL(2.3), NCU], changes=[["kernel.cu"]])
    ctx = _ctx(tmp_path, be)
    append_record(ctx.status_path, _keep(0, "sha0"))
    assert run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0",
                               green=True).decision_id == "M2-gate"


@pytest.mark.parametrize("case", ["revert", "unfinished_package", "placeholder_open",
                                  "placeholder_on_row", "switch_off", "pre_green"])
def test_a_closes_tag_is_not_honoured_where_a_gate_row_would_not_land(tmp_path, case):
    change = {"unfinished_package": "part 1 of 2 — first half [CLOSES: M2-gate]",
              "placeholder_on_row": "[PLACEHOLDER: D2.3 → full form] lever [CLOSES: M2-gate]",
              }.get(case, "lever [CLOSES: M2-gate]")
    gpu = [EVAL(2.3, ok=(case != "revert")), NCU]
    be = TreeBackend([_row("D2.3", change)], gpu, changes=[["kernel.cu"]])
    ctx = _ctx(tmp_path, be)
    seed = _keep(0, "sha0", change=("[PLACEHOLDER: D0.2 → the real form]"
                                    if case == "placeholder_open" else "built"))
    append_record(ctx.status_path, seed)
    if case == "switch_off":
        ctx.declared_gate_close = False
    green = False if case == "pre_green" else True
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=green)
    assert "M2-gate" not in [r.decision_id for r in load_records(ctx.status_path)]
    assert rec.decision_id == "D2.3"


def test_a_closed_gate_drives_the_loop_like_a_submitted_one(tmp_path):
    """The returned row is the gate: the reopened-M0 audit stops on it, its iter
    number is taken (one row per iter) but max_iters does not pay for it."""
    be = TreeBackend([_row("D0.4", "redo [CLOSES: M0-gate]")], [EVAL(2.4), NCU],
                     changes=[["kernel.cu"]])
    ctx = _ctx(tmp_path, be)
    append_record(ctx.status_path, _keep(0, "sha0"))
    assert run_inner_loop(ctx, start_iter=1, audit_on_gate="M0-gate") == "first_green"
    assert [r.decision_id for r in load_records(ctx.status_path)] == ["D1.1", "D0.4", "M0-gate"]

    be2 = TreeBackend([_row("D1.1", "a [CLOSES: M1-gate]"), REC("D2.3", 0)],
                      [EVAL(2.4), NCU, EVAL(2.35), NCU], changes=[["kernel.cu"], ["kernel.cu"]])
    (tmp_path / "b").mkdir()
    ctx2 = _ctx(tmp_path / "b", be2)
    ctx2.max_iters = 2
    append_record(ctx2.status_path, _keep(0, "sha0"))
    assert run_inner_loop(ctx2, start_iter=1) == "max_iters"
    rows = load_records(ctx2.status_path)
    assert [(r.iter, r.decision_id) for r in rows] == [(0, "D1.1"), (1, "D1.1"), (2, "M1-gate"),
                                                       (3, "D2.3")]
    assert len(be2.prompts) == 2


def test_a_reused_row_can_close_its_gate_too(tmp_path):
    be = TreeBackend([_row("D2.1+D2.2", "re-filed as built [CLOSES: M2-gate]", "none")], [],
                     changes=[[]])
    ctx = _ctx(tmp_path, be)
    append_record(ctx.status_path, _keep(0, "sha0"))
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.decision_id == "M2-gate" and rec.reading_of == 0 and _gpu_calls(be) == []


# --- reopen_max_fruitless -----------------------------------------------------

def _meta(step, pc, outcome, violations=()):
    return MetaRecord(meta_step=step, cause_class="R1" if pc != "additive" else "R2",
                      deficient_locus="D0.2", patch_class=pc, changed_entries=["D0.2"],
                      patch_direction="d", outcome=outcome, milestone_violations=list(violations))


@pytest.mark.parametrize("seq, want", [
    ([], 0),
    ([("structural", "REVERT_PATCH")], 1),
    ([("structural", "REVERT_PATCH"), ("split", "REVERT_PATCH")], 2),
    # a held reopen joined by a second one is ONE transaction
    ([("structural", "HOLD_PATCH"), ("structural", "REVERT_PATCH")], 1),
    # a KEEP_PATCH of any class resets
    ([("structural", "REVERT_PATCH"), ("additive", "KEEP_PATCH"), ("structural", "REVERT_PATCH")], 1),
    ([("structural", "REVERT_PATCH"), ("structural", "KEEP_PATCH")], 0),
    # an additive-only REVERT neither counts nor resets
    ([("structural", "REVERT_PATCH"), ("additive", "REVERT_PATCH"), ("structural", "REVERT_PATCH")], 2),
    # an additive patch that joins a held reopen makes the transaction a reopen
    ([("structural", "HOLD_PATCH"), ("additive", "REVERT_PATCH")], 1),
    # still open: not counted
    ([("structural", "REVERT_PATCH"), ("structural", "HOLD_PATCH")], 1),
    ([("structural", "PENDING")], 0),
    # refusals, KB steps and audits are not transactions
    ([("structural", "REVERT_PATCH"), ("unpriced-reopen", "UNPRICED_REOPEN"),
      ("kb-example", "KB_CERTIFIED"), ("audit-clean", "AUDIT_CLEAN"),
      ("reopen-capped", "REOPEN_CAPPED"), ("structural", "REVERT_PATCH")], 2),
])
def test_fruitless_reopen_streak_counts_transactions(seq, want):
    assert _fruitless_reopen_streak([_meta(i, pc, o) for i, (pc, o) in enumerate(seq)]) == want


def test_a_patch_that_never_landed_is_not_a_reopen():
    recs = [_meta(0, "structural", "REVERT_PATCH"),
            _meta(1, "structural", "REVERT_PATCH", violations=["§9: missing gate"])]
    assert _fruitless_reopen_streak(recs) == 1


def _spent(ctx, n=2):
    for i in range(n):
        append_meta_record(ctx.meta_status_path, _meta(i, "structural", "REVERT_PATCH"))


_R1 = "cause_class: R1\ndeficient_locus: D1.1\npatch_direction: flip the stage split axis\n"


def test_with_the_budget_spent_an_r1_is_bounced_then_refused(tmp_path):
    be = FakeBackend([_R1, _R1], [])
    ctx = _ctx(tmp_path, be)
    _spent(ctx)
    before = ctx.milestone.read_text(encoding="utf-8")
    res = run_meta_step(ctx)
    assert res.patch_class == "reopen-capped" and res.changed_entries == []
    assert "## Reopen budget exhausted" in be.prompts[0]          # announced before the diagnosis
    assert "You classified `R1`" in be.prompts[1]                # the one bounce
    assert len(be.prompts) == 2                          # no patch turn
    assert ctx.milestone.read_text(encoding="utf-8") == before
    rec = load_meta_records(ctx.meta_status_path)[-1]
    assert (rec.outcome, rec.patch_class) == ("REOPEN_CAPPED", "reopen-capped")
    assert "[reopen budget exhausted, refused]" in reflect_summary(load_meta_records(ctx.meta_status_path))


def test_with_the_budget_spent_a_structural_patch_off_a_g1_is_withdrawn(tmp_path):
    g1 = "cause_class: G1\ndeficient_locus: D1.1\npatch_direction: re-tile\n"
    be = FakeBackend([g1, _patching(tmp_path, "\n### D1.1 re-tiled\nx\n",
                                    "patch_class: structural\nchanged_entries: [D1.1]\n")], [])
    ctx = _ctx(tmp_path, be)
    _spent(ctx)
    before = ctx.milestone.read_text(encoding="utf-8")
    assert run_meta_step(ctx).patch_class == "reopen-capped"
    assert ctx.milestone.read_text(encoding="utf-8") == before


def test_an_additive_patch_lands_with_the_budget_spent(tmp_path):
    r2 = "cause_class: R2\ndeficient_locus: D2.3\npatch_direction: add a lever\n"
    be = FakeBackend([r2, _patching(tmp_path, "\n### D2.4 added\nx\n",
                                    "patch_class: additive\nchanged_entries: []\n")], [])
    ctx = _ctx(tmp_path, be)
    _spent(ctx)
    assert run_meta_step(ctx).patch_class == "additive"


@pytest.mark.parametrize("cap, spent", [(0, 5), (2, 1), (3, 2)])
def test_a_reopen_lands_under_the_budget_or_with_it_off(tmp_path, cap, spent):
    structural = _patching(tmp_path, "\n### D1.1 flipped\nx\n",
                           "patch_class: structural\nchanged_entries: [D1.1]\n")
    be = FakeBackend([_R1, structural], [])
    ctx = _ctx(tmp_path, be)
    ctx.reopen_max_fruitless = cap
    _spent(ctx, spent)
    assert run_meta_step(ctx).patch_class == "structural"
    assert "Reopen budget exhausted" not in be.prompts[0]


def test_plateau_reopen_capped_retries_the_meta_step(tmp_path):
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
    _spent(ctx)
    summary = run_two_layer(ctx, max_meta=3)
    assert summary["stop"] == "target"
    outcomes = [r.outcome for r in load_meta_records(ctx.meta_status_path)][2:]
    assert outcomes[0] == "REOPEN_CAPPED" and len(outcomes) == 2
    assert [r.decision_id for r in load_records(ctx.status_path)] == list("ABCDE")


def test_m0_audit_reopen_capped_resumes_the_segment(tmp_path):
    dag = [Decision(id="D0.1"), Decision(id="D0.2", depends_on=["D0.1"]),
           Decision(id="D1.1", depends_on=["D0.2"])]
    r1 = "cause_class: R1\ndeficient_locus: D0.2\npatch_direction: flip the split axis\n"
    be = FakeBackend([REC("D0.1", 2.66), r1, r1, REC("D1.1", 1.8)],
                     [EVAL(2.66), NCU, EVAL(1.8), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.9, dag=dag)
    ctx.m0_audit = True
    _spent(ctx)
    summary = run_two_layer(ctx, max_meta=4)
    assert summary["stop"] == "target"
    assert [r.decision_id for r in load_records(ctx.status_path)] == ["D0.1", "D1.1"]
    assert load_meta_records(ctx.meta_status_path)[-1].outcome == "REOPEN_CAPPED"


# --- plumbing -----------------------------------------------------------------

def test_reading_of_round_trips_and_old_rows_read_none(tmp_path):
    p = tmp_path / "s.jsonl"
    r = _keep(3, "shaX")
    r.reading_of = 1
    append_record(p, r)
    p.write_text(p.read_text() + '{"iter": 4, "decision_id": "D1", "change": "c", '
                 '"correctness": true, "verdict": "KEEP"}\n')
    a, b = load_records(p)
    assert a.reading_of == 1 and b.reading_of is None


def test_real_backend_lists_a_commits_changes(tmp_path):
    from forgekernel.loop.cli import RealBackend

    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c",
                               "user.email=t@t", *a], check=True, capture_output=True,
                              text=True).stdout.strip()
    git("init", "-q")
    (tmp_path / "kernel.cu").write_text("k\n")
    git("add", "-A"); git("commit", "-q", "-m", "a")
    a = git("rev-parse", "HEAD")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "smoke.py").write_text("p\n")
    git("add", "-A"); git("commit", "-q", "--allow-empty", "-m", "b")
    b = git("rev-parse", "HEAD")
    git("commit", "-q", "--allow-empty", "-m", "c")
    c = git("rev-parse", "HEAD")
    be = SimpleNamespace(workspace=tmp_path)
    assert RealBackend.commit_paths(be, a, b) == ["notes/smoke.py"]
    assert RealBackend.commit_paths(be, b, c) == []
    assert RealBackend.commit_paths(be, a, "nope") is None
