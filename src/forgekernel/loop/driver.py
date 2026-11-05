"""Two-layer loop driver — the orchestration that was previously hand-driven.

The driver chains the primitives (scaffold / submit / trajectory / ncu / resume)
into the inner student loop and the outer meta loop, driving sub-agents and GPU
jobs through a single injected ``backend`` seam. All control flow here is
deterministic given the backend, so it is unit-tested with a fake; the real
backend (agent CLI + GPU jobs + git) lives in ``forgekernel.loop.cli``.

Inner loop per iteration:
  render student prompt -> agent -> (probe sub-loop until a record) -> commit +
  push -> GPU ``all`` (correctness+perf) -> GPU ``ncu`` (profiler) -> append
  IterationRecord. relaxed-revert: a perf regression is KEPT (scaffolding), only a
  correctness break REVERTs (and rolls the workspace back). Pre-green — before the
  lineage's first correctness PASS, e.g. while M0 is under construction — committed
  sub-steps are NOT evaluated (NEUTRAL rows, POLICY §3 build-phase semantics, executed): a full
  eval of a half-built kernel decides nothing, and the deferred verification is
  spent by the gate row ("M0-gate"), which always gets the full eval.

Outer loop: when the inner loop plateaus, run meta diagnose + patch, compute the
prefix-reuse resume point, reset the workspace there, and resume the inner loop.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shlex
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Protocol

from forgekernel.inner.ncu import kernels_from_ncu, profiler_from_ncu
from forgekernel.skills.sass_census import parse_census_line, parse_serial_line
from forgekernel.loop.job_env import env_failure
from forgekernel.loop import harness_guard, precision_gate
from forgekernel.loop.kb_lint import lint_kb_dir
from forgekernel.loop import lang_mode
from forgekernel.inner.trajectory import (
    GATE_WINDOW_DEFAULT,
    GATE_WINDOW_DEVICE,
    GATE_WINDOWS,
    IterationRecord,
    append_record,
    constant_output_flag,
    gate_ms,
    load_records,
)
from forgekernel.meta_trajectory import (
    MetaRecord,
    append_meta_record,
    load_meta_records,
    reflect_summary,
    update_last_outcome,
)
from forgekernel.window_contract import PERF_WORKLOADS
from forgekernel.loop.parse import (
    describe_workload_readings,
    ENTRY_DONE_RE,
    Exhausted,
    JOB_END_KILLED,
    ProbeRequest,
    ProbeVerdict,
    Record,
    detect_plateau,
    job_abend,
    parse_eval,
    parse_probe_verdict,
    parse_student_turn,
    ReconRequest,
)
from forgekernel.baseline import RefBaseline, workloads_mismatch
from forgekernel.resume import Decision, compute_reopen, compute_resume
# a group card's meta readings, one workload per job
from forgekernel.loop import meta_shapes


class DriverBackend(Protocol):
    """The side-effecting seam: agent spawn, GPU submit, git. Faked in tests."""

    def run_agent(self, prompt: str) -> str: ...
    def submit_gpu(self, branch: str, mode: str) -> str: ...  # returns gpu_entry stdout text
    def commit(self, message: str) -> str: ...  # returns the new HEAD sha
    def push_branch(self, branch: str) -> None: ...
    def reset(self, ref: str) -> None: ...
    def head(self) -> str: ...  # current HEAD sha (baseline for meta resume)


AGENT_ROLE_STUDENT = "student"
AGENT_ROLE_TEACHER = "teacher"

# ENTRY_DONE_RE (the operator entry script's closing line) now lives in parse,
# where parse_probe_verdict demands it behind a PASS; imported above so loop.cli
# and the tests keep finding it here.


def archive_is_whole(log: str, task: str = "") -> bool:
    """Is ``log`` a complete job archive: the entry script's closing line is its
    LAST non-empty line and names ``task`` (the submitted gpu_entry task, when
    the caller knows it)?

    Anywhere else the line proves nothing: a ``cmd`` job that wraps another
    entry run (orch_census_run streams ``bash gpu_entry.sh ncu`` line by line
    and prints its ORCH-* verdict after it; teacher probes shell out the same
    way) carries the inner ``task=ncu finished`` mid-log, and an archive read
    at that point is exactly the truncated tail this check exists to refuse.
    """
    lines = [ln for ln in log.splitlines() if ln.strip()]
    if not lines:
        return False
    m = ENTRY_DONE_RE.match(lines[-1].strip())
    return bool(m) and (not task or m["task"] == task)


def _backend_takes_role(be) -> bool:
    """Does ``be.run_agent`` accept ``role=``? Backends that predate the split
    (test fakes, single-model runs) take only the prompt and serve both roles."""
    try:
        return "role" in inspect.signature(be.run_agent).parameters
    except (TypeError, ValueError):
        return False


class _TeacherView:
    """``be`` with ``run_agent`` bound to the teacher role; everything else passes
    through. Teacher turns (recon, distill, diagnose/patch, KB certify/harvest,
    lever design) go here so a run can use the teacher on one model and the
    student on another (``--teacher-agent-cmd``); a backend without a teacher
    command serves the turn from its only agent, as before."""

    def __init__(self, be):
        self._be = be

    def run_agent(self, prompt: str) -> str:
        if _backend_takes_role(self._be):
            return self._be.run_agent(prompt, role=AGENT_ROLE_TEACHER)
        return self._be.run_agent(prompt)

    def __getattr__(self, name):
        return getattr(self._be, name)


def _teacher(be):
    """The teacher-role view of a backend (idempotent)."""
    return be if isinstance(be, _TeacherView) else _TeacherView(be)


class AgentError(RuntimeError):
    """The agent backend failed unrecoverably (after its own retries).

    Raised by the real backend when the agent CLI keeps failing; the loop catches
    it and stops cleanly instead of crashing mid-run.
    """


class BackendError(RuntimeError):
    """A backend side-effect (GPU submit, git push/commit/reset) failed after
    retries. The loop catches it and stops cleanly rather than crashing."""


class NoRunError(BackendError):
    """A GPU submit that ended with NO job producing output: every job lost its
    runner, or the GPU would not hold the board's clock lock (loop.cli raises the
    concrete kinds). Nothing ran, so nothing may be
    charged to the payload — a caller that counts attempts against a tree
    (_ensure_ref_pricing: "the same tree would hang again") must not count one
    of these. Its own type because recognising the case by a word in the
    message is control flow keyed on prose: a no-run verdict worded differently
    slipped past it, wrote the attempt marker, and (same tree, same key)
    forfeited the reference pricing for good."""


@dataclass
class ProbeBattle:
    """One student probe path's GPU-round bill, kept honest per *fight*.

    A probe that needs many rounds before its PASS is a knowledge gap that never
    surfaces as a G3: the student wins, so the inner loop never stops ``stuck``
    and the meta layer never diagnoses it. A fight is the consecutive rounds on
    one path up to a PASS; it routinely spans iterations (max_probe_rounds caps
    one iteration, not the fight), so ``open_rounds`` carries across them. The
    bill (``rounds`` + evidence) is always ONE winning fight, taken whole: a
    cheap win followed by later re-probes must not add up to a phantom epic, and
    a re-won fight must not cite the previous fight's verdict — rounds, pass
    iter, verdict and log tail travel together or not at all."""
    path: str
    open_rounds: int = 0  # rounds in the fight currently in progress (no PASS yet)
    rounds: int = 0       # the billed fight: the dearest winning fight's rounds
    passed: bool = False
    pass_iter: int | None = None
    verdict_line: str = ""
    log_tail: str = ""  # bounded tail of the billed fight's winning job log


@dataclass
class LoopContext:
    """Driver configuration plus M0 audit / stall pricing / timeline toggles.

    **STRUCT channel**: on blackbox runs the per-segment construction table of
    the reference is a mandatory reconnaissance evidence channel
    (``_struct_recon`` gates it on ``oracle_mode``), decoded by the vendored
    ``struct_tools`` and banked as the ref's construction table for the distill
    brief — see ``_read_ref_struct_table``.
    """
    op: str
    workspace: Path
    milestone: Path
    kb_dir: Path
    status_path: Path
    meta_dir: Path
    kernel_file: str
    branch_prefix: str
    target_ms: float
    deadline_epoch: float
    backend: DriverBackend
    now: Callable[[], float]
    language: str = "cutedsl"  # kernel implementation language (governs the student prompt)
    # Which hw_kb/<language>/<arch>/ tree this run reads AND writes -- see kb_read_dir. One
    # tree per architecture, no shared root (hw_kb/<language>/README.md): a recipe earned on
    # Hopper is not evidence about Blackwell, so a run sees its own tree only.
    # From loop.toml [loop].kb_arch (or --kb-arch); "sm90" is the default.
    kb_arch: str = "sm90"
    # Which timing window every verdict reads: best, target stop, plateau, rollback
    # anchor, digest ranking. "device" (pipelined batch ≈ device time — the
    # kernel-quality claim, the default) or "e2e" (per-call+sync, host
    # tax included; per-task opt-in). Per-task config because neither is right in
    # general — see GATE_WINDOWS in inner/trajectory.py. target_ms must be anchored
    # in this same window; the CLI derives it that way and refuses a device gate
    # over a baseline that never measured the device window.
    gate_window: str = GATE_WINDOW_DEFAULT
    meta_status_path: Path | None = None  # meta-loop trajectory (meta agent's memory)
    fresh_recipe: str | None = None  # a just-certified KB recipe to point the student at
    # In-place reopening of structural / split patches: keep the best green tree
    # and redo only the dependency closure of the changed entries on it, instead
    # of resetting to the checkpoint before the earliest affected decision and
    # rebuilding everything after it (a sparse-attention run once spent six of ten
    # meta steps on whole-M0 rebuilds, two of them regressing to 3.9 ms). The closure
    # is announced to the student as a REOPEN row in status.jsonl plus a driver
    # context note. Off per task via loop.toml [loop] reopen_in_place = false.
    reopen_in_place: bool = True
    reopen_note: str = ""
    g5_note: str = ""  # G5 in-segment reordering directive for the student (cleared with fresh_recipe)
    # A phase the price table read as exhausted and the diagnosis closed:
    # {"gate": "M1-gate", "phase": "M1", "note": <student-facing text>}. The
    # note rides every student turn until the gate row is submitted (any
    # verdict) or a structural resume replaces the milestone premise. None
    # when no phase is closed this way. See _phase_exhaustion.
    phase_close: dict | None = None
    # status.jsonl index ranges [lo, hi) of post-patch segments a REVERT_PATCH
    # rolled back. The rows stay on file (the student's ledger is append-only)
    # but describe a tree that no longer exists: the same-stage gate check
    # must not take a reverted segment's gate as the "previous" reading, or a
    # rebuild that regressed and was reverted lowers the bar for the next one.
    # In-memory like the pending transaction itself; a driver restart starts
    # both empty.
    voided_ranges: list = field(default_factory=list)
    # Reference tokens the certify-time KB lint scans for, on top of kb_lint's
    # static list — derived from the resolved oracle tree by the CLI at startup.
    kb_lint_tokens: list[str] = field(default_factory=list)
    # Diagnosis carried into the NEXT student turn when the last evaluated round was
    # host-bound (the student's own host path, never the reference's). Set by
    # _check_host_bound, consumed and cleared by _student_prompt.
    host_bound_note: str = ""
    # Compiler-fallback note standing until the chains un-serialize: the last
    # evaluated KEEP's wgmma serialization (DEPBAR-0x0 waits per wgmma ~1).
    # Set from the sass_census fields the orchestration census emits; cleared
    # when a later KEEP prices pipelined again.
    sass_serial_note: str = ""
    # Standing notice for every student turn of a post-patch segment whose workspace
    # was reset by _resolve_resume: names the checkpoint and the invalidated
    # decisions, because the student otherwise reads the trajectory's old records as
    # proof those decisions are "delivered" and reports EXHAUSTED without landing a
    # single row (a GDN run burned 4 meta steps on exactly this). Set at each
    # meta resume, overwritten by the next one.
    resume_note: str = ""
    decisions_dag: list[Decision] = field(default_factory=list)
    oracle_path: str | Path | None = None
    # Which channel the teacher distills through: "source" reads oracle_path,
    # "blackbox" withholds it and distills from black-box reconnaissance of the
    # reference instead, "none" does not distill. Kept separate from oracle_path
    # being None because those used to be the same thing, and "forbidden to read
    # source" then silently meant "do not distill, and do not measure the baseline
    # that distillation needs" — the loop would start against whatever milestone
    # happened to be on disk.
    oracle_mode: str = "source"
    recon_path: Path | None = None  # the reconnaissance report distilled from (blackbox
                                    # always; a source-mode run that wired a recon tree)
    recon_dir: Path | None = None   # the teacher-only tree recon scripts are written into
    # A second backend, pointed at the reconnaissance tree. Reconnaissance must not run
    # through the student's submission tree: the scripts and everything they learn about
    # the reference would ride the student's job tarball, and what the student is meant
    # to receive is a milestone, not the analysis behind it.
    recon_backend: "DriverBackend | None" = None
    max_recon_rounds: int = 12
    simulator_dir: Path | None = None  # hardware calculator; pure-python, no GPU needed
    candidate: str | None = None       # which candidate architecture this run implements
    ref_baseline: "RefBaseline | None" = None  # library baseline: "better than the library?"
    # The timed-window CONTENT contract (operators/<op>/window.toml, see
    # forgekernel.window_contract): which inputs the reference's timed call
    # receives that the student's does not, and how each is classed. The CLI
    # refuses to start a run with a reference and no consistent declaration;
    # the driver renders it to the teacher next to both per-kernel tables and
    # prices the declared rebuilds at confirmation. None = not declared (the
    # binding then says so out loud rather than staying silent).
    window_contract: "WindowContract | None" = None
    arch_baseline: "RefBaseline | None" = None  # the implementation the milestone teaches;
    # target_ms anchors here. Separate from ref_baseline because they answer different
    # questions and are ~20% apart on this operator — deriving the stop threshold from
    # the library one let the loop plan to finish at three quarters of the oracle.
    max_probe_rounds: int = 4
    max_patch_rounds: int = 3
    # GPU verdict-probe budget per diagnose step (0 disables). Before this existed the
    # diagnose agent's only verification granularity was a whole inner segment judged
    # on scalar best-progress: a poisoned best waterline read three correct patches as
    # useless (an attention fwd run, 7.5h), and an unfalsified recon hypothesis steered
    # a 33.6h bwd-MHA run. A verdict probe collapses the hypothesis space at the cost
    # of one GPU job instead.
    max_verdict_probe_rounds: int = 4
    # INFEASIBLE GATE: the fewest diagnoses (this one included; the distillation
    # and a clean M0 audit do not count) before a stop request can be accepted.
    # 1 = the reachability check alone gates the stop (see _infeasible_refusal).
    # loop.toml [loop] infeasible_min_diagnoses / --min-meta-before-infeasible.
    infeasible_min_diagnoses: int = 1
    ncu_retries: int = 2  # retry NCU when a passing kernel yields an empty profiler
    # LEVER PRICING ROUND (plateau diagnosis AND the M0 audit). Before the
    # teacher names a cause_class it designs N composable levers on the best tree
    # and the driver prices them in ONE job — base / +each / all / all-each — so
    # the diagnosis reads MEASURED prices, and a lever that
    # is worthless alone but required by the others (single-add ≈ 0, all-minus
    # < 0) is seen as a package member rather than dropped. A dsa_sparse run
    # named one locus per audit off the ledger; three ablation probes on its
    # best tree afterwards priced three levers in two hours, one of which (ring
    # acquire granularity, the largest) was never a ledger line at all. The audit
    # is included because that is where structural reopens are decided: the
    # same run flipped its stage split axis at the M0 audit on an inferred
    # masking window, with no price on either the flip or the levers it passed
    # over. The value is the number of interleaved bench ROUNDS the pricing job
    # times each variant (the in-job repeat count); 0 disables the round. A
    # PRICE-FAIL (an edit that does not apply) re-prompts the design turn
    # _LEVER_PRICE_RETRIES times, independently of this value. A table priced
    # on the same best tree earlier in the run is reused, not re-bought.
    lever_pricing_rounds: int = 3
    lever_pricing_max_levers: int = 5
    # A PRICE IS A READING ON ONE TREE. Every row the pricing round banks carries
    # the best sha it was measured on (`priced_on`), and when a later KEEP takes
    # the watermark the driver re-prices — one job, base / +each only, the
    # recipes banked with the table — every lever the milestone cites
    # (`lever:<name>`) whose price was read on another tree. A citation whose
    # price reversed or faded on the new best is marked in the milestone and
    # suspended for the student until the teacher's next entry reads the
    # revision (observed on a GDN run: a +25.6% rung priced on the first green was
    # walked two trees later, where the same lever read −11.6% and serialised
    # every wgmma; the gate row and the next lever were built on that tree).
    # Off via loop.toml [loop] reprice_on_new_best = false. A new best that
    # beats the tree the citation was last priced on by less than
    # reprice_min_gain_pct (same window, same-cluster rows) is not re-priced —
    # one job per watermark move would otherwise ride every KEEP of a climbing
    # segment; the teacher's next entry prices the current best anyway.
    reprice_on_new_best: bool = True
    reprice_min_gain_pct: float = 3.0
    # TIMELINE PROBE (recon/timeline_probe.py): the student's `_seg(...)`
    # markers rewritten into %clock64 stamps in a probe build — per role, per
    # segment, cycles per chunk, who waits on whom — the unit a latency-bound
    # kernel is judged in (the stall / role account only says who waits). "on":
    # every teacher entry (M0 audit, plateau), every gate row, every new best
    # beyond reprice_min_gain_pct, and on request (`PROBE_REQUEST: timeline`
    # from the student or the teacher); "auto": on when the milestone declares
    # `bottleneck: latency`; "off": never. The reference is never instrumented.
    timeline_probe: str = "off"
    # NO REOPEN WITHOUT A PRICE. An R1 diagnosis (or any structural / split patch)
    # must cite a price the driver measured: a verdict probe run this step, a
    # lever on this step's price table, or a job-certified hw_kb entry
    # (`reopen_price: <delta> via probe:<path>#<n> | lever:<name> | KB:<entry>`).
    # Without one the diagnosis is bounced once with the three routes, then
    # refused — recorded UNPRICED_REOPEN, nothing patched. Off via loop.toml
    # [loop] reopen_requires_price = false (control runs); auto-off when the run
    # has no in-step way to measure (no verdict probes and no pricing round).
    reopen_requires_price: bool = True
    # REOPEN BUDGET. A reopen redoes a whole closure of delivered decisions on the
    # best tree (often the entire M0 package), so a run of them that keep coming
    # back REVERT_PATCH spends iteration after iteration re-delivering what was
    # already built — and a re-delivery read on a lucky node looks like a gain.
    # Once this many reopen transactions IN A ROW have closed without a new best
    # (REVERT_PATCH; a KEEP_PATCH of any patch class resets the count; a held
    # transaction counts once, when it closes), the teacher is told reopens are
    # closed and a structural / split patch is refused — recorded REOPEN_CAPPED,
    # milestone restored, nothing reopened. Additive / KB patches and the
    # INFEASIBLE stop request are untouched. 0 = no budget (the old behavior).
    # loop.toml [loop] reopen_max_fruitless.
    reopen_max_fruitless: int = 2
    # UNCHANGED-TREE READING. A row whose commit changes nothing outside notes/
    # (a gate acceptance row submitted alone, a decision re-filed "as built", a
    # pricing-only row) is not re-measured: the driver archives it with the
    # reading of the evaluated KEEP on that same tree (IterationRecord.reading_of)
    # — no eval / NCU job, and no extra noise draw for best / patch judgment to
    # select on. "all" (default) | "gate" (gate rows only) | "off" (re-measure
    # every row, the old behavior). loop.toml [loop] reuse_unchanged_reading.
    reuse_unchanged_reading: str = "all"
    # DECLARED GATE CLOSE. A delivery whose change text carries `[CLOSES: M<k>-gate]`
    # and that KEEPs on a full eval gets its Mk-gate row archived by the driver
    # right after it, on the same tree and the same reading — the separate
    # student turn (plus smoke probe and eval) a no-code gate row costs is not
    # spent. The gate stays its own ledger row and every gate consumer (phase
    # close, same-stage regression, the reopened-M0 audit, the timeline bank)
    # reads it as before. Not on a pre-green row (no eval to close on), an
    # unfinished `part k of n` step, or over an open placeholder. loop.toml
    # [loop] declared_gate_close = false turns the tag into plain text.
    declared_gate_close: bool = True
    # [loop] harness_guard (loop/harness_guard.py): "revert" refuses a delivery whose
    # commits edit the card-owned evaluation harness (gpu_entry.sh, eval/, ref_bin/,
    # .gitattributes) before any job is bought, and puts a drifted lineage back to the
    # launch tree's harness at the start of every iteration; "warn" logs only; "off".
    # harness_anchor is the launch tree (run_two_layer sets it).
    harness_guard: str = "revert"
    harness_anchor: str = ""
    # [loop] precision_contract (loop/precision_gate.py): "strict" makes a failed
    # precision check a correctness failure, "official" records it only, "off" does not
    # read it. The CLI resolves the card kind's default (leaderboard card: official,
    # own bench: strict); "off" here keeps scripted contexts neutral.
    precision_contract: str = "off"
    # The card's bench is wired to report the precision block (precision_gate.bench_findings
    # found nothing missing at launch): under strict an eval that comes back without one
    # is then a correctness failure, not "unreported" -- the block's absence is something
    # the kernel, running in the bench's process, could have caused.
    precision_bench_reports: bool = False
    precision_unreported_warned: bool = False
    plateau_window: int = 3
    max_iters: int = 60
    max_consecutive_reverts: int = 3
    # kb-tripwire: a student probe that needs >= this many GPU rounds before its
    # PASS is queued for the end-of-run KB harvest write-up (run_kb_harvest).
    # 0 disables. The threshold is on the *fight* (cumulative rounds per probe
    # path), not on one iteration's sub-loop.
    kb_tripwire_rounds: int = 6
    probe_battles: dict[str, ProbeBattle] = field(default_factory=dict)
    # Force a meta audit the first time a segment goes green (= the M0 gate lands).
    # The gate verifies the student against the milestone's OWN declared values — it
    # cannot verify the milestone. Every iteration after M0 builds on the foundation,
    # so a foundation defect audited at the M0 close costs one meta turn; discovered
    # at the next plateau it costs the whole segment built on top of it (observed on a
    # bwd run: bankconf 73M was severe-flagged early and scheduled for reckoning at D2.4
    # per the milestone — 26 iterations and ~10h later).
    m0_audit: bool = True
    # Per-SASS stall pricing: a fourth recon evidence channel (STALL-OK on the ref)
    # and a driver-run pricing job on the best student kernel before every plateau
    # diagnosis. One fwd run's meta chose two loci the aggregate digest could
    # not falsify — both priced at ~1pp once measured — and burned 10h proving it;
    # the ledger is the evidence face that diagnosis was missing. Off per task via
    # loop.toml [loop] stall_pricing = false.
    stall_pricing: bool = True
    # Set by the pricing round when a job was bought but its log carried no
    # verdict (archive read short / tool never ran): the diagnosis prompt says
    # so once, in place of the table, instead of looking like pricing never ran.
    lever_price_lost: str = ""
    # Consecutive job-environment failures per harness channel (ncu / timeline /
    # stall-price / lever-price), see loop/job_env.py: a tool that cannot start on
    # the image is not a reading about the tree. Counted so the log says how long
    # a channel has been blind; cleared by that channel's next good reading.
    env_broken: dict = field(default_factory=dict)
    # Set when a reset threw away teacher edits outside notes/ (a pricing round,
    # a verdict probe): the next diagnosis prompt says so once, so a teacher that
    # "fixed" a job-environment file in passing does not believe it landed.
    discarded_edits: str = ""
    # Per-iteration orchestration census: the NCU job of every evaluated
    # iteration also renders the tree's SASS contract axes (orch_census_run,
    # vendored with the pricing toolkit), recorded as IterationRecord.orch_summary.
    # A KEEP whose axes differ from the previous KEEP's is billed for the KB
    # harvest (kb_axis_debt) — the harvest channel cheap wins never had (the
    # cp.async cache-mode line shipped twice at ~4 % and was priced at 22 % later;
    # only probe fights and stuck diagnoses used to reach the harvest). Rides the
    # stall_pricing seeding; off per task via loop.toml [loop] orch_census = false.
    orch_census: bool = True
    # Single-ref contract: whether the declared reference rides the student eval job
    # (bench times it back-to-back with the student — legitimate only when the
    # comparator IS this run's ref). False when the ref is measured through a
    # separate submission tree (ref_mode tasks — blackbox runs where the ref package
    # is readable source): the bench must then time ONLY the student, and any in-job
    # comparator reading is a smuggled second reference (the retired "instrument"
    # concept) — dropped loudly, never recorded, never gated on.
    ref_rides_job: bool = True
    # Where the ref-side and student-side pricing evidence (stall ledger,
    # segment account, orchestration dossier, role exposure) is banked. A
    # blackbox run banks it beside the recon report; a source-mode run — which
    # never reconnoitres and used to have nowhere to put it, so its role and
    # dossier tables were one-sided forever — under operators/<op>/evidence/.
    evidence_dir: Path | None = None
    # Source-mode runs: a teacher-side tree (its own git repo, carrying eval/ncu_run.py
    # like the student's) the driver prices ONCE with the same pricing recipe to
    # produce the ref side of those tables. loop.toml [loop] ref_price_op.
    ref_price_dir: Path | None = None
    ref_price_backend: "DriverBackend | None" = None
    # The launch target stall_price_run.py opens the profiler window around in
    # that tree, when it is not the student contract (eval/ncu_run.py): loop.toml
    # [loop] ref_price_app = ["python3", "dsa_ref.py", "ncu"]. None = default.
    ref_price_app: list | None = None
    # PM-sampling interval (cycles) for the frozen pricing recipe on BOTH trees:
    # 8192 resolves phase structure on ~ms kernels; a sub-0.5 ms kernel starves
    # at it (too few samples -> PMTL-FAIL by construction) and needs 2048.
    # 0 = pick from the run's target reading. loop.toml [loop] ref_price_pm_interval.
    ref_price_pm_interval: int = 0
    ref_price_submit_failures: int = 0   # transient BackendErrors this session (not attempts)
    # the same, per non-representative workload of a group card (_ref_workload_pricing)
    ref_wl_price_submit_failures: dict = field(default_factory=dict)
    ref_price_note: str = ""             # this session's transient-failure note (markers carry the rest)
    # Probe files an iteration's reset would otherwise erase (probe-blocked /
    # probe-undelivered): re-written into the workspace when the next student
    # iteration starts — the meta step's own verdict-probe commit+reset would
    # sweep an untracked file away in between.
    kept_probes: dict = field(default_factory=dict)
    # ref_in_job: the same-job residual the last paired re-anchor measured for
    # the current best (best_ms, residual) — the residual the ledger gate must
    # judge on, not the cross-node absolute gap.
    paired_residual_memo: "tuple[float, float] | None" = None
    # The comparator the bench times INSIDE the student job is the declared
    # reference itself (same build, same image) — not a stand-in. The separate
    # ref tree still measures the launch baseline, the NCU signature and the
    # reconnaissance; but the same-job reading is the one verdict currency that
    # is node- and period-matched, so with this on the single-ref contract keeps
    # it and the loop judges parity by it (loop.toml [loop] ref_in_job).
    ref_in_job: bool = False
    # --equivalence-ratio: a same-job ratio (ref / student) at or above it is parity.
    equivalence_ratio: float = 1.0
    # shas whose same-job parity was re-measured and NOT confirmed: never re-bought
    # (sha -> account, with the cluster it was measured on).
    paired_memo: dict = field(default_factory=dict)
    # the account of a same-job parity stop: which sha stopped the run and the
    # samples already bought — the finish confirmation confirms THAT sha and
    # counts these samples instead of re-measuring the absolute best.
    paired_stop: dict | None = None
    # A KEEP that beats the running best by more than this fraction is re-measured
    # before its reading becomes the watermark (0 disables). One eval job is one
    # node draw: a single-job outlier that takes the watermark poisons every later
    # meta transaction — one fwd run's 0.2585 (re-measured 0.282, +9%) held
    # best for 9h and had two rebuilds judged "no gain" against a number that was
    # never real. Within-noise improvements (the common case) cost nothing extra.
    watermark_margin: float = 0.03
    # Machine awareness. Records are stamped with the machine their eval ran on
    # (backend.last_cluster), and every absolute-ms comparison — running best,
    # plateau, patch judgment, the target stop — reads only rows from the machine
    # submissions currently go to: one unchanged kernel's absolute latency moves
    # more between machines than most single-round gains (the same DSA kernel:
    # 2.1441 ms on one, 2.234 ms on another), so a mixed ledger reads a slower
    # machine as a REVERT streak and a faster one as a new best. home_cluster is
    # the machine the run started on; unstamped rows are taken to be from it. ""
    # (the default) means everything is one machine. On another machine (a resume
    # elsewhere) the target is restated through the reanchor_ref channel below
    # (_cluster_scale).
    home_cluster: str = ""
    # Phantom-gap re-anchor (the ref-side mirror of watermark_margin, which only
    # ever re-measures the STUDENT): before a gap-driven meta diagnosis, a residual
    # within this fraction of target is checked against a fresh same-period ref
    # measurement. The stop target derives from a ref measured once at launch, and
    # the H100 pool is bimodal (~±6% absolute), so a residual inside the band is as
    # likely to be anchor drift as kernel deficit — a bwd-GQA run's meta
    # chased a +3.6% residual through a 20h structural rebuild; the finish confirm
    # measured the same sha at 1.0271x. 0 disables. loop.toml [loop] reanchor_band.
    reanchor_band: float = 0.06
    # The measurement channel for the re-anchor: returns (fresh gate-window ref
    # samples in ms, the cluster they ran on) — same path the finish confirmation
    # uses. Wired by the CLI from the task's declared ref; None on a target-ms-only
    # run (no ref to measure).
    reanchor_ref: "Callable[[], tuple[list[float], str]] | None" = None
    # Prompt note set when the fresh anchor CONFIRMS the gap. Read by run_two_layer
    # and passed into every run_meta_step of the SAME meta entry (a malformed-patch
    # retry re-runs the diagnosis and must see the same re-anchored residual, not
    # fall back to the stale one), then cleared.
    reanchor_note: str | None = None
    # The launch reading the transport pairs with: the fastest-of-N gate-window
    # sample of the SAME channel reanchor_ref measures (baseline_cli keeps the
    # fastest), plus the cluster it ran on. None/"" when the target was not derived
    # from a measured baseline (--target-ms, a toml constant): a number that was
    # never node-sampled has no launch-period node conditions to transport from.
    target_anchor_ms: float | None = None
    target_anchor_cluster: str = ""
    # (best at the last confirmed-gap check, its note): a re-check with an
    # unchanged best cannot change the answer except by further drift — reuse the
    # verdict instead of re-buying ref jobs at every subsequent plateau.
    reanchor_memo: "tuple[float, str] | None" = None


# --- prompt rendering -------------------------------------------------------

def _oracle_binding(ctx: LoopContext) -> str:
    """What the teacher is told about the reference implementation's source.

    Under "blackbox" the source is withheld on purpose, and saying so matters: a
    teacher told only that no oracle is "provided" reads that as a missing input and
    either stalls or invents one. It is not missing — the reference is a binary that
    can be run and profiled but not read, which is the situation this mode exists to
    reproduce, and the reconnaissance report is the input that replaces it.
    """
    if ctx.oracle_mode == "blackbox":
        return ("(withheld by design — this run distills WITHOUT reading reference "
                "source. The reference is a closed binary: you may run it, time it and "
                "profile it, but there is no source to read and none will be supplied. "
                "The reconnaissance report is the input that replaces it.)")
    if ctx.oracle_path:
        return str(ctx.oracle_path)
    return "(oracle not provided)"


def _arch_ref_binding(ctx: LoopContext) -> str:
    """The architecture reference, or why there isn't one.

    Absence has two very different causes and the teacher has to tell them apart. On an
    operator whose reference *is* the library — a GEMM measured against cuBLAS — the
    two references coincide and the library baseline is exactly what the run
    reproduces; template A.4 asks that the coincidence be stated rather than left to
    look like an omission. On an operator that has a separate reference which simply
    could not be measured, anchoring to the library instead understates the target.
    """
    if ctx.arch_baseline:
        return ctx.arch_baseline.render(ctx.gate_window)
    if ctx.oracle_mode == "blackbox":
        return ("(none declared, and correctly so: this run's reference IS the library "
                "baseline above — the same implementation is both the thing being "
                "reconnoitred and the thing being reproduced. Anchor target_ms and the "
                "performance ladder to it, and state in §0 that the architecture "
                "reference and the library baseline coincide here.)")
    return ("(no architecture reference declared — target falls back to the library "
            "baseline. If this operator's reference genuinely IS the library, say so "
            "in §0; otherwise the target understates what the milestone teaches.)")


def _verdict_probe_brief(ctx: LoopContext, name: str) -> str:
    """The verdict-probe protocol section, or nothing when the budget is zero.

    A zero-budget run must use the instructions it would have used before the
    mechanism existed (the control-run principle _distill_prompt_path states):
    leaving the section in with a "0 rounds" budget both changes the control
    prompt and invites PROBE_REQUEST turns that can only spin on the
    budget-spent path. The round count is substituted here rather than left to
    render_prompt because the section arrives as another binding's value, after
    the token pass may already have moved on.
    """
    if ctx.max_verdict_probe_rounds <= 0:
        return ""
    text = _meta_prompt(ctx, name).read_text(encoding="utf-8")
    return text.replace("{MAX_VERDICT_PROBE_ROUNDS}", str(ctx.max_verdict_probe_rounds))


def kb_read_dir(ctx: LoopContext) -> Path:
    """The one KB tree this run reads and writes: ``hw_kb/<language>/<kb_arch>/``.

    hw_kb/ is one tree per architecture with no shared root. A hardware recipe
    earned on one architecture is not evidence about another -- the MMA trunk,
    the register file it was budgeted against and the asynchrony it pipelines
    against all change -- so the run gets its own tree and nothing else: the
    architecture-neutral entries it needs are transcribed into that tree (with a
    ``Certification source`` line saying where their readings came from), which is what lets
    the read set be a single directory.

    Recipe lookup (``KB:<entry>`` citations), the three leak-lint gates (loop
    start, certify, harvest) and the quarantine all go through this function, so
    what the run may read and what is held to the leak rules are the same set by
    construction. ``ctx.kb_dir`` stays the root: ``{KB_DIR}/TEMPLATE.md`` and the
    harvest are per-KB, not per-architecture.
    """
    return Path(ctx.kb_dir) / ctx.kb_arch


def _meta_prompt(ctx: LoopContext, name: str) -> Path:
    """The prompt file ``name`` for this run's language mode (loop/lang_mode.py): the
    CUDA mode reads ``meta/cuda/<name>`` where that overlay exists; every other run,
    and every language-neutral prompt, reads ``meta/<name>`` exactly as before."""
    return lang_mode.prompt_path(ctx.meta_dir, name, ctx.language)


def _is_cuda(ctx: LoopContext) -> bool:
    """A CUDA-mode run: its student edits a .cu the card builds with nvcc."""
    return lang_mode.is_cuda(ctx.language) or lang_mode.is_cuda_kernel_file(ctx.kernel_file)


def _bindings(ctx: LoopContext) -> dict[str, str]:
    repo_root = ctx.meta_dir.parent
    return {
        "{MILESTONE_PATH}": str(ctx.milestone),
        "{MILESTONE}": str(ctx.milestone),
        "{KB_DIR}": str(ctx.kb_dir),
        "{KB_ARCH}": ctx.kb_arch,
        # The read set, rendered from the same function the lookup and the lint
        # gates use, so a prompt cannot name a tree the loop does not enforce.
        "{KB_READ_DIR}": str(kb_read_dir(ctx)),
        "{STATUS_JSONL}": str(ctx.status_path),
        "{TRAJECTORY_PATH}": str(ctx.status_path),
        "{WORKSPACE}": str(ctx.workspace),
        "{KERNEL_FILE}": ctx.kernel_file,
        "{LANGUAGE}": ctx.language,
        "{META_MILESTONE_PATH}": str(repo_root / "meta_milestone.md"),
        "{MILESTONES_TEMPLATE_PATH}": str(repo_root / "docs" / "milestones-template.md"),
        # Absolute on purpose: the teacher runs with cwd = the student workspace,
        # so a bare relative docs/ path would land the oracle-provenance archive
        # inside the tree the student reads and the kbprobe branch force-pushes.
        "{PROVENANCE_DIR}": str(lang_mode.provenance_dir(repo_root, ctx.language)),
        "{META_TRAJECTORY_PATH}": str(ctx.meta_status_path) if ctx.meta_status_path else "(none)",
        "{ORACLE}": _oracle_binding(ctx),
        "{RECON_REPORT}": (str(ctx.recon_path) if ctx.recon_path
                           else "(no reconnaissance report)"),
        "{CANDIDATES_PATH}": (str(ctx.recon_path.parent / "candidates.md")
                              if ctx.recon_path else "(no candidates file)"),
        "{RECON_DIR}": str(ctx.recon_dir) if ctx.recon_dir else "(no reconnaissance tree)",
        "{SIMULATOR_DIR}": str(ctx.simulator_dir) if ctx.simulator_dir else "(no simulator)",
        "{MAX_RECON_ROUNDS}": str(ctx.max_recon_rounds),
        "{VERDICT_PROBE_BRIEF}": _verdict_probe_brief(ctx, "verdict_probe_brief.md"),
        "{VERDICT_PROBE_BRIEF_AUDIT}": _verdict_probe_brief(
            ctx, "verdict_probe_brief_audit.md"),
        "{CANDIDATE}": ctx.candidate or "(not pinned — take the first in the candidates table)",
        "{REF_BASELINE}": (ctx.ref_baseline.render(ctx.gate_window) if ctx.ref_baseline
                           else "(library baseline not measured)"),
        "{ARCH_REF}": _arch_ref_binding(ctx),
        "{GATE_WINDOW}": ctx.gate_window,
        "{WINDOW_CONTRACT}": _window_contract_binding(ctx),
        # the run's configured equivalence ratio and the target the driver derived
        # from it (anchor / ratio): §0's target_ms must be THIS number, not the
        # 1.0x reading (a self-evolution run asks for >1)
        "{EQUIVALENCE_RATIO}": f"{ctx.equivalence_ratio:g}",
        "{TARGET_MS}": (f"{ctx.target_ms:.4f}" if ctx.target_ms else "(pending measurement)"),
    }


def _ref_for_window(ctx: LoopContext):
    """The reference whose per-kernel table the ledger is read against: the arch
    ref when it banked one, else the library baseline, else the arch ref."""
    if ctx.arch_baseline and ctx.arch_baseline.kernels:
        return ctx.arch_baseline
    return ctx.ref_baseline if ctx.ref_baseline else ctx.arch_baseline


def _window_contract_binding(ctx: LoopContext) -> str:
    """The timed-window ledger for the teacher: declared boundary (both
    directions), the reference's per-kernel table and the sitting best KEEP's,
    plus the credit or why there is none. Rendered from disk state every time it
    is bound, so a regex added to window.toml mid-run prices the next entry."""
    from forgekernel.window_contract import baseline_label, render_window_ledger
    ref = _ref_for_window(ctx)
    student = _best_keep_record(load_records(ctx.status_path), ctx.gate_window) \
        if ctx.status_path and Path(ctx.status_path).exists() else None
    return render_window_ledger(
        ctx.window_contract,
        ref_kernels=ref.kernels if ref else None,
        ref_ms=ref.device_anchor_ms if ref else None,
        ref_label=baseline_label(ref),
        ref_profiler=ref.ncu if ref else None,
        student_kernels=student.kernels if student else None,
        student_ms=gate_ms(student, ctx.gate_window) if student else None,
        student_label=f"best KEEP iter {student.iter}" if student else "",
        student_profiler=student.profiler if student else None,
        student_prep=student.prep if student else None)


def _window_student_note(ctx: LoopContext) -> str:
    """Standing student note: which extra work its window carries relative to
    the reference's (inputs it must rebuild, outputs only it produces), how the
    pricing reads its launches, and which inputs reach its timed call from the
    bench's prep step outside the window (and that the step is fenced).
    Measurement convenience must not steer the design: fusing a rebuild into
    the main kernel can be the real lever, so the note asks the student to SAY
    when it does, not to avoid it."""
    contract = ctx.window_contract
    if contract is None or not contract.declared:
        return ""
    from forgekernel.window_contract import (KIND_REF_INPUT, PAYER_STUDENT,
                                             window_findings)
    if window_findings(contract):
        return ""
    items = contract.paid_by(PAYER_STUDENT)
    prep = contract.prep_entries
    if not items and not prep:
        return ""
    lines = ["\n## Timed-window contract (window.toml)"]
    if items:
        lines.append("Compared with the reference's timed call, your timed call pays for the extra "
                     "work below (the reference gets it outside its window / does not have to "
                     "produce it):")
        for a in items:
            what = ("an input the reference gets outside its window; you must rebuild it inside yours"
                    if a.kind == KIND_REF_INPUT
                    else "an output the reference does not produce; you must produce it inside your window")
            how = f" (derived from {a.derived_from})" if a.derived_from else ""
            if a.student_kernel_regex:
                tag = (f"; the finish-time pricing deducts this part by matching kernel names "
                       f"against `{a.student_kernel_regex}`")
            else:
                tag = "; no pricing regex configured yet"
            lines.append(f"- {a.name}: {what}{how}{tag}")
        lines.append("This work is contract tax: it does not go away as the kernel gets better, and "
                     "it is not a reason to KEEP or REVERT. Whether it is a separate launch or fused "
                     "into the main kernel is decided by performance, not by the pricing — if you fuse "
                     "it (or rename the kernel), say so in the decision note: the pricing then stops "
                     "applying and the finish reports only the raw ratio.")
    if prep:
        lines.append("The inputs below are produced by the bench's prep step **outside** the timed "
                     "window and reach your timed call directly (symmetric with the reference's "
                     "autograd path, which also gets them for free from the forward):")
        for a in prep:
            spec = f"{tuple(a.shape) if a.shape else '?'} {a.dtype or '?'}"
            how = f", derived from {a.derived_from}" if a.derived_from else ""
            lines.append(f"- {a.name}: {spec}{how}{f'; {a.note}' if a.note else ''}")
        lines.append("The prep step may produce only the tensors above, matching the declaration in "
                     "name / shape / dtype item by item — producing more or fewer, or a mismatched "
                     "shape or dtype, REVERTs the round as a correctness failure; the prep time is "
                     "recorded for the teacher every round. Moving in-window work into prep does not "
                     "improve the ratio; it only exposes the work in the ledger.")
    lines.append("")
    return "\n".join(lines)


def _precision_reading(ctx: LoopContext, ev, iter_no: int) -> dict:
    """The precision gate (loop/precision_gate.py): summarize the eval JSON's
    ``precision`` block for the record, and under ``[loop] precision_contract = strict``
    turn a failed check into a correctness failure (``ev`` is changed in place).
    ``official`` records and logs only -- the bench's own verdict stays what the board
    would say."""
    contract = ctx.precision_contract
    summary = precision_gate.summarize(contract, ev.precision)
    if not summary:
        return summary
    if summary.get("status") == "unreported":
        if contract == precision_gate.STRICT and ctx.precision_bench_reports:
            if ev.correctness:
                ev.correctness = False
                ev.error = precision_gate.MISSING_BLOCK
            return summary
        if not ctx.precision_unreported_warned:
            ctx.precision_unreported_warned = True
            _log(f"iter {iter_no}: WARNING precision ({contract}): the eval JSON carries no "
                 f"`precision` block — the bench does not report one (copy "
                 f"eval/precision.py from src/forgekernel/card_template and call PrecisionRecorder; a "
                 f"leaderboard card needs the current eval/leaderboard.py), so the "
                 f"precision_contract cannot be applied; rows record `unreported`")
        return summary
    if (why := precision_gate.gate_failure(contract, summary)) and ev.correctness:
        ev.correctness = False
        ev.error = why
    elif summary.get("status") == "fail":
        how = ("the eval already failed correctness" if why else "recorded only")
        _log(f"iter {iter_no}: precision ({contract}, {how}): "
             f"{precision_gate.describe(summary)}")
    return summary


def _sass_precision(ctx: LoopContext, student: "dict | None", iter_no: int) -> dict:
    """The SASS cross-check of the precision gate (warning only): the census's MMA
    operand / accumulator types against the reference dossier's and the card's input
    dtypes (loop/precision_gate.sass_findings). Logged when it finds a drop; {} when the
    census carried no MMA reading."""
    try:
        if not student or ctx.precision_contract == "off":
            return {}
        cache = ctx.__dict__.setdefault("_sass_precision_ref", {})
        if "ref" not in cache:
            cache["ref"] = ((_orch_summary(_read_ref_orch_dossier(ctx)) or {})
                            .get("mma_precision"))
            dtypes = []
            try:
                prob = json.loads((Path(ctx.workspace) / precision_gate.LEADERBOARD_PROBLEM)
                                  .read_text(encoding="utf-8"))
                dtypes = [str(v.get("dtype")) for v in
                          (prob.get("definition") or {}).get("inputs", {}).values()]
            except (OSError, ValueError, AttributeError):
                pass
            cache["dtypes"] = dtypes
        out = precision_gate.sass_findings(student, cache["ref"], cache["dtypes"])
    except Exception as e:  # noqa: BLE001 — a cross-check, never a reason to stop
        _log(f"iter {iter_no}: sass precision check skipped ({type(e).__name__}: {e})")
        return {}
    for w in out.get("warnings") or ():
        _log(f"iter {iter_no}: WARNING SASS precision: {w} (census {out['student']['ops']}; "
             f"recorded only — the precision gate reads the error ratio)")
    return out


def _enforce_prep_contract(ctx: LoopContext, ev):
    """Hold the bench's prep report to window.toml; a mismatch is a REVERT.

    The prep step is the one place work legitimately leaves the timed window,
    and the declaration (symmetrize entries with shape / dtype) is what bounds
    it. The bench ships to the job without window.toml, so the comparison is
    made here, on the eval JSON, where the student cannot edit it away: an
    undeclared extra tensor, a missing one, a wrong shape or dtype turns the
    eval into a correctness FAIL with the mismatch as its error text.
    """
    contract = ctx.window_contract
    if not ev.correctness or contract is None or not contract.declared:
        return ev
    from forgekernel.window_contract import check_prep_report, window_findings
    if not contract.prep_entries or window_findings(contract):
        return ev
    problems = check_prep_report(contract, ev.prep)
    if not problems:
        return ev
    return replace(ev, correctness=False, error="prep contract: " + "; ".join(problems))


_PREP_ERROR_PREFIX = "correctness FAIL: prep contract: "


def _prep_structural_stop(records) -> str | None:
    """The inner loop's own stop reason for a prep-contract failure that is the
    bench's, not the kernel's: ``prep_unreported`` the first time the bench
    reports no prep block at all; ``prep_contract`` when the same mismatch
    (shape / dtype / undeclared tensor / mutation) repeats verbatim on two
    consecutive iterations — a student edit that did not change the verdict
    text did not touch the cause."""
    if not records:
        return None
    last = records[-1]
    if last.verdict != "REVERT" or not last.error.startswith(_PREP_ERROR_PREFIX):
        return None
    from forgekernel.window_contract import PREP_UNREPORTED
    if PREP_UNREPORTED in last.error:
        return "prep_unreported"
    if len(records) >= 2 and records[-2].verdict == "REVERT" and records[-2].error == last.error:
        return "prep_contract"
    return None


def _best_keep_record(records, gate_window: str):
    """The sitting best KEEP record (same identity rule as ``_best_keep_profiler``)."""
    kept = [r for r in records
            if r.verdict == "KEEP" and gate_ms(r, gate_window) is not None]
    return min(kept, key=lambda r: gate_ms(r, gate_window)) if kept else None


def render_prompt(template_path: Path, bindings: dict[str, str], extra: str = "") -> str:
    text = template_path.read_text(encoding="utf-8")
    for token, value in bindings.items():
        text = text.replace(token, value)
    return text + extra


def _student_prompt(ctx: LoopContext, iter_no: int, probe_feedback: str = "") -> str:
    extra = f"\n\n## Driver context\n- This is iteration {iter_no}.\n"
    if ctx.fresh_recipe:
        extra += (f"- A new KB recipe was just certified into the KB: `{ctx.fresh_recipe}`, for the "
                  f"hardware interface your last segment got stuck on. **Read and apply it first** "
                  f"(it is GPU-probe certified and is the direct way past your current blocker).\n")
    if ctx.reopen_note:
        extra += ctx.reopen_note
    if ctx.g5_note:
        extra += ctx.g5_note
    if ctx.phase_close:
        # Standing until the gate row lands: the phase's remaining levers were
        # priced null on the best tree, so walking them one iteration at a time
        # would only re-measure the table (a dsa_sparse run spent five
        # iterations and two meta steps doing exactly that).
        extra += ctx.phase_close["note"]
    if note := _price_revision_note(ctx):
        # Standing until the teacher's next entry prices a fresh table on the
        # current best; read from the evidence file, so a driver restart keeps it.
        extra += note
    if note := _timeline_student_note(ctx):
        extra += note
    if ctx.sass_serial_note:
        # Re-prompted verbatim, un-anchored: the compiler's whole-function verdict
        # persists across the change (it re-fires on every recompile), so it must
        # not be mistaken for yesterday's perf reading. Cleared only when a KEEP
        # prices pipelined again.
        extra += ctx.sass_serial_note
    if note := _placeholder_note(ctx):
        # Standing until every [PLACEHOLDER:] has its [RESTORED:] — the gate row is
        # refused before then, so the student sees the debt on every turn.
        extra += note
    if ctx.resume_note:
        # Standing for the whole post-patch segment (unlike host_bound_note): every
        # turn of the segment operates under the same rolled-back premise.
        extra += ctx.resume_note
    if ctx.host_bound_note:
        # One-shot: it describes the LAST evaluated round, so it must not survive
        # into a turn whose measurement it no longer describes.
        extra += ctx.host_bound_note
        ctx.host_bound_note = ""
    if note := _window_student_note(ctx):
        # Standing: the contract does not change between turns, and the rebuild
        # launches must stay recognisable for the whole run to be priced.
        extra += note
    if probe_feedback:
        extra += probe_feedback
    return render_prompt(_meta_prompt(ctx, "student_prompt.md"), _bindings(ctx), extra)


# --- inner loop -------------------------------------------------------------

# A gate closure row ("M0-gate") vs an ordinary decision ("D0.7"). Gate rows always get
# the full eval — they are where a scaffold phase's deferred verification is spent.
_GATE_ROW = re.compile(r"gate", re.IGNORECASE)
# The ref's per-segment construction table travels WHOLE into the distill prompt:
# a real table (five regions, seven segments, hand-offs, precision, warp rows) is
# ~9k characters; a 4k cut fell inside the decisive region's first hand-off line.
REF_STRUCT_PROMPT_CHARS = 12000


# --- placeholder ledger --------------------------------------------------------
#
# POLICY §3 ③ lets a decision land in a conservative form only as a failure
# response, and only on the promise that the real form comes back after first
# green. The promise had no keeper: scaffold rows carry no timing, the M0 gate
# sets no performance line, and "after first green" names no iteration — when
# every sub-step is scaffolded, the first green IS the gate row. A scalar
# stand-in for the one decision the milestone had spelled out in full crossed
# the gate as the phase's best, dragged the register split with it (spill,
# then whole-function wgmma serialization), and the audit hours later was the
# first thing to notice. The ledger makes the promise a debt the driver holds:
# the row that lands a placeholder declares it in its change text,
#     [PLACEHOLDER: D0.9 → <the form the milestone specifies>]
# the row that restores the real form declares
#     [RESTORED: D0.9]
# and a gate row is refused — not committed, not evaluated — while anything is
# open. The open set rides every student turn. REVERT rows are skipped: the
# tree they described was rolled back.
_PLACEHOLDER_TAG = re.compile(r"\[PLACEHOLDER:\s*([^\]]+)\]", re.I)
_RESTORED_TAG = re.compile(r"\[RESTORED:\s*([^\]]+)\]", re.I)
_DECISION_IDS = re.compile(r"D\d+\.\d+")
_TAG_ARROW = re.compile(r"→|->|=>")
# Prose that reads as a placeholder declaration without the tag. The row is
# bounced for the tag (or a rewording) before it is committed: the ledger never
# depends on the driver guessing which member of a package is the stand-in.
_PLACEHOLDER_PROSE = re.compile(
    r"first[ -]?green\s+form|sanctioned[^.\n]{0,40}?(?:interim|placeholder|stand-in|first[ -]?green)|"
    r"transitional\s+(?:form|implementation|version)|placeholder|"
    r"stand-in\s+(?:form|solve|solver|implementation)|"
    r"interim\s+(?:form|solve|solver|implementation|path|version)|"
    r"after\s+green[^.;\n]{0,24}?swap|swap(?:ped)?\s+back\b[^.;\n]{0,24}?(?:real\s+form|D\d+\.\d+)|"
    r"swap(?:ped)?\s+(?:back|in)\s+(?:after|once|when)\b|"
    r"(?:until|once|after)\s+(?:correctness\s+is\s+|first\s+)?green\b|replaces?\s+it\s+once", re.I)


# A mention of the ledger itself, or a denial, is not a declaration: "placeholder
# ledger cleared" on a gate row / "no placeholder" on any row must not be bounced
# for a tag.
_PLACEHOLDER_PROSE_NEG = re.compile(
    r"no\s+placeholders?|placeholder[- ]free|not\s+a\s+(?:placeholder|stand-in)|non-placeholder|"
    r"without\s+(?:a\s+|any\s+)?placeholders?|ledger", re.I)


def _placeholder_ids(tag_body: str, row_decision_id: str) -> list[str]:
    """Decision ids a `[PLACEHOLDER: …]` / `[RESTORED: …]` tag names — the part
    before the arrow; a tag naming none stands for every member of the row."""
    head = _TAG_ARROW.split(tag_body, maxsplit=1)[0]
    ids = _DECISION_IDS.findall(head)
    return ids or _DECISION_IDS.findall(row_decision_id or "")


def _lineage_roots(rows) -> set:
    """The tree(s) the run's first row(s) built on — the launch baseline. A chain
    that reaches one of these has been walked to its root; any other sha no row
    owns is a break in the chain (a driver-side commit that was not reset, a
    row whose sha never made it into the record)."""
    if not rows:
        return set()
    first = min(r.iter for r in rows)
    return {getattr(r, "base_sha", "") or "" for r in rows if r.iter == first} - {""}


def _lineage_rows(rows, head: str) -> list:
    """The landed rows on the ancestry of ``head`` (each row's ``base_sha`` is the
    tree it built on), in iteration order. A head no row owns — the baseline, a
    probe commit, a pre-tag record — falls back to every non-REVERT row, and so
    does a chain that BREAKS before reaching the run's root (a ``base_sha`` no
    row owns that is not the launch baseline): rows above the break may still
    be ancestors, so they are replayed. The ledger errs towards refusing a gate
    row, never towards passing one."""
    live = [r for r in rows if r.verdict != "REVERT"]
    by_sha = {r.sha: r for r in live if r.sha}
    if not head or head not in by_sha:
        return live
    chain, seen, cur = [], set(), head
    while cur and cur in by_sha and cur not in seen:
        seen.add(cur)
        chain.append(by_sha[cur])
        cur = getattr(by_sha[cur], "base_sha", "") or ""
    if cur and cur not in seen and cur not in _lineage_roots(rows):
        return live  # broken chain: the rows above the break are not provably dropped
    return sorted(chain, key=lambda r: r.iter)


def _placeholder_ledger(records, head: str = "") -> dict[str, str]:
    """Open placeholders after replaying the rows on ``head``'s lineage in order:
    decision id -> the declared target form ('' when the tag named none). A
    workspace reset (REVERT_PATCH, a kb certification, a REVERT) rewinds the
    lineage, and a restore that landed on a branch the reset dropped no longer
    closes anything — nor does a placeholder the reset dropped stay open."""
    open_: dict[str, str] = {}
    for r in _lineage_rows(list(records), head):
        change = r.change or ""
        for m in _PLACEHOLDER_TAG.finditer(change):
            parts = _TAG_ARROW.split(m.group(1), maxsplit=1)
            target = parts[1].strip() if len(parts) > 1 else ""
            for did in _placeholder_ids(m.group(1), r.decision_id):
                open_[did] = target
        for m in _RESTORED_TAG.finditer(change):
            for did in _placeholder_ids(m.group(1), r.decision_id):
                open_.pop(did, None)
    return open_


def _placeholder_items(open_: dict[str, str]) -> str:
    return "; ".join(d + (f" (→ {t})" if t else "") for d, t in open_.items())


def _workspace_head(ctx: LoopContext) -> str:
    try:
        return ctx.backend.head() or ""
    except Exception:  # noqa: BLE001 — lineage is bookkeeping, never a precondition
        return ""


def _placeholder_note(ctx: LoopContext) -> str:
    """Standing student note: the open placeholders, until the ledger is clear."""
    try:
        open_ = _placeholder_ledger(load_records(ctx.status_path), _workspace_head(ctx))
    except (OSError, ValueError):
        return ""
    if not open_:
        return ""
    return (f"- **Placeholder ledger not cleared (no gate row accepted)**: {_placeholder_items(open_)}. "
            f"These decisions are currently in a placeholder / conservative form; when you swap "
            f"in the real form the milestone specifies, submit it as a normal decision row whose "
            f"change text opens with `[RESTORED: {', '.join(open_)}]` (you may probe first). A gate "
            f"row can be submitted only once the ledger is clear.\n")


def _placeholder_bounce(ctx: LoopContext, turn: Record,
                        head: str = "") -> tuple[str, str, str] | None:
    """Why a record block is not accepted yet: ("gate-refused" | "untagged", log line,
    feedback for the re-prompt), or None when it may land. ``head`` is the tree the
    iteration builds on (captured before its probe commits), so the ledger is read
    on the lineage the gate row would close."""
    change = turn.change or ""
    if _GATE_ROW.search(turn.decision_id or ""):
        open_ = _placeholder_ledger(load_records(ctx.status_path), head)
        if open_:
            ids = ", ".join(open_)
            return (
                "gate-refused",
                f"gate row {turn.decision_id} REFUSED — open placeholder(s) "
                f"{_placeholder_items(open_)}; [RESTORED: {ids}] must land first",
                f"- **Gate row `{turn.decision_id}` not accepted**: placeholder ledger not cleared — "
                f"{_placeholder_items(open_)}. A placeholder form must not cross a gate (the build "
                f"phase has no performance reading and the gate sets no performance line, so it "
                f"would become this phase's \"best\" as is). First swap these decisions back to the "
                f"real form the milestone specifies, submitted as normal decision rows whose change "
                f"text opens with `[RESTORED: {ids}]` (you may probe the local contract first), then "
                f"submit the gate row on its own.\n")
        if (m := _PLACEHOLDER_TAG.search(change)):
            # the debt declared ON the gate row: a shape that once crossed an
            # M0 gate (a stand-in landed with the gate row itself)
            ids = ", ".join(_placeholder_ids(m.group(1), turn.decision_id)) or "?"
            return (
                "gate-refused",
                f"gate row {turn.decision_id} REFUSED — the row itself declares "
                f"[PLACEHOLDER: {m.group(1).strip()}]; a gate row cannot land a stand-in",
                f"- **Gate row `{turn.decision_id}` not accepted**: this row itself declares a "
                f"placeholder `[PLACEHOLDER: {m.group(1).strip()}]`. A gate row must not carry a "
                f"placeholder: first land {ids} in the real form the milestone specifies as a normal "
                f"decision row (or record it in the ledger and, once swapped back, "
                f"`[RESTORED: {ids}]`), then submit the gate row on its own.\n")
        if _RESTORED_TAG.search(change):
            return None
        # a gate row whose prose reads as a placeholder is bounced for the tag
        # like any other row (below) — it must not slip past the ledger untagged
    elif _PLACEHOLDER_TAG.search(change) or _RESTORED_TAG.search(change):
        return None
    if (m := _PLACEHOLDER_PROSE.search(change)):
        seen = change[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
        if _PLACEHOLDER_PROSE_NEG.search(seen):
            return None
        return (
            "untagged",
            f"{turn.decision_id}: change text reads as a placeholder declaration "
            f"(…{seen.strip()}…) without a [PLACEHOLDER: …] tag — bounced for the tag",
            f"- Your change text reads as landing a decision in a placeholder / transitional form "
            f"first (\"…{seen.strip()}…\"), but it carries no `[PLACEHOLDER: D<x>.<y> → <real form>]` "
            f"tag. A placeholder must be recorded in the ledger: put that tag in the first sentence "
            f"of the change (which decision, and what it will become), and before the gate row it "
            f"must be swapped back with `[RESTORED: D<x>.<y>]`; if this is not a placeholder, reword "
            f"it and resend the record block.\n")
    return None


def _best_ms(records, gate_window: str = GATE_WINDOW_DEFAULT) -> float | None:
    vals = [gate_ms(r, gate_window) for r in records
            if gate_ms(r, gate_window) is not None]
    return min(vals) if vals else None


def _gate_ratio(rec, gate_window: str = GATE_WINDOW_DEFAULT) -> float | None:
    """The record's same-job ratio (ref / student) in the gate window; None when
    the reference did not ride that job."""
    return rec.vs_ref_device if gate_window == GATE_WINDOW_DEVICE else rec.vs_ref


def _best_ratio(records, gate_window: str = GATE_WINDOW_DEFAULT) -> float | None:
    """The best same-job ratio among the measured KEEP rows, or None."""
    vals = [_gate_ratio(r, gate_window) for r in records
            if r.verdict == "KEEP" and r.correctness and _gate_ratio(r, gate_window)]
    return max(vals) if vals else None


# --- machine-aware comparisons ----------------------------------------------
#
# A run resumed on another machine is a measurement problem: absolute latency is a
# property of (kernel, machine), and the loop's verdicts are all absolute-ms
# comparisons. So: every record carries the machine ("cluster") its eval ran on,
# and the comparisons below read same-machine rows only. NCU kernel time is deliberately NOT partitioned — it replays at
# locked base clocks, so it is the one reading that travels.


def _active_cluster(ctx: LoopContext) -> str:
    """The cluster submissions currently resolve to (the last one the backend
    submitted to), falling back to home. "" on backends that cannot say."""
    return getattr(ctx.backend, "last_cluster", "") or ctx.home_cluster


def _rec_cluster(rec, ctx: LoopContext) -> str:
    return getattr(rec, "cluster", "") or ctx.home_cluster


def _on_cluster(records, ctx: LoopContext, cluster: str | None = None) -> list:
    """The rows measured on ``cluster`` (default: the active one). On one machine
    every row and the active cluster resolve to the same string — possibly "" —
    and this is the identity."""
    c = _active_cluster(ctx) if cluster is None else cluster
    return [r for r in records if _rec_cluster(r, ctx) == c]


def _best_good_ref_here(records, ctx: LoopContext) -> str | None:
    """Rollback checkpoint: the best KEEP on the active cluster, else the best
    anywhere — a checkpoint from another cluster is still a verified kernel,
    just not a comparable reading."""
    return (_best_good_ref(_on_cluster(records, ctx), ctx.gate_window)
            or _best_good_ref(records, ctx.gate_window))


_ANCHORS_FILENAME = "cluster_anchors.json"
_REANCHOR_ATTEMPTS = 2


def _home_ref_ms(ctx: LoopContext) -> float | None:
    """The launch reading of the declared reference, in the gate window — the
    same channel the in-job readings and reanchor_ref measure, so a foreign
    cluster's fresh reading divides by a like-for-like number. (The target may
    derive from the architecture ref instead; the cluster speed ratio is what
    is being transported, and one reference kernel measures it as well as
    another.) Falls back to the architecture ref when the run has no library
    baseline."""
    ref = ctx.ref_baseline or ctx.arch_baseline
    if ref is None:
        return None
    # The PLAIN device reading, not device_anchor_ms: the anchor prefers a
    # CUDA-graph replay when one exists, and neither the in-job ref reading
    # nor measure_ref_fresh measures that window — dividing a plain-window
    # numerator by a graph-window denominator would loosen the scaled target
    # by the host tax, inside the noise band, and hand out stop=target for
    # a mediocre KEEP. Same field the phantom-gap anchor uses.
    return ref.ms_device if ctx.gate_window == "device" else ref.ms


def _anchors_path(ctx: LoopContext) -> Path:
    return Path(ctx.status_path).parent / _ANCHORS_FILENAME


def _load_anchors(ctx: LoopContext) -> dict:
    try:
        return json.loads(_anchors_path(ctx).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _target_workloads_mismatch(ctx: LoopContext, records=None) -> str | None:
    """Why this run's gate readings cannot be held against its absolute target.

    The target, the cluster scale and the phantom-gap transport divide a gate
    reading by an archived reference reading (architecture / library baseline).
    On a multi-workload card the gate reading is the geomean over the bench's
    workloads (timing.workloads_performance); an archive read over another input
    set is a different quantity, and the "gap" is then the workloads' cost mix,
    not the kernel, in either direction. The CLI refuses such a fresh start
    (_target_workloads_problems); this is the runtime half — resumed runs, and
    the workload NAMES only an eval reveals. None when comparable or before
    the first KEEP.
    """
    records = load_records(ctx.status_path) if records is None else records
    best = _best_keep_record(records, ctx.gate_window)
    if best is None:
        return None
    names = [str(w.get("name")) for w in best.workloads or [] if isinstance(w, dict)]
    for ref in (ctx.arch_baseline, ctx.ref_baseline):
        if ref is None:
            continue
        why = workloads_mismatch(getattr(ref, "n_workloads", 1), getattr(ref, "workloads", []),
                                 best.n_workloads, names)
        if why:
            return f"{getattr(ref, 'role', 'reference')} baseline: {why}"
    return None


def _say_target_suspended(ctx: LoopContext, where: str, why: str) -> None:
    """Log the absolute-target suspension once per distinct reason."""
    if getattr(ctx, "_target_wl_warned", None) == why:
        return
    ctx._target_wl_warned = why
    _log(f"{where}: absolute target suspended — {why}. No stop, re-anchor or residual "
         f"is read off it until the reference is re-archived as the geomean over the "
         f"bench's workloads; same-job ratios are unaffected")


def _cluster_scale(ctx: LoopContext) -> float | None:
    """What the target must be multiplied by on the active cluster; 1.0 at
    home; None when the reference cannot be anchored there (the target stop is
    then suspended on that cluster rather than judged against home's number).

    Three sources, cheapest first — every numerator is a MIN, because the
    launch reading it divides by is the fastest of its samples (baseline_cli
    keeps the fastest): a median numerator over a min denominator reads the
    median-vs-min offset (~2% on quiet nodes) as cluster slowness and loosens
    the target by exactly that, inside the noise band.
      1. same-job reference readings already recorded on this cluster (a run
         whose ref rides the job pays nothing: the ratio travels by
         construction, and the min firms up as rows accumulate);
      2. a fresh re-anchor persisted from earlier in this run
         (``cluster_anchors.json`` next to status.jsonl);
      3. one fresh measurement through ``ctx.reanchor_ref`` (the channel the
         phantom-gap re-anchor also uses; it follows the student's stay) —
         GPU jobs, so it runs at most _REANCHOR_ATTEMPTS times per cluster per
         run, under the same admission as the phantom-gap check (>= 2 usable
         samples, spread <= 10%, and the samples must have landed on THIS
         cluster). Min against min: the launch reading it is paired with is
         the fastest of its samples (baseline_cli keeps the fastest).
    """
    cluster = _active_cluster(ctx)
    if not cluster or cluster == ctx.home_cluster:
        return 1.0
    home = _home_ref_ms(ctx)
    if not home:
        return None
    key = "ref_device_ms" if ctx.gate_window == "device" else "ref_median_ms"
    seen = [getattr(r, key) for r in
            _on_cluster(load_records(ctx.status_path), ctx, cluster)
            if getattr(r, key, None)]
    if seen:
        return min(seen) / home
    anchors = _load_anchors(ctx)
    entry = anchors.get(cluster) or {}
    if entry.get("ref_ms"):
        return float(entry["ref_ms"]) / home
    attempts = int(entry.get("attempts", 0))
    if ctx.reanchor_ref is None or attempts >= _REANCHOR_ATTEMPTS:
        return None
    _log(f"cluster {cluster!r}: no reference anchor yet — re-measuring the "
         f"reference here (attempt {attempts + 1}/{_REANCHOR_ATTEMPTS})")
    fresh = None
    try:
        samples, ran_on = ctx.reanchor_ref()
        samples = [s for s in samples if s and s > 0]
        if len(samples) < 2:
            _log(f"cluster {cluster!r}: re-anchor admission failed — "
                 f"{len(samples)} usable sample(s), need >= 2")
        elif (max(samples) - min(samples)) / min(samples) > 0.10:
            _log(f"cluster {cluster!r}: re-anchor admission failed — spread "
                 f"{(max(samples) - min(samples)) / min(samples):.1%} > 10%")
        elif ran_on and ran_on != cluster:
            _log(f"cluster {cluster!r}: re-anchor samples ran on {ran_on!r} — "
                 f"the reference channel did not follow the stay; unusable here")
        else:
            fresh = min(samples)
    except Exception as e:  # a failed anchor must not take the loop down
        _log(f"cluster {cluster!r}: reference re-anchor failed ({e})")
    entry = {"attempts": attempts + 1, "home_ref_ms": home,
             "window": ctx.gate_window, "ref_ms": fresh}
    anchors[cluster] = entry
    try:
        _anchors_path(ctx).write_text(json.dumps(anchors, indent=1), encoding="utf-8")
    except OSError:
        pass
    if not fresh:
        _log(f"cluster {cluster!r}: reference re-anchor yielded no reading — the "
             f"target stop is suspended on this cluster")
        return None
    _log(f"cluster {cluster!r}: reference re-anchored at {fresh} ms "
         f"({ctx.gate_window} window) vs {home} ms at home — the target scales "
         f"x{fresh / home:.4f} while the run is here")
    return fresh / home


def _log(msg: str) -> None:
    print(f"[loop] {msg}", flush=True)


def _enforce_single_ref(ctx: LoopContext, ev):
    """Drop in-job reference readings on runs whose ref does not ride the job.

    On such runs (ref measured through a separate submission tree) any comparator
    the bench timed in-job is by construction NOT the ref — a stand-in library kept
    "to calibrate node speed". That concept is retired: it made every vs_* number
    ambiguous (one fwd run's same-job 1.0418x read as "beat FA" when the
    denominator was cuDNN). Dropped here, at the single entry point where eval logs
    become records, so no verdict, gate, digest, or report downstream ever sees a
    second reference. Loud once per run — a bench still timing a stand-in is a
    config leftover the task author should remove.
    """
    if ctx.ref_rides_job or ctx.ref_in_job or not any(
            v is not None for v in (ev.ref_median_ms, ev.vs_ref,
                                    ev.ref_device_ms, ev.vs_ref_device)):
        # ref_in_job: the in-job comparator IS the declared reference (the
        # image-baked library the ref tree also measures). Dropping it here
        # threw away the only node-matched reading the run had — a
        # grouped-GEMM run chased a +2.3% gap against anchors read on other nodes while
        # its own jobs read the student at parity with deep_gemm on the same
        # card, every iteration.
        return ev
    if not getattr(ctx, "_second_ref_warned", False):
        _log("WARNING single-ref contract: this run's ref is measured out-of-job, but "
             "the bench timed an in-job comparator — dropping those readings (remove "
             "the stand-in library from eval/bench.py)")
        ctx._second_ref_warned = True
    return replace(ev, ref_median_ms=None, vs_ref=None,
                   ref_device_ms=None, vs_ref_device=None)


# NCU replays with clocks locked to base frequency, so its per-kernel time is
# node-invariant where the bench reading is not. A new watermark whose NCU time
# REGRESSED past this band against the sitting best is a worse kernel on a fast
# node — the free cross-signal that convicts it sits in the same record.
_NCU_REGRESSION_BAND = 1.05


def _best_keep_ncu_ns(records, gate_window: str) -> float | None:
    """NCU kernel time of the sitting best KEEP, if it was profiled."""
    kept = [r for r in records
            if r.verdict == "KEEP" and gate_ms(r, gate_window) is not None]
    if not kept:
        return None
    t = min(kept, key=lambda r: gate_ms(r, gate_window)).profiler.get("time_ns")
    return float(t) if t else None


def _best_keep_profiler(records, gate_window: str) -> tuple[dict, int] | None:
    """The sitting best KEEP's NCU profiler dict + its iter, or None if IT lacks one.

    Strict on identity, like ``_best_keep_ncu_ns`` above: NCU is best-effort, so
    the best KEEP can land with ``profiler={}`` (retries exhausted). Falling back
    to an older profiled KEEP would reconcile a kernel that no longer exists —
    and could stamp OVERLAP-BOUND on it — so the answer is None, not a substitute.
    """
    kept = [r for r in records
            if r.verdict == "KEEP" and gate_ms(r, gate_window) is not None]
    if not kept:
        return None
    best = min(kept, key=lambda r: gate_ms(r, gate_window))
    return (best.profiler, best.iter) if best.profiler else None


def _confirm_suspicious_watermark(ctx: LoopContext, be, branch: str, ev, metric_before,
                                  profiler: dict | None = None):
    """Re-measure a KEEP whose reading is about to take the watermark on suspect
    evidence, BEFORE it does.

    One eval job is one node draw. An outlier that takes the watermark is not one
    bad number — it poisons every later meta transaction: rebuilds get attributed
    "no gain" against a best that was never real, and equal-or-better segments
    cannot accumulate. Within-noise improvements — the common case — trigger
    nothing and cost nothing.

    Two triggers:
    * the reading beats the best by more than the noise floor (a jump this size
      is exactly what one lucky node draw can fake), or
    * the reading takes the watermark while the record's own NCU kernel time
      REGRESSED past the sitting best's by more than _NCU_REGRESSION_BAND — a
      worse kernel on a fast node. This one costs nothing to detect and catches
      fakes whose device-reading jump stays under the margin.

    Protocol: one confirmation job; agreement within the margin confirms the
    jump, and the watermark takes the median of the two draws — not the first,
    faster one: keeping it let a reading 1.5% fast stand as the bar every later
    patch was priced against — except under the NCU trigger, where
    ms agreement alone proves nothing (see below) and the branch is re-profiled
    to clear the kernel clock too. Disagreement beyond it gets a third job and
    the median of the three becomes the gate-window reading (the other window's
    fields keep the first job's values — the watermark only reads the gate
    window). A failed FIRST confirmation keeps the original reading: a flaky
    queue must not veto a measured KEEP. But once a dissenting confirmation is
    in hand, that excuse is spent — a failed tie-break settles on the median of
    what was measured rather than re-instating the unconfirmed jump.
    """
    margin = ctx.watermark_margin
    ms = gate_ms(ev, ctx.gate_window)
    if not margin or metric_before is None or ms is None or ms >= metric_before:
        return ev
    why = None
    ncu_suspect_ns = None  # set iff the NCU-regression trigger raised the alarm
    if ms < metric_before * (1 - margin):
        why = (f"beats best {metric_before} by more than {margin:.0%}")
    else:
        new_t = (profiler or {}).get("time_ns")
        best_t = _best_keep_ncu_ns(load_records(ctx.status_path), ctx.gate_window)
        if new_t and best_t and float(new_t) > best_t * _NCU_REGRESSION_BAND:
            why = (f"takes the watermark while its NCU kernel time regressed "
                   f"{float(new_t) / best_t:.3f}x vs the sitting best "
                   f"({new_t} vs {best_t:.0f} ns) — worse kernel on a fast node?")
            ncu_suspect_ns = best_t
    if why is None:
        return ev
    _log(f"watermark: {ms} {why} — confirming before it takes the watermark")
    key = "device_median_ms" if ctx.gate_window == "device" else "bwd_median_ms"
    readings = [ms]

    def _confirmed():
        both = statistics.median(readings)
        if both != ms:
            _log(f"watermark: confirmed — recording median {both} of {readings}, "
                 f"not the first draw")
        return replace(ev, **{key: both})
    for _ in range(2):  # at most two extra jobs: confirm, then tie-break
        try:
            ev_n = _enforce_single_ref(ctx, parse_eval(be.submit_gpu(branch, "all")))
            ms_n = gate_ms(ev_n, ctx.gate_window) if ev_n.correctness else None
        except BackendError as e:
            ms_n, e_txt = None, str(e)
        else:
            e_txt = None
        if ms_n is None:
            what = f"failed ({e_txt})" if e_txt else "run unusable"
            if len(readings) > 1:
                # The dissenting confirmation already measured the jump as
                # unconfirmed; losing the tie-break job must not hand the
                # watermark back to the outlier.
                _log(f"watermark: tie-break {what} — settling on what was measured")
                break
            _log(f"watermark: confirmation {what} — keeping the single reading")
            return ev
        readings.append(ms_n)
        _log(f"watermark: confirmation job {getattr(be, 'last_job_id', '') or '?'} "
             f"read {ms_n}")
        if len(readings) == 2 and abs(ms_n - ms) <= margin * ms:
            if ncu_suspect_ns is None:
                return _confirmed()  # two independent draws agree — the jump is real
            # ms agreement cannot clear the NCU trigger: on a bimodal queue the
            # confirmation draws from the same fast-node pool the suspect did,
            # so band-consistent readings are near coin-flip likely even when
            # the kernel really regressed. The alarm came from the kernel
            # clock; only the kernel clock clears it.
            try:
                t_n = _ncu_profiler(ctx, branch, correctness=True).get("time_ns")
            except BackendError:
                t_n = None
            if t_n and float(t_n) <= ncu_suspect_ns * _NCU_REGRESSION_BAND:
                return _confirmed()  # kernel time recovered too — the jump is real
            _log(f"watermark: ms agrees but the re-profiled kernel time "
                 f"({t_n or 'unavailable'} ns) does not clear the regression vs "
                 f"{ncu_suspect_ns:.0f} ns — tie-breaking")
    final = statistics.median(readings)
    _log(f"watermark: recording median {final} of {readings} — the single-job jump "
         f"did not confirm")
    return replace(ev, **{key: final})


_PROBE_LOG_TAIL_LINES = 25
_PROBE_LOG_TAIL_CHARS = 2000


def _probe_label(pv: ProbeVerdict) -> str:
    """How a verdict reads in the loop log. NO VERDICT is its own state on purpose:
    it means the job never got far enough to judge anything."""
    if pv.passed:
        return "PASS"
    return "FAIL" if pv.found else "NO VERDICT"


def _probe_feedback(path: str, pv: ProbeVerdict, log: str, *, what: str = "probe",
                    on_pass: str = "") -> str:
    """One probe round's feedback, split by failure mode.

    Every non-PASS used to read as "PROBE FAIL (max_diff=None) — iterate the probe",
    which sent the agent looking for a numerical bug when the job had in fact failed
    to build, or when the driver had rejected its PASS. Each of those now says so.
    """
    # Every branch carries a bounded log tail. The FAIL branch used to return the
    # verdict counts alone, which silently discarded whatever diagnostic payload the
    # probe had printed (sanitizer reports, shape dumps, per-rung readings) in exactly
    # the case the agent needed it — a probe's stdout is its instrument, not noise.
    # The PASS branch then did the same for longer: a probe is told to put its
    # candidate sweep on the FACT channel and keep the verdict lines to contracts
    # (gpu_probe_ladder), so a PASS is the round with the MOST payload — and it came
    # back as one line. One fp4 probe swept six relaunch variants, named the one
    # that survives, PASSed, and the student re-derived that answer eight hours later.
    tail = "\n".join(log.strip().splitlines()[-_PROBE_LOG_TAIL_LINES:])
    tail = tail[-_PROBE_LOG_TAIL_CHARS:]
    if pv.passed:
        return (f"- Your {what} {path} returned PROBE PASS (max_diff={pv.diff_text}). "
                f"{on_pass} Tail of the job log (the FACTS / readings your probe "
                f"printed live here and are not re-sent — carry them into the "
                f"decision):\n{tail}\n")
    if not pv.found:
        return (f"- Your {what} {path} produced NO verdict: {pv.reason}. This is NOT a "
                f"numerical failure — do not go hunting for one. Make the probe run and "
                f"print exactly one PROBE PASS/FAIL line per contract it checks. Tail of "
                f"the job log:\n{tail}\n")
    # A multi-contract probe names the contracts that failed: "2 of 5 reported
    # FAIL" sent the agent re-reading all five when the log already said which two.
    table = f"Per-contract verdicts:\n{pv.contract_table()}\n" if pv.summary else ""
    if pv.reason:
        return (f"- Your {what} {path} did NOT certify — the driver rejected the verdict: "
                f"{pv.reason}. Fix this and re-emit PROBE_REQUEST. {table}Tail of the job "
                f"log:\n{tail}\n")
    return (f"- Your {what} {path} returned PROBE FAIL (max_diff={pv.diff_text}). "
            f"Iterate the probe. {table}Tail of the job log (your probe's own diagnostics "
            f"live here):\n{tail}\n")


def _probe_log_suffix(pv: ProbeVerdict) -> str:
    """What follows the verdict in a loop-log line: the per-contract tally when the
    probe printed several lines (it names the failing contracts and their readings),
    and the driver's reason when it is not already told by that tally.

    The tally replaces the reason only when a contract FAILed — the reason then is
    the generic "k of n lines reported FAIL". An all-PASS probe the audit rejected
    has a tally of "n/n PASS", which next to a FAIL label explains nothing: the
    audit's reason (ceiling, unparseable, missing max_diff) must stay on the line.
    """
    a_contract_failed = pv.n_pass < len(pv.contracts)
    parts = [pv.summary] if pv.summary else []
    if pv.reason and not (pv.summary and a_contract_failed):
        parts.append(pv.reason)
    return "".join(f" — {p}" for p in parts)


# A verdict probe's payload is its printed readings (tables, per-variant ratios), not
# the PASS bit — so the feedback carries a much longer log tail than the certification
# probes, whose only question is "did it certify".
_VERDICT_PROBE_TAIL_LINES = 150
_VERDICT_PROBE_TAIL_CHARS = 8000


def _run_ablation_probe(ctx: LoopContext, be, path: str, probe_branch: str):
    """Run a JSON ablation spec (``notes/<x>.json``: ``ablations`` only) through
    the frozen pricing recipe on the tree as it stands. (rows, why, log): the
    ablation rows of the table (decisions attributed), the reason there are
    none, the job log.

    Any ``levers`` in the file are ignored: candidates are priced at the
    design turn on the best tree, a probe answers a cause question on the
    current one. The spec is normalised in place (kernel_file, rounds,
    roles off) and committed with the toolkit, like a script probe."""
    spec_path = Path(ctx.workspace) / path
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [], f"ablation spec unreadable ({e})", ""
    if not isinstance(raw, dict):
        return [], "ablation spec is not a JSON object", ""
    known_ids = {d.id for d in _milestone_dag(ctx)}
    ablations = _normalise_ablations(ctx, raw.get("ablations") or [], known_ids)
    if not ablations:
        return [], "the spec names no `ablations`", ""
    spec = {"ablations": ablations, "levers": [], "roles": False,
            "ablation_rounds": int(raw.get("ablation_rounds") or 0) or None,
            "kernel_file": str(ctx.kernel_file)}
    if spec["ablation_rounds"] is None:
        del spec["ablation_rounds"]
    if fail := _lever_spec_refused(ctx, spec):
        return [], f"spec refused before the job — {fail}", ""
    try:
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
        _seed_pricing_toolkits(ctx.workspace, be)
        be.commit(f"meta-probe: {path}")
        be.push_branch(probe_branch)
        # a dirty node or a lost archive is bought once more (_lever_price_job); a group
        # card's probe runs on each of its meta workloads, merged
        log, _job = _lever_price_submit(ctx, be, probe_branch,
                                        f"cmd:python3 notes/lever_pricing_run.py {path}",
                                        f"meta: ablation probe {path}")
        rows, _stack, fail = parse_lever_prices(log)
    except BackendError as e:
        return [], f"job failed ({e})", ""
    if _price_log_verdict(log) is None:
        # a cut archive is not "no ABLATE line": the probe ran, its reading was
        # never read — the round says so instead of scoring the probe n/a
        return [], (f"the job log carries no PRICE/ABLATE verdict "
                    f"({_price_log_lost_reason(log)}) — the probe's reading was not read"), log
    rows = [r for r in rows if r.get("kind") == "ablation"]
    attribution = {a["name"]: a["decisions"] for a in ablations}
    for r in rows:
        r["decisions"] = list(attribution.get(r["name"], []))
        r["probe"] = path
    if fail:
        return rows, f"PRICE-FAIL {fail}", log
    if (afail := _ABLATE_FAIL_RE.search(log)):
        return rows, f"ABLATE-FAIL {afail.group(1).strip()}", log
    return rows, (None if rows else "no ABLATE line in the job log"), log


def _ablation_probe_feedback(path: str, rows: list, why: str | None, log: str,
                             round_no: int, budget: int) -> str:
    """One JSON ablation probe's feedback: the ablation table with its reading
    rules, or the reason there is none plus the log tail."""
    head = f"\n## Verdict experiment {round_no}/{budget}: `{path}` (ablation, driver-frozen recipe)\n"
    if rows:
        body = _ablation_block(rows)
        body += ("Citing it: write the ablation name + Δ in `evidence`; for R1 use "
                 "`reopen_price: <Δ> via ablation:<name>`"
                 f" (or `via probe:{path}#<name>`).\n")
        if why:
            body += f"(driver note: {why})\n"
        return head + body
    tail = "\n".join(log.strip().splitlines()[-_VERDICT_PROBE_TAIL_LINES:])[-_VERDICT_PROBE_TAIL_CHARS:]
    return (head + f"No ablation reading: {why}. This is not a numerical failure — fix the JSON "
            f"(every `ablations` item needs name / hypothesis / edits, its `old` must occur exactly "
            f"once in the kernel, and its name must not equal a lever's name) and send "
            f"PROBE_REQUEST again, or send a script probe instead. Job log tail:\n{tail}\n")


def _verdict_probe_feedback(path: str, pv: ProbeVerdict, log: str,
                            round_no: int, budget: int) -> str:
    """One diagnose-step verdict-probe round's feedback.

    Unlike a certification probe, FAIL here is not a defect in the probe — PASS/FAIL
    is the experiment's *answer* to a pre-registered hypothesis, and either answer is
    evidence. Only the no-verdict case is a protocol problem, and even then the log
    tail still flows back: a probe that crashed after printing half its table has
    still measured something.
    """
    tail = "\n".join(log.strip().splitlines()[-_VERDICT_PROBE_TAIL_LINES:])
    tail = tail[-_VERDICT_PROBE_TAIL_CHARS:]
    head = (f"\n## Verdict experiment {round_no}/{budget}: `{path}` — "
            f"PROBE {_probe_label(pv)} (max_diff={pv.diff_text})\n")
    if pv.summary:
        # Several pre-registered contracts: each is its own answer. The single
        # PASS/FAIL bit above is "all passed"; a control contract that could not
        # run must not read as the decisive hypothesis being falsified.
        head += (f"Per-contract verdicts ({pv.summary.split(';')[0]}; each row is one "
                 f"hypothesis's answer, readings recorded exactly as the script printed them):"
                 f"\n```\n{pv.contract_table()}\n```\n")
    if not pv.found:
        head += ("(The log has no `PROBE PASS/FAIL` verdict line — a verdict experiment must "
                 "pre-register a falsifiable criterion: print one line "
                 "`PROBE PASS/FAIL max_diff=<reading>` per hypothesis, so the script lands the "
                 "verdict in the log instead of you interpreting it afterwards. Fix this next "
                 "round.)\n")
    elif pv.reason:
        head += f"(the driver's audit of the verdict line: {pv.reason})\n"
    return head + f"\n### Job log tail (your experiment's readings)\n```\n{tail}\n```\n"


def _ncu_profiler(ctx: LoopContext, branch: str, *, correctness: bool) -> dict:
    """Submit the NCU job and parse the profiler, retrying when a PASSING kernel comes
    back empty (best-effort NCU failed on the GPU side). A correctness-broken kernel
    yields no NCU legitimately, so it is not retried. Neither is a launch target that
    cannot start on the image (loop/job_env.py): the retry would replay the same
    import error, and the empty digest is the job's blindness, not the kernel's."""
    mode = _ncu_mode(ctx)
    out = ctx.backend.submit_gpu(branch, mode)
    prof = profiler_from_ncu(_without_census_lines(out))
    tries = 0
    while (not prof and correctness and tries < ctx.ncu_retries
           and env_failure(out) is None):
        tries += 1
        _log(f"ncu empty on a passing kernel -> retry {tries}/{ctx.ncu_retries}")
        out = ctx.backend.submit_gpu(branch, mode)
        prof = profiler_from_ncu(_without_census_lines(out))
    if correctness:
        if prof:
            _note_env_ok(ctx, "ncu")
        elif (why := env_failure(out)) is not None:
            _note_env_failure(ctx, "ncu", why)
    # Operand-source census (opt-in per operator): when the op's ncu task ran
    # sass_census over the kernel dump, its fields join the digest so milestone
    # gate lines can anchor [VERIFY: sass-derived] the way they anchor bankconf_st.
    # Absent line -> absent fields; digests of census-less operators are unchanged.
    if prof:
        # The whole decomposition (every launch of the window, by share) rides
        # under a private key the iteration pops into IterationRecord.kernels —
        # the timed-window contract prices the rebuild launches off it.
        prof[_KERNELS_PROF_KEY] = kernels_from_ncu(_without_census_lines(out))
        prof.update(parse_census_line(out))
        # The compiler's wgmma verdict (orch_census_run prints WGMMA-SERIAL off
        # the same dump): waits per wgmma, plus whether the log carries ptxas's
        # own "serialized due to" advisory. Flat digest fields, so the NCU
        # thresholds flag ratio ~1 as the binding pathology of the round.
        prof.update(parse_serial_line(out))
        # The orchestration census (orch_census_run) rides the same log. Parked
        # under a private key the iteration pops into IterationRecord.orch_summary
        # — it must not join the flat profiler digest.
        if (summ := _orch_summary(out)) is not None:
            prof[_ORCH_PROF_KEY] = summ
        # The kernel's MMA operand / accumulator types (the precision gate's SASS
        # cross-check), printed even when the dump is too small for a dossier.
        if (m := _SASS_PRECISION_RE.search(out)) is not None:
            try:
                prof[_SASS_PRECISION_PROF_KEY] = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return prof


def _note_env_failure(ctx: LoopContext, channel: str, why: str) -> None:
    """Say — every time, with a running count — that a harness channel died on the
    job's environment rather than on the tree (loop/job_env.py). A fused-norm-rope
    run once went 2.5 days with its NCU, timeline and stall channels dead on two
    imports while every log line read like a transient or a tree failure."""
    counts = _env_counts(ctx)
    n = counts.get(channel, 0) + 1
    counts[channel] = n
    _log(f"ENV-BROKEN {channel} (x{n} in a row): {why} — this tool cannot start on the "
         f"job image, so the job is no reading about the tree and a retry replays it. "
         f"Fix the job environment: the card's gpu_entry.sh / eval/ (overlaid on every "
         f"job when listed in [loop] job_env_files) or the image")


def _note_env_ok(ctx: LoopContext, channel: str) -> None:
    if _env_counts(ctx).pop(channel, None):
        _log(f"ENV-OK {channel}: the channel reads again")


def _env_counts(ctx) -> dict:
    """ctx.env_broken, created on a context built without it (test doubles)."""
    counts = getattr(ctx, "env_broken", None)
    if counts is None:
        counts = {}
        try:
            ctx.env_broken = counts
        except AttributeError:
            pass
    return counts


def _edits_outside_notes(be, base: str) -> list:
    """Tracked paths outside notes/ whose content differs between ``base`` and the
    working tree — what a hard reset to ``base`` throws away that is not a probe
    script or a pricing spec. [] when the backend cannot say (test doubles) or git
    fails: this is a visibility aid, never a reason to stop a round."""
    lister = getattr(be, "changed_paths", None)
    if lister is None or not base:
        return []
    try:
        return sorted(p for p in lister(base) if p and not p.startswith("notes/"))
    except Exception:  # noqa: BLE001
        return []


def _note_discarded_edits(ctx: LoopContext, where: str, paths: list) -> None:
    """Log — and tell the next diagnosis once — that a reset is discarding edits
    outside notes/. The reset itself is right (the student line stays teacher-free);
    the silence was not: a teacher once fixed the job environment in a pricing
    turn, wrote itself a memory that it was fixed, and the fix was gone 21 minutes
    later without a line anywhere."""
    shown = ", ".join(paths[:8]) + (f" (+{len(paths) - 8} more)" if len(paths) > 8 else "")
    _log(f"{where}: discarding edits outside notes/ with the reset: {shown} — the student "
         f"line stays teacher-free, so these reach no job. A job-environment fix belongs in "
         f"the card and reaches every job through [loop] job_env_files")
    ctx.discarded_edits = (
        f"{where}: the reset discarded changes outside notes/: {shown}. The student lineage does "
        f"not take the teacher's changes, so these files reached **no** later job. If they are a "
        f"job-environment fix (gpu_entry.sh / eval/), make it in the card directory and list it "
        f"under [loop] job_env_files in loop.toml, so the driver overlays it on every job; if it "
        f"is a kernel change, write it into the milestone for the student to land.")


_ORCH_PROF_KEY = "_orch_summary"
_KERNELS_PROF_KEY = "_kernels"
_ORCH_CENSUS_CMD = "cmd:python3 notes/orch_census_run.py"
_ORCH_CENSUS_RUNNER = "orch_census_run.py"
# The census runner prints its lines AFTER the NCU CSV (the job log archive
# keeps only the tail). The CSV parser reads every line past the header as a
# row, and a JSON line has enough commas to land a phantom kernel in the digest.
_CENSUS_LINE_RE = re.compile(r"^(?:ORCH-CENSUS|ORCH-SUMMARY-JSON:|\[orch-census\]|WGMMA-SERIAL"
                             r"|SASS-PRECISION-JSON:).*$\n?", re.M)
_SASS_PRECISION_RE = re.compile(r"^SASS-PRECISION-JSON: (\{.*\})$", re.M)
_SASS_PRECISION_PROF_KEY = "_sass_precision"


def _without_census_lines(log: str) -> str:
    return _CENSUS_LINE_RE.sub("", log)


def _ncu_mode(ctx: LoopContext) -> str:
    """The per-iteration profile job: the op's plain ``ncu`` task, or the same
    task wrapped by the vendored orchestration census when the runner is in the
    workspace (seeded with the pricing toolkit) and the run has not opted out."""
    if getattr(ctx, "orch_census", False) and (
            Path(ctx.workspace) / "notes" / _ORCH_CENSUS_RUNNER).exists():
        if _is_cuda(ctx):
            # no JIT dump for a .cu: the runner reads the build of this kernel file
            return f"{_ORCH_CENSUS_CMD} --kernel {ctx.kernel_file}"
        return _ORCH_CENSUS_CMD
    return "ncu"


def _pop_orch_summary(prof: dict) -> dict:
    """Detach the census summary from a profiler dict (in place); {} when absent."""
    summ = prof.pop(_ORCH_PROF_KEY, None) if prof else None
    return summ if isinstance(summ, dict) else {}


def _pop_kernels(prof: dict) -> list:
    """Detach the per-kernel table from a profiler dict (in place); [] when absent."""
    rows = prof.pop(_KERNELS_PROF_KEY, None) if prof else None
    return rows if isinstance(rows, list) else []


def _parse_serial_census(prof: dict) -> dict:
    """The WGMMA serialization fields of a profiler dict (a view, not a pop).

    ``orch_census_run`` prints the ``WGMMA-SERIAL`` line off the same dump the
    orchestration census uses, so it rides the same job; ``_ncu_profiler`` folds
    the parsed fields into the flat digest, where ``wgmma_serial_ratio`` is read
    by the NCU thresholds and the trace columns like any other metric. This
    returns that subset for the student-facing verdict.
    """
    return {k: prof[k] for k in ("wgmma_census_kernel", "wgmma_total", "wgmma_depbar0",
                                 "wgmma_tight", "wgmma_serial_ratio",
                                 "wgmma_ptxas_serialized") if k in (prof or {})}


def _check_profile_covers_the_kernel(prof: dict, bwd_median_ms: float | None,
                                     ev=None) -> str | None:
    """Warn when profiled kernel time cannot account for the timed window.

    NCU reports whichever launches it captured. If the kernel under test never
    reached the CSV, what remains is some setup kernel — and nothing downstream can
    tell, because a profile of the wrong kernel looks exactly like a profile of a
    fast one. That happened for a whole 27-iteration run: the timed kernel went
    2871 ms -> 4 ms while the recorded profile sat at 117 us / 40 registers, and every
    NCU-anchored gate line was silently being compared against the preprocess pass.

    Two changes since, both aimed at the same false-positive:

    * the share is computed from ALL profiled kernels (``total_time_ns``), not the
      dominant one, so a legitimately decomposed operator no longer trips it;
    * a second, physically different cause of the same shortfall — the GPU sitting
      idle behind a host bottleneck — is separated out using the bench's enqueue
      clock, because "the profile is aimed wrong" and "the host is the pipeline"
      call for opposite fixes.

    Always a warning, never a stop: how many of the bench's calls the profiled
    launches correspond to is not recorded anywhere, so the share is a smell, not a
    measurement. 20% is deliberately loose.
    """
    from forgekernel.timing_audit import classify, ncu_reconciliation_warning
    verdict = classify(host_enqueue_ms=getattr(ev, "host_enqueue_ms", None),
                       device_median_ms=getattr(ev, "device_median_ms", None),
                       ratio=getattr(ev, "host_bound_ratio", None),
                       basis=getattr(ev, "device_window_basis", None))
    return ncu_reconciliation_warning(prof, bwd_median_ms, verdict) or None


def _check_host_bound(ctx: LoopContext, iter_no: int, ev) -> None:
    """Audit the device window's precondition on an evaluated round, both sides.

    The two sides need opposite handling, and conflating them is how a measurement
    defect becomes a lesson:

    * STUDENT host-bound is a real defect — per-call recompiles, python-side
      allocation, redundant launches, a synchronize inside the call. It inflates the
      student's own gate reading and the deployment pays it too, so it is logged AND
      handed to the student as the next round's diagnosis.
    * REFERENCE host-bound is a defect of the measurement. It is logged for the
      operator and deliberately NOT given to the student: "the reference's python
      wrapper is thick" reads to a kernel writer as "a thin python path wins", which
      teaches metric-gaming. The fix belongs to whoever archives the baseline —
      re-measure the reference through a host-free window.
    * Under a HOST-FREE basis (graph replay / CUPTI kernel span, the bench's
      device_window_basis) neither verdict applies: the reading cannot contain
      host time, so the enqueue clock is not divided into it. A thick student
      wrapper is then logged and handed over as a deployment NOTE — its own
      line, explicit that the gate is unaffected — instead of as a gate defect
      (a dsa_sparse card set host_bound_ratio to null for exactly this
      reason and the driver recomputed 3–6x from the raw timings anyway).
    """
    from forgekernel.timing_audit import (classify, host_path_feedback, host_path_note,
                                          student_feedback, student_warning)
    basis = getattr(ev, "device_window_basis", None)
    student = classify(host_enqueue_ms=getattr(ev, "host_enqueue_ms", None),
                       device_median_ms=getattr(ev, "device_median_ms", None),
                       ratio=getattr(ev, "host_bound_ratio", None), basis=basis)
    if student.host_free:
        ref_enqueue = getattr(ev, "ref_host_enqueue_ms", None)
        if note := host_path_note(ev.host_enqueue_ms, ev.device_median_ms, basis=basis,
                                  ref_host_enqueue_ms=ref_enqueue):
            _log(f"iter {iter_no}: NOTE {note}")
        ctx.host_bound_note = host_path_feedback(ev.host_enqueue_ms, ev.device_median_ms,
                                                basis=basis, ref_host_enqueue_ms=ref_enqueue)
    else:
        if warn := student_warning(student, device_ms=ev.device_median_ms):
            _log(f"iter {iter_no}: WARNING {warn}")
        ctx.host_bound_note = student_feedback(student, device_ms=ev.device_median_ms)

    ref = classify(ratio=getattr(ev, "ref_host_bound_ratio", None))
    if ref.host_bound:
        _log(f"iter {iter_no}: WARNING HOST-BOUND IN-JOB REFERENCE "
             f"(host_bound_ratio={ref.ratio:.2f}): the ref's device reading is its own "
             f"python dispatch cost, so vs_ref_device this round is not a "
             f"kernel-vs-kernel ratio. Re-measure the reference through a host-free "
             f"window (graph replay / NCU kernel time). NOT reported to the student: "
             f"a thick reference wrapper is a measurement fact, and feeding it to a "
             f"kernel writer teaches metric-gaming.")


def _keep_probe_across_reset(ctx: LoopContext, probe_path: str) -> str | None:
    """The probe file's text before a workspace reset wipes the iteration's
    commits — a fight that spans iterations (ProbeBattle) needs the probe to
    survive the reset that ends the iteration, or the next student starts the
    apparatus from scratch (observed: five rounds lost to rig debugging)."""
    try:
        return (ctx.workspace / probe_path).read_text()
    except OSError:
        return None


def _restore_probe_after_reset(ctx: LoopContext, probe_path: str, text: str) -> None:
    """Write the probe back after the reset AND remember it: a meta step between
    two student iterations commits its own verdict probe with `git add -A` and
    resets to the pre-probe head, which erases an untracked file — so the next
    student iteration re-writes it from ctx.kept_probes before it starts."""
    ctx.kept_probes[probe_path] = text
    dst = ctx.workspace / probe_path
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
    except OSError as e:
        _log(f"could not keep {probe_path} across the reset: {e}")


def _restore_kept_probes(ctx: LoopContext) -> None:
    """Re-materialise the kept probes a reset erased. A file that is present
    with OTHER text is the student's later edit (or its deliberate rewrite):
    it is left alone and the kept copy is dropped — the keep exists to survive
    a reset, never to overrule the student's tree."""
    for path, text in list(ctx.kept_probes.items()):
        dst = ctx.workspace / path
        if dst.is_file():
            try:
                same = dst.read_text() == text
            except OSError:
                same = False
            if not same:
                ctx.kept_probes.pop(path, None)
                _log(f"kept probe {path} differs from the workspace copy — keeping "
                     f"the workspace's, dropping the kept copy")
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text)
            _log(f"restored kept probe {path} into the workspace")
        except OSError as e:
            _log(f"could not restore kept probe {path}: {e}")


# --- the card-owned evaluation harness (loop/harness_guard.py) ------------------------

def _harness_hygiene(ctx: LoopContext, iter_no: int) -> None:
    """Put the lineage's harness back to the launch tree's before the student turn.

    The path check below sees what ONE delivery changed; three things reach the lineage
    without being a delivery: an iteration that ends with no record keeps its probe
    commits while there is no green tree to reset to, a teacher turn runs in the same
    working directory and a reset keeps the untracked files it wrote, and the next
    ``git add -A`` commits whatever is there. Restored here in a commit of their own,
    so they are neither judged by nor blamed on the next delivery."""
    if ctx.harness_guard == "off" or not ctx.harness_anchor:
        return
    be = ctx.backend
    drift_of = getattr(be, "harness_drift", None)
    restore = getattr(be, "restore_harness", None)
    if drift_of is None or restore is None:
        return
    try:
        drift = drift_of(ctx.harness_anchor)
    except Exception as e:  # noqa: BLE001 — a hygiene step must not become the failure
        _log(f"iter {iter_no}: harness guard: cannot read the lineage's harness ({e})")
        return
    if not drift:
        return
    shown = harness_guard.describe(drift)
    if ctx.harness_guard != "revert":
        _log(f"iter {iter_no}: WARNING harness guard: the lineage's {shown} differ from the "
             f"launch tree (harness_guard = warn: left as is; jobs run the card's copies)")
        return
    try:
        restore(ctx.harness_anchor, drift,
                f"harness guard: restore the card-owned {shown} to the launch tree")
    except Exception as e:  # noqa: BLE001
        _log(f"iter {iter_no}: harness guard: could not restore {shown} ({e})")
        return
    _log(f"iter {iter_no}: harness guard: restored {shown} to the launch tree "
         f"({str(ctx.harness_anchor)[:10]}) before the student turn — card-owned evaluation "
         f"files that reached the workspace outside a delivery")


def _harness_edit_revert(ctx: LoopContext, iter_no: int, record, sha: str, base_sha: str,
                         metric_before) -> "IterationRecord | None":
    """REVERT a delivery whose commits (probe commits included) edit the card-owned
    evaluation harness, before any job is bought; None = nothing to refuse."""
    if ctx.harness_guard == "off" or not base_sha:
        return None
    changes_of = getattr(ctx.backend, "harness_changes", None)
    if changes_of is None:
        return None
    try:
        touched = changes_of(base_sha, sha) or []
    except Exception:  # noqa: BLE001 — bookkeeping, never a reason to stop
        return None
    if not touched:
        return None
    shown = harness_guard.describe(touched)
    if ctx.harness_guard != "revert":
        _log(f"iter {iter_no}: WARNING {record.decision_id} edits the card-owned {shown} "
             f"(harness_guard = warn: kept; jobs run the card's copies)")
        return None
    err = (f"harness guard: the delivery edits card-owned evaluation files ({shown}). "
           f"gpu_entry.sh, eval/, ref_bin/ and .gitattributes belong to the card — jobs run "
           f"the card's copies whatever the tree holds, and only {ctx.kernel_file} (plus "
           f"notes/) is the student's to change. Reverted without an eval; deliver the "
           f"decision again without touching them.")
    _log(f"iter {iter_no}: {record.decision_id} REVERT ({err})")
    ctx.backend.reset(base_sha)
    # correctness None: nothing was evaluated, so the lineage's green state is not
    # moved (a False would drop a never-evaluated workspace into scaffold mode)
    rec = IterationRecord(
        iter=iter_no, decision_id=record.decision_id, change=record.change,
        correctness=None, verdict="REVERT", metric_before=metric_before,
        error=err, profiler={}, cluster=_active_cluster(ctx), base_sha=base_sha,
    )
    append_record(ctx.status_path, rec)
    return rec


def run_inner_iteration(
    ctx: LoopContext, iter_no: int, *, metric_before: float | None,
    last_good_ref: str | None = None, green: bool | None = True,
) -> IterationRecord | None:
    """One inner iteration. Returns the record, or None if the student is exhausted.

    ``green`` is the lineage's correctness state: True once the kernel has passed
    correctness since the last workspace reset, False while it is under construction,
    None when unknown (the next full eval decides). Pre-green (False), a committed
    decision is NOT evaluated — see the scaffold branch below."""
    be = ctx.backend
    entry_cluster = _active_cluster(ctx)
    _restore_kept_probes(ctx)
    _harness_hygiene(ctx, iter_no)
    prompt = _student_prompt(ctx, iter_no)
    _log(f"iter {iter_no}: spawning student agent…")
    try:
        base_sha = be.head() or ""       # the tree this iteration builds on
    except Exception:  # noqa: BLE001 — lineage is bookkeeping, never a precondition
        base_sha = ""

    record: Record | None = None
    last_probe: str = ""
    last_probe_passed = False
    last_pv, last_log = None, ""
    gate_refused = ""   # the last gate-row refusal (open placeholders), for the record
    for _ in range(ctx.max_probe_rounds):
        out = be.run_agent(prompt)
        try:
            turn = parse_student_turn(out)
        except ValueError:
            # student emitted no parseable result (prose only / truncated) — re-prompt
            # with a strict format reminder instead of aborting the loop.
            _log(f"iter {iter_no}: unparseable student output -> re-prompt")
            prompt = _student_prompt(
                ctx, iter_no,
                probe_feedback="- Your last reply had NO parseable result. End your turn with "
                               "EXACTLY one of: the record block (decision_id: / change: / "
                               "files_touched:), 'PROBE_REQUEST: <path>', or 'EXHAUSTED: <reason>' "
                               "— and nothing after it.\n")
            continue
        if isinstance(turn, Exhausted):
            _log(f"iter {iter_no}: student EXHAUSTED ({turn.reason})")
            return None
        if isinstance(turn, Record):
            if bounce := _placeholder_bounce(ctx, turn, base_sha):
                # A gate row over an open placeholder, or a placeholder landed
                # without its tag: neither is committed — the student is re-prompted
                # with the debt (see the placeholder ledger above).
                kind, why, feedback = bounce
                _log(f"iter {iter_no}: {why}")
                if kind == "gate-refused":
                    gate_refused = why
                prompt = _student_prompt(ctx, iter_no, probe_feedback=feedback)
                continue
            record = turn
            break
        if _is_timeline_request(turn.path):
            # a timeline of the student's own tree: one probe round, no verdict
            if not _timeline_on(ctx):
                prompt = _student_prompt(
                    ctx, iter_no,
                    probe_feedback="- This run has the timeline probe off (loop.toml [loop] "
                                   "timeline_probe), so `PROBE_REQUEST: timeline` is unavailable; "
                                   "send an ordinary probe or submit a record block.\n")
                continue
            be.commit("probe: timeline")
            probe_branch = f"{ctx.branch_prefix}-i{iter_no}probe"
            be.push_branch(probe_branch)
            tl = _run_timeline(ctx, be.head(), f"iter{iter_no}:student", branch=probe_branch)
            prompt = _student_prompt(ctx, iter_no, probe_feedback=_timeline_feedback(tl))
            continue
        # ProbeRequest: commit + run on GPU, feed the verdict back to the student
        last_probe, last_probe_passed = turn.path, False
        be.commit(f"probe: {turn.path}")
        probe_branch = f"{ctx.branch_prefix}-i{iter_no}probe"
        be.push_branch(probe_branch)
        log = be.submit_gpu(probe_branch, f"cmd:python3 {turn.path}")
        pv = parse_probe_verdict(log)
        last_pv, last_log = pv, log
        _log(f"iter {iter_no}: probe {turn.path} -> PROBE {_probe_label(pv)} "
             f"(max_diff={pv.diff_text}){_probe_log_suffix(pv)}")
        battle = ctx.probe_battles.setdefault(turn.path, ProbeBattle(path=turn.path))
        battle.open_rounds += 1
        if pv.passed:
            last_probe_passed = True
            fight = battle.open_rounds
            battle.open_rounds = 0  # a PASS closes the fight; re-probes start a new one
            # Bill the dearest winning fight, whole: on a tie the later fight wins
            # (its PASS certifies the probe file as it now ships). A cheaper re-win
            # leaves the billed fight's record intact — still internally coherent.
            if fight >= battle.rounds:
                battle.passed = True
                battle.rounds = fight
                battle.pass_iter = iter_no
                battle.verdict_line = f"PROBE PASS (max_diff={pv.diff_text})"
                tail = "\n".join(log.strip().splitlines()[-_PROBE_LOG_TAIL_LINES:])
                battle.log_tail = tail[-_PROBE_LOG_TAIL_CHARS:]
                if ctx.kb_tripwire_rounds and fight >= ctx.kb_tripwire_rounds:
                    _log(f"iter {iter_no}: kb-tripwire: {turn.path} passed after "
                         f"{fight} GPU rounds (>= {ctx.kb_tripwire_rounds}) — "
                         f"queued for the end-of-run KB harvest write-up")
        prompt = _student_prompt(
            ctx, iter_no,
            probe_feedback=_probe_feedback(
                turn.path, pv, log,
                on_pass="Implement the full decision and emit the record block."),
        )
    if record is None and last_probe_passed:
        # The budget counts agent turns, probe turns included, so a probe that
        # PASSes on the last turn has no turn left to deliver the decision it
        # certified. Observed on dsa_sparse runs: eight
        # rounds, r20 PASS 2/2, recorded as "could not certify", two hours of
        # kernel work reset. One delivery-only turn closes that hole — a probe
        # request on it is not honoured (no GPU), the record block is.
        _log(f"iter {iter_no}: {last_probe} passed on the last probe round — "
             f"one delivery-only turn")
        # A fresh subprocess: it needs the verdict table and log tail the
        # normal on_pass path carries, not just "it passed".
        out = be.run_agent(_student_prompt(
            ctx, iter_no,
            probe_feedback=_probe_feedback(
                last_probe, last_pv, last_log,
                on_pass=("This iteration's probe rounds are used up: **this turn can only "
                         "deliver** — implement the full decision and end with a record block. "
                         "Another PROBE_REQUEST will not be executed."))
            + "- This turn can only deliver (probe rounds used up).\n"))
        try:
            turn = parse_student_turn(out)
        except ValueError:
            turn = None
        if isinstance(turn, Record):
            if bounce := _placeholder_bounce(ctx, turn, base_sha):
                _log(f"iter {iter_no}: {bounce[1]}")
                if bounce[0] == "gate-refused":
                    gate_refused = bounce[1]
            else:
                record = turn
    if record is not None:
        # A landed decision ends every kept fight: the probe files are in the
        # student's tree now (or the student dropped them), and a keep that
        # outlived the iteration it was made for re-wrote stale text into
        # every later workspace (and every tree a later reset restored).
        ctx.kept_probes.clear()
    if record is None:
        # budget exhausted with no landed decision (probe never certified, or the
        # student never emitted a parseable decision). Objective REVERT, roll back.
        kept = _keep_probe_across_reset(ctx, last_probe) if last_probe else None
        if last_good_ref:
            be.reset(last_good_ref)
        if kept is not None:
            _restore_probe_after_reset(ctx, last_probe, kept)
        if gate_refused:
            # The student spent the iteration re-submitting a gate row over an
            # open placeholder. Say which debt blocked it — "no parseable
            # decision" would send the next iteration looking for a format bug.
            did, change = "gate-refused", "(gate row refused: placeholder ledger not clear)"
            err = gate_refused
        elif last_probe and last_probe_passed:
            # The fight was WON; only the delivery is missing. Say so — the next
            # iteration's student reads this row, and "could not certify" sent
            # the next iteration straight past a certified probe into a -17.9% commit.
            did, change = "probe-undelivered", "(probe passed; decision not delivered in budget)"
            err = (f"{last_probe} PASSED on the last probe round but the decision was "
                   f"not delivered — the probe file is kept in the workspace (uncommitted); "
                   f"the next iteration continues from that PASS, it does not re-fight it")
        elif last_probe:
            did, change = "probe-blocked", "(probe budget exhausted)"
            err = (f"{last_probe} did not pass in {ctx.max_probe_rounds} probe rounds — "
                   f"the probe file is kept in the workspace (uncommitted) so the fight "
                   f"can continue next iteration")
        else:
            did, change = "unparseable", "(no parseable decision emitted)"
            err = f"no parseable decision from the student in {ctx.max_probe_rounds} rounds"
        rec = IterationRecord(
            iter=iter_no, decision_id=did, change=change,
            correctness=False, verdict="REVERT", metric_before=metric_before,
            error=err, profiler={}, cluster=_active_cluster(ctx),
        )
        append_record(ctx.status_path, rec)
        _log(f"iter {iter_no}: {did} -> REVERT ({err})")
        return rec

    sha = be.commit(f"{record.decision_id}: {record.change[:60]}")
    if (rec := _harness_edit_revert(ctx, iter_no, record, sha, base_sha,
                                    metric_before)) is not None:
        return rec
    for m in _PLACEHOLDER_TAG.finditer(record.change or ""):
        _log(f"iter {iter_no}: placeholder opened — [{m.group(1).strip()}] "
             f"(gate rows are refused until [RESTORED: …] lands)")
    for m in _RESTORED_TAG.finditer(record.change or ""):
        _log(f"iter {iter_no}: placeholder restored — [{m.group(1).strip()}]")
    declared = _declared_construction(record.change) if green is None else None
    if (green is False or declared) and not _GATE_ROW.search(record.decision_id):
        # SCAFFOLD (POLICY §3 build-phase semantics, executed): the kernel has not passed
        # correctness in this lineage, so a full eval of this sub-step can only measure
        # "still under construction" — the verdict is decided before the job runs, NCU
        # yields nothing on a broken bench, and the REVERT it would record is
        # bookkeeping, not information. The decision's own [VERIFY:] declaration backs
        # the skip: "none" is never per-step-measurable, and the ncu-* launch facts are
        # all harvested by the single full eval the gate row triggers. The commit still
        # lands (the sha is a resume checkpoint); the record says NOT-evaluated
        # (NEUTRAL, correctness None) rather than pretending a verdict.
        #
        # An UNKNOWN lineage (green None: fresh workspace, or a checkpoint no record
        # owns) is normally decided by its first full eval — a source-mode run that starts
        # from a working kernel must not lose its first readings to scaffolding. But a
        # delivery whose change OPENS with the prompt's own construction declaration
        # (`part k of n`, k < n; `[PLACEHOLDER: …]`) has told the driver the tree is
        # mid-build: evaluating it buys a REVERT that says "under construction" (the
        # observed: a GDN run paid two such evals, the second right after a budget resume).
        from forgekernel.milestone import parse_verify_tags
        tag = parse_verify_tags(_read_milestone(ctx)).get(record.decision_id, "none")
        rec = IterationRecord(
            iter=iter_no, decision_id=record.decision_id, change=record.change,
            correctness=None, verdict="NEUTRAL", metric_before=metric_before, sha=sha,
            cluster=_active_cluster(ctx), base_sha=base_sha,
        )
        append_record(ctx.status_path, rec)
        why = (f"declared construction: {declared}; lineage state unknown"
               if declared else "pre-green")
        _log(f"iter {iter_no}: {record.decision_id} SCAFFOLD ({why}, [VERIFY: {tag}] "
             f"-> full eval deferred to the gate row)")
        if ctx.declared_gate_close and (m := _CLOSES_TAG.search(record.change or "")):
            _log(f"iter {iter_no}: [CLOSES: {_closes_gate_id(m)}] not honoured — a "
                 f"scaffold row has no eval to close on; the gate row is the student's")
        return rec
    if (base := _unchanged_tree_reading(ctx, record, sha, base_sha)) is not None:
        # Nothing the eval reads changed since a tree this trajectory already
        # measured: archive the row on that reading instead of re-measuring it.
        rec = _archive_reused_reading(ctx, iter_no, record, sha, base_sha, base,
                                      metric_before)
        return _declared_gate_close(ctx, record, rec, iter_no) or rec
    branch = f"{ctx.branch_prefix}-i{iter_no}"
    be.push_branch(branch)
    _log(f"iter {iter_no}: decision {record.decision_id} committed; evaluating on GPU…")
    ev = parse_eval(be.submit_gpu(branch, "all"))
    ev = _enforce_single_ref(ctx, ev)
    ev = _enforce_prep_contract(ctx, ev)
    precision = _precision_reading(ctx, ev, iter_no)
    cluster = _active_cluster(ctx)
    if cluster != entry_cluster:
        # The eval failed over mid-iteration: the best it is about to be judged
        # against was read on another cluster. Re-read it from this cluster's
        # rows (usually nothing yet) — the watermark check and the record's
        # metric_before must not compare across clusters.
        metric_before = _best_ms(_on_cluster(load_records(ctx.status_path), ctx, cluster),
                                 ctx.gate_window)
        _log(f"iter {iter_no}: eval ran on {cluster!r} (the iteration started on "
             f"{entry_cluster!r}) — best re-read from that machine's rows: "
             f"{metric_before}")
    # NCU on EVERY committed iteration (KEEP and REVERT) — negative iterations need
    # measured hardware truth too, so the meta agent never has to guess a diagnosis.
    # NCU is best-effort on the GPU side: a PASSING kernel with an empty profiler means
    # NCU itself failed (e.g. a lost job), so retry; a correctness-broken kernel yields
    # no NCU legitimately (the bench crashes), so don't waste GPU retrying that.
    profiler = _ncu_profiler(ctx, branch, correctness=ev.correctness)
    orch_summary = _pop_orch_summary(profiler)
    kernels = _pop_kernels(profiler)
    serial_raw = _parse_serial_census(profiler)
    sass_mp = profiler.pop(_SASS_PRECISION_PROF_KEY, None) if profiler else None
    if sass := _sass_precision(ctx, sass_mp or (orch_summary or {}).get("mma_precision"),
                               iter_no):
        precision = {**precision, "sass": sass}

    if not ev.correctness:
        # relaxed-revert keeps perf regressions, but a correctness break is a hard REVERT.
        # Carry the bench's own verdict text into the record: the student and the meta
        # diagnoser debug from this string, and "correctness FAIL" alone once hid a
        # harness-level rejection (forbidden-op scan) behind 14h of phantom-bug chasing.
        detail = f"correctness FAIL: {ev.error}" if ev.error else "correctness FAIL"
        _log(f"iter {iter_no}: {record.decision_id} REVERT ({detail})")
        if last_good_ref:
            be.reset(last_good_ref)
        rec = IterationRecord(
            iter=iter_no, decision_id=record.decision_id, change=record.change,
            correctness=False, verdict="REVERT", metric_before=metric_before,
            error=detail, profiler=profiler, cluster=cluster,
            orch_summary=orch_summary, base_sha=base_sha, kernels=kernels,
            prep=ev.prep or {}, precision=precision,
        )
        append_record(ctx.status_path, rec)
        return rec

    ev = _confirm_suspicious_watermark(ctx, be, branch, ev, metric_before,
                                       profiler=profiler)
    rec = IterationRecord(
        iter=iter_no, decision_id=record.decision_id, change=record.change,
        correctness=True, verdict="KEEP", bwd_median_ms=ev.bwd_median_ms, mfu=ev.mfu,
        ref_median_ms=ev.ref_median_ms, vs_ref=ev.vs_ref,
        device_median_ms=ev.device_median_ms, ref_device_ms=ev.ref_device_ms,
        vs_ref_device=ev.vs_ref_device,
        host_enqueue_ms=ev.host_enqueue_ms, host_bound_ratio=ev.host_bound_ratio,
        ref_host_bound_ratio=ev.ref_host_bound_ratio,
        device_window_basis=ev.device_window_basis,
        metric_before=metric_before, profiler=profiler, sha=sha, cluster=cluster,
        orch_summary=orch_summary, base_sha=base_sha, kernels=kernels,
        prep=ev.prep or {}, n_workloads=ev.n_workloads, workloads=ev.workloads,
        precision=precision,
    )
    append_record(ctx.status_path, rec)
    _log_axis_change(ctx, rec)
    _check_host_bound(ctx, iter_no, ev)
    # NCU reports device time, so the reconciliation compares against the device
    # window when the bench measured one — against the e2e window a host-heavy
    # call shrinks the kernel share and false-fires the alarm.
    if (warn := _check_profile_covers_the_kernel(
            profiler, ev.device_median_ms or ev.bwd_median_ms, ev)):
        _log(f"iter {iter_no}: WARNING {warn}")
    _log(f"iter {iter_no}: {record.decision_id} KEEP  "
         f"metric={gate_ms(ev, ctx.gate_window)} ({ctx.gate_window}) mfu={ev.mfu}"
         + (f" vs_ref={ev.vs_ref}" if ev.vs_ref is not None else "")
         + (f" vs_ref_device={ev.vs_ref_device}" if ev.vs_ref_device is not None else "")
         + (f"  (best was {metric_before})" if metric_before else "")
         + (f"  [cluster {cluster}]" if cluster and cluster != ctx.home_cluster else "")
         + (f"  {precision_gate.describe(precision)}" if precision else ""))
    if ev.workloads:
        _log(f"iter {iter_no}: gate = geomean over {ev.n_workloads} workloads: "
             + describe_workload_readings(ev.workloads))
    if _too_few_workloads(ctx, ev) and str(ctx.status_path) not in _SINGLE_WORKLOAD_WARNED:
        _SINGLE_WORKLOAD_WARNED.add(str(ctx.status_path))
        _log(f"iter {iter_no}: WARNING the gate reading covers {ev.n_workloads} workload(s), "
             f"not {PERF_WORKLOADS} — a kernel kept on it can be tuned to that input set "
             f"(timing.measure_windows_workloads)")
    new_ms = gate_ms(ev, ctx.gate_window)
    ctx.sass_serial_note = _sass_serial_status(ctx, serial_raw, exclude_iter=iter_no)
    if new_ms is not None and (metric_before is None or new_ms < metric_before):
        # the tree the milestone's cited prices were read on just changed under
        # them: re-price those citations here, on the tree they now apply to
        try:
            _reprice_cited_levers(ctx, rec)
        except (BackendError, OSError, ValueError) as e:
            _log(f"iter {iter_no}: price-check skipped ({e})")
    if _timeline_on(ctx):
        # a delivery that closes its phase is read like the gate row it files
        is_gate = bool(_GATE_ROW.search(rec.decision_id or "")) or bool(
            ctx.declared_gate_close and _CLOSES_TAG.search(record.change or ""))
        is_best = new_ms is not None and (metric_before is None or new_ms < metric_before)
        why = "gate row" if is_gate else ("new best" if is_best and _timeline_due(ctx, new_ms) else "")
        if why:
            try:
                _run_timeline(ctx, rec.sha, f"iter{iter_no}:{why}")
            except (BackendError, OSError, ValueError) as e:
                _log(f"iter {iter_no}: timeline skipped ({e})")
        elif is_best:
            _log(f"iter {iter_no}: new best within {ctx.reprice_min_gain_pct:g}% of the last "
                 f"timeline's tree — not re-read")
    return _declared_gate_close(ctx, record, rec, iter_no) or rec


# --- rows that buy no eval ------------------------------------------------------
#
# Two shapes of row carry no new information for a GPU job to measure:
#   * a commit that changes nothing outside notes/ on top of a tree this
#     trajectory already evaluated — the gate acceptance row "submitted on its
#     own with no code change", a reopened decision re-filed as built, a
#     pricing-only row. Its eval re-reads the same bytes; the reading it buys is
#     a second noise draw of one tree, and best / patch judgment select on the
#     maximum of the draws, so a lucky one moves "the best" for nothing.
#   * the gate row of a delivery that declares it closes its phase
#     (`[CLOSES: Mk-gate]`): the delivery's own full eval already read the tree
#     the gate accepts, so a separate gate turn would only re-read it.
# Both rows still land in the ledger as their own rows (the student's ledger,
# phase close, the same-stage gate check and the reopened-M0 audit read them by
# decision id); they carry the evaluated row's reading and say so (reading_of).

# Paths a commit may touch and still be the same tree for the eval: the
# student's probe scripts. Everything else (the kernel, any other file the job
# ships) counts as a change.
_READING_NEUTRAL_PREFIXES = ("notes/",)
_REUSE_MODES = ("all", "gate", "off")
# `[CLOSES: M2-gate]` on a delivery: this tree completes phase M2 (see
# LoopContext.declared_gate_close). "M2 gate" / "M2-Gate" spellings are read too.
_CLOSES_TAG = re.compile(r"\[CLOSES:\s*(M\d+)\s*[- ]?\s*gate\s*\]", re.I)


def _closes_gate_id(m: re.Match) -> str:
    return f"{m.group(1).upper()}-gate"


def _unchanged_tree_reading(ctx: LoopContext, record: Record, sha: str,
                            base_sha: str) -> IterationRecord | None:
    """The evaluated KEEP whose reading ``sha`` may carry, or None (evaluate).

    Applies when the commit changes nothing outside notes/ relative to the tree
    the iteration built on (``base_sha``), and that tree has a KEEP on file
    that passed correctness and read the gate window, on the cluster the run is
    submitting to now. A backend that cannot list a commit's changes (no
    ``commit_paths``) always evaluates."""
    mode = str(ctx.reuse_unchanged_reading or "off").lower()
    if mode not in _REUSE_MODES or mode == "off" or not sha or not base_sha or sha == base_sha:
        return None
    if mode == "gate" and not _GATE_ROW.search(record.decision_id or ""):
        return None
    lister = getattr(ctx.backend, "commit_paths", None)
    if lister is None:
        return None
    try:
        paths = lister(base_sha, sha)
    except Exception:  # noqa: BLE001 — unknown means evaluate, never a crash
        return None
    if paths is None or any(not p.startswith(_READING_NEUTRAL_PREFIXES) for p in paths):
        return None
    try:
        records = load_records(ctx.status_path)
    except (OSError, ValueError):
        return None
    for r in reversed(records):
        if (r.sha == base_sha and r.verdict == "KEEP" and r.correctness
                and gate_ms(r, ctx.gate_window) is not None):
            return r if _rec_cluster(r, ctx) == _active_cluster(ctx) else None
    return None


def _archive_reused_reading(ctx: LoopContext, iter_no: int, record: Record, sha: str,
                            base_sha: str, base: IterationRecord,
                            metric_before: float | None) -> IterationRecord:
    """Land ``record`` as a KEEP carrying ``base``'s reading (no GPU job)."""
    origin = base.reading_of if base.reading_of is not None else base.iter
    rec = replace(base, iter=iter_no, decision_id=record.decision_id, change=record.change,
                  metric_before=metric_before, sha=sha, base_sha=base_sha, error="",
                  reading_of=origin)
    append_record(ctx.status_path, rec)
    if base.sha in ctx.paired_memo:
        # the same tree's same-job parity was already re-measured and refused
        ctx.paired_memo[sha] = ctx.paired_memo[base.sha]
    _log(f"iter {iter_no}: {record.decision_id} changes nothing outside notes/ on "
         f"{base_sha[:10]} — KEEP on the reading of iter {origin} "
         f"(metric={gate_ms(rec, ctx.gate_window)} {ctx.gate_window}); no eval "
         f"([loop] reuse_unchanged_reading={ctx.reuse_unchanged_reading})")
    if _timeline_on(ctx) and (_GATE_ROW.search(record.decision_id or "") or (
            ctx.declared_gate_close and _CLOSES_TAG.search(record.change or ""))):
        # the gate row banks a table of the tree it accepts; on an unchanged
        # tree that is the evaluated row's table when one is banked
        try:
            _run_timeline(ctx, base.sha, f"iter{iter_no}:gate row")
        except (BackendError, OSError, ValueError) as e:
            _log(f"iter {iter_no}: timeline skipped ({e})")
    return rec


def _declared_gate_close(ctx: LoopContext, record: Record, rec: IterationRecord,
                         iter_no: int) -> IterationRecord | None:
    """The Mk-gate row a `[CLOSES: Mk-gate]` delivery files, archived on the
    delivery's tree and reading; None when the tag is absent or not honoured
    (the student then submits the gate row as before).

    Holds the gate row's own preconditions: the delivery KEEPs on a measured
    reading, it is not an unfinished `part k of n` step, and the placeholder
    ledger on its tree is clear (a gate row is refused over an open one)."""
    if not ctx.declared_gate_close:
        return None
    m = _CLOSES_TAG.search(record.change or "")
    if not m:
        return None
    gid = _closes_gate_id(m)
    why = ""
    if _GATE_ROW.search(record.decision_id or ""):
        why = "the row is a gate row itself"
    elif rec.verdict != "KEEP" or not rec.correctness or gate_ms(rec, ctx.gate_window) is None:
        why = "the delivery did not KEEP on a measured reading"
    elif (d := _declared_construction(record.change)) or _PLACEHOLDER_TAG.search(record.change or ""):
        why = f"the delivery declares construction ({d or 'a placeholder'})"
    else:
        open_ = _placeholder_ledger(load_records(ctx.status_path), rec.sha)
        if open_:
            why = f"open placeholder(s) {_placeholder_items(open_)}"
    if why:
        _log(f"iter {iter_no}: [CLOSES: {gid}] not honoured — {why}; the gate row is "
             f"the student's to submit")
        return None
    origin = rec.reading_of if rec.reading_of is not None else rec.iter
    records = load_records(ctx.status_path)
    gate = replace(
        rec, iter=iter_no + 1, decision_id=gid,
        change=(f"{gid} closed by the driver on the delivery of iter {iter_no} "
                f"({record.decision_id}, [CLOSES: {gid}]): same tree {rec.sha[:10]}, "
                f"reading of iter {origin} — no separate gate turn or eval"),
        metric_before=_best_ms(_on_cluster(records, ctx), ctx.gate_window),
        base_sha=rec.sha, reading_of=origin)
    append_record(ctx.status_path, gate)
    _log(f"iter {iter_no}: {record.decision_id} declares [CLOSES: {gid}] — {gid} archived "
         f"as iter {iter_no + 1} on {rec.sha[:10]} (reading of iter {origin})")
    return gate


def _sass_serial_status(ctx: LoopContext, serial_raw: dict,
                        exclude_iter: int | None = None) -> str:
    """The standing compiler-fallback note, or '' when the chains are pipelined.

    ``serial_raw`` is the census's wgmma fields for THIS KEEP (the ordinates the
    census printed). Reads the previous KEEP's ratio off the trajectory so a
    change that *introduced* serialization reads as a regression ("pipelined ->
    serialized") rather than a standing state — the thing a GDN run once
    could not see and forbade a correct direction over.

    ``exclude_iter`` is THIS iteration: its record is already on status.jsonl
    when the note is written, and reading it back as "the previous KEEP" made
    every verdict a standing state (prev == cur), never a regression.
    """
    from forgekernel.skills.sass_census import serialization_verdict
    if not serial_raw:
        return ctx.sass_serial_note  # census absent this round: leave the last note
    # the previous KEEP's ratio (the status.jsonl rows, gate-window agnostic)
    prev = None
    try:
        for r in reversed(load_records(ctx.status_path)):
            if exclude_iter is not None and getattr(r, "iter", None) == exclude_iter:
                continue
            raw = getattr(r, "profiler", None) or {}
            if raw.get("wgmma_serial_ratio") is not None:
                prev = raw
                break
    except Exception:  # noqa: BLE001 — a note, never a crash
        prev = None
    v = serialization_verdict(prev, serial_raw)
    return v if v is not None else ""


def _paired_ratio(rec, gate_window: str) -> float | None:
    """The same-job ref/student ratio of a record in the gate window (> 1 =
    student faster), or None when the job carried no ref reading."""
    if gate_window == "device":
        ratio, ms, ref = rec.vs_ref_device, rec.device_median_ms, rec.ref_device_ms
    else:
        ratio, ms, ref = rec.vs_ref, rec.bwd_median_ms, rec.ref_median_ms
    if ratio is None and ms and ref:
        ratio = ref / ms
    return ratio





def _persist_paired(ctx: LoopContext, record: dict) -> None:
    """Append a same-job parity account to <op_dir>/paired_verdict.jsonl."""
    try:
        path = Path(ctx.milestone).parent / "paired_verdict.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        _log(f"paired: could not persist the account ({e})")


# Two same-job ratios that disagree by more than this represent no single
# period (a node-mode flip mid-run, a killed-and-resumed job): the same
# floor the docstrings quote for a same-job ratio, not the 10% band the
# absolute path allows two different nodes.
_PAIRED_SPREAD_MAX = 0.05


def _admit_paired(conf: dict | None) -> "tuple[float | None, str | None]":
    """(median same-job ratio, None) when the re-measure is admissible, else
    (None, why). Admission: >= 2 usable ratios, spread within
    _PAIRED_SPREAD_MAX — one draw is still one draw, and two draws that
    disagree by more than the noise floor decide nothing either way."""
    if not conf:
        return None, "no re-measure came back"
    ratios = [s["ratio"] for s in conf.get("jobs", []) if s.get("ratio") is not None]
    if len(ratios) < 2:
        return None, f"{len(ratios)} usable same-job ratio(s), need >= 2"
    spread = (max(ratios) - min(ratios)) / min(ratios)
    if spread > _PAIRED_SPREAD_MAX:
        return None, (f"ratios {sorted(round(r, 4) for r in ratios)} spread {spread:.1%} "
                      f"> {_PAIRED_SPREAD_MAX:.0%} (no single period)")
    return statistics.median(ratios), None


def _paired_parity_stop(ctx: LoopContext, rec, iter_no: int) -> bool:
    """Same-job parity as a stop: a KEEP whose in-job ratio meets the
    equivalence ratio is re-measured (2 jobs, same sha) and stops the run when
    the median same-job ratio still meets it.

    Only on ref_in_job runs. The absolute check above it compares one period's
    reading against a target derived from another period's draws — nodes vary by
    ~6% — and cannot decide anything inside that band;
    the same-job ratio is measured on one card in one minute and can. One draw
    is still one draw (same-job ratios carry a 1-5% floor), so the claim is
    repeated before it ends the run, and a sha whose re-measure disagreed is
    memoised so a later plateau does not re-buy the same two jobs.
    """
    if not ctx.ref_in_job or rec.verdict != "KEEP" or not rec.sha:
        return False
    ratio = _paired_ratio(rec, ctx.gate_window)
    eq = ctx.equivalence_ratio
    if ratio is None or ratio < eq or rec.sha in ctx.paired_memo:
        return False
    _log(f"iter {iter_no}: same-job ratio {ratio:.4f} >= {eq} ({ctx.gate_window} "
         f"window) — parity on paper; re-measuring {rec.sha[:10]} (2 jobs) before stopping")
    try:
        conf = run_best_confirmation(ctx, jobs=2, max_extra=1, ref=rec.sha)
    except Exception as e:  # noqa: BLE001 — a starved queue must not kill the run
        conf = None
        _log(f"iter {iter_no}: paired re-measure failed ({e})")
    med, why = _admit_paired(conf)
    record = {"ts": ctx.now(), "iter": iter_no, "sha": rec.sha, "window": ctx.gate_window,
              "cluster": _active_cluster(ctx),
              "ratio": round(ratio, 4), "equivalence_ratio": eq,
              "jobs": [s.get("job") for s in (conf or {}).get("jobs", [])],
              "samples": list((conf or {}).get("jobs", [])),
              "median_ratio": med}
    if med is None:
        # not admissible: neither a stop nor a verdict — not memoised either,
        # so a later plateau may try again under better conditions
        record["verdict"] = "remeasure_unavailable"
        record["why"] = why
        _persist_paired(ctx, record)
        _log(f"iter {iter_no}: paired re-measure inadmissible ({why}) — no verdict; continuing")
        return False
    if med >= eq:
        record["verdict"] = "target"
        ctx.paired_stop = record
        _persist_paired(ctx, record)
        _log(f"iter {iter_no}: same-job median ratio {med:.4f} over "
             f"{len(record['jobs'])} job(s) >= {eq} — parity confirmed on the same "
             f"card; stopping as target")
        return True
    record["verdict"] = "unconfirmed"
    ctx.paired_memo[rec.sha] = record
    _persist_paired(ctx, record)
    _log(f"iter {iter_no}: paired parity NOT confirmed (median ratio {med:.4f} < {eq}) "
         f"— the single reading was a lucky draw; continuing")
    return False


def run_inner_loop(ctx: LoopContext, start_iter: int, *,
                   segment_records_before: int | None = None,
                   gate_contract: dict | None = None,
                   initial_green: bool | None = True,
                   initial_good_ref: str | None = None,
                   audit_on_green: bool = False,
                   audit_on_gate: str | None = None) -> str:
    """Iterate until a stop condition. Returns the reason.

    ``initial_good_ref``: the rollback anchor a green segment enters on when it is
    NOT its entry HEAD — a resumed lineage whose HEAD no record verified (a
    delivery committed before the process died mid-eval, a kept probe on top of
    the last KEEP). The anchor is the last KEEP then, HEAD stays the tree the next
    delivery builds on and its first eval decides it.

    ``initial_green`` seeds the lineage's correctness state (see run_inner_iteration):
    run_two_layer passes None for a fresh workspace and the resumed-to checkpoint's
    state after a meta reset; the default True preserves evaluate-everything behavior
    for direct callers. Green flips True on the first correctness PASS and never flips
    back within a segment — a post-green break is a real regression and stays a hard
    REVERT, not a return to scaffolding.

    ``audit_on_green``: stop with reason "first_green" the first time this segment
    turns green (= its M0 gate lands), so run_two_layer can run the forced M0 audit
    before optimization stacks a whole segment on an unverified foundation. Target
    still wins: a kernel that is already at target has nothing left to audit for.

    ``audit_on_gate``: the same stop, keyed on a gate row instead of the green
    flip — for a segment that ENTERS green (an in-place reopen keeps the best
    tree) but redoes foundation decisions on it: its M0 close is the first
    KEEP of that gate, not a correctness transition, and without this the
    rebuilt foundation would only be read at the next plateau."""
    iter_no = start_iter
    green = initial_green
    entered_not_green = initial_green is not True
    # Rollback anchor for hard REVERTs (run_inner_iteration resets the workspace to
    # it on a correctness break / dead probe budget). A green segment enters on a
    # verified tree, so its entry HEAD is the first anchor; scaffolding/unknown
    # segments start with none — construction has no verified state to restore, and
    # resetting mid-build would erase progress, not recover any. Every KEEP advances
    # the anchor. This assignment IS the revert contract's production half: it was
    # never written, so the reset branch was dead code and REVERT was a ledger row
    # while the broken commit stayed in the tree as the next iteration's foundation.
    last_good_ref: str | None = (initial_good_ref if initial_good_ref is not None
                                 else (ctx.backend.head() if initial_green else None))
    consecutive_reverts = 0
    # iter numbers taken by rows the driver archived without a turn (a gate a
    # delivery closed): they keep status rows one-iter-per-row, but they are not
    # iterations the max_iters budget pays for
    free_rows = 0
    while True:
        if ctx.now() >= ctx.deadline_epoch:
            return "budget"
        if iter_no - start_iter - free_rows >= ctx.max_iters:
            return "max_iters"
        records = load_records(ctx.status_path)
        try:
            rec = run_inner_iteration(
                ctx, iter_no,
                metric_before=_best_ms(_on_cluster(records, ctx), ctx.gate_window),
                last_good_ref=last_good_ref,
                green=green,
            )
        except AgentError as e:
            _log(f"agent failed unrecoverably: {e}")
            return "agent_failed"
        except BackendError as e:
            _log(f"backend failed unrecoverably: {e}")
            return "backend_error"
        except Exception as e:  # unattended-run safety net: never crash, stop cleanly
            _log(f"unexpected error, stopping cleanly: {e!r}")
            return "error"
        if rec is None:
            return "exhausted"
        if rec.iter > iter_no:
            # the delivery closed its gate: the iteration filed two rows, and the
            # one returned (and checked below) is the gate row it closed
            free_rows += rec.iter - iter_no
            iter_no = rec.iter
        if ctx.phase_close and rec.decision_id == ctx.phase_close["gate"]:
            if rec.verdict == "KEEP":
                # The closed phase's gate row is on file and passed: the notice
                # described the walk up to it, and the gate's own lines judged it.
                _log(f"iter {iter_no}: {rec.decision_id} submitted (KEEP) — "
                     f"phase-close notice for {ctx.phase_close['phase']} retired")
                ctx.phase_close = None
            elif ctx.phase_close.get("stage") != "gate-failed":
                # The gate row failed: the phase is open again, but the table that
                # closed it still describes this best tree. Without a standing
                # note the student walks straight back into the null knobs it
                # priced (the iterations the close was meant to save), and the
                # next meta step re-asks the question on the same table. Keep the
                # "do not re-walk" half until the tree moves (a KEEP) or a
                # structural patch replaces the premise.
                ctx.phase_close["stage"] = "gate-failed"
                ctx.phase_close["note"] = _phase_priced_note(ctx.phase_close, rec.verdict)
                _log(f"iter {iter_no}: {rec.decision_id} submitted ({rec.verdict}) — "
                     f"{ctx.phase_close['phase']} stays open; its levers stay priced null "
                     f"on this best tree (notice reduced to 'do not re-walk')")
        elif (ctx.phase_close and ctx.phase_close.get("stage") == "gate-failed"
                and rec.verdict == "KEEP"):
            _log(f"iter {iter_no}: {rec.decision_id} KEEP moved the tree — the priced-null "
                 f"notice for {ctx.phase_close['phase']} retired")
            ctx.phase_close = None
        if stop := _prep_structural_stop(load_records(ctx.status_path)):
            # Same family as gate_unmeasured below: the bench is the card's
            # frozen file, so a prep block it never reports — or a prep
            # mismatch that repeats verbatim — does not heal with the next
            # student edit. Left as per-iteration REVERTs it would run
            # three-REVERT stuck, a meta diagnosis, and repeat to the budget.
            _log(f"iter {iter_no}: {rec.error} — structural, stopping ({stop}); fix "
                 f"eval/bench.py / window.toml, not the kernel")
            return stop
        if (ctx.gate_window == "device" and rec.verdict == "KEEP"
                and rec.bwd_median_ms is not None and rec.device_median_ms is None):
            # Structural bench-contract violation: the bench timed the e2e window
            # but not the device window this run gates on — that never heals on
            # its own, and letting it run means every KEEP carries no gate
            # reading, so target is never reached, plateau never fires, and the
            # loop grinds to its wall-clock budget with nothing steering it.
            # Checked HERE, after the record is appended, so the trajectory and
            # the pushed commit agree on resume; and only for a KEEP that DID
            # measure e2e — a wholly missing perf block (job killed mid-eval) is
            # transient and stays the tolerated no-metric row it always was, in
            # both windows. Its own stop reason, because "backend_error" reads
            # as GPU flake and sends the operator restarting instead of fixing
            # eval/bench.py (see src/forgekernel/card_template/eval/timing.py).
            _log(f"iter {iter_no}: KEEP measured bwd_median_ms but no "
                 f"device_median_ms while gate_window=device — the bench does not "
                 f"time the gate window; stopping (fix eval/bench.py, not the GPU)")
            return "gate_unmeasured"
        if rec.correctness is not None:  # scaffold NEUTRAL rows carry no verdict on green
            green = True if rec.correctness else (False if green is None else green)
        if rec.verdict == "KEEP" and rec.sha:
            last_good_ref = rec.sha  # the anchor follows the last verified state
        if rec.verdict == "REVERT":
            consecutive_reverts += 1
            flag = constant_output_flag(load_records(ctx.status_path))
            if flag:
                # the diagnosis reaches student and meta through digest["flags"];
                # this line is for the human reading the log at 2am
                _log(f"iter {iter_no}: CONSTANT-OUTPUT flag — bit-identical diff "
                     f"signature across iters {flag['iters']}; the scored outputs "
                     f"are not responding to code changes (see digest flags)")
            if consecutive_reverts >= ctx.max_consecutive_reverts:
                return "stuck"
        else:
            consecutive_reverts = 0
        records = load_records(ctx.status_path)
        best = _best_ms(_on_cluster(records, ctx), ctx.gate_window)
        if best is not None and (
                wl_why := _target_workloads_mismatch(ctx, _on_cluster(records, ctx))):
            # the cluster scale's in-job readings divide the same mismatched pair,
            # so neither the absolute hit nor its restatement is read at all;
            # the same-job parity stop below keeps its ratio currency
            _say_target_suspended(ctx, f"iter {iter_no}", wl_why)
        elif best is not None:
            scale = _cluster_scale(ctx)
            if scale is None:
                if getattr(ctx, "_scale_suspended_on", None) != _active_cluster(ctx):
                    ctx._scale_suspended_on = _active_cluster(ctx)
                    _log(f"iter {iter_no}: target stop suspended on cluster "
                         f"{_active_cluster(ctx)!r} — the reference has no reading "
                         f"there, so the home target cannot be restated")
            elif best <= _resolve_target(ctx) * scale and not ctx.ref_in_job:
                # On a ref_in_job run this absolute comparison is a cross-node
                # number that must not be reasoned from: the
                # target derives from the reference tree's node while the
                # student reads on its own node (one self-evolution run's
                # reference itself read 0.278 ms on one and 0.249 ms on the
                # other). There `_paired_parity_stop` below is the only stop
                # currency, and no fresh-ref jobs are bought for the absolute hit.
                if scale != 1.0:
                    _log(f"iter {iter_no}: target reached on cluster "
                         f"{_active_cluster(ctx)!r} against the home target scaled "
                         f"x{scale:.4f} ({_resolve_target(ctx) * scale:.4f} ms)")
                # An absolute hit is a claim about two readings drawn hours apart
                # on a bimodal cluster, so it buys a same-period check rather than
                # a stop. False (a node draw) keeps the run climbing with the
                # residual threaded into the next diagnosis; None means the check
                # itself was unavailable, and a run that cannot check must still
                # be able to finish.
                confirmed = _confirm_apparent_target(ctx, best, iter_no)
                if confirmed is not False:
                    if confirmed is None:
                        _log(f"iter {iter_no}: target stop NOT verified same-period "
                             f"(no usable re-measurement) — stopping on the launch "
                             f"anchor alone; treat the ratio as unconfirmed")
                    return "target"
                # a node draw, not a win: fall through and keep climbing
        if _paired_parity_stop(ctx, rec, iter_no):
            return "target"
        if segment_records_before is not None:
            # Post-patch segment: hold each gate row against the previous
            # segment's same gate. Checked before first_green so a rebuild whose
            # M0 already lost stops here instead of walking M1 to a plateau.
            msg = _gate_regression(records, rec, segment_records_before,
                                   gate_contract or {}, ctx.gate_window,
                                   voided=ctx.voided_ranges)
            if msg:
                _log(f"iter {iter_no}: gate regression — {msg}")
                return "gate_regression"
        if audit_on_green and entered_not_green and green is True:
            return "first_green"
        if audit_on_gate and rec.verdict == "KEEP" and rec.decision_id == audit_on_gate:
            return "first_green"
        # plateau is scoped to the post-resume SEGMENT (records since start_iter), not the
        # whole history. Otherwise a meta patch's first non-improving iteration re-fires the
        # pre-meta plateau instantly, so each patch gets only ONE iteration to prove itself
        # and meta intervenes every iteration. Segment-scoping gives each patch a real window.
        if detect_plateau(_on_cluster(records[start_iter:], ctx),
                          window=ctx.plateau_window, gate_window=ctx.gate_window):
            # A standing plateau reading with a declared package in flight is not a
            # stop: the package's measured intermediates are exactly the rows that
            # read as "no new best" (its payoff lands at part n), so an n-step
            # package all but guarantees one mid-package plateau — and stopping here
            # burns a meta step plus a diagnose round on construction the judgment
            # lane already knows to hold (_judge_patch HOLDs this same declaration).
            # The stop lane defers to the same audited plea under the same forfeit
            # rules: package finished / moved on / over its KEEP budget / stale
            # trail flips _open_package_tail to None, and the standing plateau
            # routes to meta on the very next reading. The plea buys continuation
            # only, never exemption — the eventual judgment still prices the whole
            # segment, and budget/max_iters walls above are untouched.
            pkg = _open_package_tail(records[start_iter:])
            if pkg is None:
                return "plateau"
            k, n = pkg
            _log(f"iter {iter_no}: plateau reading with a declared package in flight "
                 f"(part {k} of {n} is the segment's latest delivery) -> continuing "
                 f"construction instead of stopping")
        iter_no += 1


# --- meta loop --------------------------------------------------------------

@dataclass
class MetaResult:
    diagnosis: str
    patch_class: str
    changed_entries: list[str]
    resume_after: str | None
    invalidated: list[str]
    certified: bool | None = None
    milestone_violations: list[str] = field(default_factory=list)
    recipe: str | None = None  # a just-certified KB recipe path (G3 branch)
    gate_contract: dict = field(default_factory=dict)
    # In-place reopen plan (structural / split): the dependency closure of the
    # changed entries, and what stays built. Empty for additive / kb-example.
    reopened: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    # G5 only: the in-segment reordering the diagnose prompt promises the
    # student ("KB recipe + student intra-segment instruction reordering"). The KB branch alone delivered
    # the recipe and nothing to the student; this rides to the next student
    # turns as ctx.g5_note so the second half of the promise exists.
    student_directive: str | None = None
    # The lever pricing table this diagnosis read (plateau path), for the ledger.
    lever_prices: list = field(default_factory=list)
    # The phase the diagnosis closed on an exhausted price table (see
    # _phase_exhaustion): {"gate", "phase", "note"}; None when no phase closed.
    phase_close: dict | None = None


@dataclass
class KbCertifyResult:
    certified: bool
    probe: str | None
    rounds: int
    recipe: str | None = None  # RECIPE_CERTIFIED path — the student is pointed at it
    job: str = ""              # the certifying probe's job id (the entry's provenance pointer)


def _parse_list(text: str, field_name: str) -> list[str]:
    m = re.search(rf"{field_name}\s*:\s*\[([^\]]*)\]", text)
    if not m:
        return []
    # The reply is YAML-ish prose and the teacher quotes ids as often as not
    # (`changed_entries: ["D0.1", "D0.3"]`). A quoted id matches no DAG node, so
    # compute_resume / compute_reopen saw nothing to invalidate or reopen and the
    # landing check (which keys on the id shape) skipped it: an attn_bwd R1
    # rebuild was planned as "resume after the tip, nothing invalidated" and only
    # restarted from the baseline because no record owned that tip.
    items = (x.strip().strip("\"'`").strip() for x in m.group(1).split(","))
    return [x for x in items if x]


def _parse_field(text: str, field_name: str) -> str:
    m = re.search(rf"{field_name}\s*:\s*(\S+)", text)
    return m.group(1) if m else ""


def _parse_line(text: str, field_name: str) -> str:
    """Grab the rest of a ``field: ...`` line (multi-word values like patch_direction)."""
    m = re.search(rf"{field_name}\s*:\s*(.+)", text)
    return m.group(1).strip() if m else ""


# the reachability line, like cause_class below, must tolerate a full-width colon —
# an agent typing CJK punctuation (`reachability：`) would otherwise burn the one
# re-prompt the gate is allowed.
_REACH_LINE = re.compile(r"reachability\s*[:：]\s*(.+)")


def _parse_reachability(text: str) -> str:
    m = _REACH_LINE.search(text)
    return m.group(1).strip() if m else ""


# INFEASIBLE GATE: a stop request is accepted only on the diagnosis's own
# reachability disproof (diagnose_prompt §1c) — `-> BLOCKED` with both numbers
# and floor_opt above target x 1.20, the margin §1c itself sets for BLOCKED.
# OPEN means the disproof failed: the optimistic floor sits inside the target,
# which is a reason to keep walking, not to stop. Before this gate the first
# INFEASIBLE ended the run whatever the line said, and stops on an OPEN line
# ("every lever priced null" = levers exhausted, not target out of reach) cost
# runs most of their budget.
_REACH_VERDICT = re.compile(r"(?:->|→|=>)\s*\**\s*(BLOCKED|OPEN)\b", re.I)
_REACH_VERDICT_BARE = re.compile(r"\b(BLOCKED|OPEN)\b", re.I)
_REACH_UNIT_MS = {"ms": 1.0, "µs": 1e-3, "μs": 1e-3, "us": 1e-3}
_REACH_BLOCKED_MARGIN = 1.20


def _reach_quantity(line: str, name: str) -> float | None:
    """The first ``<name>=<number> <unit>`` on the line, in ms (µs / us accepted)."""
    m = re.search(rf"{name}\s*[=:：]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*(ms|µs|μs|us)\b", line)
    if not m:
        return None
    return float(m.group(1)) * _REACH_UNIT_MS[m.group(2)]


def _parse_reach_verdict(line: str) -> str:
    """BLOCKED / OPEN from the reachability line: the arrowed verdict (the last
    one, if the teacher restated it), else the single bare token; '' if neither
    or both appear bare."""
    arrowed = _REACH_VERDICT.findall(line)
    if arrowed:
        return arrowed[-1].upper()
    bare = {t.upper() for t in _REACH_VERDICT_BARE.findall(line)}
    return bare.pop() if len(bare) == 1 else ""


def _plateau_diagnoses(records) -> int:
    """Diagnoses this run has made that could have carried a stop request: every
    meta record except the distillation (AUTHORED) and a clean M0 audit."""
    return sum(1 for r in records if r.outcome not in ("AUTHORED", "AUDIT_CLEAN"))


def _infeasible_refusal(text: str, *, diagnoses: int, min_diagnoses: int,
                        target_ms: float | None) -> str:
    """Why an INFEASIBLE diagnosis is NOT accepted as a stop ('' = accepted).

    ``diagnoses`` counts this one; ``target_ms`` is the driver's stop threshold
    (None/0 = unknown, then the teacher's stated target alone is the bar)."""
    if diagnoses < min_diagnoses:
        return (f"this is diagnosis step {diagnoses} of this run; the loop accepts a stop "
                f"request only after at least {min_diagnoses} diagnosis steps")
    line = _parse_reachability(text)
    if not line:
        return "missing the `reachability:` line"
    verdict = _parse_reach_verdict(line)
    if verdict != "BLOCKED":
        return (f"the reachability verdict is "
                f"{verdict or 'unparseable (the line must end with -> BLOCKED or -> OPEN)'}; "
                f"only BLOCKED can support a stop")
    floor = _reach_quantity(line, "floor_opt")
    stated = _reach_quantity(line, "target")
    if floor is None or stated is None:
        return ("the BLOCKED line lacks a machine-readable `floor_opt=<number> ms` or "
                "`target=<number> ms`")
    bar = max(stated, target_ms or 0.0)
    if floor <= bar * _REACH_BLOCKED_MARGIN:
        src = ("the driver's stop threshold" if target_ms and target_ms > stated
               else "the target you wrote")
        return (f"floor_opt={floor:.6g} ms does not exceed target×{_REACH_BLOCKED_MARGIN:.2f}="
                f"{bar * _REACH_BLOCKED_MARGIN:.6g} ms (target taken from {src}: {bar:.6g} ms)")
    return ""


def _infeasible_refused_block(why: str) -> str:
    return (
        "\n## INFEASIBLE stop request not accepted (driver check)\n"
        f"Not satisfied: {why}.\n"
        "The driver accepts a stop request only when **all** of the following hold: (1) the "
        "`reachability` line reads `BLOCKED` and writes out `floor_opt=<number> ms` and "
        "`target=<number> ms`; (2) `floor_opt > target × "
        f"{_REACH_BLOCKED_MARGIN:.2f}` (target = the larger of the value you wrote and the "
        "driver's stop threshold); (3) the loop's configured minimum number of diagnosis steps "
        "has been reached. `OPEN` only says the disproof failed — the optimistic floor is still "
        "inside the target, so some routes have not been walked yet; it is no ground for a stop. "
        "\"Every in-tree lever priced null\" means the levers are exhausted, not that the target "
        "is out of reach.\n"
        "Resend a **complete** diagnosis under the usual classes (G1–G5 / R1 / R2); if you can add "
        "to `evidence` a floor construction that meets the conditions above (term-by-term "
        "arithmetic, `[MEASURED]` first), you may raise INFEASIBLE again together with the "
        "corrected `reachability` line. If it still does not meet them, the stop request is "
        "refused: this step applies no patch and re-diagnoses on the same state.\n")


# `reopen_price: <delta> via probe:<path>#<n> | lever:<name> | KB:<entry>` — the
# measured price an R1 / structural patch cites. Only the source is machine-
# checked (it must be something the driver measured this step, or a certified KB
# entry); the delta is the teacher's reading of it and travels to the record.
_REOPEN_PRICE_LINE = re.compile(r"reopen_price\s*[:：]\s*(.+)")
# the name stops at whitespace, list punctuation or EITHER paren family:
# teachers append fullwidth glosses ("lever:ring（independent）")
_REOPEN_PRICE_SRC = re.compile(r"\b(probe|lever|kb|ablation)\s*[:：]\s*([^\s,;，；()（）]+)", re.I)
_NO_READING = {"", "unreported", "n/a", "none"}
# A KB entry counts as a price source only when it is GPU-certified: its text
# carries a probe verdict, a job pointer or a [PRICED] tag. A transcribed
# recipe with none of these is a description, not a measurement. Job ids are
# 5-7 digits on the Hopper pools and 10 on the Blackwell pool (the sm100 tree's
# KEEP-census entries cite nothing else); the width matches kb_lint._JOB_ID.
_KB_CERT_MARK = re.compile(r"PROBE PASS|\[PRICED|GPU-certified|✅ Verified|probe_kb_|"
                           r"(?<![A-Za-z0-9_])job\s+\*{0,2}[0-9]{5,12}(?![0-9])")
_KB_NOT_ENTRIES = {"index", "README", "TEMPLATE"}


def _parse_reopen_price(text: str) -> str:
    m = _REOPEN_PRICE_LINE.search(text or "")
    return m.group(1).strip() if m else ""


def _reopen_gate_on(ctx: LoopContext) -> bool:
    """The no-reopen-without-a-price rule applies when the run has an in-step way
    to measure one: verdict probes or the pricing round. A control run with
    neither could only ever cite a KB entry, so the rule stands down there."""
    return bool(ctx.reopen_requires_price
                and (ctx.max_verdict_probe_rounds > 0
                     # the pricing round only runs on a priced tree
                     or (ctx.lever_pricing_rounds > 0 and ctx.stall_pricing)))


def _verify_reopen_price(cite: str, probe_readings: list, lever_table: list,
                         ctx: LoopContext, probes_run: list | None = None,
                         best_ref: str | None = None) -> str | None:
    """None when ``cite`` names a price the driver measured (or a certified KB
    entry); otherwise the reason it does not. ``probe_readings`` are this step's
    verdict-probe contract lines, ``lever_table`` this step's price table,
    ``probes_run`` the probe paths submitted this step (a probe that ran but
    printed no verdict line is "no reading", not "not run"), ``best_ref`` the
    tree the reopen would be applied to — a lever price read on another tree
    is stale and does not verify (see _reprice_cited_levers)."""
    if not cite:
        return "no reopen_price line"
    m = _REOPEN_PRICE_SRC.search(cite)
    if not m:
        return "reopen_price names no probe:/lever:/KB:/ablation: source"
    kind, name = m.group(1).lower(), m.group(2).strip("`'\"")
    if kind == "ablation":
        # an ablation delta is what the mechanism charges today — the ceiling
        # on what a reopen that fixes it can return. Only a POSITIVE delta
        # above the pass's noise floor is a price: n/a (hung, unbuilt) is no
        # reading, within-noise says the mechanism is not the term, negative
        # says removing it is slower — it is helping, and there is nothing to
        # return by reopening it.
        for lv in lever_table:
            if lv.get("kind") == "ablation" and lv.get("name") == name:
                d = lv.get("delta_pct")
                noise = lv.get("noise_pct") if lv.get("noise_pct") is not None else _LEVER_NOISE_DEFAULT_PCT
                if d is None:
                    return (f"ablation {name} has no measured delta (n/a: status="
                            f"{lv.get('status') or 'unknown'})")
                if d < -noise:
                    return (f"ablation {name} reads {d:+.2f}%: removing the mechanism is SLOWER — "
                            f"it is helping, not charging; not a price")
                if d <= noise:
                    return (f"ablation {name} reads {d:+.2f}%, within the {noise:.2f}% noise floor — "
                            f"the mechanism is not the term; not a price")
                return None
        return f"ablation {name} is not on the price table (this step or an earlier one)"
    if kind == "probe":
        path, _, idx = name.partition("#")
        path = path.lstrip("./")
        rows = [r for r in probe_readings if str(r.get("path", "")).lstrip("./") == path]
        if not rows:
            if any(str(p).lstrip("./") == path for p in (probes_run or [])):
                return f"probe {path} ran but printed no PROBE PASS/FAIL line (no reading)"
            return f"probe {path} was not run this step"
        if idx:
            rows = [r for r in rows if str(r.get("contract")) == idx]
            if not rows:
                return f"probe {path} printed no contract #{idx}"
        if not any(str(r.get("reading", "")).strip().lower() not in _NO_READING for r in rows):
            return f"probe {path} printed no reading (max_diff unreported / n/a)"
        return None
    if kind == "lever":
        for lv in lever_table:
            if lv.get("kind") == "lever" and lv.get("name") == name:
                if lv.get("add_pct") is None and lv.get("remove_pct") is None:
                    return f"lever {name} has no measured add / remove"
                return _stale_price(lv, name, best_ref)
            if lv.get("kind") == "variant" and lv.get("name") == name:
                if lv.get("delta_pct") is None:
                    return f"subset {name} has no measured delta"
                return _stale_price(lv, name, best_ref)
        return f"lever {name} is not on the price table (this step or an earlier one)"
    name = re.sub(r"\.md$", "", name.split("/")[-1])
    if name in _KB_NOT_ENTRIES:
        return f"KB:{name} is not a recipe entry"
    try:
        hit = next(kb_read_dir(ctx).rglob(f"{name}.md"), None)
        text = hit.read_text(encoding="utf-8") if hit else ""
    except OSError:
        hit, text = None, ""
    if hit is None:
        return f"KB entry {name} not found under {kb_read_dir(ctx)}"
    if not _KB_CERT_MARK.search(text):
        return (f"KB entry {name} carries no GPU certification (no PROBE PASS / "
                f"[PRICED] / job pointer) — a transcribed recipe is not a price")
    return None


def _stale_price(row: dict, name: str, best_ref: str | None) -> str | None:
    """None when ``row``'s price was read on ``best_ref`` (or either side is
    unknown — rows from before provenance was banked); else why it does not
    price a reopen of that tree."""
    on = str(row.get("priced_on") or "")
    if not best_ref or not on or on == best_ref:
        return None
    return (f"lever {name} was priced on tree {on[:10]}, not on the current best "
            f"{best_ref[:10]} — a stale price is not a price for this tree; list it in "
            f"levers.json again so the driver re-prices it here, then cite the new reading")


_UNPRICED_REOPEN_BLOCK = (
    "\n## A reopen must be priced first (driver check)\n"
    "You classified `R1` (structural reopen), but the diagnosis carries no price the driver "
    "measured. **A reopen without a measured price must not rewrite the milestone** — an observed "
    "failure mode: an audit inferred from \"where the wait instruction sits\" that the reference "
    "hid a wait, flipped the stage-split axis on that basis and had the whole group of decisions "
    "redone; the reopen itself was never priced, while the producer lever that was really worth "
    "double-digit percent never made it onto the candidates table. Take any one of the three "
    "routes, then resend a **complete** diagnosis:\n"
    "1. **Price it with a verdict experiment**: send `PROBE_REQUEST: notes/<name>.py` that builds "
    "the **minimal variant** of the reopen on the best tree (change only this one structural "
    "variable; do not deliver the full implementation), measured as a same-job pair, with the "
    "script printing `PROBE PASS/FAIL max_diff=<reading>`; then write "
    "`reopen_price: <±x.x%> via probe:notes/<name>.py#<contract index>` in the diagnosis.\n"
    "2. **Cite the pricing table**: the lever behind the reopen is already on this run's pricing "
    "table (this step or earlier, marked `priced` in the memory; **a price tag holds only for "
    "the best tree it was measured on** — a tag from an old table, read before best moved, is "
    "judged stale and must be re-listed in levers.json for a re-measure) → "
    "`reopen_price: <add or −remove> via lever:<name in the table>`; or the ablation table's "
    "reading for removing that mechanism → `reopen_price: <Δ> via ablation:<name>` (the driver "
    "records it as a ceiling: fixing it returns at most this much). If no ablation exists yet, "
    "write the ablation as JSON and send `PROBE_REQUEST: notes/<name>.json` (format: see "
    "`ablations` in the pricing round); the driver runs it with a frozen recipe, so you need "
    "not write a script.\n"
    "3. **Cite a certified KB entry**: the structural change has a job-certified KB recipe → "
    "`reopen_price: <measured delta in the entry> via KB:<entry name>`.\n"
    "If you can do none of these, land the diagnosis in a non-reopen class (G1/G2/G4/G5, or an "
    "additive R2) and write the reopen direction into `evidence` as an [ESTIMATE] candidate. "
    "If you submit an unpriced R1 again, the driver refuses this step and applies no patch.\n")


# --- reopen budget -------------------------------------------------------------

_MILESTONE_PATCH_CLASSES = ("additive", "structural", "split")
_REOPEN_PATCH_CLASSES = ("structural", "split")


def _fruitless_reopen_streak(meta_records) -> int:
    """How many reopen transactions in a row have closed without a new best.

    A transaction is what _judge_patch prices: one milestone patch, or a held
    one (HOLD_PATCH / still PENDING) plus every patch that joined it until a
    KEEP_PATCH / REVERT_PATCH closed it — the closing verdict is written on its
    last record only. A transaction reopens when any member is structural /
    split. KEEP_PATCH of any class resets the count (the tree moved, a reopen's
    premise may be new); a REVERT_PATCH transaction that reopened adds one; an
    additive-only REVERT_PATCH leaves it. A still-open transaction is not
    counted, and a patch that never landed (malformed: violations on record)
    is not a member of anything."""
    streak, open_, reopens = 0, False, False
    for r in meta_records:
        if r.patch_class not in _MILESTONE_PATCH_CLASSES or r.milestone_violations:
            continue
        if not open_:
            open_, reopens = True, False
        reopens = reopens or r.patch_class in _REOPEN_PATCH_CLASSES
        if r.outcome == "KEEP_PATCH":
            streak, open_ = 0, False
        elif r.outcome == "REVERT_PATCH":
            streak += 1 if reopens else 0
            open_ = False
    return streak


def _reopen_capped_block(streak: int, cap: int, *, bounce: bool = False) -> str:
    """The teacher-facing note once the reopen budget is spent."""
    head = ("\n## Reopen budget exhausted (driver check)\n"
            + ("You classified `R1` (structural reopen), but this" if bounce else "This")
            + f" run's last {streak} reopens in a row (structural / split patches) all ended in "
            f"REVERT_PATCH without a new best (budget {cap}). **Reopening delivered decisions is no "
            "longer accepted**: R1 diagnoses and structural / split patches are both refused "
            "(recorded as `REOPEN_CAPPED`; the milestone is not changed and the student redoes no "
            "decision). A reopen means the student re-delivers a whole group of already-delivered "
            "decisions on the best tree, and these readings say that route pays nothing on this "
            "tree.\n")
    ask = ("Routes still open: add on the current tree a new decision that does **not** reopen "
           "delivered decisions (R2 / G1 / G2 / G4, additive), or take the KB route of G3 / G5; if "
           "`reachability` reads BLOCKED and the target is out of reach without a reopen, report "
           "`INFEASIBLE` per the stop rules. Any KEEP_PATCH (a new best) restores the reopen "
           "budget.\n")
    tail = ("Resend a **complete** diagnosis along the routes above; if you submit R1 again, this "
            "step is refused and no patch is applied.\n") if bounce else ""
    return head + ask + tail


_INFEASIBLE_MENTION_BLOCK = (
    "\n## INFEASIBLE has only one channel (driver check)\n"
    "Your diagnosis mentions INFEASIBLE, but `cause_class` is not `INFEASIBLE`. The driver reads "
    "a stop request only from `cause_class`: INFEASIBLE written into the milestone or the "
    "evidence does not stop the loop; it only makes the student report EXHAUSTED and you diagnose "
    "yet another class with no lesion, idling until the budget runs out. Pick one, then resend a "
    "**complete** diagnosis:\n"
    "1. **You judge this run's target out of reach and request a stop** → "
    "`cause_class: INFEASIBLE`. The driver stops the whole loop at once (no more student rounds, "
    "no patch), and this run finishes on its current best; your diagnosis text and the "
    "`reachability` line go into the stop record. This is the end of this run, not a request "
    "for help.\n"
    "2. **It is not a stop request** (only a citation / negation, e.g. \"no ground for "
    "INFEASIBLE\") → resend under the original class; `deficient_locus` / `patch_direction` "
    "must point at a direction the student can build.\n")


# the diagnose cause_class is a closed enum; the agent occasionally editorializes in the
# field ("R1 (atypical…"). Extract just the enum token, tolerating trailing prose.
# INFEASIBLE is the plateau diagnosis's stop request, not a cause: the teacher
# judges the target out of reach and the run ends there — the run's verdict,
# recorded for a human to read (see _write_infeasible_notice), never a request
# for a human to hand the teacher a new direction. The milestone template tells the teacher to
# "declare INFEASIBLE and stop", but a declaration written only into the
# milestone had no channel to the driver — the run kept diagnosing "G4 @ none"
# against an EXHAUSTED student until the budget ran out.
_CAUSE_ENUM = re.compile(r"cause_class\s*[:：]\s*(G[1-5]|R[12]|INFEASIBLE)")
# The M0 audit adds one verdict the plateau diagnosis has no use for: OK — the
# milestone survived the reconciliation and there is nothing to patch. It has
# no INFEASIBLE: at M0 the distance to target is the expected state.
_CAUSE_ENUM_AUDIT = re.compile(r"cause_class\s*[:：]\s*(G[1-5]|R[12]|OK)")
_INFEASIBLE_MENTION = re.compile(r"INFEASIBLE")


def _parse_cause_class(text: str, *, allow_ok: bool = False) -> str:
    """The clean cause_class enum (G1-G5 / R1 / R2 / INFEASIBLE on the plateau
    path, G1-G5 / R1 / R2 / OK when ``allow_ok``), or "" if none present."""
    m = (_CAUSE_ENUM_AUDIT if allow_ok else _CAUSE_ENUM).search(text)
    return m.group(1) if m else ""


def _teacher_prompt(ctx: LoopContext, diagnosis: str, probe_feedback: str = "") -> str:
    extra = f"\n\n## DIAGNOSIS\n{diagnosis}\n"
    if probe_feedback:
        extra += probe_feedback
    return render_prompt(_meta_prompt(ctx, "kb_recipe_prompt.md"), _bindings(ctx), extra)


def _kb_certify_lint_gate(ctx: LoopContext, diagnosis: str, *, fix_rounds: int = 2) -> list[str]:
    """Lint what the teacher just wrote into the run's KB tree, bouncing findings
    back for a bounded number of fix turns. Returns whatever findings survive.

    This is the second enforcement point (loop start is the other):
    without it, a leaky recipe certified mid-run sits in the run's tree — which
    the student reads in full — for every remaining iteration, and the next start
    gate only finds out after the run's result is already contaminated. The tree
    was lint-clean at loop start, so any finding here is the teacher's new
    content. Scoped to kb_read_dir: another architecture's tree is not in this
    run's read set, so its findings are not this run's to fix."""
    findings = lint_kb_dir(kb_read_dir(ctx), extra_tokens=ctx.kb_lint_tokens)
    for _ in range(fix_rounds):
        if not findings:
            return []
        _log(f"meta/kb: leak lint — {len(findings)} finding(s) in the fresh recipe, "
             f"bouncing back to the teacher")
        listing = "\n".join(f"- {f}" for f in findings)
        prov_dir = lang_mode.provenance_dir(ctx.meta_dir.parent, ctx.language)
        _teacher(ctx.backend).run_agent(_teacher_prompt(ctx, diagnosis, probe_feedback=(
            f"\n\n## KB LEAK LINT FAILED\n"
            f"What you just wrote into the KB contains banned provenance narrative (lint "
            f"findings below). Following the Sources discipline of {ctx.kb_dir}/TEMPLATE.md, "
            f"**edit these files in place**: move the reference solution's file names / function "
            f"names / line numbers / paraphrases / footprints to {prov_dir}/<entry name>.md, and "
            f"leave only the archive-pointer line in the entry. When done, output "
            f"RECIPE_CERTIFIED: <recipe-path> as the final line.\n"
            f"{listing}\n")))
        findings = lint_kb_dir(kb_read_dir(ctx), extra_tokens=ctx.kb_lint_tokens)
    return findings


def _quarantine_leaky_kb(ctx: LoopContext, findings: list[str]) -> None:
    """Move entries that failed the certify lint out of the student-readable tree.

    The run's tree (kb_read_dir) is what every student iteration reads; leaving
    a leaky entry there while merely not announcing it would not contain
    anything. Files are moved (not deleted) to <repo>/kb_quarantine/ for the
    human to inspect; a moved tracked entry shows up as a deletion in git status,
    so the next run's kb_blockers guard forces a human through here before
    anything restarts.

    A finding's path is relative to the lint root's parent (lint_kb_dir's
    rel_to), i.e. ``<arch>/<entry>.md`` under hw_kb/; the quarantine hangs off
    the repo root, because a quarantine inside hw_kb/ would still sit in the
    tree this is trying to take the entry out of."""
    qdir = lang_mode.kb_repo_root(ctx.kb_dir) / "kb_quarantine"
    for rel in sorted({f.split(":", 1)[0] for f in findings}):
        src = kb_read_dir(ctx).parent / rel
        if src.is_file():
            qdir.mkdir(exist_ok=True)
            dst = qdir / src.name
            src.rename(dst)
            _log(f"meta/kb: QUARANTINED leaky entry {rel} -> {dst} — review by hand")


def run_kb_certify(ctx: LoopContext, diagnosis: str) -> KbCertifyResult:
    """G3 fix: teacher writes a hw_kb recipe + probe, GPU-certifies it (PASS/FAIL loop).

    Mirrors the student probe sub-loop but for the teacher: the recipe is only
    accepted once its certification probe returns PROBE PASS on real hardware.

    The certification's `kb-probe:` commits live on the -kbprobe branch (pushed
    per round, provenance); the student's lineage does not inherit them. The
    workspace is reset to the pre-certification head afterwards and the probe
    files re-materialised untracked, so the student's next KEEP is built on a
    tree the records know — its base_sha pairs with its censused ancestor and
    kb_axis_debt bills exactly the KEEP that applies the fresh recipe — and the
    student line stays teacher-free, as after a verdict probe.
    """
    be = ctx.backend
    pre_head = be.head()
    probes: dict = {}
    try:
        return _kb_certify_rounds(ctx, diagnosis, probes)
    finally:
        if probes:
            _detach_kb_probes(ctx, pre_head, probes)


def _detach_kb_probes(ctx: LoopContext, pre_head: str, probes: dict) -> None:
    be = ctx.backend
    if lost := _edits_outside_notes(be, pre_head):
        _note_discarded_edits(ctx, "meta/kb (certification reset)", lost)
    try:
        be.reset(pre_head)
    except BackendError as e:
        _log(f"meta/kb: could not reset the workspace to {pre_head[:10]} after the "
             f"certification ({e}) — the kb-probe commits stay in the student lineage")
        return
    kept = []
    for path, text in probes.items():
        if text is None:
            continue
        dst = ctx.workspace / path
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text)
            kept.append(path)
        except OSError as e:
            _log(f"meta/kb: could not re-materialise {path}: {e}")
    _log(f"meta/kb: workspace reset to the pre-certification head {pre_head[:10]}; "
         f"probe file(s) {kept or list(probes)} re-materialised untracked "
         f"(the -kbprobe branch keeps the commits)")


def _kb_certify_rounds(ctx: LoopContext, diagnosis: str, probes: dict) -> KbCertifyResult:
    be = ctx.backend
    prompt = _teacher_prompt(ctx, diagnosis)
    for r in range(ctx.max_probe_rounds):
        # safe variant: the teacher's finalize turn is naturally prose, and a
        # stray RECON_REQUEST must not parse here (recon runs in another tree).
        turn = parse_student_turn_safe(_teacher(be).run_agent(prompt))
        if not isinstance(turn, ProbeRequest):
            if leaks := _kb_certify_lint_gate(ctx, diagnosis):
                _quarantine_leaky_kb(ctx, leaks)
                _log("meta/kb: recipe REJECTED — leak lint findings survived the fix rounds")
                return KbCertifyResult(False, None, r)
            _log(f"meta/kb: teacher finalized recipe (no probe) after {r} rounds")
            return KbCertifyResult(True, None, r)
        be.commit(f"kb-probe: {turn.path}")
        probes[turn.path] = _keep_probe_across_reset(ctx, turn.path)
        kb_branch = f"{ctx.branch_prefix}-kbprobe"
        be.push_branch(kb_branch)
        kb_log = be.submit_gpu(kb_branch, f"cmd:python3 {turn.path}")
        pv = parse_probe_verdict(kb_log)
        # the job id is the entry's provenance pointer (index / status cell
        # `job N`, the harvest's `[MEASURED, job N]`): a certification whose
        # job is not on the log cannot be cited, and the loop log is the only
        # place it is ever written
        job = str(getattr(be, "last_job_id", "") or "")
        _log(f"meta/kb: recipe probe {turn.path} -> PROBE {_probe_label(pv)} "
             f"(max_diff={pv.diff_text}){_probe_log_suffix(pv)} (round {r + 1}, "
             f"job {job or '?'})")
        if pv.passed:
            # finalize: teacher marks the recipe verified in the KB index
            final_out = _teacher(be).run_agent(_teacher_prompt(
                ctx, diagnosis,
                probe_feedback=_probe_feedback(
                    turn.path, pv, kb_log, what="recipe probe",
                    on_pass="Finalize: mark the recipe verified in the KB index and "
                            "emit RECIPE_CERTIFIED: <recipe-path>. Cite the certifying "
                            f"job in the entry's status line and its index cell: "
                            f"`PROBE PASS max_diff={pv.diff_text}, job {job or '?'}` "
                            f"(round {r + 1} of this certification).")))
            recipe = _parse_field(final_out, "RECIPE_CERTIFIED")
            if leaks := _kb_certify_lint_gate(ctx, diagnosis):
                _quarantine_leaky_kb(ctx, leaks)
                _log("meta/kb: recipe REJECTED — leak lint findings survived the fix rounds")
                return KbCertifyResult(False, None, r + 1, job=job)
            _log(f"meta/kb: recipe CERTIFIED and available to the student ({recipe or '?'}; "
                 f"job {job or '?'})")
            return KbCertifyResult(True, turn.path, r + 1, recipe=recipe or None, job=job)
        prompt = _teacher_prompt(
            ctx, diagnosis,
            probe_feedback=_probe_feedback(turn.path, pv, kb_log, what="recipe probe"))
    _log("meta/kb: could not certify a recipe within the probe budget")
    return KbCertifyResult(False, None, ctx.max_probe_rounds)


def kb_harvest_debt(ctx: LoopContext) -> list[ProbeBattle]:
    """The probe victories expensive enough to owe a KB write-up, dearest first."""
    if not ctx.kb_tripwire_rounds:
        return []
    debt = [b for b in ctx.probe_battles.values()
            if b.passed and b.rounds >= ctx.kb_tripwire_rounds]
    return sorted(debt, key=lambda b: -b.rounds)


@dataclass
class AxisBattle:
    """One KEEP that moved a SASS contract axis — a harvest item without a fight.

    The kb-tripwire bills probe fights; a stuck stop bills a diagnosis. A
    one-round KEEP that changes the built structure (sync topology, in-flight
    depth, supply order, cache-policy bits — the orch_tools summary axes) had
    no channel at all, and that is exactly how a 22 % knob shipped twice at
    ~4 % and evaporated with both runs (dsa_sparse_fwd). The
    evidence is the pair of per-iteration census readings around the KEEP plus
    its own eval — the KEEP's NCU job IS the certification, the same way a
    probe's PASS is for a fight. Computed from status records, so a resumed
    run bills the same KEEPs as the one that made them.

    "Moved an axis" is read on the count-free contract signature
    (`_contract_signature`), not on raw summary equality: the
    counting fields (mbarrier object / tier / per-role counts, nbar widths,
    itc candidates, per-loop DEPBAR lines) still move with codegen after every
    real fix (11 residual `!` on the grouped-gemm pages), and billing those
    would hand the teacher compiler noise to write recipes for. The raw axes
    that changed alongside ride as ``noise_axes`` for the reader."""
    iter: int
    decision_id: str
    sha: str
    axes: dict            # axis -> {"before": ..., "after": ...}
    ms_before: float | None
    ms_after: float | None
    change: str
    prev_iter: int | None = None
    noise_axes: list = field(default_factory=list)

    @property
    def delta_pct(self) -> float | None:
        if self.ms_before and self.ms_after:
            return round(100.0 * (self.ms_after - self.ms_before) / self.ms_before, 2)
        return None


def _axis_changes(prev: dict, cur: dict) -> tuple[dict, list[str]]:
    """({axis: {before, after}} over the contract axes whose count-free
    signature differs, [raw summary keys that differ but are not a contract
    move]). The before/after readings are the raw axis values (the evidence
    the teacher reads), the trigger is the signature."""
    from forgekernel.skills.orch_tools import _DIFF_KEYS
    a = json.loads(json.dumps(prev))
    b = json.loads(json.dumps(cur))
    sa, sb = _contract_signature(a), _contract_signature(b)
    moved = {k: {"before": a.get(k), "after": b.get(k)}
             for k in _CONTRACT_DIFF_AXES if sa.get(k) != sb.get(k)}
    noise = [k for k in _DIFF_KEYS if k not in moved and a.get(k) != b.get(k)]
    return moved, noise


def _census_ancestor(rec, by_sha: dict):
    """The nearest censused KEEP in ``rec``'s own lineage: walk base_sha ->
    record until one carries a census. None when the chain leaves the records
    (a baseline / meta-patched tree) — then there is nothing to compare with."""
    sha = rec.base_sha
    seen: set = set()
    while sha and sha in by_sha and sha not in seen:
        seen.add(sha)
        p = by_sha[sha]
        if p.verdict == "KEEP" and p.orch_summary:
            return p
        sha = p.base_sha
    return None


def kb_axis_debt(records, gate_window: str = GATE_WINDOW_DEFAULT) -> list[AxisBattle]:
    """The KEEPs whose census differs from the previous censused KEEP's IN THEIR
    LINEAGE, in iteration order. Records without a summary (census skipped,
    pre-census runs) are transparent: the comparison is between the two nearest
    KEEPs that carry one, and the first censused KEEP has nothing to compare
    against (it is the baseline, not a move).

    Lineage, not adjacency: after a meta reopen / reset the next KEEP is built
    on an earlier tree, and diffing it against the rolled-back KEEP would bill
    the rollback to its change record. A record that knows the tree it was
    built on (``base_sha``) is paired through that chain; records from before
    the field existed fall back to the previous censused KEEP."""
    debt: list[AxisBattle] = []
    prev_adjacent = None
    by_sha: dict = {}
    for r in records:
        if r.sha:
            by_sha[r.sha] = r
        if r.verdict != "KEEP" or not r.orch_summary:
            continue
        prev = _census_ancestor(r, by_sha) if r.base_sha else prev_adjacent
        if prev is not None:
            axes, noise = _axis_changes(prev.orch_summary, r.orch_summary)
            if axes:
                debt.append(AxisBattle(
                    iter=r.iter, decision_id=r.decision_id, sha=r.sha, axes=axes,
                    ms_before=gate_ms(prev, gate_window), ms_after=gate_ms(r, gate_window),
                    change=r.change, prev_iter=prev.iter, noise_axes=noise))
        prev_adjacent = r
    return debt


def _log_axis_change(ctx: LoopContext, rec) -> None:
    """Name the axis a just-landed KEEP moved, at the moment it lands (the bill
    itself is settled from the records at harvest time)."""
    if not rec.orch_summary:
        return
    try:
        records = load_records(ctx.status_path)
        debt = [b for b in kb_axis_debt(records, ctx.gate_window) if b.iter == rec.iter]
    except Exception as e:  # noqa: BLE001 — a log line, never a precondition
        _log(f"iter {rec.iter}: kb-axis: could not compare census ({e.__class__.__name__}: {e})")
        return
    for b in debt:
        d = f" ({b.ms_before} -> {b.ms_after} ms, {b.delta_pct:+.2f}%)" if b.delta_pct is not None else ""
        _log(f"iter {rec.iter}: kb-axis: {rec.decision_id} moved SASS contract axes "
             f"{sorted(b.axes)}{d} — queued for the end-of-run KB harvest write-up")


def _axis_battle_block(b: AxisBattle) -> str:
    d = f"{b.ms_before} -> {b.ms_after} ms"
    if b.delta_pct is not None:
        d += f" ({b.delta_pct:+.2f}%)"
    axes = "\n".join(
        f"  - `{k}`:\n      before (iter {b.prev_iter}): {json.dumps(v['before'], ensure_ascii=False)}\n"
        f"      after  (iter {b.iter}): {json.dumps(v['after'], ensure_ascii=False)}"
        for k, v in b.axes.items())
    noise = (f"- Counting axes that moved at the same time (codegen follow-on; not billed, "
             f"no recipe): {', '.join(b.noise_axes)}\n"
             if b.noise_axes else "")
    return (f"### iter {b.iter} {b.decision_id} (sha {b.sha[:10] if b.sha else '?'}) — "
            f"gate window {d}\n- Contract axes changed:\n{axes}\n{noise}"
            f"- Student change record (first 600 chars):\n```\n{b.change[:600]}\n```")


def _harvest_lint_findings(ctx: LoopContext) -> list[str]:
    try:
        return lint_kb_dir(kb_read_dir(ctx), extra_tokens=ctx.kb_lint_tokens)
    except Exception as e:  # noqa: BLE001 — a lint crash must not sink the harvest
        _log(f"meta/kb-harvest: kb_lint unavailable ({e.__class__.__name__}: {e})")
        return []


def run_kb_harvest(ctx: LoopContext) -> list[str]:
    """End-of-run KB write-up for expensive probe victories (the kb-tripwire)
    and for KEEPs that moved a SASS contract axis (kb_axis_debt).

    The G3 path certifies fresh recipes with its own GPU loop; this path exists
    for the fights G3 never sees — the student won, so the run has no ``stuck``
    stop and no diagnosis, and the lesson lives only in probe notes and commit
    messages. The winning probe already returned PROBE PASS on real hardware, so
    that run IS the certification evidence: one teacher write-up turn citing the
    probe as provenance. The axis items are the same shape with a different
    evidence face: the census before/after and the KEEP's own eval reading.
    Zero GPU jobs, zero meta steps, and any failure here degrades to the status
    quo (lessons stay in the notes).

    Returns the recipe paths the teacher reported via ``RECIPE_RECORDED:`` lines.
    """
    debt = kb_harvest_debt(ctx)
    try:
        axis_debt = kb_axis_debt(load_records(ctx.status_path), ctx.gate_window)
    except Exception as e:  # noqa: BLE001 — never let the axis bill block the probe bill
        _log(f"meta/kb-harvest: axis bill unavailable ({e.__class__.__name__}: {e})")
        axis_debt = []
    # The gap queue (unverified recon rows) is deliberately NOT a harvest item:
    # it is the reference's structure as narrated, and hw_kb takes only
    # probe-certified recipes (see the gap-queue section below).
    if not debt and not axis_debt:
        return []
    # Everything from here on is best-effort, and the caller sits BEFORE the
    # harvest report: an exception escaping this function would
    # take down the one code path that keeps uncommitted hw_kb recipes from
    # being a `git checkout` away from gone. So the promise is total — a missing
    # prompt template (version skew) degrades exactly like a dead agent.
    try:
        extra = ""
        if debt:
            battles = "\n\n".join(
                f"### {b.path} — {b.rounds} GPU rounds before PASS (iter {b.pass_iter})\n"
                f"- Final verdict: {b.verdict_line}\n"
                f"- Tail of the winning job log:\n```\n{b.log_tail}\n```"
                for b in debt)
            extra += (f"\n\n## Probe battle list (expensive probe wins in this run, most costly "
                      f"first)\n\n{battles}\n")
        if axis_debt:
            moves = "\n\n".join(_axis_battle_block(b) for b in axis_debt)
            extra += (f"\n\n## Contract-axis change list (KEEP patches that changed a SASS contract "
                      f"axis, in iter order; readings = that iter's own eval + the dossier census "
                      f"before and after)\n\n{moves}\n")
        prompt = render_prompt(
            _meta_prompt(ctx, "kb_harvest_prompt.md"), _bindings(ctx), extra=extra)
        _log(f"meta/kb-harvest: {len(debt)} expensive probe victor(ies) at threshold "
             f"{ctx.kb_tripwire_rounds} rounds + {len(axis_debt)} axis-moving KEEP(s) "
             f"— one write-up turn, no GPU")
        out = _teacher(ctx.backend).run_agent(prompt)
        # What the teacher wrote is the next run's exam paper: the same lint the
        # loop start enforces runs here, once, with the findings —
        # otherwise a harvest that slipped narrative into hw_kb blocks every
        # later `forge loop` start until someone hand-edits the entry.
        findings = _harvest_lint_findings(ctx)
        if findings:
            _log(f"meta/kb-harvest: {len(findings)} lint finding(s) in the write-up "
                 f"-> one re-prompt: {findings[0]}")
            out = _teacher(ctx.backend).run_agent(prompt + (
                "\n\n## What you wrote into the KB last time failed kb_lint (fix it and resend)\n"
                "None of the lines below may stay in any file under `{KB_READ_DIR}` "
                "(index.md included): run names / dates, iter numbers, commit / cubin shas, "
                "where the reference implementation came from. Move the narrative to the "
                "merge-request description or `docs/kb-provenance/`; evidence brackets carry "
                "only job numbers / probe paths / readings.\n"
                + "\n".join(f"- {f}" for f in findings[:40]) + "\n").replace(
                    "{KB_READ_DIR}", str(kb_read_dir(ctx))))
            left = _harvest_lint_findings(ctx)
            if left:
                _log(f"meta/kb-harvest: {len(left)} lint finding(s) remain after the "
                     f"re-prompt — the next loop start will refuse them: {left[0]}")
    except Exception as e:  # noqa: BLE001 — degrade to the status quo, never raise
        _log(f"meta/kb-harvest: failed ({e.__class__.__name__}: {e}) — "
             f"lessons stay in the probe notes")
        return []
    recipes = re.findall(r"^RECIPE_RECORDED:\s*(\S+)", out, re.M)
    if recipes:
        for r in recipes:
            _log(f"meta/kb-harvest: recorded {r}")
    else:
        _log("meta/kb-harvest: no RECIPE_RECORDED line from the teacher — "
             "check the KB diff by hand (NO_RECIPE or unparseable turn)")
    return recipes


# --- dossier contract signature ------------------------------------------------------
#
# The count-free structural core of an orch dossier, per contract axis. Read by
# the KB axis harvest (`_axis_changes`): a KEEP "moved an axis" when this
# signature differs between the censused KEEPs of its lineage, not when the raw
# summary does. (It also gated the residual-ledger window, a late-plateau
# diagnosis mode since removed: on the two runs that reached its regime the
# window never opened — the `!` it keys on is written by wait-encoding choices
# as much as by structure — and the lever pricing round now answers "where is
# the remaining time" with a measurement instead of a teacher-written list.)
#
# Dossier axes that encode the milestone's declared contracts ([SYNC:] ->
# named-barrier verdicts + mbarrier rings, [GATHER:] -> supply order, [PHASE:]
# -> merge placement / commit depth). The other axes count things codegen
# moves without any structural difference — spin micro-loops, unresolved
# mbarrier objects, barrier widths, a copy-producer -> spin edge — and stayed
# `!` on the grouped-gemm pages after every real fix (an offline check: 11
# residual `!`, all counting axes). Billing them would hand the teacher
# compiler noise to write recipes for.
_CONTRACT_DIFF_AXES = ("nbar_verdicts", "rings", "supply", "commit", "cache_ops")
# cache_ops rides with the contract axes: a pricing probe
# put the cp.async `.BYPASS` bit at -22.2 % device time on a dsa_sparse_fwd
# best tree — a `!` on it is as structural as a missing ring. The axis is
# compared as the set of cache-MODE classes, not the verbatim opcode set: the
# same probe read the `.L2::256B` prefetch hint (`LTC256B`) at ~0 and reachable
# only through inline PTX, and a closed-source ref carries it plus its
# `CREATEPOLICY` descriptors — verbatim equality would bill exactly the kernel
# that already has the 22 % bit right.
_L2_HINT_RE = re.compile(r"\.LTC\d+B")
_CACHE_DESCRIPTOR_FAMS = {"CREATEPOLICY"}


def _cache_class(op: str) -> str | None:
    """A cache-policy opcode with its L2 prefetch hint stripped; None for the
    policy-descriptor instructions (they carry no load/store of their own)."""
    if op.split(".")[0] in _CACHE_DESCRIPTOR_FAMS:
        return None
    return _L2_HINT_RE.sub("", op)
# Even those axes carry counts in their printed form (a verdict repeated per
# barrier id, "copies 40" vs "copies 16", one DEPBAR line per loop region), so
# a plain equality read them as `!` on the grouped-gemm pages (an offline
# replay) while both kernels carry the same contracts. The signature is
# count-free per axis: the SET of barrier verdict kinds, the ring
# set, the supply KIND (serial gather vs prefetched), the DEPBAR depth set per
# role. Merge placement is a nomination axis (§2g: a ledger reading only earns a nomination) and stays out.
_SUPPLY_TIGHT_RE = re.compile(r"tight-chained copies (\d+)")


def _contract_signature(summary: dict) -> dict:
    """The count-free structural core of a dossier summary, per contract axis."""
    s = json.loads(json.dumps(summary or {}))
    kinds = sorted({re.sub(r"\s*width=\S+", "", v).strip()
                    for v in (s.get("nbar_verdicts") or [])})
    supply = []
    for line in (s.get("supply") or []):
        # orch_tools now emits the class itself ("...; serial-gather" /
        # "...; prefetched"); summaries banked before that carried the raw
        # counted line, which the tight-chain count still classifies.
        if "serial-gather" in line or "prefetched" in line:
            supply.append("serial-gather" if "serial-gather" in line else "prefetched")
            continue
        m = _SUPPLY_TIGHT_RE.search(line)
        tight = int(m.group(1)) if m else 0
        supply.append("serial-gather" if tight > 0 else "prefetched")
    depths = {}
    for line in (s.get("commit") or []):
        role, _, rest = line.partition(":")
        depths.setdefault(role.strip(), set()).update(re.findall(r"LE (0x[0-9a-fA-F]+)", rest))
    return {"nbar_verdicts": kinds,
            "rings": sorted(s.get("rings") or []),
            "supply": sorted(set(supply)),
            "commit": {r: sorted(v) for r, v in sorted(depths.items())},
            "cache_ops": sorted({c for c in map(_cache_class, s.get("cache_ops") or [])
                                 if c})}
_LEVER_LINE_RE = re.compile(
    r"^LEVER (?P<name>\S+) add=(?P<add>[+-]?[\d.]+%|n/a) remove=(?P<remove>[+-]?[\d.]+%|n/a) "
    r"add_ms=(?P<add_ms>\S+) remove_ms=(?P<remove_ms>\S+) class=(?P<cls>\S+) "
    r"spin_add=(?P<spin_add>\S+) spin_all_minus=(?P<spin_rem>\S+)", re.M)
_STACK_LINE_RE = re.compile(
    r"^STACK all=(?P<all>[+-]?[\d.]+%|n/a) sum_singles=(?P<sum>[+-]?[\d.]+%|n/a) "
    r"coupling=(?P<coupling>[+-]?[\d.]+%|n/a)"
    r"(?: all_pos=(?P<all_pos>[+-]?[\d.]+%|n/a) all_pos_levers=(?P<pos_levers>\S+) "
    r"dropped=(?P<dropped>\S+))?", re.M)
# the teacher's own subsets (and the tool's automatic all-pos), one line each
_VARIANT_LINE_RE = re.compile(
    r"^VARIANT (?P<name>\S+) levers=(?P<levers>\S+) delta=(?P<delta>[+-]?[\d.]+%|n/a) "
    r"ms=(?P<ms>\S+) spin=(?P<spin>\S+) q=(?P<q>.*)$", re.M)
_PRICE_OK_RE = re.compile(r"^PRICE-OK levers=(\d+) rounds=(\d+) base_ms=(\S+) all_ms=(\S+)"
                          r"(?: spin_base=(\S+) spin_all=(\S+) tensor_base=(\S+) tensor_all=(\S+))?", re.M)
_PRICE_FAIL_RE = re.compile(r"^PRICE-FAIL reason=(.*)$", re.M)
_PRICE_NOISE_RE = re.compile(r"^PRICE noise_floor=([\d.]+)% metric=(\S+)", re.M)
_PRICE_TIMED_RE = re.compile(r"^PRICE-TIMED levers=", re.M)
# a group card priced on its representative workload only (lever_pricing_run
# WORKLOAD_SCOPE_ENV): the table is a reading on that shape, the gate the group's geomean
_PRICE_SCOPE_RE = re.compile(r"^PRICE-SCOPE workloads=(\S+) rep=(\S+) of=(\S+)", re.M)
_ABLATE_OK_RE = re.compile(r"^ABLATE-OK ablations=", re.M)
_TOOL_TRACEBACK = "Traceback (most recent call last)"
# the ablation pass: one line per ablation (an edit that removes a mechanism the
# tree has, timed on the NCU kernel clock so a broken kernel still reads)
_ABLATE_LINE_RE = re.compile(
    r"^ABLATE (?P<name>\S+) delta=(?P<delta>[+-]?[\d.]+%|n/a) ncu_ns=(?P<ns>\S+) "
    r"base_ncu_ns=(?P<base_ns>\S+) correctness=(?P<corr>\S+) device_delta=(?P<dd>[+-]?[\d.]+%|n/a) "
    # the tool prints the kernel as one token; a log from before it did may
    # carry a demangled name with spaces — the field runs to the next key
    r"kernel=(?P<kernel>.+?)(?: status=(?P<status>\S+))? h=(?P<h>.*)$", re.M)
_ABLATE_NOISE_RE = re.compile(r"^PRICE ablation_noise_floor=([\d.]+)% metric=(\S+)", re.M)
_ABLATE_FAIL_RE = re.compile(r"^ABLATE-FAIL reason=(.*)$", re.M)
# The seeded pricing tool's verdict when a timed-out variant left the GPU held
# (lever_pricing_run.NODE_DIRTY_MARKER): PRICE-FAIL / ABLATE-FAIL reason=node-dirty
# and a NODE-DIRTY line. Readings after that line are the orphan's share of the
# GPU, not the variant's; the design is re-priced once.
_NODE_DIRTY_MARKER = "NODE-DIRTY"


def _price_log_verdict(log: str) -> str | None:
    """The closing verdict a lever_pricing_run log carries, None when it carries
    none. The tool ends every run with one of PRICE-OK / PRICE-FAIL / ABLATE-OK /
    ABLATE-FAIL; PRICE-TIMED (the timing copy of the table, roles pending) counts
    only when the log says how the job ended — the archive is whole (the role
    pass died on its own and the entry script closed), or the job was KILLED
    after the timing copy (observed: the per-variant `stall_price_run --pm` role
    pass outlived TASK_TIMEOUT on two B200 runs, `timeout --signal=KILL` took
    the job at rc=137, and a killed entry script never prints its closing line;
    the table those 4 jobs printed was complete and consistent, and refusing it
    bought the same table again, lost it the same way, and diagnosed blind).
    Not when the log simply ends there, which is an archive read between the
    two copies. A log with none of these is not a table, whatever LEVER rows it
    holds: the rows are the first copy, cut wherever the archiver was."""
    for name, rx in (("PRICE-OK", _PRICE_OK_RE), ("PRICE-FAIL", _PRICE_FAIL_RE),
                     ("ABLATE-OK", _ABLATE_OK_RE), ("ABLATE-FAIL", _ABLATE_FAIL_RE)):
        if rx.search(log or ""):
            return name
    if _PRICE_TIMED_RE.search(log or "") and (archive_is_whole(log) or _price_role_pass_killed(log)):
        return "PRICE-TIMED"
    return None


def _price_role_pass_killed(log: str) -> str | None:
    """The kill sentence when ``log`` holds the timing copy of a price table
    (PRICE-TIMED) and then the job's kill evidence (``parse.job_abend``), None
    otherwise. The timing copy is printed once every timed variant has its
    rounds; whatever the kill took was the role pass (spin / tensor columns) or
    the ablation pass — the LEVER / VARIANT / STACK rows stand as timed."""
    if not _PRICE_TIMED_RE.search(log or ""):
        return None
    end = job_abend(log or "")
    return end.reason if end.kind == JOB_END_KILLED else None


def _price_log_settled(be, log: str, what: str) -> str:
    """``log`` once it carries a verdict: as read, or the job's archive read
    again: a first read that stopped short of the verdict may be complete on a
    second read — a re-read is the free recovery. Unchanged when the backend cannot re-read (test fakes)
    or the re-read is no longer."""
    if _price_log_verdict(log) is not None:
        return log
    job = str(getattr(be, "last_job_id", "") or "")
    reread = getattr(be, "reread_log", None)
    if not job or not callable(reread):
        return log
    _log(f"{what}: job {job} log carries no PRICE/ABLATE verdict — re-reading the archive")
    try:
        again = reread(job)
    except BackendError as e:
        _log(f"{what}: re-read of job {job} failed ({e})")
        return log
    return again if len(again or "") > len(log or "") else log


def _price_log_lost_reason(log: str) -> str:
    """Why a verdict-less pricing log is not a table, for the log and the teacher."""
    if _TOOL_TRACEBACK in (log or ""):
        return "the pricing tool crashed"
    return "archive incomplete or the tool never ran"


def _node_dirty_reason(fail: str | None, log: str) -> str | None:
    if fail and fail.startswith("node-dirty"):
        return f"PRICE-FAIL {fail}"
    if (m := _ABLATE_FAIL_RE.search(log or "")) and m.group(1).strip().startswith("node-dirty"):
        return f"ABLATE-FAIL {m.group(1).strip()}"
    if _NODE_DIRTY_MARKER in (log or ""):
        return "the job reported NODE-DIRTY"
    return None


def _lever_price_job(be, branch: str, cmd: str, what: str,
                     rerun: bool = True) -> "tuple[str, str]":
    """One pricing job, settled: (log, job id). Re-bought once (``rerun``) when its GPU
    was dirty or its log still carries no verdict (a crashed tool is not re-bought). BackendError
    propagates."""
    attempts = 2 if rerun else 1
    for attempt in range(attempts):
        log = be.submit_gpu(branch, cmd)
        log = _price_log_settled(be, log, what)
        _, _, fail = parse_lever_prices(log)
        last = attempt == attempts - 1
        if not last and _node_dirty_reason(fail, log):
            # the job's GPU held an orphan (a hung variant's launch target): the
            # readings after it are the orphan's share of the GPU — the same design
            # is re-priced once, no design turn spent
            _log(f"{what}: {_node_dirty_reason(fail, log)} — re-pricing the same "
                 f"design once")
            continue
        if not last and _price_log_verdict(log) is None and _TOOL_TRACEBACK not in log:
            # still no verdict after the archive was re-read: the job is lost to the
            # archiver, not to the design — the same spec is bought once more (a tool
            # that crashed would crash again: no resubmit for that)
            _log(f"{what}: job {getattr(be, 'last_job_id', '') or '?'} log still carries no "
                 f"PRICE/ABLATE verdict — resubmitting the same design once")
            continue
        break
    return log, str(getattr(be, "last_job_id", "") or "")


def _lever_price_submit(ctx: LoopContext, be, branch: str, cmd: str, what: str,
                        rerun: bool = True) -> "tuple[str, str]":
    """The pricing log the parser reads, and the job it cites: one job, or -- a group card
    listing its meta workloads (loop/meta_shapes.py) -- one job per workload
    (``--workload <name>``) merged into one table. The representative's own failure (an
    edit that does not apply, the job environment, a lost archive) is the table's and
    comes back as its raw log, handled as on a one-workload card; another workload's is
    named in the merged table and leaves it out. BackendError on the representative's
    job propagates. The workloads are priced one after the other, the representative
    first."""
    shapes = meta_shapes.card_meta_workloads(ctx.workspace)
    if not shapes:
        return _lever_price_job(be, branch, cmd, what, rerun)

    def price(wl):
        if wl != shapes[0] and ctx.now() > ctx.deadline_epoch - 1800:
            return ("budget",)
        try:
            # the role pass on the representative alone: on the others its columns were
            # never read and took a third of each job
            roles = "" if wl == shapes[0] else " --no-roles"
            log, job = _lever_price_job(be, branch, f"{cmd}{roles} --workload {wl}",
                                        f"{what} [{wl}]", rerun)
        except BackendError as e:
            return ("error", e)
        return ("log", log, job)

    results: dict = {shapes[0]: price(shapes[0])}
    r = results[shapes[0]]
    if r[0] == "error":
        raise r[1]
    _, _, fail = parse_lever_prices(r[1])
    if _price_log_verdict(r[1]) is None or fail:
        return r[1], r[2]
    for wl in shapes[1:]:
        results[wl] = price(wl)
    per: list = []
    missing: dict = {}
    for wl in shapes:
        r = results.get(wl) or ("budget",)
        if r[0] == "budget":
            missing[wl] = "budget"
            continue
        if r[0] == "error":
            missing[wl] = f"job failed ({r[1]})"[:80]
            continue
        _, log, job = r
        _, _, fail = parse_lever_prices(log)
        if _price_log_verdict(log) is None or fail:
            missing[wl] = fail or "no verdict"
            _log(f"{what}: workload {wl} has no table ({missing[wl]}) — merged without it")
            continue
        per.append((wl, log, job))
    return meta_shapes.merge_lever_logs(per, missing=missing), per[0][2]


# ablations per pricing job / per JSON verdict probe (== lever_pricing_run.MAX_ABLATIONS)
_LEVER_MAX_ABLATIONS = 4
_LEVER_NOISE_DEFAULT_PCT = 0.5   # == lever_pricing_run.NOISE_FLOOR_PCT
_LEVER_SPEC_NAME = "levers.json"
_LEVER_PRICING_CMD = "cmd:python3 notes/lever_pricing_run.py notes/levers.json"
# design-turn re-prompts after a PRICE-FAIL (an `old` that does not apply)
_LEVER_PRICE_RETRIES = 1
# the teacher's own subsets per pricing job (each is one more variant timed)
_LEVER_MAX_SUBSETS = 6


def _pct_or_none(tok: str):
    return None if tok in (None, "n/a") else float(tok.rstrip("%"))


def parse_lever_prices(log: str) -> "tuple[list, dict | None, str | None]":
    """(levers, stack, fail_reason) from a lever_pricing_run log."""
    # the run prints the table twice — once when timing is done (PRICE-TIMED),
    # once with the role columns filled (PRICE-OK); a job killed in between
    # still left the first copy. Keep the last copy of each lever / STACK line.
    by_name: dict = {}
    for m in _LEVER_LINE_RE.finditer(log):
        by_name[m["name"]] = {
            "name": m["name"], "add_pct": _pct_or_none(m["add"]),
            "remove_pct": _pct_or_none(m["remove"]),
            "add_ms": None if m["add_ms"] == "None" else _float_or_none(m["add_ms"]),
            "remove_ms": None if m["remove_ms"] == "None" else _float_or_none(m["remove_ms"]),
            "cls": m["cls"], "spin_add_ns": _float_or_none(m["spin_add"]),
            "spin_all_minus_ns": _float_or_none(m["spin_rem"])}
    levers = list(by_name.values())
    by_variant: dict = {}
    for m in _VARIANT_LINE_RE.finditer(log):
        by_variant[m["name"]] = {
            "kind": "variant", "name": m["name"],
            "levers": [x for x in m["levers"].split(",") if x],
            "delta_pct": _pct_or_none(m["delta"]),
            "ms": None if m["ms"] == "None" else _float_or_none(m["ms"]),
            "spin_ns": _float_or_none(m["spin"]), "question": m["q"].strip()}
    levers += list(by_variant.values())
    by_ablation: dict = {}
    for m in _ABLATE_LINE_RE.finditer(log):
        by_ablation[m["name"]] = {
            "kind": "ablation", "name": m["name"], "delta_pct": _pct_or_none(m["delta"]),
            "ncu_ns": _float_or_none(m["ns"]), "base_ncu_ns": _float_or_none(m["base_ns"]),
            "correctness": m["corr"], "device_delta_pct": _pct_or_none(m["dd"]),
            "kernel": m["kernel"], "status": m["status"] or ("ok" if m["delta"] != "n/a" else "n/a"),
            "hypothesis": m["h"].strip()}
    if by_ablation:
        an = None
        for an in _ABLATE_NOISE_RE.finditer(log):
            pass
        for row in by_ablation.values():
            row["noise_pct"] = _float_or_none(an.group(1)) if an else _LEVER_NOISE_DEFAULT_PCT
    levers += list(by_ablation.values())
    sm = None
    for sm in _STACK_LINE_RE.finditer(log):
        pass
    stack = None
    if sm:
        stack = {"all_pct": _pct_or_none(sm["all"]), "sum_singles_pct": _pct_or_none(sm["sum"]),
                 "coupling_pp": _pct_or_none(sm["coupling"]),
                 "all_pos_pct": _pct_or_none(sm["all_pos"]),
                 "all_pos_levers": [x for x in (sm["pos_levers"] or "").split(",") if x and x != "none"],
                 "dropped": [x for x in (sm["dropped"] or "").split(",") if x and x != "none"]}
        om = None
        for om in _PRICE_OK_RE.finditer(log):
            pass
        if om:
            stack.update({"base_ms": _float_or_none(om.group(3)), "all_ms": _float_or_none(om.group(4)),
                          "spin_base_ns": _float_or_none(om.group(5)), "spin_all_ns": _float_or_none(om.group(6)),
                          "tensor_base": _float_or_none(om.group(7)), "tensor_all": _float_or_none(om.group(8))})
        nm = None
        for nm in _PRICE_NOISE_RE.finditer(log):
            pass
        if nm:
            stack.update({"noise_pct": _float_or_none(nm.group(1)), "metric": nm.group(2)})
        sc = None
        for sc in _PRICE_SCOPE_RE.finditer(log):
            pass
        if sc:
            stack["workload_scope"] = {"workloads": sc.group(1), "rep": sc.group(2), "of": sc.group(3)}
    meta_shapes.attach_shape_rows(log, levers, stack)   # a table merged over workloads
    fm = _PRICE_FAIL_RE.search(log)
    return levers, stack, (fm.group(1).strip() if fm else None)


def _float_or_none(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None


# --- phase exhaustion -----------------------------------------------------------
#
# The price table is a batch measurement of exactly the knobs a phase asks the
# student to A/B one iteration at a time. When every lever a phase owns reads
# null on the best tree, the phase's remaining walk can only re-measure the
# table — but nothing consumed the table: the driver handed it to the teacher
# and the student, who never sees it, kept walking the phase (observed on a
# dsa_sparse run: five levers null at the M0 audit, then six iterations and
# two meta steps of D1.1 wait-depth A/Bs inside the noise band, one of them
# −32%). The reading below turns the table into a phase-level fact and makes
# the diagnosis answer it: close the phase (the student submits the gate row
# next, judged by the gate's own lines) or name the ONE decision of the phase
# no lever priced and walk that. Unanswered twice, the driver closes — a gate
# row that fails just returns the student to normal work with its diagnosis,
# while a phase walked for nothing is a budget the run never gets back.
_PHASE_EXHAUSTED_CLASSES = ("null", "noop", "harmful")


def _table_noise_pct(table: list) -> float:
    """The noise floor the pricing job read its table at (the tool's `PRICE
    noise_floor=` line, banked on the stack row); the tool's own default when
    a table predates the field."""
    for t in table or []:
        if t.get("kind") == "stack" and t.get("noise_pct") is not None:
            return float(t["noise_pct"])
    return _LEVER_NOISE_DEFAULT_PCT
# the value stays on the line: `\s*` would run through a newline and read the
# NEXT field's first token as the verdict of an empty `phase_verdict:` line
_PHASE_VERDICT_LINE = re.compile(r"phase_verdict[ \t]*[:：][ \t]*([^\n]*)")


def _current_phase(records, phases) -> int | None:
    """The phase the student is walking: the lowest of ``phases`` (the k of every
    ``D<k>.<n>`` the milestone declares) whose ``M<k>-gate`` has no KEEP row.
    None when every declared gate is closed (or nothing is declared)."""
    closed = {int(m.group(1)) for r in records
              if r.verdict == "KEEP" and (m := re.match(r"M(\d+)-gate$", r.decision_id or ""))}
    return next((k for k in sorted(set(phases)) if k not in closed), None)


def _delivered_decisions(records) -> set:
    """Decision ids with a KEEP row (``D1.1+D1.3`` counts both)."""
    out: set = set()
    for r in records:
        if r.verdict == "KEEP":
            out.update(p.strip() for p in (r.decision_id or "").split("+")
                       if _DECISION_ID.match(p.strip()))
    return out


_REOPEN_CLOSURE_RE = re.compile(r"closure \[([^\]]*)\] is reopened")


def _live_records(records, voided=None) -> list:
    """The rows that still describe a standing tree — what every phase /
    parity reading is computed on.

    Two kinds of row are on file but void (the student's ledger is append-only):
    rows inside a segment a REVERT_PATCH rolled back (``voided``: [lo, hi) index
    ranges), and KEEP rows for decisions a later in-place REOPEN row reopened
    (the row itself tells the student those KEEPs are void; the gate KEEP of a
    reopened phase is void with them — its closure was premised on the old
    values). A REOPEN-REVERT row restores what its REOPEN voided."""
    voided = voided or []
    stack: list = []          # (reopen row index, decision ids, phase ks) still standing
    for i, r in enumerate(records):
        if r.decision_id == "REOPEN":
            m = _REOPEN_CLOSURE_RE.search(r.change or "")
            ids = {p.strip() for p in (m.group(1).split(",") if m else []) if p.strip()}
            ks = {int(mm.group(1)) for d in ids if (mm := re.match(r"D(\d+)\.", d))}
            stack.append((i, ids, ks))
        elif r.decision_id == "REOPEN-REVERT" and stack:
            stack.pop()
    out = []
    for i, r in enumerate(records):
        if any(lo <= i < hi for lo, hi in voided):
            continue
        if r.verdict == "KEEP" and stack:
            parts = {p.strip() for p in (r.decision_id or "").split("+")}
            gate = re.match(r"M(\d+)-gate$", r.decision_id or "")
            if any(i < ri and ((parts & ids) or (gate is not None and int(gate.group(1)) in ks))
                   for ri, ids, ks in stack):
                continue
        out.append(r)
    return out


def _phase_exhaustion(ctx: LoopContext, records, table: list) -> dict | None:
    """The phase-exhaustion reading of a price table, or None.

    Non-None when the phase the student is walking has at least one lever on
    the table and every one of them is null / noop: {"phase", "gate",
    "levers", "priced" (phase decisions some lever moved), "unpriced" (phase
    decisions no lever moved and no KEEP delivered), "delivered"}. A phase with
    an independent or coupled lever is not exhausted (the lever IS the way
    forward); a phase past the milestone's last gate has nothing to close.
    """
    levers = [t for t in table if t.get("kind") == "lever"]
    if not levers:
        return None
    dag = _milestone_dag(ctx)
    by_phase: dict = {}
    for d in dag:
        if (m := re.match(r"D(\d+)\.", d.id)):
            by_phase.setdefault(int(m.group(1)), []).append(d.id)
    k = _current_phase(records, by_phase)
    if k is None:
        return None
    phase_ids = sorted(by_phase[k])
    own = [lv for lv in levers if set(lv.get("decisions") or []) & set(phase_ids)]
    if not own:
        return None
    if any(str(lv.get("cls")) not in _PHASE_EXHAUSTED_CLASSES for lv in own):
        return None
    # A subset (the teacher's, or the tool's all-pos) that moves one of this
    # phase's decisions and reads above the noise floor IS a way forward: the
    # single-add / remove classes of its members can all read null while the
    # combination pays (the reading rule handed to the teacher says the subset
    # wins over the class column — the phase reading must agree with it).
    noise = _table_noise_pct(table)
    for v in table:
        if v.get("kind") != "variant" or not (set(v.get("decisions") or []) & set(phase_ids)):
            continue
        if v.get("delta_pct") is not None and v["delta_pct"] > noise:
            return None
    priced = sorted({d for lv in own for d in lv["decisions"] if d in phase_ids})
    delivered = sorted(_delivered_decisions(records) & set(phase_ids))
    unpriced = [d for d in phase_ids if d not in priced and d not in delivered]
    return {"phase": f"M{k}", "gate": f"M{k}-gate",
            "levers": [lv["name"] for lv in own],
            "priced": priced, "unpriced": unpriced, "delivered": delivered}


def _phase_exhausted_block(ex: dict) -> str:
    unpriced = (", ".join(f"`{d}`" for d in ex["unpriced"])
                if ex["unpriced"] else "(none: every undelivered decision of this phase has a priced lever)")
    return (
        f"\n## Phase levers exhausted (the driver read this off the pricing table)\n"
        f"The student is working on **{ex['phase']}**. This phase's levers in the pricing table, "
        f"{', '.join(f'`{n}`' for n in ex['levers'])} (moving {', '.join(f'`{d}`' for d in ex['priced'])}), "
        f"**all measured null / noop** on the best tree — any further point-by-point A/B in this phase "
        f"would only re-measure that table (one point per iter, each inside the noise band). "
        f"Delivered in this phase: {', '.join(ex['delivered']) or '(none)'}; "
        f"decisions with no priced lever and not delivered: {unpriced}.\n"
        f"The output block **must** add one line (driver check: a missing line is bounced once, "
        f"missing again is treated as close):\n"
        f"```\n"
        f"phase_verdict: close | continue:<D id>\n"
        f"```\n"
        f"- `close`: the remaining decisions of this phase count as settled (the pricing table is "
        f"their measurement); next round the student **submits the `{ex['gate']}` row directly** "
        f"(still submitted on its own, still judged by the gate's own acceptance bar; a gate FAIL is "
        f"attributed as a FAIL, without going back to tune this phase's knobs). If the gap sits in a "
        f"next-phase decision (supply organisation, per-element refinement, ...), close is the way to "
        f"get the student there fastest; to pull a next-phase decision forward, close first and then "
        f"adjust it in the patch.\n"
        f"- `continue:<D id>`: may only name one decision from the \"no priced lever\" list above, and "
        f"`evidence` must say which lever of it the pricing table has not measured; the driver "
        f"validates the id and lets the student continue on it.\n"
        f"This line sits beside `cause_class` and the other fields and does not replace them: G3/G5 "
        f"still add to the KB, R1 still prices first.\n")


def _parse_phase_verdict(text: str) -> str:
    m = _PHASE_VERDICT_LINE.search(text or "")
    if not m:
        return ""
    # teachers gloss the token ("close (M1 fully measured)"), quote it (`close`) and
    # space the continue form ("continue: D1.4" / "continue:`D1.4`"): cut at
    # either paren family (as _REOPEN_PRICE_SRC and the locus parser do), drop
    # the quoting, close the gap after "continue:", then keep the first token.
    raw = re.split(r"[（(]", m.group(1))[0].strip().strip("`'\"* ")
    raw = re.sub(r"^continue\s*[:：]\s*[`'\"*]*", "continue:", raw, flags=re.I)
    tok = raw.split()[0] if raw.split() else ""
    return tok.strip("`'\"*.;，。").strip()


def _verify_phase_verdict(verdict: str, ex: dict) -> str | None:
    """None when ``verdict`` answers the exhaustion reading; else why not."""
    if not verdict:
        return "no phase_verdict line"
    v = verdict.lower()
    if v == "close":
        return None
    if v.startswith("continue:"):
        did = verdict.split(":", 1)[1].strip().strip("`'\"*")
        if did in ex["unpriced"]:
            return None
        if did in ex["priced"]:
            return f"continue:{did} names a decision the table already priced null"
        if did in ex["delivered"]:
            return f"continue:{did} names a decision already delivered (KEEP row)"
        return f"continue:{did} is not a decision of {ex['phase']} without a priced lever"
    return f"phase_verdict {verdict!r} is neither close nor continue:<D id>"


def _phase_close_note(ex: dict, verdict: str, job: str = "") -> str:
    """The student-facing notice for a phase verdict (rides every turn until
    the gate row lands; see LoopContext.phase_close)."""
    priced = ", ".join(f"`{d}`" for d in ex["priced"]) or "(none)"
    head = (f"- **Phase close (driver read of the pricing table, teacher verdict `{verdict}`)**: the "
            f"pricing round measured {ex['phase']}'s levers {', '.join(f'`{n}`' for n in ex['levers'])} "
            f"on the best tree, all null / noop"
            f"{f' (job {job})' if job else ''}; the decisions they move, {priced}, are settled by "
            f"that measurement — **no more point-by-point A/B**. ")
    if verdict.startswith("continue:"):
        walk = verdict.split(":", 1)[1].strip()
        return (head + f"The teacher ruled that only `{walk}` is still worth doing in this phase "
                f"(it has a lever the pricing table has not measured): do `{walk}` this round, and "
                f"once it is delivered, **submit the `{ex['gate']}` row directly next round** "
                f"(the gate row is still submitted on its own and judged by the gate's own acceptance "
                f"bar; a FAIL is attributed as a FAIL).\n")
    left = ", ".join(f"`{d}`" for d in ex["unpriced"]) or "(none)"
    return (head + f"The other undelivered decisions of this phase, {left}, count as settled too; "
            f"**submit the `{ex['gate']}` row directly this round** "
            f"(the gate row is still submitted on its own and judged by the gate's own acceptance "
            f"bar; a FAIL is attributed as a FAIL, without going back to tune this phase's knobs). "
            f"Once the gate passes, move to the next phase in phase order.\n")


def _phase_priced_note(pc: dict, verdict: str) -> str:
    """The reduced notice after a closed phase's gate row FAILED: the phase is
    open again, but the table that closed it still describes this best tree."""
    levers = ", ".join(f"`{n}`" for n in pc.get("levers", [])) or "(the levers on the pricing table)"
    return (f"- **Phase levers already priced (driver)**: the `{pc['gate']}` row was submitted but "
            f"judged {verdict}, so {pc['phase']} is open again; but the pricing round's reading on this "
            f"best tree — {pc['phase']}'s levers {levers} all null / noop — still holds: **do not go "
            f"back to point-by-point A/B of these knobs** (every point would land inside the noise "
            f"band); act on the teacher's attribution of the gate FAIL instead. This notice is "
            f"withdrawn automatically when the best tree changes (any KEEP lands) or a structural "
            f"patch lands.\n")


def _lever_price_lost_block(note: str) -> str:
    return (f"\n## The pricing job produced no table\n{note}**None of the levers designed this time "
            f"was priced**: a citation of one may only be marked `est:<x>%`, never `priced_by=`; a "
            f"structural reopen still needs a `reopen_price` that cites a measured source.\n")


def _lever_price_block(table: list) -> str:
    """The measured price table for the diagnose prompt, with the reading rules."""
    levers = [t for t in table if t.get("kind") == "lever"]
    stack = next((t for t in table if t.get("kind") == "stack"), None)
    rows = ["| lever | add-one | remove-one | class | Δspin(ns, +L) | Δspin(ns, all−L) |",
            "|---|---|---|---|---|---|"]
    for lv in levers:
        rows.append(f"| `{lv['name']}` | {_fmt_pct(lv['add_pct'])} | {_fmt_pct(lv['remove_pct'])} | "
                    f"**{lv['cls']}** | {lv.get('spin_add_ns')} | {lv.get('spin_all_minus_ns')} | "
                    f"{', '.join(lv.get('decisions') or []) or '—'} |")
    rows[0] += " decisions |"
    rows[1] += "---|"
    block = ("\n## Pricing table (measured by the driver on the best tree: add-one / remove-one / "
             "all-on, same-job paired ratios)\n")
    block += meta_shapes.scope_note(stack)
    if levers:
        block += "\n".join(rows) + "\n"
    else:
        block += "(No candidate levers this step; only the ablation table follows.)\n"
    trees = {str(lv.get("priced_on")) for lv in levers if lv.get("priced_on")}
    if len(trees) == 1:
        (tree,) = trees
        jobs = sorted({str(lv.get("job")) for lv in levers if lv.get("job")})
        block += (f"(Measured tree = best `{tree[:10]}`" + (f", job {', '.join(jobs)}" if jobs else "")
                  + ". **The price tags hold only for this tree**: once best moves, the driver "
                  "re-measures every lever the milestone cites on the new tree automatically, and "
                  "citations that reversed / faded are marked SUSPENDED — see the \"Price-tag review\" "
                  "block.)\n")
    variants = [t for t in table if t.get("kind") == "variant"]
    if variants:
        vrows = ["| subset | members | vs base | Δspin(ns) | question it answers |", "|---|---|---|---|---|"]
        for v in variants:
            vrows.append(f"| `{v['name']}` | {', '.join(v.get('levers') or [])} | "
                         f"{_fmt_pct(v.get('delta_pct'))} | {v.get('spin_ns')} | {v.get('question') or '—'} |")
        block += ("\n**Subsets (the combinations you named in the design turn; `all-pos` is the "
                  "driver's automatic \"all-on minus the harmful levers\", re-measured in a second pass "
                  "interleaved with base and read against that pass's own base)**:\n"
                  + "\n".join(vrows) + "\n")
    if stack:
        block += (f"\nall-on = {_fmt_pct(stack.get('all_pct'))}; sum of add-ones = "
                  f"{_fmt_pct(stack.get('sum_singles_pct'))}; "
                  f"coupling = {_fmt_pct(stack.get('coupling_pp'))} (negative = they eat each other, "
                  f"positive = they enable each other). "
                  f"base {stack.get('base_ms')} ms → all {stack.get('all_ms')} ms; "
                  f"spin exposure {stack.get('spin_base_ns')} → {stack.get('spin_all_ns')} ns; "
                  f"tensor {stack.get('tensor_base')} → {stack.get('tensor_all')}%.\n")
        if stack.get("dropped"):
            pos = stack.get("all_pos_pct")
            block += (f"**Harmful levers {', '.join(f'`{d}`' for d in stack['dropped'])} "
                      f"(negative add-one, and all-on does not depend on them) went into all-on**, "
                      f"and they skew every `remove` reading (remove is measured on the all-on that "
                      f"contains them) — "
                      + (f"all-on without them, `all-pos` = {_fmt_pct(pos)} (members "
                         f"{', '.join(stack.get('all_pos_levers') or [])}); all-on and the class column "
                         f"go by it.\n"
                         if pos is not None else
                         "the driver could not measure `all-pos` (fewer than two members left), so "
                         "all-on and the class column are not trustworthy; go by the add-ones and "
                         "the subsets.\n"))
    block += (
        "\n**How to read it (hard rules)**: `independent` = this group pays on its own; fix in "
        "add order. `coupled` = ≈ 0 alone but taking it out of all-on loses — you split a group "
        "that must go in together: it is the companion of another candidate in the table, and the "
        "diagnosis **must merge it with the candidate that depends on it into one ⊗ package** "
        "(price the package with −remove); it must not be judged useless on its own. "
        "`dependent` = the add-one pays but taking it out of all-on loses nothing — someone else "
        "swallowed the gain; rank it last. `null` = ≈ 0 both ways; drop it. `harmful` = negative "
        "add-one **and** taking it out of all-on loses nothing — the direction is backwards and must "
        "not be proposed as a direction again (base never had it; its reverse is not a gain). A "
        "negative add-one whose removal from all-on does lose is `coupled`, not harmful — it costs "
        "on its own but all-on cannot do without it; treat it as a ⊗ package. `noop` = the edit did "
        "not change the cubin; the spelling is ineffective. `broken` = the variants containing it "
        "failed (did not compile / were incorrect): this lever **was not priced**, and the all-on "
        "and remove-one readings that contain it are n/a too — the direction is not falsified, the "
        "spelling is rejected; either rewrite it and propose it again, or mark it unpriced in the "
        "diagnosis. `unstacked` = the add-one has a reading but the all-on variant failed "
        "correctness, so its coupling with the others is unmeasured. **Every price the diagnosis "
        "cites must be a measured value from this table** (add for an independent item; −remove, a "
        "subset delta or an all-on difference for a package); write `priced_by=<lever or subset name "
        "from the table>` to point at it; a quantity the table does not have may only be marked "
        "`est:<x>%` and must not be cited as measured. "
        "The class is derived mechanically by the driver from add-one / remove-one: `coupled` only "
        "says that **some** lever depends on it, not which one — the subset deltas you specified "
        "are the judge; when a subset and the class disagree, the subset wins. "
        "The ceiling of the pricing table: the spin exposure that all-on still leaves belongs to "
        "something outside this batch of candidates; the diagnosis must say whose it is.\n")
    block += meta_shapes.shape_block(table)
    block += _ablation_block(table)
    return block


def _ablation_block(table: list) -> str:
    """The ablation rows of a price table (design-turn `ablations` or a JSON
    verdict probe) with their reading rules, or nothing."""
    abl = [t for t in table if t.get("kind") == "ablation"]
    if not abl:
        return ""
    noise = next((a.get("noise_pct") for a in abl if a.get("noise_pct") is not None),
                 _LEVER_NOISE_DEFAULT_PCT)
    rows = ["| ablation | Δ(NCU kernel clock) | status | correctness | Δdevice | decisions | hypothesis |",
            "|---|---|---|---|---|---|---|"]
    for a in abl:
        rows.append(f"| `{a['name']}` | {_fmt_pct(a.get('delta_pct'))} | {a.get('status') or '—'} | "
                    f"{a.get('correctness')} | {_fmt_pct(a.get('device_delta_pct'))} | "
                    f"{', '.join(a.get('decisions') or []) or '—'} | {a.get('hypothesis') or '—'} |")
    return (
        "\n**Ablation table (in the same job the driver removes / degrades a mechanism already on the "
        "tree and re-measures the NCU kernel clock in the same pass as base; "
        f"noise floor {noise:.2f}%)**:\n" + "\n".join(rows) + "\n"
        "**How to read it (hard rules)**: Δ positive and above the noise floor = faster without it = "
        "**the cost this mechanism charges now** (hypothesis holds); inside the band = this mechanism "
        "is not the current main term (hypothesis falsified); **Δ negative beyond the noise = slower "
        "without it = this mechanism is helping** (hypothesis falsified, and it is not something to "
        "reopen); `n/a` = no reading (status: `hung` the variant hung, `nodata` ncu captured nothing, "
        "`budget` the ablation budget ran out), **not 0**. "
        "An ablation reading is the upper bound on **how much fixing this mechanism can win back at "
        "most**, not the gain of any particular fix: it can convict, and it can price an R1 "
        "(`reopen_price: <Δ> via ablation:<name>`, recorded by the driver as an upper bound), but it "
        "must not be cited as the gain of an additive patch (a patch's gain goes through a measured "
        "candidate lever). `correctness=fail` is expected (an ablation may compute wrong results); "
        "Δdevice has a reading only when the ablation happens to stay correct.\n")


def _fmt_pct(x):
    return "n/a" if x is None else f"{x:+.2f}%"


def _merge_lever_tables(records, current: list) -> list:
    """The levers this run has priced: earlier steps' tables (the reflection
    shows them as priced, so a citation of one must verify) under the current
    step's (which wins on a name clash). Only lever rows travel; the stack row
    is the current step's own."""
    merged: dict = {}
    for r in records:
        for lv in (getattr(r, "lever_prices", None) or []):
            if lv.get("kind") in ("lever", "variant", "ablation") and lv.get("name"):
                merged[lv["name"]] = dict(lv, step=r.meta_step)
    out = list(merged.values())
    names = {lv["name"] for lv in current if lv.get("kind") in ("lever", "variant", "ablation")}
    out = [lv for lv in out if lv["name"] not in names] + list(current)
    return out


def _run_lever_pricing(ctx: LoopContext, priced: "PricedTrees", bindings: dict,
                       diagnose_extra: str, prompt_file: str = "diagnose_prompt.md") -> list:
    """The lever pricing round: one design turn, one GPU job, a measured table.

    Runs at every teacher entry that has a priced tree — the plateau diagnosis
    and the M0 audit alike (``prompt_file`` is the entry's own prompt, so the
    design turn reads the audit's question at an audit).

    Design turn: the teacher reads the same evidence the diagnosis will read
    (ledger, dossier diff, role exposure) and writes ``notes/levers.json`` — up
    to ``lever_pricing_max_levers`` composable exact-text edits of the best
    tree's kernel file, each a lever it would otherwise have to estimate a
    price for. The driver seeds ``lever_pricing_run.py``, commits, runs it on
    the best tree and parses the LEVER / STACK lines. No table (design turn
    wrote nothing, job refused an edit, budget spent) is logged and the
    diagnosis proceeds as before — the round never blocks a step.
    """
    if ctx.lever_pricing_rounds <= 0 or ctx.stall_pricing is False:
        return []
    be = ctx.backend
    best_ref = priced.best_sha or priced.head_sha
    if not best_ref:
        return []
    banked = _banked_lever_table(ctx, best_ref)
    if banked:
        # the same best tree was priced earlier in this run (a retried meta step,
        # an audit followed by a plateau on the same first green): the readings
        # are on the record, another design turn + job would only re-buy them
        _log(f"meta/lever-price: reusing the table priced on best {best_ref[:10]} "
             f"earlier this run ({len([t for t in banked if t.get('kind') == 'lever'])} lever(s))")
        return banked
    if ctx.now() > ctx.deadline_epoch - 3600:
        _log("meta/lever-price: < 60 min of budget left — skipping the pricing round")
        return []
    spec_path = Path(ctx.workspace) / "notes" / _LEVER_SPEC_NAME
    pre_head = be.head()
    if lost := _edits_outside_notes(be, pre_head):
        _note_discarded_edits(ctx, "meta/lever-price (checkout of the best tree)", lost)
    try:
        be.reset(best_ref)
    except BackendError as e:
        _log(f"meta/lever-price: could not check out best {best_ref[:10]} ({e}) — skipping")
        return []
    table: list = []
    storm_budget = [TRUNCATION_STORM_BUDGET]
    try:
        _seed_pricing_toolkits(ctx.workspace, be)
        design_extra = diagnose_extra + _LEVER_DESIGN_BLOCK.replace(
            "{MAX_LEVERS}", str(ctx.lever_pricing_max_levers)).replace(
            "{KERNEL_FILE}", str(ctx.kernel_file)).replace(
            "{PRICING_JOBS}", _lever_pricing_jobs_text(ctx))
        for attempt in range(1 + _LEVER_PRICE_RETRIES):
            try:
                spec_path.unlink()
            except OSError:
                pass
            out = _agent_turn(
                be, render_prompt(_meta_prompt(ctx, prompt_file), bindings,
                                  extra=design_extra),
                label="meta/lever-design", storm_budget=storm_budget,
                # a legal design turn either wrote the lever file or declined
                left_a_trace=lambda o: (spec_path.is_file()
                                        or re.search(r"^LEVERS:\s*none\b", o, re.M) is not None))
            if not spec_path.is_file():
                _log("meta/lever-price: the teacher declared no composable levers "
                     "(or wrote no levers.json) — diagnosing without a price table")
                return []
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                levers = spec.get("levers") or []
                ablations = spec.get("ablations") or []
            except (OSError, json.JSONDecodeError) as e:
                _log(f"meta/lever-price: levers.json unreadable ({e}) — skipping")
                return []
            if not levers and not ablations:
                _log("meta/lever-price: levers.json names no lever and no ablation "
                     "— diagnosing without a price table")
                return []
            levers = levers[:ctx.lever_pricing_max_levers]
            known_ids = {d.id for d in _milestone_dag(ctx)}
            ablations = _normalise_ablations(ctx, ablations, known_ids)
            for lv in levers:
                # one token on the LEVER line, so the parser and every later
                # citation (`priced_by=` / `lever:<name>`) see the same name
                lv["name"] = re.sub(r"\s+", "_", str(lv.get("name", "")).strip()) or "unnamed"
                # the milestone decisions this lever moves the value of — the
                # link the phase-exhaustion reading needs (a table row alone
                # cannot say which phase it priced); unknown ids are dropped
                # loudly, never bounced (a mis-typed id is not worth a turn)
                ids = [str(d).strip() for d in (lv.get("decisions") or [])]
                bad = [d for d in ids if d not in known_ids]
                if bad:
                    _log(f"meta/lever-price: lever {lv['name']} names decision(s) not in "
                         f"the milestone {bad} — dropped from its attribution")
                lv["decisions"] = [d for d in ids if d in known_ids]
                if not lv["decisions"]:
                    _log(f"meta/lever-price: lever {lv['name']} names no milestone decision "
                         f"— priced, but it cannot count toward any phase's exhaustion")
            # the teacher's subsets: names one token, members lever names
            subsets = []
            for sub in (spec.get("variants") or [])[:_LEVER_MAX_SUBSETS]:
                if not isinstance(sub, dict):
                    continue
                sub = dict(sub)
                sub["name"] = re.sub(r"\s+", "_", str(sub.get("name", "")).strip())
                sub["levers"] = [re.sub(r"\s+", "_", str(m).strip()) for m in (sub.get("levers") or [])]
                subsets.append(sub)
            spec["variants"] = subsets
            spec["levers"] = levers
            spec["ablations"] = ablations
            shapes = meta_shapes.card_meta_workloads(ctx.workspace)
            spec["rounds"] = meta_shapes.rounds_for(ctx.lever_pricing_rounds, len(shapes))
            spec["kernel_file"] = str(ctx.kernel_file)
            # The tool's own build check, run here on the same tree first: a
            # subset that shadows a lever, a member that is not a stack lever,
            # an `old` that does not apply — each used to cost a GPU job to
            # learn (PRICE-FAIL lever-edit). Skipped only when the kernel file
            # cannot be read locally; the job then judges as before.
            if fail := _lever_spec_refused(ctx, spec):
                _log(f"meta/lever-price: levers.json refused before the job — {fail}")
                if attempt < _LEVER_PRICE_RETRIES:
                    design_extra += _lever_refusal_block(ctx, fail)
                    continue
                return []
            spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
            be.commit(f"meta-price: {len(levers)} lever(s) on {best_ref[:10]}")
            branch = f"{ctx.branch_prefix}-leverprice"
            be.push_branch(branch)
            solo = [lv["name"] for lv in levers if lv.get("in_stack", True) is False]
            _log(f"meta/lever-price: pricing {len(levers)} lever(s) "
                 f"[{', '.join(lv.get('name', '?') for lv in levers)}] on best {best_ref[:10]} "
                 + (f"(one job per workload of {', '.join(shapes)}, merged: " if shapes else "(one job: ")
                 + f"base / +each / all / all-each x {spec['rounds']} rounds"
                 + (f"; {len(subsets)} teacher subset(s) [{', '.join(sb['name'] for sb in subsets)}]"
                    if subsets else "")
                 + (f"; solo-only [{', '.join(solo)}]" if solo else "")
                 + (f"; {len(ablations)} ablation(s) [{', '.join(a['name'] for a in ablations)}] "
                    f"on the NCU clock" if ablations else "") + ")")
            try:
                log, price_job = _lever_price_submit(ctx, be, branch, _LEVER_PRICING_CMD,
                                                     "meta/lever-price")
            except BackendError as e:
                _log(f"meta/lever-price: job failed ({e}) — diagnosing without a price table")
                return []
            parsed, stack, fail = parse_lever_prices(log)
            if _price_log_verdict(log) is None:
                # No verdict: the archive was read before the job finished
                # writing (observed in an offline replay: a job log came back 60 lines
                # long, cut mid-variant, and an empty table was banked as if
                # measured) or the tool never ran. Not a table — whatever LEVER
                # rows the cut log holds are the timing copy, incomplete.
                job = str(getattr(be, "last_job_id", "") or "")
                why = _price_log_lost_reason(log)
                _log(f"meta/lever-price: job {job or '?'} log carries no PRICE-OK / PRICE-FAIL / "
                     f"ABLATE verdict ({why}) — diagnosing without a price table")
                ctx.lever_price_lost = (
                    f"Pricing job {job or '?'} was bought but its log has no PRICE-OK / PRICE-FAIL / "
                    f"ABLATE verdict line ({why}); this entry **has no measured price table**. ")
                return []
            if (killed := _price_role_pass_killed(log)) and _price_log_verdict(log) == "PRICE-TIMED":
                # the timing copy is whole; the kill took the role pass (and any
                # ablation pass after it) — priced from the timing copy, role
                # columns n/a, rather than bought again to be killed again
                _log(f"meta/lever-price: job {getattr(be, 'last_job_id', '') or '?'}: {killed} after "
                     f"the timing table (PRICE-TIMED) — the role pass was cut by the job timeout; "
                     f"pricing from the timing copy (spin / tensor columns n/a)")
            if fail and (why := env_failure(log)) is not None:
                # The base IS the best tree, which passed its own eval: an import
                # that dies in the pricing job is the job's environment (the tool's
                # bare `python3`, loop/job_env.py). The tool still says
                # "base-failed-correctness", and a retry design turn read that as
                # its tree or its levers.json — neither can fix an interpreter.
                _note_env_failure(ctx, "lever-price", why)
                job = str(getattr(be, "last_job_id", "") or "")
                _log(f"meta/lever-price: PRICE-FAIL {fail} — job environment ({why}), not "
                     f"the base tree and not levers.json: no design retry, diagnosing "
                     f"without a price table")
                ctx.lever_price_lost = (
                    f"Pricing job {job or '?'} could not start in the job environment (`{why}`) — the "
                    f"base tree's own eval passed, so this is neither the base tree nor levers.json "
                    f"but the job environment (the operator tree's gpu_entry.sh / eval/, or the "
                    f"image); ")
                return []
            if fail:
                _log(f"meta/lever-price: PRICE-FAIL {fail}")
                if attempt < _LEVER_PRICE_RETRIES:
                    design_extra += _lever_refusal_block(ctx, fail)
                    continue
                return []
            _note_env_ok(ctx, "lever-price")
            if (afail := _ABLATE_FAIL_RE.search(log)):
                # the lever table stands; only the ablation pass is lost
                _log(f"meta/lever-price: ABLATE-FAIL {afail.group(1).strip()}")
            for lv in parsed:
                lv.setdefault("kind", "lever")
            attribution = {lv["name"]: lv["decisions"] for lv in levers}
            attribution.update({a["name"]: a["decisions"] for a in ablations})
            for lv in parsed:
                if lv["kind"] in ("lever", "ablation"):
                    lv["decisions"] = list(attribution.get(lv["name"], []))
                else:
                    # a subset moves whatever its members move
                    lv["decisions"] = sorted({d for m in lv.get("levers", [])
                                              for d in attribution.get(m, [])})
            table = list(parsed)
            if stack:
                stack["kind"] = "stack"
                table.append(stack)
            for lv in table:
                # provenance on every row: a price is a reading on ONE tree, and
                # a later citation of it (a reopen, a milestone rung) is checked
                # against the tree it is about to be applied to
                lv["priced_on"] = best_ref
                lv["job"] = price_job
            _log("meta/lever-price: " + "; ".join(
                f"{lv['name']} add={_fmt_pct(lv['add_pct'])} remove={_fmt_pct(lv['remove_pct'])} "
                f"[{lv['cls']}]" for lv in parsed if lv["kind"] == "lever")
                + (f"; all={_fmt_pct(stack['all_pct'])} coupling={_fmt_pct(stack['coupling_pp'])}"
                   if stack else "")
                + (f"; all_pos={_fmt_pct(stack['all_pos_pct'])} (dropped {','.join(stack['dropped'])})"
                   if stack and stack.get("all_pos_pct") is not None else "")
                + meta_shapes.scope_log(stack))
            if variant_rows := [lv for lv in parsed if lv["kind"] == "variant"]:
                _log("meta/lever-price: subsets " + "; ".join(
                    f"{lv['name']}={{{','.join(lv['levers'])}}} {_fmt_pct(lv['delta_pct'])}"
                    for lv in variant_rows))
            if abl_rows := [lv for lv in parsed if lv["kind"] == "ablation"]:
                _log("meta/lever-price: ablations " + "; ".join(
                    f"{lv['name']} {_fmt_pct(lv['delta_pct'])} (ncu, correctness={lv['correctness']})"
                    for lv in abl_rows))
            _persist_lever_table(ctx, table, best_ref, price_job, spec_levers=levers)
            break
    finally:
        # The pricing commits are `add -A` snapshots of whatever the design turns
        # touched; the reset below throws all of it away so the student line stays
        # teacher-free. Anything outside notes/ in there was never part of the
        # pricing spec — a teacher's job-environment fix once rode one pricing
        # job this way and then vanished without a word.
        if lost := _edits_outside_notes(be, best_ref):
            _note_discarded_edits(ctx, "meta/lever-price", lost)
        try:
            be.reset(pre_head)
        except BackendError as e:
            _log(f"meta/lever-price: could not restore head {pre_head[:10]} ({e})")
    return table


def _lever_spec_refused(ctx: LoopContext, spec: dict) -> str | None:
    """The tool's PRICE-FAIL reason for ``spec`` on the workspace's kernel file
    (the tree is checked out at best when this runs), or None when the job
    would accept it. None too when the file cannot be read here."""
    from forgekernel.skills.lever_pricing_run import (build_ablation_variants, build_variants,
                                                       check_ablations, check_subsets,
                                                       lever_name, stack_names)
    try:
        src = (Path(ctx.workspace) / str(spec.get("kernel_file") or ctx.kernel_file)
               ).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    levers = spec.get("levers") or []
    try:
        subsets = (check_subsets(spec.get("variants") or [], stack_names(levers),
                                 [lever_name(lv) for lv in levers]) if levers else [])
        if levers:
            build_variants(src, levers, subsets)
        abl = check_ablations(spec.get("ablations") or [],
                              [lever_name(lv) for lv in levers] + [sb["name"] for sb in subsets])
        build_ablation_variants(src, abl)
    except (ValueError, KeyError, TypeError) as e:
        return f"lever-edit ({e})"
    return None


def _normalise_ablations(ctx: LoopContext, ablations: list, known_ids: set) -> list:
    """The teacher's ablations the way the tool and the table will see them:
    at most _LEVER_MAX_ABLATIONS, one-token names, `decisions` restricted to
    milestone ids (unknown ones dropped loudly, as for levers)."""
    out = []
    for a in list(ablations)[:_LEVER_MAX_ABLATIONS]:
        if not isinstance(a, dict):
            continue
        a = dict(a)
        a["name"] = re.sub(r"\s+", "_", str(a.get("name", "")).strip()) or "unnamed"
        ids = [str(d).strip() for d in (a.get("decisions") or [])]
        bad = [d for d in ids if d not in known_ids]
        if bad:
            _log(f"meta/lever-price: ablation {a['name']} names decision(s) not in "
                 f"the milestone {bad} — dropped from its attribution")
        a["decisions"] = [d for d in ids if d in known_ids]
        a["hypothesis"] = " ".join(str(a.get("hypothesis", "")).split())
        out.append(a)
    return out


def _lever_refusal_block(ctx: LoopContext, fail: str) -> str:
    return (f"\n## The pricing job rejected the lever table\n`{fail}`. Every `old` must occur "
            f"exactly once in {ctx.kernel_file}, and still exactly once each after all levers are "
            f"stacked in order; every subset name in `variants` must be unique, **must not equal any "
            f"lever's name**, and its members must be lever names from the table (with `in_stack` not "
            f"switched off); `ablations` holds at most {_LEVER_MAX_ABLATIONS} entries, whose names "
            f"must not equal a lever or subset name, whose `old` must likewise occur exactly once, "
            f"and whose edit must not be a no-op. "
            f"Fix levers.json, then end the turn.\n")


def _banked_lever_table(ctx: LoopContext, best_ref: str) -> list:
    """The last table this run priced on ``best_ref`` (evidence/lever_prices.jsonl),
    or [] when that tree was never priced."""
    ev = _evidence_dir(ctx, create=False)
    if ev is None or not (ev / "lever_prices.jsonl").is_file():
        return []
    found: list = []
    try:
        for line in (ev / "lever_prices.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("reprice"):
                continue        # a price-check's partial table, not a teacher table
            if rec.get("best_ref") == best_ref and rec.get("table"):
                found = [_stamp_price_row(dict(lv), str(rec.get("best_ref") or ""),
                                          str(rec.get("job") or "")) for lv in rec["table"]]
    except (OSError, ValueError) as e:
        _log(f"meta/lever-price: could not read lever_prices.jsonl ({e}) — pricing afresh")
        return []
    return found


def _persist_lever_table(ctx: LoopContext, table: list, best_ref: str, job: str,
                         spec_levers: list | None = None, reprice: bool = False) -> None:
    """Bank a priced table under the tree it was read on. ``spec_levers`` (the
    teacher's lever recipes: name + edits + in_stack) ride along so a later new
    best can re-price a cited lever without a design turn. A ``reprice`` table
    (cited levers only, no stack, no roles) is history for the price-check and
    never the teacher's table: _banked_lever_table skips it."""
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    rec = {"ts": ctx.now(), "best_ref": best_ref, "job": job, "table": table}
    if reprice:
        rec["reprice"] = True
    if spec_levers is not None:
        rec["spec_levers"] = [{k: lv.get(k) for k in ("name", "edits", "in_stack", "expect", "decisions")
                               if k in lv} for lv in spec_levers]
    try:
        with (ev / "lever_prices.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        _log(f"meta/lever-price: could not persist lever_prices.jsonl: {e}")
    else:
        _bank_evidence(ctx, "lever prices")


# --- price staleness ----------------------------------------------------------
#
# A price is a reading on one tree. The pricing round banks its table under the
# best sha it ran on, the teacher writes those numbers into the milestone
# (`price +25.59% via lever:regsplit_184_120 [MEASURED, job <id>]`), and the
# student walks the rung several trees later — nothing compared the two trees.
# A GDN run priced a register-split lever at +25.6% on its first green
# (the aux role spilling 8.9M sectors); the teacher wrote it in as the last rung
# of a mandatory ladder; two KEEPs later the aux spill was gone and the same
# lever read −11.6% (it serialises every wgmma), the student walked it anyway,
# the M0 gate row and the next lever were built on that tree, and a full meta
# step went to discovering the reversal. The check below runs where the tree
# changes: every KEEP that takes the watermark re-prices the milestone's cited
# levers whose price came from another tree (one job, base / +each only, the
# recipes banked with the table), marks each citation in the milestone with the
# verdict, and suspends a reversed / faded rung for the student until the
# teacher's next entry reads the revision and rewrites the rung.
_LEVER_NAME = r"[A-Za-z0-9_][A-Za-z0-9_\-]*(?:\.[A-Za-z0-9_\-]+)*"
_LEVER_CITE_RE = re.compile(r"\blever\s*[:：]\s*`?(" + _LEVER_NAME + r")`?(?![A-Za-z0-9_\-])")
# a job pointer on the citation's line, after the citation: `[MEASURED, job <id>]`
_CITE_JOB_RE = re.compile(r"job\s*\**\s*([0-9]{4,})")
_REPRICE_TAG = "RE-PRICED"
_REVISIONS_NAME = "price_revisions.jsonl"
# verdicts that change what a citation may be used for
_REVISION_SUSPENDS = ("reversed", "faded")
_REVISION_FLAGGED = _REVISION_SUSPENDS + ("emerged", "drifted", "unverifiable")


def _stamp_price_row(row: dict, best_ref: str, job: str) -> dict:
    row.setdefault("priced_on", best_ref)
    row.setdefault("job", job)
    return row


def _priced_best(priced) -> str | None:
    """The tree a reopen decided at this entry would be applied to."""
    return (priced.best_sha or priced.head_sha) if priced is not None else None


def _cited_levers(text: str) -> list:
    """Lever names the milestone cites as prices (`lever:<name>`), in order, once each."""
    seen: list = []
    for m in _LEVER_CITE_RE.finditer(text or ""):
        name = m.group(1)
        if name and name not in seen:
            seen.append(name)
    return seen


def _cited_price_jobs(text: str, name: str) -> list:
    """Job pointers written on the same line as a `lever:<name>` citation (the
    `[MEASURED, job N]` tag the teacher copied from the table) — they name the
    reading the milestone actually cites."""
    jobs: list = []
    for line in (text or "").splitlines():
        for m in re.finditer(r"\blever\s*[:：]\s*`?" + re.escape(name) + r"`?(?![A-Za-z0-9_\-])", line):
            # only the teacher's own tag counts: the driver's [RE-PRICED … job N]
            # marker after it names a re-price, and reading THAT as the citation
            # would move the baseline to the latest re-price — the drift the
            # cited-reading rule exists to stop
            after = line[m.end():m.end() + 120]
            after = after.split("[" + _REPRICE_TAG, 1)[0]
            for j in _CITE_JOB_RE.findall(after):
                if j not in jobs:
                    jobs.append(j)
    return jobs


def _lever_price_history(ctx: LoopContext, name: str) -> list:
    """Every banked table that priced ``name``, oldest first: dicts of
    ``best_ref`` / ``job`` / ``row`` (the lever or variant row, provenance
    stamped) / ``spec`` (the recipe banked with that table; None for tables
    from before recipes were banked)."""
    ev = _evidence_dir(ctx, create=False)
    if ev is None or not (ev / "lever_prices.jsonl").is_file():
        return []
    out: list = []
    try:
        for line in (ev / "lever_prices.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            row = next((lv for lv in rec.get("table") or []
                        if lv.get("kind") in ("lever", "variant") and lv.get("name") == name), None)
            if row is None:
                continue
            ref, job = str(rec.get("best_ref") or ""), str(rec.get("job") or "")
            spec = next((sp for sp in rec.get("spec_levers") or [] if sp.get("name") == name), None)
            stack = next((t for t in rec.get("table") or [] if t.get("kind") == "stack"), None)
            out.append({"best_ref": ref, "job": job, "reprice": bool(rec.get("reprice")),
                        "base_ms": (stack or {}).get("base_ms"),
                        "row": _stamp_price_row(dict(row), ref, job), "spec": spec})
    except (OSError, ValueError) as e:
        _log(f"price-check: could not read lever_prices.jsonl ({e})")
        return []
    return out


def _ms_of_tree(ctx: LoopContext, sha: str) -> float | None:
    """The gate-window reading of the KEEP that produced ``sha`` (same-cluster
    rows only), or None when no record owns it."""
    if not sha:
        return None
    try:
        rows = _on_cluster(load_records(ctx.status_path), ctx)
    except (OSError, ValueError):
        return None
    for r in reversed(rows):
        if r.sha == sha and r.verdict == "KEEP":
            return gate_ms(r, ctx.gate_window)
    return None


def _price_legs(row: dict) -> "tuple[float | None, float | None]":
    """(add, remove) of a lever row; a subset row prices as (delta, None)."""
    if row.get("kind") == "variant":
        return row.get("delta_pct"), None
    return row.get("add_pct"), row.get("remove_pct")


def _reprice_verdict(before: dict, now: dict, noise_pct: float) -> "tuple[str, str]":
    """(verdict, note) for a lever priced ``before`` on one tree and ``now`` on
    another. Read on the add leg (the reading a single-lever citation is), on
    the remove leg when the older table had no add. ``noise_pct`` is the newer
    job's own noise floor."""
    def value(row):
        # what having the lever is worth on that tree: its add leg, else the
        # negated remove leg (a landed lever is priced by taking it out)
        add, rem = _price_legs(row)
        if add is not None:
            return add, "add"
        return (None, "add") if rem is None else (-rem, "-remove")
    old, b_leg = value(before)
    new, n_leg = value(now)
    if old is None or new is None:
        return "unmeasured", "no add / remove reading on one of the two trees"
    big = lambda x: abs(x) > noise_pct  # noqa: E731
    span = f"{b_leg} {old:+.2f}% -> {n_leg} {new:+.2f}% (noise {noise_pct:.2f}%)"
    if big(old) and big(new) and (old > 0) != (new > 0):
        return "reversed", span + ": the sign flipped — on this tree the lever costs what it paid before"
    if big(old) and not big(new):
        return "faded", span + ": inside the noise band on this tree — the lever no longer pays"
    if not big(old) and big(new):
        return "emerged", span + ": a null reading became a real one on this tree"
    if big(old) and big(new) and abs(new - old) > max(2 * noise_pct, 0.25 * abs(old)):
        return "drifted", span + ": same sign, different size — cite the new reading"
    return "confirmed", span


def _append_revisions(ctx: LoopContext, revisions: list) -> None:
    if not revisions:
        return
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    try:
        with (ev / _REVISIONS_NAME).open("a", encoding="utf-8") as fh:
            for rv in revisions:
                fh.write(json.dumps(rv, ensure_ascii=False) + "\n")
    except OSError as e:
        _log(f"price-check: could not persist {_REVISIONS_NAME}: {e}")
    else:
        _bank_evidence(ctx, "price revisions")


def _unretired_revisions(ctx: LoopContext) -> list:
    """The standing revisions, newest per lever."""
    ev = _evidence_dir(ctx, create=False)
    if ev is None or not (ev / _REVISIONS_NAME).is_file():
        return []
    latest: dict = {}
    try:
        for line in (ev / _REVISIONS_NAME).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rv = json.loads(line)
                if not rv.get("retired"):
                    latest[rv["lever"]] = rv
    except (OSError, ValueError) as e:
        _log(f"price-check: could not read {_REVISIONS_NAME} ({e})")
        return []
    return list(latest.values())


def _retire_revisions(ctx: LoopContext) -> None:
    """Mark every standing revision retired: a fresh table on the current best
    supersedes them, and the teacher has read them at this entry."""
    ev = _evidence_dir(ctx, create=False)
    p = (ev / _REVISIONS_NAME) if ev is not None else None
    if p is None or not p.is_file():
        return
    try:
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for rv in rows:
            rv["retired"] = True
        p.write_text("".join(json.dumps(rv, ensure_ascii=False) + "\n" for rv in rows),
                     encoding="utf-8")
    except (OSError, ValueError) as e:
        _log(f"price-check: could not retire {_REVISIONS_NAME} ({e})")


def _legs_text(d: dict | None) -> str:
    if not d:
        return "n/a"
    if d.get("delta_pct") is not None and d.get("add_pct") is None:
        return f"delta {_fmt_pct(d.get('delta_pct'))}"
    return f"add {_fmt_pct(d.get('add_pct'))} / remove {_fmt_pct(d.get('remove_pct'))}"


def _reprice_marker(rv: dict) -> str:
    """The in-milestone tag a re-priced citation gets, after its [MEASURED] tag."""
    tag = (f"[{_REPRICE_TAG} @{str(rv['priced_on_now'])[:10]}"
           + (f" job {rv['job_now']}" if rv.get("job_now") else "")
           + f": {_legs_text(rv.get('now'))} — {rv['verdict'].upper()}")
    if rv["verdict"] in _REVISION_SUSPENDS:
        tag += ("; SUSPENDED by driver — the rung citing this price may not be landed "
                "until the teacher revises it")
    elif rv["verdict"] == "unverifiable":
        tag += f"; {rv['note']}"
    return tag + "]"


def _mark_lever_citations(text: str, name: str, marker: str) -> str:
    """Put ``marker`` after every `lever:<name>` citation in ``text`` (after its
    [MEASURED …] tag when one follows), replacing any earlier RE-PRICED marker
    on that citation — one standing verdict per citation, the latest."""
    pat = re.compile(r"(\blever\s*[:：]\s*`?" + re.escape(name) + r"`?(?![A-Za-z0-9_\-]))"
                     r"((?:\s*\[MEASURED[^\]\n]*\])?)((?:\s*\[" + _REPRICE_TAG + r"[^\]\n]*\])*)")
    return pat.sub(lambda m: f"{m.group(1)}{m.group(2)} {marker}", text)


def _revision_line(rv: dict) -> str:
    return (f"`{rv['lever']}`: was {_legs_text(rv.get('before'))} @`{str(rv['priced_on_before'])[:10]}`"
            + (f" (job {rv['job_before']})" if rv.get("job_before") else "")
            + f" → now {_legs_text(rv.get('now'))} @`{str(rv['priced_on_now'])[:10]}`"
            + (f" (job {rv['job_now']})" if rv.get("job_now") else "")
            + f" → **{rv['verdict'].upper()}** ({rv['note']})")


def _price_revision_note(ctx: LoopContext) -> str:
    """Student-facing: the standing revisions of milestone-cited lever prices."""
    revs = [rv for rv in _unretired_revisions(ctx) if rv.get("verdict") in _REVISION_FLAGGED]
    if not revs:
        return ""
    lines = ["\n## Price-tag review (driver: whenever the best tree moves, the lever price tags "
             "the milestone cites are re-measured on the new tree)"]
    for rv in revs:
        lines.append("- " + _revision_line(rv))
        if rv["verdict"] in _REVISION_SUSPENDS:
            lines.append("  → the milestone clauses / ladder rungs citing it have been marked "
                         "`SUSPENDED` by the driver: **do not land them this round and do not base a "
                         "change on them**; the teacher re-checks them at its next entry.")
        elif rv["verdict"] == "unverifiable":
            lines.append("  → this price tag cannot be re-measured on the current tree: treat the "
                         "clauses citing it as **unpriced**, not as grounds to land anything.")
        else:
            lines.append("  → the reading on the new tree is what counts; read every place that "
                         "quotes the old number with the new value.")
    return "\n".join(lines) + "\n"


def _price_revision_teacher_block(revs: list) -> str:
    lines = ["\n## Price-tag review (measured by the driver in the inner loop: when the best tree "
             "changes, the levers the milestone cites are re-measured)",
             "These levers' price tags were written into the milestone (`lever:<name>` citations); "
             "below are their re-measurements on the later best trees. "
             "**Citations that came back REVERSED / FADED have been marked `SUSPENDED` in the "
             "milestone by the driver, and the student is forbidden to land them** — "
             "this diagnosis / patch must dispose of every one: retract the rung, rewrite it at the "
             "new price, or reopen at the new price; "
             "no SUSPENDED mark may be left in the milestone."]
    for rv in revs:
        lines.append("- " + _revision_line(rv))
    return "\n".join(lines) + "\n"


def _reprice_cited_levers(ctx: LoopContext, rec) -> list:
    """After a KEEP took the watermark: re-price, on that tree, every lever the
    milestone cites whose price was read on another tree; bank the readings,
    mark each citation with its verdict, suspend reversed / faded ones for the
    student. Returns the revision records written (possibly empty)."""
    if not (ctx.reprice_on_new_best and ctx.lever_pricing_rounds > 0 and ctx.stall_pricing):
        return []
    if not rec.sha:
        return []
    ev = _evidence_dir(ctx, create=False)
    if ev is None or not (ev / "lever_prices.jsonl").is_file():
        return []
    text = _read_milestone(ctx)
    new_ms = gate_ms(rec, ctx.gate_window)
    stale: list = []          # (name, the cited reading, the latest reading)
    skipped: list = []
    for name in _cited_levers(text):
        hist = _lever_price_history(ctx, name)
        if not hist:
            continue          # cited but never priced here: the reopen gate's business
        latest = hist[-1]
        if not latest["best_ref"] or latest["best_ref"] == rec.sha:
            continue          # already read on this very tree
        # the reading the milestone cites: the table whose job the citation
        # names, else the oldest — a verdict against the latest re-price would
        # let a slow drift read "confirmed" at every step
        jobs = _cited_price_jobs(text, name)
        cited = next((h for h in hist if h["job"] and h["job"] in jobs), hist[0])
        # bound the job count: a best that moved by less than the margin since
        # the lever was last read is not re-read (the teacher's entry prices it)
        last_ms = _ms_of_tree(ctx, latest["best_ref"]) or latest.get("base_ms")
        if (new_ms is not None and last_ms
                and new_ms > last_ms * (1.0 - ctx.reprice_min_gain_pct / 100.0)):
            skipped.append(name)
            continue
        stale.append((name, cited, latest))
    if skipped:
        _log(f"iter {rec.iter}: new best {rec.sha[:10]} — cited lever price(s) {skipped} were read "
             f"on a tree within {ctx.reprice_min_gain_pct:g}% of this one; not re-pricing them")
    if not stale:
        return []
    names = [n for n, _, _ in stale]
    if ctx.now() > ctx.deadline_epoch - 3600:
        _log(f"iter {rec.iter}: new best {rec.sha[:10]} — the milestone cites lever price(s) "
             f"{names} read on other trees, but < 60 min of budget left; not re-pricing")
        return []
    be = ctx.backend
    revisions: list = []

    def revise(name, cited, verdict, note, now_row=None, job="", landed=False):
        keys = ("add_pct", "remove_pct", "delta_pct", "cls")
        revisions.append({
            "ts": ctx.now(), "iter": rec.iter, "lever": name,
            "priced_on_before": cited["best_ref"], "job_before": cited["job"],
            "before": {k: cited["row"].get(k) for k in keys},
            "priced_on_now": rec.sha, "job_now": job, "landed": landed,
            "now": ({k: now_row.get(k) for k in keys} if now_row else None),
            "verdict": verdict, "note": note, "retired": False})

    # the recipes: a lever without a banked recipe cannot be re-priced; each is
    # priced ALONE (in_stack off): the citation is a single-add reading, and
    # unrelated cited levers need not compose. A lever the student has since
    # landed (its `new` text is in the tree, its `old` is not) is priced by
    # taking it out: the reversed recipe's add is the cost of removing it.
    todo: list = []
    for name, cited, latest in stale:
        spec = latest.get("spec") or cited.get("spec")
        if cited["row"].get("kind") == "variant":
            revise(name, cited, "unverifiable", "a subset row; only single levers are re-priced")
            continue
        if not spec or not spec.get("edits"):
            revise(name, cited, "unverifiable",
                   "no edit recipe was banked with its price (priced before recipes were banked)")
            continue
        lv = {"name": name, "edits": spec["edits"], "in_stack": False}
        landed = False
        if fail := _lever_spec_refused(ctx, {"levers": [lv], "kernel_file": str(ctx.kernel_file)}):
            undo = {"name": name, "in_stack": False,
                    "edits": [{"old": e["new"], "new": e["old"]} for e in reversed(spec["edits"])]}
            if _lever_spec_refused(ctx, {"levers": [undo], "kernel_file": str(ctx.kernel_file)}):
                revise(name, cited, "unverifiable",
                       f"its edit no longer applies on {rec.sha[:10]} ({fail})")
                continue
            lv, landed = undo, True
        todo.append((name, cited, lv, landed, spec))
    if todo:
        job, parsed, stack, fail_note = "", [], None, ""
        pre_head = be.head()
        try:
            be.reset(rec.sha)
            _seed_pricing_toolkits(Path(ctx.workspace), be)
            spec_path = Path(ctx.workspace) / "notes" / _LEVER_SPEC_NAME
            # a group card re-prices on the workloads its table was priced on, merged alike
            shapes = meta_shapes.card_meta_workloads(ctx.workspace)
            spec = {"levers": [lv for _, _, lv, _, _ in todo], "variants": [],
                    "rounds": meta_shapes.rounds_for(ctx.lever_pricing_rounds, len(shapes)),
                    "roles": False, "kernel_file": str(ctx.kernel_file)}
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
            be.commit(f"meta-reprice: {len(todo)} cited lever(s) on {rec.sha[:10]}")
            branch = f"{ctx.branch_prefix}-reprice"
            be.push_branch(branch)
            _log(f"iter {rec.iter}: new best {rec.sha[:10]} — re-pricing {len(todo)} milestone-cited "
                 f"lever(s) [{', '.join(n + (' (landed: by removal)' if ld else '') for n, _, _, ld, _ in todo)}] "
                 f"whose price was read on another tree ("
                 + (f"one job per workload of {', '.join(shapes)}" if shapes else "one job")
                 + f": base / +each x {spec['rounds']}, no role pass)")
            log, job = _lever_price_submit(ctx, be, branch, _LEVER_PRICING_CMD,
                                           f"iter {rec.iter}: re-price", rerun=False)
            parsed, stack, fail = parse_lever_prices(log)
            if fail:
                _log(f"iter {rec.iter}: re-price PRICE-FAIL {fail}")
                parsed, fail_note = [], f"the re-price job refused the recipe ({fail})"
            elif _price_log_verdict(log) is None:
                # not a table (see _price_log_verdict): the cited prices stay
                # unverified rather than being "re-read" off a cut log
                why = _price_log_lost_reason(log)
                _log(f"iter {rec.iter}: re-price job {job or '?'} log carries no PRICE verdict "
                     f"({why}) — the cited prices stay unverified")
                parsed, fail_note = [], f"the re-price job's log carries no PRICE verdict ({why})"
            elif (killed := _price_role_pass_killed(log)):
                _log(f"iter {rec.iter}: re-price job {job or '?'}: {killed} after the timing "
                     f"table (PRICE-TIMED) — re-priced from the timing copy")
        except BackendError as e:
            _log(f"iter {rec.iter}: re-price job failed ({e}) — the cited prices stay unverified")
            parsed, fail_note = [], f"the re-price job failed ({e})"
        finally:
            try:
                be.reset(pre_head)
            except BackendError as e:
                _log(f"iter {rec.iter}: could not restore head {pre_head[:10]} after the re-price ({e})")
        by_name = {lv.get("name"): lv for lv in parsed}
        landed_names = {n for n, _, _, ld, _ in todo if ld}
        if parsed:
            attribution = {n: list(cited["row"].get("decisions") or []) for n, cited, _, _, _ in todo}
            table = list(parsed)
            for lv in table:
                lv.setdefault("kind", "lever")
                if lv.get("name") in landed_names:
                    # the reversed recipe's add IS the lever's remove leg
                    lv["remove_pct"], lv["remove_ms"] = lv.get("add_pct"), lv.get("add_ms")
                    lv["add_pct"], lv["add_ms"], lv["landed"] = None, None, True
                lv["decisions"] = attribution.get(lv.get("name"), [])
                lv["priced_on"], lv["job"] = rec.sha, job
            if stack:
                stack.update(kind="stack", priced_on=rec.sha, job=job)
                table.append(stack)
            # the forward recipes stay the banked recipes (a landed lever's undo
            # is derived from them whenever it is needed again)
            _persist_lever_table(ctx, table, rec.sha, job, reprice=True,
                                 spec_levers=[sp for _, _, _, _, sp in todo])
        noise = _table_noise_pct([stack] if stack else [])
        for name, cited, _, landed, _ in todo:
            row = by_name.get(name)
            if row is None:
                revise(name, cited, "unverifiable",
                       fail_note or "the re-price job printed no reading for it", job=job, landed=landed)
                continue
            verdict, note = _reprice_verdict(cited["row"], row, noise)
            if landed:
                note += " — the lever is landed on this tree; read by taking it out"
            revise(name, cited, verdict, note, now_row=row, job=job, landed=landed)
    _append_revisions(ctx, revisions)
    for rv in revisions:
        _log(f"iter {rec.iter}: price-check `{rv['lever']}` {rv['verdict'].upper()} — {rv['note']}"
             + (" (the citing rung is SUSPENDED for the student)"
                if rv["verdict"] in _REVISION_SUSPENDS else ""))
    marked = text
    for rv in revisions:
        marked = _mark_lever_citations(marked, rv["lever"], _reprice_marker(rv))
    if marked != text:
        _write_milestone(ctx, marked)
    return revisions


# --- timeline probe -------------------------------------------------------------
#
# A latency-bound kernel (a serial recurrence: one CTA walks its chunks one
# after another, nothing hides an exposed cycle) is judged in cycles per chunk
# per segment, and the stall / role account cannot say that: it says WHO waits,
# not WHERE the cycles sit. A GDN run spent 15 iterations and three meta
# steps on register accounting while the aux role's inverse level and gate
# epilogue were ~2400 cycles/chunk over the reference and the state roles waited
# ~1300 cycles/chunk on them — a per-segment clock account read that off one
# job. recon/timeline_probe.py is that instrument (the student's `_seg(...)`
# markers rewritten into %clock64 stamps in a probe build; the reference is never
# instrumented, so blackbox runs use the same path). The driver runs it at every
# teacher entry, at gate rows, at every new best beyond reprice_min_gain_pct,
# and on request, banks the table, puts it first among the teacher's evidence
# and makes the diagnosis name the segment (`locus_segment:`).
_TIMELINE_CMD = "cmd:python3 notes/timeline_probe.py"


def _timeline_cmd(ctx: LoopContext) -> str:
    """The probe job's command: a CUDA-mode run names its kernel.cu (the probe
    stamps `FK_SEG` markers in C++ and runs the bench under ``--cuda-run``); every
    other run uses exactly ``_TIMELINE_CMD``, which defaults to kernel.py."""
    if _is_cuda(ctx):
        return f"{_TIMELINE_CMD} --kernel {ctx.kernel_file}"
    return _TIMELINE_CMD


_TIMELINE_OVERLAY_MSG = ("timeline probe: harness copy of notes/timeline_probe.py overlaid for one "
                         "probe job (the tree under judgment is the parent, unchanged)")
_TIMELINES_NAME = "timelines.jsonl"
_TIMELINE_REQUESTS = ("timeline", "notes/timeline", "notes/timeline_probe.py")
_LOCUS_SEG_LINE = re.compile(r"locus_segment\s*[:：]\s*([^\n]*)")
_BOTTLENECK_LATENCY = re.compile(r"^\s*bottleneck\s*[:：]\s*latency\b", re.M | re.I)


def _is_timeline_request(path: str) -> bool:
    """`PROBE_REQUEST: timeline` however the agent quoted it (backticks, quotes,
    a trailing full stop) — the parser's `\\S+` keeps the punctuation."""
    return str(path or "").strip().strip("`'\"“”「」").rstrip(".,;:：。").lower() in _TIMELINE_REQUESTS


def _timeline_on(ctx: LoopContext) -> bool:
    mode = str(getattr(ctx, "timeline_probe", "off") or "off").lower()
    if mode == "on":
        return True
    if mode == "auto":
        return _BOTTLENECK_LATENCY.search(_read_milestone(ctx)) is not None
    return False


def _timeline_markers(ctx: LoopContext) -> "tuple[bool, list]":
    """(the kernel keeps the no-op `_seg` definition, its marker site names) — the
    `#define FK_SEG` line and `FK_SEG("...")` calls of a CUDA-mode kernel."""
    from forgekernel.skills.timeline_probe import has_marker_def, markers
    try:
        src = (Path(ctx.workspace) / str(ctx.kernel_file)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False, []
    if _is_cuda(ctx):
        return has_marker_def(src, cuda=True), markers(src, cuda=True)
    return has_marker_def(src), markers(src)


def _timeline_unparsed_markers(ctx: LoopContext) -> list:
    """The kernel's `_seg(` calls whose site name is not a string literal — the
    probe's site table cannot see them and the stamped variant reads them as no-ops
    (timeline_probe.unparsed_markers); the student note names them."""
    from forgekernel.skills.timeline_probe import unparsed_markers
    try:
        src = (Path(ctx.workspace) / str(ctx.kernel_file)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    if _is_cuda(ctx):
        return unparsed_markers(src, cuda=True)
    return unparsed_markers(src)


def _parse_locus_segment(text: str) -> str:
    m = _LOCUS_SEG_LINE.search(text or "")
    return m.group(1).strip() if m else ""


def _timeline_short_coverage(tl: dict) -> bool:
    """A renderable table whose stamps do not span the whole device call.

    Two spellings, one condition: the probe's own ``partial`` class, and a table
    banked before the TIMELINE-COVERAGE line existed that ``table_usable`` read
    through on its span (``ok`` false, coverage unknown). Both leave part of the
    device window outside the table, so both get the caveat where the numbers
    are shown (see ``timeline_probe.coverage_class`` for what a short span does
    and does not say: segment cycles are exact, the rows do not add up to the
    call).
    """
    if tl.get("coverage") == "partial":
        return True
    return tl.get("coverage") in (None, "unknown") and not tl.get("ok") and bool(tl.get("roles"))


def _timeline_probe_stale(ctx: LoopContext) -> "str | None":
    """The harness copy of the probe when the tree's ``notes/timeline_probe.py``
    differs from it (or is missing); None when the two are identical.

    The job runs the TREE's copy (``_TIMELINE_CMD``), and nothing in the harness
    refreshes it: an operator tree keeps whatever probe it was scaffolded with,
    so a probe change lands in a job only once someone copies the file over
    (a GDN bwd tree's copy once differed from the harness's). The driver
    reads such a log through its legacy rule (``table_usable``); this names the
    gap so the operator and the student can close it."""
    from forgekernel.skills import timeline_probe as probe_mod
    src = Path(probe_mod.__file__)
    tree = Path(ctx.workspace) / "notes" / "timeline_probe.py"
    try:
        if tree.is_file() and tree.read_bytes() == src.read_bytes():
            return None
    except OSError:
        pass
    return str(src)


def _timeline_preamble_fence_hit(ctx: LoopContext) -> "str | None":
    """The card-fence token the probe's injected preamble would trip, read from
    the workspace's ``eval/bench.py`` (the file the stamped job runs); None when
    the card has no literal ``_FORBIDDEN`` fence or nothing in the preamble
    matches it. Never raises: an unreadable bench is the job's problem."""
    try:
        from forgekernel.skills.timeline_probe import preamble_fence_hit
        bench = Path(ctx.workspace) / "eval" / "bench.py"
        if not bench.is_file():
            return None
        return preamble_fence_hit(bench.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a self-check must not become the failure
        return None


def _timeline_probe_ref(ctx: LoopContext, ref: str, label: str) -> str:
    """The commit the timeline job archives for ``ref``: ``ref`` itself when its
    notes/timeline_probe.py is the harness copy, else ``ref`` with the harness
    copy overlaid as a child commit (backend ``with_file``; a backend without
    it probes the tree's own copy, as before).

    The job runs the TREE's copy of the probe and nothing refreshed it, so a
    probe fix never reached a tree committed before it: one run's
    plateau timeline would have run on a best sha carrying a probe one fix old
    and read dead. What is banked and logged stays ``ref`` — only the probe
    file differs, and the probe is the instrument, not the thing measured."""
    be = ctx.backend
    with_file = getattr(be, "with_file", None)
    if not ref or with_file is None:
        return ref
    from forgekernel.skills import timeline_probe as probe_mod
    try:
        text = Path(probe_mod.__file__).read_text(encoding="utf-8")
        out = str(with_file(ref, "notes/timeline_probe.py", text, _TIMELINE_OVERLAY_MSG) or ref)
    except (BackendError, OSError) as e:
        _log(f"timeline[{label}]: could not overlay the harness probe on {ref[:10]} ({e}) — "
             f"probing the tree's own copy")
        return ref
    if out != ref:
        _log(f"timeline[{label}]: notes/timeline_probe.py at {ref[:10]} is not the harness copy — "
             f"probing {ref[:10]} with the harness probe overlaid ({out[:10]}; the tree under "
             f"judgment is unchanged)")
    return out


def _run_timeline(ctx: LoopContext, ref: str | None, label: str, *,
                  branch: str | None = None, workload: str | None = None) -> dict | None:
    """Run the timeline probe on ``ref`` (a pushed ``branch`` when given),
    bank and log the table. None when the job ran no probe at all; a dict
    with ``ok`` False when the probe failed closed (banked, not rendered).
    ``workload``: one of a group card's other meta workloads (loop/meta_shapes.py),
    banked under its name -- every machine reading stays on the representative's."""
    be = ctx.backend
    hit = _timeline_preamble_fence_hit(ctx)
    if hit:
        # The card's kernel fence would refuse the driver's own preamble: every
        # stamped bench dies before the kernel imports and the job can only
        # report no-slab-dumped (observed: three GPU jobs on a GDN run, zero
        # tables). A configuration fault, caught here for the price of a regex.
        _log(f"timeline[{label}]: preamble hits the card fence: {hit!r} — skipping the "
             f"probe; give the fence a word boundary (`\\bos\\s*\\.\\s*environ`) or keep "
             f"that token out of recon/timeline_probe._PREAMBLE, then re-run")
        return None
    stale = _timeline_probe_stale(ctx)
    if stale and not getattr(ctx, "_timeline_stale_logged", False):
        # once per loop: the tree's probe is not this harness's, so its verdict
        # lines (TIMELINE-COVERAGE among them) may be missing or older
        _log(f"timeline: notes/timeline_probe.py in the tree differs from the harness copy "
             f"({stale}) — each probe job gets the harness copy overlaid on the probed ref "
             f"(logged per probe; a backend without with_file runs the tree's copy, whose "
             f"verdict lines may predate this harness and are read through the legacy span "
             f"rule); copy the harness file over notes/timeline_probe.py and commit it to "
             f"refresh the tree itself")
        try:
            ctx._timeline_stale_logged = True
        except AttributeError:
            pass
    push_ref = getattr(be, "push_ref", None)
    if branch is None and ref and (banked := _banked_timeline_of(ctx, ref, workload)) is not None:
        # TIMELINE REUSE: the teacher entry re-reads the best tree, which the
        # gate row / new-best probe (or the previous entry) usually read
        # already — in one large batch of runs, 1672 of 1887 entry probes were of
        # a sha with a renderable table banked. Same tree, same
        # harness probe: the banked table is that reading. A student-requested
        # probe (``branch`` given) always runs.
        _log(f"timeline[{label}] @ {str(ref)[:10]}: reusing the banked table "
             f"({banked.get('label')}, job {banked.get('job') or '?'}) — same tree, same probe")
        return banked["table"]
    if branch is None:
        branch = f"{ctx.branch_prefix}-timeline"
        try:
            if ref and push_ref is not None:
                push_ref(_timeline_probe_ref(ctx, ref, label), branch)
            else:
                be.push_branch(branch)
                ref = be.head()
        except BackendError as e:
            _log(f"timeline[{label}]: could not push {str(ref)[:10]} ({e})")
            return None
    elif ref and push_ref is not None and (probe_ref := _timeline_probe_ref(ctx, ref, label)) != ref:
        # the caller pushed ``branch`` at ``ref`` already: re-point it at the overlay
        try:
            push_ref(probe_ref, branch)
        except BackendError as e:
            _log(f"timeline[{label}]: could not push the overlay {probe_ref[:10]} ({e}) — "
                 f"probing the tree's own copy")
    log = None
    for rerun in range(2):
        try:
            log = be.submit_gpu(branch, _timeline_cmd(ctx)
                                + (f" --workload {workload}" if workload else ""))
        except BackendError as e:
            _log(f"timeline[{label}]: job failed ({e})")
            return None
        if rerun == 0 and _node_dirty_reason(None, log):
            # the job's GPU held an orphan (the probe's own clock-stamped bench
            # timed out; its launch target is still on the GPU): the SAME design
            # is re-probed once — same contract as the ablation / lever-price
            # probes
            _log(f"timeline[{label}]: {_node_dirty_reason(None, log)} — re-probing "
                 f"once")
            continue
        break
    if log is None:
        return None
    from forgekernel.skills.timeline_probe import parse_timeline
    tl = parse_timeline(log)
    job = str(getattr(be, "last_job_id", "") or "")
    if tl is None:
        _log(f"timeline[{label}] @ {str(ref)[:10]}: the job printed no TIMELINE line "
             f"(is notes/timeline_probe.py in the tree?)")
        return None
    # The log line reads the SAME verdict the retrieval does. It used to read
    # ``tl["ok"]``, which stopped being the question when a partial table became
    # renderable: the probe no longer prints TIMELINE-OK for one, so a short-
    # coverage run logged "FAIL — None" and the annotation below never printed.
    if _timeline_renderable(tl):
        roles = "; ".join(f"{r} period={_fmt_cyc(v.get('period'))} busy={_fmt_cyc(v.get('busy'))} "
                          f"wait={_fmt_cyc(v.get('wait'))}" for r, v in tl["roles"].items())
        # A partial table is not a failure: name the shortness here, so a run
        # that reads 0.6 span on every probe is visibly "usable, short" rather
        # than a wall of FAILs that hide the table.
        short = _timeline_short_coverage(tl)
        _log(f"timeline[{label}] @ {str(ref)[:10]}: OK mode={tl['mode']} sites={tl['sites']} "
             f"perturbation={tl.get('perturbation')} span={tl.get('span_ratio')}"
             + (f" (table usable; stamps bracket {tl.get('span_ratio')} of the device window — "
                f"segment cycles exact, rows do not add up to the call)" if short else "")
             + f" — {roles}")
        _note_env_ok(ctx, "timeline")
    else:
        # "base-bench-failed-or-incorrect" over a tree whose own eval passed, with an
        # import error in the log, is the probe's `python3` (loop/job_env.py) — say
        # so here and to whoever reads the feedback, or markers get "fixed" forever
        if (why := env_failure(log)) is not None:
            tl["env"] = why
            _note_env_failure(ctx, "timeline", why)
        _log(f"timeline[{label}] @ {str(ref)[:10]}: FAIL — {tl.get('fail')}"
             + (f" [job environment: {why}]" if why else ""))
    _bank_timeline(ctx, ref or "", job, label, tl, workload)
    return tl


def _fmt_cyc(x) -> str:
    return "n/a" if x is None else f"{int(round(x))}"


def _timeline_renderable(tl: dict) -> bool:
    """Whether a parsed or banked table can be handed to the teacher / student.

    A thin read of ``table_usable`` (the one place the rule lives — see its
    docstring for why ``ok`` and "worth diagnosing from" are separate questions).
    Kept as a name because call sites read better for it, not because the rule
    is re-decided here.
    """
    from forgekernel.skills.timeline_probe import table_usable
    return table_usable(roles=tl.get("roles") or {}, ok=bool(tl.get("ok")),
                        coverage=tl.get("coverage"), span_ratio=tl.get("span_ratio"),
                        sm_src=tl.get("sm_src"))


def _bank_timeline(ctx: LoopContext, sha: str, job: str, label: str, tl: dict,
                   workload: str | None = None) -> None:
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    rec = {"ts": ctx.now(), "sha": sha, "job": job, "label": label, "ok": bool(tl.get("ok")),
           # a group card's other meta workload: a reading beside the representative's,
           # never the one the loop's timeline decisions read (_latest_timeline)
           **({"workload": workload} if workload else {}),
           # ``tl["usable"]`` when the parse produced one (it always does since the
           # field landed; ``_timeline_renderable`` is the fallback for a dict a
           # test or an older path built by hand).
           "usable": bool(tl.get("usable")) if "usable" in tl else _timeline_renderable(tl),
           "coverage": tl.get("coverage"),
           "mode": tl.get("mode"), "ms": tl.get("base_ms"), "table": tl,
           # the instrument the table was read with: reuse requires the same one
           "probe": _timeline_probe_digest()}
    try:
        with (ev / _TIMELINES_NAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        _log(f"timeline: could not persist {_TIMELINES_NAME}: {e}")
    else:
        _bank_evidence(ctx, "timeline table")


def _banked_timelines(ctx: LoopContext) -> list:
    ev = _evidence_dir(ctx, create=False)
    if ev is None or not (ev / _TIMELINES_NAME).is_file():
        return []
    try:
        return [json.loads(ln) for ln in (ev / _TIMELINES_NAME).read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    except (OSError, ValueError) as e:
        _log(f"timeline: could not read {_TIMELINES_NAME} ({e})")
        return []


def _banked_timeline_usable(rec: dict) -> bool:
    """A banked record the driver can render. Records written before ``usable``
    existed are judged on the table they carry, so a resumed run's evidence
    does not go dark mid-run."""
    if "usable" in rec:
        return bool(rec.get("usable"))
    return _timeline_renderable(rec.get("table") or {})


def _latest_timeline(ctx: LoopContext, sha: str | None = None) -> dict | None:
    """The last banked *renderable* timeline (of ``sha`` when given).

    Renderable, not ``ok``: a partial-coverage table is a real table and the
    whole point of the split (see ``_timeline_renderable``). Returning None
    here would silence the meta entries."""
    for rec in reversed(_banked_timelines(ctx)):
        if rec.get("workload"):
            continue      # another meta workload of a group card: not the loop's reading
        if _banked_timeline_usable(rec) and (sha is None or rec.get("sha") == sha):
            return rec
    return None


def _timeline_probe_digest() -> str:
    """A short digest of the harness probe every timeline job runs (overlaid by
    ``_timeline_probe_ref``); '' when it cannot be read."""
    from forgekernel.skills import timeline_probe as probe_mod
    try:
        return hashlib.sha256(Path(probe_mod.__file__).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _banked_timeline_of(ctx: LoopContext, sha: str, workload: str | None = None) -> dict | None:
    """The latest banked renderable timeline of ``sha`` read with today's
    harness probe (on ``workload``: a group card's other meta workload; None =
    the card's own reading), or None. Records banked before the ``probe`` field,
    or by another probe version, never match: their table may be one fix old."""
    digest = _timeline_probe_digest()
    if not digest:
        return None
    for rec in reversed(_banked_timelines(ctx)):
        if (rec.get("sha") == sha and rec.get("probe") == digest
                and rec.get("workload") == workload
                and isinstance(rec.get("table"), dict) and _banked_timeline_usable(rec)):
            return rec
    return None


def _timeline_due(ctx: LoopContext, new_ms) -> bool:
    """A new best is re-read when it beats the tree of the last *renderable*
    timeline by the re-price margin (the same bound that keeps the price-check
    to one job per real move); always when nothing is banked yet.

    Renderable, not ``ok``: a partial-coverage table is a real reading of a
    real tree, so it anchors the margin like any other. (Whether a stale
    anchor should also force a re-read is a separate question — it fires on
    every KEEP and would defeat the margin, so it is not decided here.)"""
    last = _latest_timeline(ctx)
    if last is None:
        return True
    last_ms = _ms_of_tree(ctx, str(last.get("sha") or ""))
    if last_ms is None and ctx.gate_window == GATE_WINDOW_DEVICE:
        last_ms = last.get("ms")          # the probe's own base variant, a device reading
    if new_ms is None or not last_ms:
        return True
    return new_ms <= last_ms * (1.0 - ctx.reprice_min_gain_pct / 100.0)


def _ref_region_lines(ctx: LoopContext) -> str:
    """The reference's recon-time stall share per SASS region (the [tensor]
    line and the REGION rows of ref_role_exposure.txt) — the blackbox-safe
    reference column: the teacher maps regions to segments and reads
    cycles ≈ share × the reference's chunk period."""
    ev = _evidence_dir(ctx, create=False)
    p = (ev / "ref_role_exposure.txt") if ev is not None else None
    if p is None or not p.is_file():
        return ""
    try:
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines()
                 if ln.startswith("[tensor]") or ln.startswith("REGION ")]
    except OSError:
        return ""
    return "\n".join(lines[:12])


def _timeline_tables(tl: dict) -> str:
    """The per-role tables of a timeline (shared by the teacher block, the
    student note and the probe feedback)."""
    out = [f"Checks: mode={tl.get('mode')}, numeric={tl.get('numeric')}, perturbation {tl.get('perturbation')}×, "
           f"span/device {tl.get('span_ratio')}, non-monotonic {tl.get('mono_bad')}/{tl.get('mono_tot')}, "
           + (f"row restarts {tl.get('restarts')}/{tl.get('restart_pairs')}, " if tl.get('restart_pairs') is not None else "") +
           f"sites {tl.get('sites')}, steady rows {tl.get('rows')}, SM {tl.get('sm_mhz')} MHz"
           f" ({tl.get('sm_src', 'default')})."]
    pert = tl.get("perturbation")
    if pert is not None and pert > 1.03:
        out.append(f"**Note**: instrumentation stretched device time by {(pert - 1) * 100:.1f}%, so every "
                   "segment's absolute value is inflated by about that ratio; the relative structure "
                   "between segments, the waits and the phases are trustworthy — scale absolute cycle "
                   "counts down by that ratio.")
    if _timeline_short_coverage(tl):
        # Say it in the rendered table, not only in the log — and say only what
        # the span can say. It is the longest slot's max-minus-min over all its
        # stamps, so a short one means the stamps do not reach both ends of the
        # device window; which stretch is missing (an unmarked prologue /
        # epilogue, sites mode=waits dropped, a launch with no stamps) it cannot
        # tell. The segment cycles are stamp differences inside a row and are
        # exact; what the reader must not do is add the rows up into the call,
        # or scale the segments up by 1/coverage (one GDN bwd tree's
        # mode=all and mode=waits tables read 0.947 vs 0.60 with the same
        # per-row periods). Also the caveat for a table banked by a probe that
        # predates TIMELINE-COVERAGE and was read through on its span.
        out.append(f"{meta_shapes.TL_COVERAGE_NOTE} (partial)**: this table's stamps cover only "
                   f"{tl.get('span_ratio')} of device time (mode={tl.get('mode')}). The part they do not "
                   "cover **is not in this table** — it may be a stretch of the marked kernel that no "
                   "pair of sites brackets (mode=waits keeps only `.begin`/`.end` wait pairs), or another "
                   "launch in the same call that carries no stamps; the probe cannot tell which. "
                   "**Segment cycles are in-row stamp differences; read the absolute values as they "
                   "are** (scaled only by the perturbation above), and do not inflate them by "
                   "1/coverage. Only, the segments do not add up to the whole call: do not use this "
                   "table as the time account of the whole call.")
    for w in tl.get("warn") or []:
        out.append(f"{meta_shapes.TL_SELFCHECK_NOTE}**: {w}")
    out += ["", "| role | warps | period per block | busy | wait | wrap (unmarked tail) | period IQR |",
            "|---|---|---|---|---|---|---|"]
    for role, r in tl.get("roles", {}).items():
        out.append(f"| {role} | {r.get('warps')} | {_fmt_cyc(r.get('period'))} | {_fmt_cyc(r.get('busy'))} | "
                   f"{_fmt_cyc(r.get('wait'))} | {_fmt_cyc(r.get('wrap'))} | {r.get('iqr')} |")
    for role, r in tl.get("roles", {}).items():
        out += ["", f"**{role}** {meta_shapes.TL_SEGMENT_LEDGER_HEADING}",
                "| segment | kind | median | p25 | p75 | n |", "|---|---|---|---|---|---|"]
        for sg in r.get("segments", []):
            out.append(f"| `{sg['from']}` → `{sg['to']}` | {sg['kind']} | {_fmt_cyc(sg['median'])} | "
                       f"{_fmt_cyc(sg['p25'])} | {_fmt_cyc(sg['p75'])} | {sg['n']} |")
    if tl.get("phase"):
        out += ["", "Phase (each site's offset from the block's earliest stamp; aligned across roles — "
                "who is late, who waits for whom)",
                "| role | site | offset |", "|---|---|---|"]
        for role, sites in tl["phase"].items():
            for site, off in sorted(sites.items(), key=lambda kv: (kv[1] is None, kv[1] or 0)):
                out.append(f"| {role} | `{site}` | {_fmt_cyc(off)} |")
    return "\n".join(out) + "\n"


def _timeline_block(tl: dict, ref_lines: str = "") -> str:
    """The teacher's timeline evidence block."""
    block = ("\n## Timeline (measured by the driver with %clock64 stamps on the student tree — cycles "
             "per block; the reference is not instrumented)\n"
             + _timeline_tables(tl))
    if ref_lines:
        block += ("\nReference sampling share per SASS region (from recon, available to blackbox runs "
                  "too; you map region ↔ segment through the milestone timeline, cycles ≈ share × the "
                  "reference's period per block):\n```\n" + ref_lines + "\n```\n")
    block += (
        "\n**How to read it (hard rules)**: the diagnosis output block must carry "
        "`locus_segment: <role>.<segment>` (the segment the gap falls in; if it cannot be placed in "
        "a segment, write `none` and say why), and `deficient_locus` points at the decision that owns "
        "that segment. A large wait segment is not the wait's fault — look in the phase table at what "
        "the awaited role is doing at that moment; the lesion is on the side being waited for. Only a "
        "large busy segment is this role's own instruction stream. "
        "busy + wait + wrap ≈ period; wrap is the unmarked tail from the last site to the next "
        "block's first site — when it is large, have the student add markers there before judging. "
        "A `mode=waits` table keeps only the wait sites (the fallback when full-site instrumentation "
        "perturbs too much): its busy segments are merged spans between adjacent waits, and "
        "cross-role phases are slightly off (the waiting side reads ~10% high). An `.end` site after "
        "`bar.sync` can fire early (DEFER_BLOCKING), and the rendezvous time then lands in the next "
        "segment — segments with a wide p25/p75 spread are this case. When one role is split into "
        "two `@wg<k>` rows, the skew between the two warpgroups is itself a reading (who arrives "
        "first, who waits for whom).\n")
    return block


def _timeline_student_note(ctx: LoopContext) -> str:
    if not _timeline_on(ctx):
        return ""
    has_def, names = _timeline_markers(ctx)
    from forgekernel.skills.timeline_probe import NSITE
    if _is_cuda(ctx):
        return _timeline_student_note_cuda(ctx, has_def, names, NSITE)
    lines = ["\n## Timeline probe (enabled for this run)",
             "- Convention: keep `def _seg(site, row=0): return None` in the kernel (the driver turns it "
             "into stamping only in the probe build); put `_seg(\"<role>.<segment>.<stem>.begin\", i)` / "
             "`\".end\"` before and after every blocking wait, and one marker after each chain commit, "
             "after each `wait_group(0)` drain, and at each role's entry; names use the segment names of "
             "the milestone timeline and **must be string literals** "
             "(the probe builds its site table from the source text: computed names such as "
             "`\"…%d…\" % k` or `NAMES[k]` cannot be read and are treated as no-ops in the probe build; "
             "to split sites per unrolled step, write N literal calls under branches such as "
             "`if cutlass.const_expr(k == 0)`), `i` = chunk / tile counter "
             "(not passed outside loops). In a persistent kernel (one CTA runs several work items / "
             "tile batches across the grid) a counter that keeps increasing over the CTA's whole life "
             "is the more convenient `i` — if the row restarts per work item, the probe automatically "
             "reads it as time-monotonic stretches, restarts do not count toward wrap/period, and the "
             "low-row-number segments only read the last consistent stretch of the loop. Send "
             "`PROBE_REQUEST: timeline` to get a table of the current tree (counts as one probe).",
             f"- **Site cap: {NSITE}**: beyond it the probe keeps only the `.begin`/`.end` wait pairs "
             f"(the other markers cannot be read) — as the marker count nears the cap, merge sites of "
             f"the same segment."]
    if has_def and names:
        lines.append(f"- Markers in the current tree ({len(names)}): " + ", ".join(f"`{n}`" for n in names[:24])
                     + ("…" if len(names) > 24 else ""))
        if len(names) > NSITE:
            lines.append(f"- **{len(names)} markers now > cap {NSITE}**: full-table mode will be "
                         f"skipped and only wait pairs measured; trim to {NSITE} or fewer.")
    else:
        lines.append("- **The current tree has no markers** (or lacks the `_seg` definition): the "
                     "driver cannot measure your timeline; add them before the next gate row.")
    if has_def and (bad := _timeline_unparsed_markers(ctx)):
        # observed: six computed site names the table could not see
        # killed the stamped variant at trace time on three trees, and the failure
        # line never said why — say it here, every turn, before the next probe
        lines.append(f"- **{len(bad)} `_seg(` calls have site names that are not string literals; "
                     f"the probe cannot read them (treated as no-ops in the probe build)**: "
                     + ", ".join(f"`{c}`" for c in bad[:8]) + ("…" if len(bad) > 8 else "")
                     + "; to split sites per unrolled step, write one literal call per step.")
    stale = _timeline_probe_stale(ctx)
    if stale:
        lines.append(f"- **The tree's `notes/timeline_probe.py` does not match the harness version (or "
                     f"is missing)**: copy `{stale}` over `notes/timeline_probe.py` and commit it with "
                     f"your next submission — the job runs the copy in the tree, and an old version's "
                     f"self-check lines (e.g. no `TIMELINE-COVERAGE`) leave the driver reading the "
                     f"table by the old rules.")
    last = _latest_timeline(ctx)
    if last is not None:
        tl = last.get("table") or {}
        lines.append(f"- Latest table (@`{str(last.get('sha'))[:10]}`, {last.get('label')}):")
        if _timeline_short_coverage(tl):
            # The student reads these cycle counts as the size of its own
            # segments. A short-coverage table's absolutes are not that, and the
            # teacher's copy already carries this note — same table, same caveat.
            lines.append(f"  - {meta_shapes.TL_COVERAGE_NOTE}**: this table's stamps cover only "
                         f"{tl.get('span_ratio')} of device time (mode={tl.get('mode')}); the uncovered "
                         f"part is not in the table (stretches no marker brackets, sites that mode=waits "
                         f"dropped, or launches in the same call without stamps); **segment cycles are "
                         f"in-row stamp differences; read the absolute values as they are** — do not "
                         f"inflate them by coverage, and do not take the sum of the segments as the whole "
                         f"call; to make the table cover everything, put sites in the prologue / epilogue "
                         f"and in the other kernels too.")
        for role, r in tl.get("roles", {}).items():
            top = sorted([sg for sg in r.get("segments", []) if sg.get("median") is not None],
                         key=lambda sg: -sg["median"])[:3]
            lines.append(f"  - {role}: period {_fmt_cyc(r.get('period'))}, busy {_fmt_cyc(r.get('busy'))}, "
                         f"wait {_fmt_cyc(r.get('wait'))}; largest three segments "
                         + ", ".join(f"`{sg['from']}`→`{sg['to']}` {_fmt_cyc(sg['median'])} ({sg['kind']})"
                                     for sg in top))
    return "\n".join(lines) + "\n"


def _timeline_student_note_cuda(ctx: LoopContext, has_def: bool, names: list, nsite: int) -> str:
    """``_timeline_student_note`` for a CUDA-mode run: the C++ marker convention
    (``timeline_probe.CU_MARKER_*``), the same site rules, cap, staleness and
    latest-table lines."""
    lines = ["\n## Timeline probe (enabled for this run)",
             f"- Convention: keep one line `#define FK_SEG(...) ((void)0)` in `{ctx.kernel_file}` (the "
             f"driver replaces it with stamping only in the probe build); "
             "put `FK_SEG(\"<role>.<segment>.<stem>.begin\", i);` / `\".end\"` before and after every "
             "blocking wait, and one marker after each chain commit, after each wait drain, and at each "
             "role's entry; names use the segment names of the milestone timeline and **must be string "
             "literals** "
             "(the probe builds its site table from the source text: names computed by macro pasting or "
             "array lookup cannot be read and are treated as no-ops in the probe build; to split sites "
             "per unrolled step, write N literal calls), `i` = chunk / tile counter (outside loops write "
             "`FK_SEG(\"<role>.entry\");`; no parentheses inside `i`). In a persistent kernel a counter "
             "that keeps increasing over the CTA's whole life is the more convenient `i` — if the row "
             "restarts per work item, the probe reads it as time-monotonic stretches. The probe build "
             "adds a readout function to your `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` block; keep "
             "that block in its standard form. Send `PROBE_REQUEST: timeline` to get a table of the "
             "current tree (counts as one probe).",
             f"- **Site cap: {nsite}**: beyond it the probe keeps only the `.begin`/`.end` wait pairs "
             f"(the other markers cannot be read) — as the marker count nears the cap, merge sites of "
             f"the same segment."]
    if has_def and names:
        lines.append(f"- Markers in the current tree ({len(names)}): " + ", ".join(f"`{n}`" for n in names[:24])
                     + ("…" if len(names) > 24 else ""))
        if len(names) > nsite:
            lines.append(f"- **{len(names)} markers now > cap {nsite}**: full-table mode will be "
                         f"skipped and only wait pairs measured; trim to {nsite} or fewer.")
    else:
        lines.append("- **The current tree has no markers** (or lacks the `#define FK_SEG` line): the "
                     "driver cannot measure your timeline; add them before the next gate row.")
    if has_def and (bad := _timeline_unparsed_markers(ctx)):
        lines.append(f"- **{len(bad)} `FK_SEG(` calls have site names that are not string literals; "
                     f"the probe cannot read them (treated as no-ops in the probe build)**: "
                     + ", ".join(f"`{c}`" for c in bad[:8]) + ("…" if len(bad) > 8 else "")
                     + "; to split sites per unrolled step, write one literal call per step.")
    stale = _timeline_probe_stale(ctx)
    if stale:
        lines.append(f"- **The tree's `notes/timeline_probe.py` does not match the harness version (or "
                     f"is missing)**: copy `{stale}` over `notes/timeline_probe.py` and commit it with "
                     f"your next submission.")
    last = _latest_timeline(ctx)
    if last is not None:
        tl = last.get("table") or {}
        lines.append(f"- Latest table (@`{str(last.get('sha'))[:10]}`, {last.get('label')}):")
        if _timeline_short_coverage(tl):
            lines.append(f"  - {meta_shapes.TL_COVERAGE_NOTE}**: this table's stamps cover only "
                         f"{tl.get('span_ratio')} of device time (mode={tl.get('mode')}); segment cycles "
                         f"are in-row stamp differences; read the absolute values as they are, and do "
                         f"not inflate them by coverage.")
        for role, r in tl.get("roles", {}).items():
            top = sorted([sg for sg in r.get("segments", []) if sg.get("median") is not None],
                         key=lambda sg: -sg["median"])[:3]
            lines.append(f"  - {role}: period {_fmt_cyc(r.get('period'))}, busy {_fmt_cyc(r.get('busy'))}, "
                         f"wait {_fmt_cyc(r.get('wait'))}; largest three segments "
                         + ", ".join(f"`{sg['from']}`→`{sg['to']}` {_fmt_cyc(sg['median'])} ({sg['kind']})"
                                     for sg in top))
    return "\n".join(lines) + "\n"


def _timeline_feedback(tl: dict | None) -> str:
    if tl is None:
        return ("\n## Timeline probe\n- The job printed no TIMELINE line (the probe did not run); check "
                "that `notes/timeline_probe.py` is in the tree.\n")
    if not _timeline_renderable(tl):
        if tl.get("env"):
            return (f"\n## Timeline probe\n- Did not run: a **job environment** problem (`{tl['env']}`), "
                    f"not a problem with the tree or the markers — adding markers or changing the kernel "
                    f"will not make it pass. Do not send `PROBE_REQUEST: timeline` again until the "
                    f"environment is fixed; take segment readings from the pricing table or a verdict "
                    f"probe instead.\n")
        return (f"\n## Timeline probe\n- Failed: {tl.get('fail')}. Add the missing markers / fix it, then "
                f"send `PROBE_REQUEST: timeline` again.\n")
    return "\n## Timeline (measured on the current tree)\n" + _timeline_tables(tl)


_LEVER_DESIGN_BLOCK = (
    "\n## Pricing round (this turn only designs candidates; no diagnosis)\n"
    "Before you write cause_class, the driver **measures** on the best tree what each candidate "
    "you list is worth: add-one (base + candidate), remove-one (all-on − candidate) and all-on, as "
    "same-job paired ratios, {PRICING_JOBS}. Your task: of **all the directions in the evidence "
    "above that you would consider patching**, take those that can be written as **local, "
    "stackable exact-text replacements** in `{KERNEL_FILE}` (at most "
    "{MAX_LEVERS} groups) and write them into `notes/levers.json`.\n\n"
    "**Group first, then list.** One candidate = a group of edits that **only work together** "
    "(one place, or three). "
    "Grouping is a mechanism judgement and it is yours: when two edits touch the two ends of the "
    "same wait, or one is the precondition of the other (registers must be freed before the "
    "prefetch can deepen; the holding scheme must change before the batch depth can rise), put "
    "them in **the same candidate's `edits` list**, and the driver measures them as one unit. "
    "Listing them apart makes the driver measure them separately — the half that is useless "
    "alone reads 0, the other half reads negative, both are judged null, and the real gain was "
    "never measured. Conversely, write two edits that **each stand on their own** as two "
    "candidates, so the driver can tell whether stacking them helps or cancels. When unsure, "
    "add one more \"merged group\" candidate "
    "(A+B as a third candidate); all-on / remove-one will tell you the answer.\n\n"
    "```json\n"
    "{\"rounds\": 3, \"levers\": [\n"
    "  {\"name\": \"<short name>\", \"decisions\": [\"D<x>.<y>\"], "
    "\"expect\": \"<one-sentence mechanism prediction: which wait of which role this group changes>\",\n"
    "   \"edits\": [{\"old\": \"<original text occurring exactly once in the kernel>\", \"new\": \"<replacement>\"},\n"
    "             {\"old\": \"...\", \"new\": \"...\"}],\n"
    "   \"in_stack\": true},\n"
    "  ...],\n"
    " \"variants\": [\n"
    "  {\"name\": \"<subset name>\", \"levers\": [\"<lever name>\", \"<lever name>\"], "
    "\"question\": \"<the question this combination answers>\"},\n"
    "  ...]}\n"
    "```\n"
    "**You decide which combinations to measure.** The driver always measures base and every "
    "add-one (the floor under every citation; not optional); "
    "all-on / all-on minus one are measured by default too, but they cannot answer two questions: "
    "whom a `coupled` lever actually depends on, and how much information the other readings keep "
    "once a lever that is harmful on its own got into all-on. You answer those with subsets in "
    "`variants` "
    "(at most {MAX_SUBSETS}, each a set of lever names + one sentence on the question it answers); "
    "each subset gets its own "
    "`VARIANT` reading against base, which the diagnosis can cite directly as "
    "`priced_by=<subset name>`. Typical uses: suspect A depends on B → "
    "list `{A, B}`; suspect C is harmful → give C `\"in_stack\": false` (only its add-one is "
    "measured; it enters neither all-on "
    "nor any subset), or list a combination without C. In addition, when a lever is judged harmful "
    "(negative add-one, and all-on does not depend on it) the driver automatically measures "
    "\"all-on minus these\" once more (`all-pos`, re-measured interleaved with base); you do not "
    "write it.\n"
    "Rules: (1) every `old` occurs exactly once in the kernel, and still exactly once after all "
    "candidates are stacked in order "
    "(the edit regions of different candidates do not overlap; several edits inside one candidate "
    "are applied in list order); (2) write only local edits of the **timing / holding scheme / "
    "issue organisation** kind (wait placement, index holding, ring acquisition granularity, "
    "DEPBAR depth, cache bits, ...); "
    "edits to **geometry / axes / stage shape** are not candidates — that is R1 reopen territory, "
    "it goes into the diagnosis, not here; but a reopen "
    "itself likewise **must not happen unpriced**: its price comes from a verdict experiment "
    "(PROBE_REQUEST builds a minimal variant on the best tree, paired same-job measurement), and "
    "the diagnosis cites the reading with `reopen_price:`; (3) no "
    "new implementation bodies and no reference code; (4) the members inside one candidate group "
    "are never priced individually — if you want to know what one place inside a group is worth "
    "on its own, also list it as an independent candidate; (5) **every candidate must carry "
    "`decisions`**: which "
    "milestone decision(s) it moves the value of (decision ids such as `D1.1`). The driver uses "
    "this column to map the pricing table back onto the phase order — "
    "when every candidate of the current phase measures null, the driver rules \"this phase's "
    "levers are exhausted\" and asks you to rule on closing the phase in the diagnosis, "
    "so the student stops re-measuring the same knobs point by point. A candidate without "
    "`decisions` is still priced but does not count toward any phase's exhaustion ruling.\n"
    "**Ablations (optional, same job, at most {MAX_ABLATIONS} entries in `ablations`)**: a "
    "candidate asks \"is it faster with this added\"; "
    "an ablation asks \"how much faster is it once a mechanism **already on** the tree is removed "
    "/ degraded\" — strip one layer of barrier, make a pipelined stretch serial, "
    "stub a stage out to its naive form. It is **a conviction tool, not a candidate**: applied "
    "alone on base only, never in all-on or any "
    "subset, and **allowed to compute wrong results** (the reading is the NCU kernel clock, "
    "re-measured in the same pass as base; bench only records whether it happens to stay "
    "correct). "
    "Each entry carries a `hypothesis` (the one sentence you want to confirm / refute) and "
    "`decisions`. Δ positive = faster without it = the upper bound of what this mechanism costs "
    "now; Δ negative = slower without it = it is helping. When to use one: when no candidate can "
    "be written — you do not know how the reference does it "
    "(blackbox run), the fix is geometry / stage shape (R1 territory), or the probe script failed "
    "to produce a verdict several times — convict with an ablation first, then decide whether "
    "a reopen is worth it; an R1 can cite `reopen_price: <Δ> via ablation:<name>` (only a "
    "positive Δ above the noise floor counts as a price). "
    "A hanging ablation: ncu runs first, a timeout marks it hung and it is not touched again (no "
    "bench, no later rounds), and the ablation pass has its own total budget; "
    "**an ablation that deadlocks still costs one ncu timeout**, so prefer a degraded form "
    "(swap in a coarser synchronisation that still runs) over a deletion that will hang. "
    "Names must not equal a lever / subset name.\n"
    "```json\n"
    "{\"ablations\": [\n"
    "  {\"name\": \"<short name>\", \"decisions\": [\"D<x>.<y>\"], "
    "\"hypothesis\": \"<why it is faster without it, and by how much>\",\n"
    "   \"edits\": [{\"old\": \"<original text occurring exactly once in the kernel>\", "
    "\"new\": \"<the removed / degraded form>\"}]}\n"
    "  ...]}\n"
    "```\n"
    "After writing the file, end this turn with a single line `LEVERS: <n>` (no cause_class); if "
    "there really is no stackable candidate, write "
    "`LEVERS: none` (if the file holds only `ablations`, the driver still runs the ablation "
    "pass). Next turn you get the pricing table and then diagnose.\n")
_LEVER_DESIGN_BLOCK = _LEVER_DESIGN_BLOCK.replace("{MAX_SUBSETS}", str(_LEVER_MAX_SUBSETS)).replace(
    "{MAX_ABLATIONS}", str(_LEVER_MAX_ABLATIONS))


def _lever_pricing_jobs_text(ctx: LoopContext) -> str:
    """How many GPU jobs the design turn's levers.json buys (_LEVER_DESIGN_BLOCK's
    {PRICING_JOBS}): one, or -- a group card listing its meta workloads -- one per
    meta workload, each timing that workload alone (_lever_price_submit)."""
    meta = meta_shapes.card_meta_workloads(ctx.workspace)
    if not meta:
        return "one GPU job"
    return (f"multi-workload tree: **one GPU job for each** of the operator tree's {len(meta)} meta "
            f"workloads ({', '.join(f'`{w}`' for w in meta)}, the representative first) — the same "
            f"levers.json (same levers, subsets and ablations), each job "
            f"timing only its own workload; the driver merges the tables by geomean and attaches "
            f"the per-workload readings under the table "
            f"(when a workload's job produced no table, the text above the table says which ones "
            f"were actually merged)")


def run_meta_step(ctx: LoopContext, *, best_before: float | None = None,
                  milestone_before: str = "", audit: bool = False,
                  reanchor_note: str | None = None) -> MetaResult:
    """Diagnose, then either certify a hw_kb recipe (G3) or patch the milestone.

    Records a MetaRecord (the meta agent's memory): the diagnosis, the patch class +
    changed entries, and the outcome — terminal for the G3 KB branch, PENDING for a
    milestone patch (judged after its inner segment runs, see run_two_layer). Prior
    meta records are reflected back into the diagnose prompt so the meta agent does
    not re-propose a direction that already regressed.

    ``audit``: the forced M0 reconciliation variant. Uses m0_audit_prompt.md (the
    question is "is the MILESTONE consistent with the hardware truth", not "why did
    the segment stall") and accepts the extra verdict ``OK`` — no defect, no patch,
    recorded AUDIT_CLEAN. Any G/R verdict flows into the normal KB / patch machinery.
    """
    be = ctx.backend
    bindings = _bindings(ctx)
    bindings["{DIAGNOSIS_PATH}"] = "(inline below)"

    prior_records = load_meta_records(ctx.meta_status_path) if ctx.meta_status_path else []
    meta_step = len(prior_records)
    reflection = reflect_summary(prior_records)
    # INJECT the NCU report inline: the meta agent runs in the student workspace and
    # cannot fetch it via ./forge, so the diagnose prompt must carry the hardware truth
    # itself — otherwise meta diagnoses blind and over-relies on the perf numbers.
    from forgekernel.inner.ncu import (render_ncu_reconciliation, render_ncu_report,
                                       render_ncu_trace)
    records = load_records(ctx.status_path)
    # the phase / parity readings must not see rows of trees that are gone
    # (a reverted segment, KEEPs an in-place REOPEN voided) — see _live_records
    live_records = _live_records(records, ctx.voided_ranges)
    # The arch ref's signature rides along so ratio-based flags (wavefront
    # volume) can fire — a volume tax is invisible to per-iteration differencing
    # AND to absolute thresholds; only student-vs-ref exposes it.
    arch_ncu = ctx.arch_baseline.ncu if (ctx.arch_baseline and ctx.arch_baseline.has_ncu) else None
    # The reconciliation reads any archived signature: the arch ref (the machine
    # being reproduced) wins when both exist; the library baseline covers runs
    # that only measured ref_baseline.json. The whole BASELINE travels, not just
    # its ncu dict: the block prints role + ref_name so the reader can tell which
    # reference is speaking, and the isomorphism verdict keys on the role — a
    # library kernel is a different implementation, and "same tensor work" against
    # it is a coincidence, not a construction target (see baseline.py on why the
    # two roles must not be collapsed).
    recon_baseline = ctx.arch_baseline if arch_ncu else (
        ctx.ref_baseline if ctx.ref_baseline and ctx.ref_baseline.has_ncu else None)
    best_keep = _best_keep_profiler(records, ctx.gate_window)
    recon = None
    if recon_baseline is not None and best_keep is not None:
        best_prof, best_iter = best_keep
        recon = render_ncu_reconciliation(
            best_prof, recon_baseline.ncu, student_iter=best_iter,
            ref_role=recon_baseline.role, ref_name=recon_baseline.ref_name)
    ncu_view = (f"\n\n## NCU hardware ground truth (inline, per-iteration bottleneck verdicts — "
                f"primary evidence, read before judging)\n"
                f"{render_ncu_report(records, baseline=arch_ncu)}"
                f"\n\n### Metric evolution table\n{render_ncu_trace(records)}\n")
    if recon:
        # Two-sided reconciliation: the digest above never quoted the ref side,
        # so "tensor occupancy gap ≈ the whole residual" — the decisive plateau
        # signal — was invisible to every diagnosis (observed on fp8 and GDN runs).
        ncu_view += (f"\n### Two-sided NCU reconciliation (student best KEEP vs reference "
                     f"signature, read before judging)\n"
                     f"{recon}\n")
    # The timed-window ledger: what each side's stopwatch covers, both
    # decompositions, and the priced contract tax. Without it a residual that is
    # entirely a rebuild the reference never pays reads as a kernel defect and
    # the diagnosis goes looking for it in register ledgers.
    ncu_view += f"\n{_window_contract_binding(ctx)}\n"
    diagnose_extra = (f"{ncu_view}\n## Your past meta steps (reflect; do not repeat failed "
                      f"directions)\n{reflection}\n")
    # REOPEN BUDGET (see LoopContext.reopen_max_fruitless): read off the meta
    # memory, so a driver restart keeps the count
    reopen_streak = _fruitless_reopen_streak(prior_records)
    reopens_capped = 0 < ctx.reopen_max_fruitless <= reopen_streak
    if reopens_capped:
        diagnose_extra += _reopen_capped_block(reopen_streak, ctx.reopen_max_fruitless)
    if not audit and reanchor_note:
        # Passed by run_two_layer for every diagnosis of the SAME meta entry —
        # including malformed-patch retries, which re-run the diagnosis and would
        # otherwise reason from the stale residual again. A later meta entry is a
        # new period: the caller measures afresh instead of forwarding this note.
        diagnose_extra += reanchor_note
    priced = None
    if ctx.stall_pricing:
        # The aggregate digest above ranks bottleneck CLASSES; only a per-instruction
        # ledger localizes a gap dispersed through mainloop interleave quality (the
        # observed on a fwd run: two loci chosen off the digest, ~1pp each once priced, 10h
        # of construction rolled back). Price the best kernel NOW, before a locus
        # is named, and put the ref's recon-time ledger beside it.
        #
        # Every teacher entry prices — the audit included. The audit used to be
        # paper-level ("no GPU job at M0 time"), which left it holding the ref's
        # dossier against the blueprint's WORDING while the student's built
        # structure went unread: a segment whose every stop is an M0 rebuild
        # (first_green -> audit -> foundation patch -> rebuild -> first_green ...)
        # never reaches the plateau path, so the student dossier was produced
        # once per run or never, and the audit kept re-legislating acceptance
        # text at the same locus while the `!` structural rows stood.
        _ensure_ref_pricing(ctx)
        priced = _price_student_trees(ctx, records)
        student_ledger, student_segments = priced.ledger, priced.segments
        ref_ledger = _read_ref_stall_ledger(ctx)
        ref_segments = _read_ref_segment_ledger(ctx)
        aligned = (align_segment_table(ref_segments, student_segments)
                   if ref_segments and student_segments else None)
        orch_diff = _orch_diff_views(ctx, priced)
        role_view = _role_exposure_view(ctx, priced)
        if role_view:
            # The priced face by role: where the segment table says WHICH
            # SEGMENT waits and the dossier says WHICH STRUCTURE differs, this
            # says WHICH ROLE owns the non-tensor exposure, in ns, both sides.
            diagnose_extra += role_view
        if orch_diff:
            # The structural face beside the priced one: the stall ledger says
            # where time leaks, this says which sync structure differs — the
            # known failure mode ("[GATHER: prefetched] in the blueprint,
            # never built") is a `!` row here, statically, before any timing
            # argument. `=` rows are equally load-bearing: a structure already
            # isomorphic to the ref is not where the residual lives.
            diagnose_extra += orch_diff
    tl_table = None
    if _timeline_on(ctx):
        # the per-segment clock account of the tree under judgment: rendered
        # before the stall ledgers, because on a latency-bound kernel it is the
        # evidence the locus is read from; a probe that produced nothing is
        # logged, not shown — but a partial-coverage table IS shown (dropping
        # it once left three meta entries diagnosing from aggregate NCU ratios
        # alone)
        tl_ref = (priced.best_sha or priced.head_sha) if priced is not None else ctx.backend.head()
        tl_table = _run_timeline(ctx, tl_ref, "audit" if audit else "plateau")
        if tl_table and _timeline_renderable(tl_table):
            diagnose_extra += _timeline_block(tl_table, _ref_region_lines(ctx))
        else:
            tl_table = None
    if not audit and ctx.stall_pricing:
        if aligned:
            # The comparison the hard rule needs, pre-aligned: same anchor, ref
            # share next to student share. Two raw ledgers made the reader do the
            # alignment, and the reading that matters (share 23.5 vs 35.4 at the
            # SAME segment, dispatch- vs wait-dominated) only exists after it.
            diagnose_extra += (
                "\n## Per-instruction stall ledger — same-position segment comparison (read "
                "before judging — `deficient_locus` must match the segments where the student "
                "exceeds the reference)\n"
                "Anchors aligned on the schedule skeleton: loop-top wait → first HG run → DEPBAR → "
                "EW segments per depth → EXIT.\n"
                "% = the segment's share of the loop body's stall samples; stalls = the dominant "
                "stall of the largest segment inside the anchor "
                "(`category:percent`). Reading: a same-segment share clearly above the "
                "reference's, with wait classes "
                "(barrier/long_sb/short_sb) dominating on the student side while dispatch / "
                "selection classes dominate the reference's same segment ⇒ the structure is "
                "right and the intra-segment instruction scheduling is worse (G5 shape; do not "
                "touch the structure).\n"
                f"```\n{aligned}\n```\n"
                f"\n### student {priced.ledger_tree} kernel raw segment ledger (measured this "
                f"time)\n"
                f"```\n{student_segments[:4000]}\n```\n")
        else:
            diagnose_extra += (
                "\n## Per-instruction stall ledger (read before judging — `deficient_locus` "
                "must match the ledger's big items)\n"
                "Ledger rows are sorted by cost, descending, accumulated to ≥90% coverage. The "
                "defect locus you name must explain the big items at the top of the ledger; if you "
                "consider them immovable and pick a locus outside the ledger, `evidence` must "
                "argue explicitly why every item at the top of the ledger is immovable.\n"
                f"\n### student {priced.ledger_tree} kernel (measured this time)\n"
                + (f"```\n{student_ledger}\n```\n" if student_ledger
                   else "(Ledger unavailable this round — the collection job failed or its "
                        "self-check did not pass. Fall back to diagnosing from the aggregate "
                        "signature, and lower `confidence` by one notch accordingly.)\n")
                + "\n### Reference (collected at recon time)\n"
                + (f"```\n{ref_ledger[:6000]}\n```\n" if ref_ledger
                   else "(Recon left no reference ledger.)\n"))
    if audit and (ref_dossier := _read_ref_orch_dossier(ctx)):
        # Paper-level contract check: the audit holds the blueprint's
        # [SYNC:]/[GATHER:]/[PHASE:] adjudications against the reference's
        # machine-read dossier — whether the blueprint transcribed the
        # reference's structure or guessed it (the two inertial wrong answers
        # §2g names). The BUILT structure is audited by the dossier diff block
        # above (the judged tree's own dossier, priced this entry); this block
        # is the blueprint's side of that reading.
        diagnose_extra += (
            "\n## Reference orchestration dossier (machine-read at recon time, static SASS "
            "transcription — for the audit's ruling on the blueprint)\n"
            "Check the milestone's [SYNC:]/[GATHER:]/[PHASE:] rulings one by one against it: "
            "where the sync topology / supply organisation the blueprint declares agrees with "
            "this dossier, record CLEAN; where it disagrees, or the blueprint is vague while the "
            "dossier has a clear reading, bounce it with a mechanism attribution — guessing a "
            "symmetric rendezvous for the sync topology and per-row inline loads for the supply "
            "were the two inertial answers from before there was a dossier; with the dossier "
            "there is nothing left to guess.\n"
            f"```\n{ref_dossier[:6000]}\n```\n")
    if priced is not None:
        # a group card: the same readings on its other meta workloads (one job each)
        diagnose_extra += _meta_workload_readings(ctx, priced, "audit" if audit else "plateau")
    lever_table: list = []
    if priced is not None:
        # plateau OR audit: the audit is where structural reopens are decided,
        # and a reopen is chosen against the levers it passes over — those must
        # carry measured prices there too, not only after a later plateau.
        lever_table = _run_lever_pricing(
            ctx, priced, bindings, diagnose_extra,
            prompt_file="m0_audit_prompt.md" if audit else "diagnose_prompt.md")
        if lever_table:
            diagnose_extra += _lever_price_block(lever_table)
        elif ctx.lever_price_lost:
            # a job was bought and its table lost (archive / tool): said once,
            # so the teacher does not read this entry as "pricing never ran"
            # and cite priced_by= for levers it just designed
            diagnose_extra += _lever_price_lost_block(ctx.lever_price_lost)
            ctx.lever_price_lost = ""
    if getattr(ctx, "discarded_edits", ""):
        # said once: a pricing turn (just now) or an earlier verdict-probe reset
        # threw away edits outside notes/ that the teacher may believe landed
        diagnose_extra += f"\n## Edits discarded by a reset\n{ctx.discarded_edits}\n"
        ctx.discarded_edits = ""
    if revs := [rv for rv in _unretired_revisions(ctx) if rv.get("verdict") in _REVISION_FLAGGED]:
        # what the inner loop found when the best moved under the milestone's
        # cited prices: the teacher reads it here and must act on the suspended
        # rungs; a fresh table on the current best supersedes (retires) it
        diagnose_extra += _price_revision_teacher_block(revs)
        # shown once: the milestone's RE-PRICED / SUSPENDED markers are the
        # durable record, the standing student note is retired here whether or
        # not this entry lists levers again
        _retire_revisions(ctx)
    # every lever this run has priced: a citation of one from an earlier step
    # (the reflection lists them as priced) verifies against the same readings
    known_levers = _merge_lever_tables(prior_records, lever_table)
    # PHASE EXHAUSTION: the table priced on the current best is a batch
    # measurement of the phase's own knobs; when they all read null the phase's
    # remaining walk is a re-measurement, and the diagnosis must say so (close
    # the phase, or name the one unpriced decision still worth walking).
    exhaustion = _phase_exhaustion(ctx, live_records, lever_table) if lever_table else None
    if exhaustion and ctx.phase_close and ctx.phase_close.get("gate") == exhaustion["gate"]:
        _log(f"meta: {exhaustion['phase']} is already closed on the price table — "
             + ("its gate row failed on this same best tree and the table has not "
                "changed; not asking again (the diagnosis attributes the FAIL)"
                if ctx.phase_close.get("stage") == "gate-failed"
                else f"awaiting its {exhaustion['gate']} row, not asking again"))
        exhaustion = None
    if exhaustion:
        _log(f"meta: {exhaustion['phase']} levers {exhaustion['levers']} all null/noop on the "
             f"best tree -> phase exhausted; demanding a phase_verdict "
             f"(unpriced: {exhaustion['unpriced'] or 'none'})")
        diagnose_extra += _phase_exhausted_block(exhaustion)
    prompt_file = "m0_audit_prompt.md" if audit else "diagnose_prompt.md"
    enum_help = ("`OK` `G1` `G2` `G3` `G4` `G5` `R1` `R2`" if audit
                 else "`G1` `G2` `G3` `G4` `G5` `R1` `R2` `INFEASIBLE`")
    diagnosis, cause = "", ""
    probe_rounds = 0
    probe_readings: list = []       # every contract line the probes printed, for the ledger
    reach_demanded = False
    price_demanded = False
    capped_demanded = False
    phase_demanded = False
    seg_demanded = False
    infeasible_asked = False
    infeasible_bounces: list[str] = []   # refusal reasons put back to the teacher this step
    diagnoses = _plateau_diagnoses(prior_records) + 1
    stop_line = None if audit else _resolve_target(ctx)
    # VERDICT PROBES: before committing to a cause_class the diagnose agent may run
    # bounded GPU experiments against the student's own tree (PROBE_REQUEST, the same
    # marker and submission plumbing as the student's probe sub-loop and the KB
    # certification — the agent decides what to ask, the driver owns the GPU). This is
    # the cheap falsification channel the meta loop lacked: without it a diagnosis
    # hypothesis was only ever tested at whole-inner-segment granularity, judged on
    # scalar best-progress. Probe turns and malformed-enum turns each draw on their own
    # budget; the loop bound is the sum, so a stubborn prober still terminates.
    pre_probe_head = be.head()
    probes_run: list = []           # paths, so a probe with no verdict line is "ran, no reading"
    for _ in range(max(1, ctx.max_patch_rounds) + ctx.max_verdict_probe_rounds):
        out = _teacher(be).run_agent(render_prompt(
            _meta_prompt(ctx, prompt_file), bindings, extra=diagnose_extra))
        cause = _parse_cause_class(out, allow_ok=audit)
        # A turn carrying a valid cause_class IS the final diagnosis, even when the
        # PROBE_REQUEST marker also appears in its text: the probe rules ask the
        # diagnosis to cite probe names + readings in `evidence`, and the marker regex
        # is a whole-text search — reading such a citation as a request would burn a
        # GPU job re-running an old script and discard a valid diagnosis (the same
        # trap the student-turn parser documents for a stray RECON_REQUEST echo).
        turn = None if cause else parse_student_turn_safe(out)
        if isinstance(turn, ProbeRequest) and _is_timeline_request(turn.path):
            if not _timeline_on(ctx):
                diagnose_extra += ("\n## Timeline probe not enabled\nThis run's loop.toml has "
                                   "[loop] timeline_probe off, so `PROBE_REQUEST: timeline` is not "
                                   "available; use a verdict probe or the pricing table.\n")
                continue
            if tl_table is not None:
                # the entry already read this tree: the block is in the prompt
                diagnose_extra += ("\n## Timeline\nThis step's entry already read the table on "
                                   "the same tree (the \"Timeline\" block above); not re-run — "
                                   "diagnose from it.\n")
                continue
            if probe_rounds >= ctx.max_verdict_probe_rounds:
                _log("meta: verdict-probe budget spent -> demanding a diagnosis")
                diagnose_extra += ("\n## Verdict-experiment budget exhausted\nNo more PROBE_REQUEST "
                                   "may be sent. Give the final diagnosis from the evidence you have; "
                                   "mark every unverified hypothesis in it explicitly as unverified.\n")
                continue
            probe_rounds += 1
            tl_ref = (priced.best_sha or priced.head_sha) if priced is not None else be.head()
            tl_table = _run_timeline(ctx, tl_ref, f"meta{meta_step}:teacher")
            if tl_table and _timeline_renderable(tl_table):
                diagnose_extra += _timeline_block(tl_table, _ref_region_lines(ctx))
            else:
                diagnose_extra += _timeline_feedback(tl_table)
                tl_table = None
            continue
        if isinstance(turn, ProbeRequest):
            if probe_rounds >= ctx.max_verdict_probe_rounds:
                _log("meta: verdict-probe budget spent -> demanding a diagnosis")
                diagnose_extra += ("\n## Verdict-experiment budget exhausted\nNo more PROBE_REQUEST "
                                   "may be sent. Give the final diagnosis from the evidence you have; "
                                   "mark every unverified hypothesis in it explicitly as unverified.\n")
                continue
            probe_rounds += 1
            probes_run.append(turn.path)
            probe_branch = f"{ctx.branch_prefix}-metaprobe"
            if turn.path.lower().endswith(".json"):
                # an ablation spec instead of a script: the driver's frozen
                # pricing recipe runs it (NCU clock, base re-timed alongside),
                # so the experiment cannot fail on the teacher's harness code
                # — the NO VERDICT shape a hand-written probe keeps taking
                rows, why, probe_log = _run_ablation_probe(ctx, be, turn.path, probe_branch)
                _log(f"meta: ablation probe {turn.path} "
                     f"({probe_rounds}/{ctx.max_verdict_probe_rounds}) -> "
                     + ("; ".join(f"{r['name']} {_fmt_pct(r['delta_pct'])}" for r in rows)
                        if rows else f"NO VERDICT ({why})"))
                diagnose_extra += _ablation_probe_feedback(
                    turn.path, rows, why, probe_log, probe_rounds, ctx.max_verdict_probe_rounds)
                noise = next((r.get("noise_pct") for r in rows if r.get("noise_pct") is not None),
                             _LEVER_NOISE_DEFAULT_PCT)
                probe_readings.extend(
                    {"path": turn.path, "round": probe_rounds, "contract": r["name"],
                     # the hypothesis "this mechanism is charging": confirmed only by a
                     # positive delta above the floor; within the floor or NEGATIVE
                     # (removing it is slower — it helps) falsifies it
                     "verdict": ("PASS" if r["delta_pct"] is not None and r["delta_pct"] > noise
                                 else "FAIL" if r["delta_pct"] is not None else "NO VERDICT"),
                     "reading": _fmt_pct(r["delta_pct"]) if r["delta_pct"] is not None else "unreported",
                     "note": (f"ablation (ncu clock, correctness={r.get('correctness')}, "
                              f"status={r.get('status')}): {r.get('hypothesis', '')}"
                              + (" — removing it is SLOWER: the mechanism helps"
                                 if r["delta_pct"] is not None and r["delta_pct"] < -noise else ""))}
                    for r in rows)
                # the rows join the step's price table: `ablation:<name>` cites
                # them, the record banks them, later steps see them as priced
                lever_table.extend(rows)
                known_levers.extend(rows)
                continue
            be.commit(f"meta-probe: {turn.path}")
            be.push_branch(probe_branch)
            probe_log = be.submit_gpu(probe_branch, f"cmd:python3 {turn.path}")
            # audit=False: these lines answer pre-registered hypotheses, and the
            # max_diff field carries the reading as printed. The certification
            # audit (error-norm ceiling, float-only) was downgrading a PASS with a
            # reading table to FAIL — i.e. telling the teacher its hypothesis had
            # been falsified when the experiment had confirmed it.
            pv = parse_probe_verdict(probe_log, audit=False)
            _log(f"meta: verdict probe {turn.path} "
                 f"({probe_rounds}/{ctx.max_verdict_probe_rounds}) -> "
                 f"PROBE {_probe_label(pv)} (max_diff={pv.diff_text})"
                 f"{_probe_log_suffix(pv)}")
            diagnose_extra += _verdict_probe_feedback(
                turn.path, pv, probe_log, probe_rounds, ctx.max_verdict_probe_rounds)
            probe_readings.extend(
                {"path": turn.path, "round": probe_rounds, "contract": c.index,
                 "verdict": c.verdict, "reading": c.reading, "note": c.note}
                for c in pv.contracts)
            continue
        diagnosis = out
        if cause:
            # REACHABILITY GATE (plateau only, ledger channel present): the floor
            # arithmetic (diagnose_prompt §1c) is what turns "levers exhausted" into
            # "architecture cannot reach target" while the evidence is fresh — the
            # observed: a DSA run derived it 6 meta steps late and lost the rebuild to the
            # shutdown. One re-prompt, then the diagnosis is taken as-is: a missing
            # line must never cost more than one turn of the shared budget.
            if (not audit and ctx.stall_pricing and not reach_demanded
                    and not _parse_reachability(out)):
                reach_demanded = True
                _log("meta: diagnosis omitted the reachability line -> re-prompt")
                diagnose_extra += (
                    "\n## Format requirement (binding)\nThe output block lacks a `reachability:` "
                    "line. Do the reachability refutation of diagnosis step 1c first (the "
                    "construction of `floor_opt` and its term-by-term arithmetic go in "
                    "`evidence`), then resend the **complete** diagnosis "
                    "(every field, not just the one missing line).\n")
                continue
            # INFEASIBLE is a stop request: none of the patch-shaping gates below
            # (segment / reopen price / phase verdict) apply to it.
            if cause.upper() == "INFEASIBLE":
                # INFEASIBLE GATE: one bounce naming the conditions; a request
                # that still fails them is refused after the loop.
                why = _infeasible_refusal(out, diagnoses=diagnoses,
                                          min_diagnoses=ctx.infeasible_min_diagnoses,
                                          target_ms=stop_line)
                if why and not infeasible_bounces:
                    infeasible_bounces.append(why)
                    infeasible_asked = True  # the refusal block already names the channel
                    _log(f"meta: INFEASIBLE stop request not accepted ({why}) -> re-prompt")
                    diagnose_extra += _infeasible_refused_block(why)
                    continue
                break
            # INFEASIBLE CHANNEL: a diagnosis that talks INFEASIBLE under another
            # class is how a run spins — the verdict lives in the milestone, the
            # driver never hears it, and every later meta step re-diagnoses a
            # "none" locus until the budget is gone. One bounce makes the teacher
            # choose: the stop request, or the class it wrote (a quote / a negation).
            #
            # Not when the diagnosis's own reachability line reads OPEN: the gate
            # refuses an INFEASIBLE stop on OPEN anyway (_infeasible_refusal), so the
            # bounce could only buy a turn that ends where this one did. §1c makes
            # every diagnosis argue reachability against the stop rule, so the word
            # is in nearly all of them as a negation — in one large batch of
            # runs, 1157 of 1163 such diagnoses read OPEN and 695 teacher
            # turns went to the bounce.
            if (not audit and not infeasible_asked
                    and _INFEASIBLE_MENTION.search(out)
                    and _parse_reach_verdict(_parse_reachability(out)) != "OPEN"):
                infeasible_asked = True
                _log(f"meta: diagnosis mentions INFEASIBLE under cause_class {cause} "
                     f"-> re-prompt (stop request or not)")
                diagnose_extra += _INFEASIBLE_MENTION_BLOCK
                continue
            # LOCUS SEGMENT: a diagnosis read against a timeline names the segment
            # the gap sits in — the reading that makes the table binding.
            if tl_table is not None and not seg_demanded and not _parse_locus_segment(out):
                seg_demanded = True
                _log("meta: diagnosis omitted the locus_segment line -> re-prompt")
                diagnose_extra += (
                    "\n## Format requirement (binding)\nThis step has a timeline table, and the "
                    "output block lacks a `locus_segment:` line. Use the table to place the lesion "
                    "in a specific segment: `locus_segment: <role>.<segment>` (if it cannot be "
                    "placed in a segment, write `none` and explain in evidence), "
                    "then resend the **complete** diagnosis (every field, not just the one missing "
                    "line).\n")
                continue
            # REOPEN BUDGET GATE: the reopen budget is spent — an R1 cannot land,
            # so one bounce asks for a class that can (before the price gate
            # would spend turns pricing a reopen the driver is going to refuse)
            if cause.upper() == "R1" and reopens_capped and not capped_demanded:
                capped_demanded = True
                _log(f"meta: R1 diagnosed with the reopen budget spent ({reopen_streak} "
                     f"fruitless reopen(s) in a row) -> re-prompt")
                diagnose_extra += _reopen_capped_block(reopen_streak, ctx.reopen_max_fruitless,
                                                       bounce=True)
                continue
            # REOPEN PRICE GATE (plateau and audit): an R1 is a structural reopen,
            # and a reopen without a measured price is a known dsa_sparse failure —
            # the stage split axis flipped on an inferred masking window, six
            # decisions redone for 2%. One bounce naming the three ways to price
            # it (verdict probe / price table / certified KB); a second unpriced
            # R1 is refused below, after the loop.
            if (cause.upper() == "R1" and _reopen_gate_on(ctx) and not price_demanded
                    and (why := _verify_reopen_price(_parse_reopen_price(out), probe_readings,
                                                     known_levers, ctx, probes_run,
                                                     best_ref=_priced_best(priced)))):
                price_demanded = True
                _log(f"meta: R1 reopen carries no measured price ({why}) -> re-prompt")
                diagnose_extra += _UNPRICED_REOPEN_BLOCK + f"(driver check result: {why})\n"
                continue
            # PHASE VERDICT GATE: the exhaustion reading was put to the teacher;
            # an answer that is neither close nor a walkable decision costs one
            # bounce, then the driver closes the phase itself (below).
            if (exhaustion and not phase_demanded
                    and (why := _verify_phase_verdict(_parse_phase_verdict(out), exhaustion))):
                phase_demanded = True
                _log(f"meta: phase_verdict for {exhaustion['phase']} missing or invalid "
                     f"({why}) -> re-prompt")
                diagnose_extra += (
                    f"\n## Phase ruling (binding)\nThe driver has ruled that {exhaustion['phase']}'s "
                    f"levers are all null on the pricing table, but the diagnosis's `phase_verdict` "
                    f"is not valid ({why}). Add `phase_verdict: close` "
                    f"(the student submits the `{exhaustion['gate']}` row directly next round) or "
                    f"`phase_verdict: continue:<D id>` "
                    f"(only a decision with no priced lever: "
                    f"{', '.join(exhaustion['unpriced']) or 'none — this phase has no such decision, so only close is possible'}), "
                    f"and resend the **complete** diagnosis. If it is missing or invalid again, the "
                    f"driver treats it as close.\n")
                continue
            break
        # ENUM GATE: cause_class must be exactly one enum token (no prose). Re-prompt.
        _log("meta: diagnosis had no valid cause_class enum -> re-prompt")
        diagnose_extra += (f"\n## Format requirement (binding)\nThe `cause_class` field must be "
                           f"exactly one of the enum values: "
                           f"{enum_help}, with no prose / commentary after it.\n")
    if probe_rounds:
        # Probe commits are add -A snapshots of whatever the diagnose agent touched —
        # not just notes/<name>.py — and every downstream branch (patch resume via
        # head(), the audit-clean and kb-example returns) continues from HEAD, so
        # without this reset a stray teacher edit to the kernel would enter the
        # student lineage silently under a "meta-probe:" message, unrecorded and
        # ungated. The -metaprobe branch keeps the scripts for provenance and the
        # readings are already in the prompt; the student line stays teacher-free.
        if lost := _edits_outside_notes(be, pre_probe_head):
            _note_discarded_edits(ctx, "meta (verdict-probe reset)", lost)
        be.reset(pre_probe_head)
        _log(f"meta: workspace reset to pre-probe head {pre_probe_head} "
             f"(probe commits preserved on the -metaprobe branch)")
    # the id alone: teachers append a parenthesised gloss ("D0.19 (C side…", or with a
    # fullwidth paren and no space) that the \S+ field parser kept, so the ledger and
    # memory rows read "D0.19(C"
    locus = re.split(r"[（(]", _parse_field(diagnosis, "deficient_locus"))[0]
    direction = _parse_line(diagnosis, "patch_direction")
    reachability = _parse_reachability(diagnosis)
    if reachability:
        _log(f"meta: reachability -> {reachability}")
    _log(f"meta{'/m0-audit' if audit else ''}: diagnosed -> {cause or '?'} @ {locus or '?'}")

    def _record(rec: MetaRecord) -> None:
        # every record of this step carries the stop requests refused on the
        # way (audit trail: a bounced INFEASIBLE that became a patch is still
        # visible in meta_status.jsonl)
        rec.infeasible_refusals = list(infeasible_bounces)
        if ctx.meta_status_path:
            append_meta_record(ctx.meta_status_path, rec)

    refused = ""
    if cause.upper() == "INFEASIBLE":
        refused = _infeasible_refusal(diagnosis, diagnoses=diagnoses,
                                      min_diagnoses=ctx.infeasible_min_diagnoses,
                                      target_ms=stop_line)
    if refused:
        # INFEASIBLE GATE, second strike: the request still fails the check after
        # the bounce. Not a stop: nothing is patched, the refusal enters the
        # memory the next diagnosis reads, and the caller re-diagnoses the same
        # state (meta budget permitting) — the UNPRICED_REOPEN shape.
        if refused not in infeasible_bounces:
            infeasible_bounces.append(refused)
        _log(f"meta: INFEASIBLE stop request REFUSED ({refused}); nothing patched, "
             f"the run continues (refusals this step: {len(infeasible_bounces)})")
        _record(MetaRecord(
            meta_step=meta_step, cause_class="INFEASIBLE", deficient_locus=locus,
            patch_class="infeasible-refused", changed_entries=[], patch_direction=direction,
            outcome="INFEASIBLE_REFUSED", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table))
        return MetaResult(diagnosis, "infeasible-refused", [], None, [],
                          lever_prices=lever_table)

    if cause.upper() == "INFEASIBLE":
        # The stop request: nothing is patched and no segment resumes — the
        # caller ends the run, and the stop is the run's verdict.
        notice = _write_infeasible_notice(ctx, meta_step=meta_step, diagnosis=diagnosis,
                                          reachability=reachability, best=best_before)
        _record(MetaRecord(
            meta_step=meta_step, cause_class="INFEASIBLE", deficient_locus=locus,
            patch_class="infeasible-stop", changed_entries=[], patch_direction=direction,
            outcome="INFEASIBLE_STOP", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table))
        _log("meta: teacher declared INFEASIBLE -> stopping the run; the stop is the "
             "run's verdict (stop=infeasible; no guard restarts it); notice -> "
             + (str(notice) if notice else "<unwritten, see this log>"))
        return MetaResult(diagnosis, "infeasible-stop", [], None, [],
                          lever_prices=lever_table)

    # The phase verdict, resolved: the teacher's answer, or close by default —
    # the student submits the gate row next either way (a `continue` narrows
    # the walk to the one decision named; the rest of the phase stays priced).
    phase_verdict, phase_close = "", None
    if exhaustion:
        answered = _parse_phase_verdict(diagnosis)
        if (why := _verify_phase_verdict(answered, exhaustion)):
            phase_verdict = "close (default: unanswered)"
            _log(f"meta: phase_verdict for {exhaustion['phase']} still {why} after the bounce "
                 f"-> closed by default")
        else:
            phase_verdict = answered
        phase_close = {"gate": exhaustion["gate"], "phase": exhaustion["phase"],
                       "verdict": phase_verdict, "levers": list(exhaustion["levers"]),
                       "note": _phase_close_note(exhaustion, phase_verdict)}
        _log(f"meta{'/m0-audit' if audit else ''}: {exhaustion['phase']} phase_verdict = "
             f"{phase_verdict} -> the student "
             + (f"walks {phase_verdict.split(':', 1)[1]} then submits {exhaustion['gate']}"
                if phase_verdict.startswith("continue:")
                else f"submits {exhaustion['gate']} next"))

    if audit and cause.upper() == "OK":
        # The milestone survived the reconciliation: every severe/elevated flag is
        # either scheduled with a sane price in the milestone or a student-side
        # matter already routed. Nothing to patch; the segment resumes where it was.
        _record(MetaRecord(
            meta_step=meta_step, cause_class="OK", deficient_locus="",
            patch_class="audit-clean", changed_entries=[], patch_direction=direction,
            outcome="AUDIT_CLEAN", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table,
            phase_verdict=phase_verdict))
        return MetaResult(diagnosis, "audit-clean", [], None, [],
                          lever_prices=lever_table, phase_close=phase_close)

    reopen_price = _parse_reopen_price(diagnosis)
    if cause.upper() == "R1" and reopens_capped:
        # the bounce named the budget; the teacher insisted: nothing is patched
        _log(f"meta{'/m0-audit' if audit else ''}: R1 @ {locus or '?'} REFUSED — reopen "
             f"budget spent ({reopen_streak} fruitless reopen(s) in a row, "
             f"[loop] reopen_max_fruitless = {ctx.reopen_max_fruitless}); nothing patched")
        _record(MetaRecord(
            meta_step=meta_step, cause_class=cause, deficient_locus=locus,
            patch_class="reopen-capped", changed_entries=[], patch_direction=direction,
            outcome="REOPEN_CAPPED", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table,
            reopen_price=reopen_price, phase_verdict=phase_verdict))
        return MetaResult(diagnosis, "reopen-capped", [], None, [],
                          lever_prices=lever_table, phase_close=phase_close)
    if cause.upper() == "R1" and _reopen_gate_on(ctx):
        why = _verify_reopen_price(reopen_price, probe_readings, known_levers, ctx, probes_run,
                                   best_ref=_priced_best(priced))
        if why:
            # The bounce above already named the routes; the teacher insisted (or
            # spent the turns without pricing). No reopen without a price: nothing
            # is patched, the refusal enters the memory the next diagnosis reads.
            _log(f"meta{'/m0-audit' if audit else ''}: R1 @ {locus or '?'} REFUSED — "
                 f"no measured price ({why}); nothing patched")
            _record(MetaRecord(
                meta_step=meta_step, cause_class=cause, deficient_locus=locus,
                patch_class="unpriced-reopen", changed_entries=[], patch_direction=direction,
                outcome="UNPRICED_REOPEN", probe_rounds=probe_rounds,
                probe_readings=probe_readings, lever_prices=lever_table,
                reopen_price=reopen_price, phase_verdict=phase_verdict))
            return MetaResult(diagnosis, "unpriced-reopen", [], None, [],
                              lever_prices=lever_table, phase_close=phase_close)

    if cause.upper() in ("G3", "G5"):
        # interface gap (G3) or in-segment ILP (G5: structure right, instruction-
        # level scheduling behind the ref's) -> the fix is a GPU-verified hw_kb
        # recipe, NOT a milestone edit; a KB addition is additive (no milestone
        # checkpoint invalidated) and the structure stays frozen.
        kb = run_kb_certify(ctx, diagnosis)
        _record(MetaRecord(
            meta_step=meta_step, cause_class=cause, deficient_locus=locus,
            patch_class="kb-example", changed_entries=[], patch_direction=direction,
            outcome="KB_CERTIFIED" if kb.certified else "KB_UNCERTIFIED",
            certified=kb.certified, probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table,
            phase_verdict=phase_verdict))
        directive = None
        if cause.upper() == "G5":
            directive = (
                f"- The previous meta step ruled **G5 intra-segment ILP** (the structure is all "
                f"right, the intra-segment instruction scheduling is worse), locus = "
                f"`{locus or '?'}`, direction: {direction or '(not given)'}. "
                "**Do not touch the M0 structure and do not change the milestone**: reorder "
                "instructions inside that segment (chain depth / interleaving two adjacent rows / "
                "the distance between a long-latency instruction and its consumer / the work "
                "distribution across a drain)"
                + (", together with the KB recipe just certified" if kb.certified
                   else " (this step's KB recipe could not be certified; only this direction is given)")
                + ". Acceptance = NCU `t_ns` goes down + that segment's stall share converges "
                "toward the reference; the single-shot absolute performance line is not the "
                "judge.\n")
        return MetaResult(diagnosis, "kb-example", [], None, [], certified=kb.certified,
                          recipe=kb.recipe, student_directive=directive,
                          lever_prices=lever_table, phase_close=phase_close)

    # HARNESS GATE: the patch must actually reach the file, and the patched milestone
    # must stay §9-conformant. Re-prompt the patch agent with the violations until clean
    # or budget spent (symmetric to the student's probe sub-loop and the KB certify loop).
    at_entry = _read_milestone(ctx)
    patch_out, violations = "", []
    for _ in range(max(1, ctx.max_patch_rounds)):
        extra = f"\n\n## DIAGNOSIS\n{diagnosis}\n"
        if violations:
            extra += ("\n## Violations in your previous patch (fix them and resend)\n"
                      + "\n".join(f"- {v}" for v in violations) + "\n")
        patch_out = _teacher(be).run_agent(render_prompt(_meta_prompt(ctx, "patch_prompt.md"), bindings, extra))
        violations = _patch_landing_violations(
            at_entry, _read_milestone(ctx), _parse_list(patch_out, "changed_entries"))
        violations += _validate_milestone_file(ctx.milestone, baseline=at_entry)
        if not violations:
            break
        _log(f"meta: patch has {len(violations)} violation(s) -> re-prompt: {violations[0]}")
    for w in _dependency_warnings_file(ctx.milestone):
        # advisory in the patch mode (a ⇐ hole / cycle / one-sided ⇒): logged
        # so the closure arithmetic below can be read against it, never
        # re-prompted — see milestone.dependency_warnings
        _log(f"meta: milestone DAG warning — {w}")
    changed = _parse_list(patch_out, "changed_entries")
    patch_class = _parse_field(patch_out, "patch_class")
    if patch_class in ("structural", "split") and reopens_capped:
        # a non-R1 diagnosis whose patch reopens delivered decisions anyway: the
        # same budget — the landed patch is withdrawn before any student turn
        _write_milestone(ctx, at_entry)
        _log(f"meta: {patch_class} patch @ {locus or '?'} REFUSED — reopen budget spent "
             f"({reopen_streak} fruitless reopen(s) in a row); milestone restored, "
             f"nothing patched")
        _record(MetaRecord(
            meta_step=meta_step, cause_class=cause, deficient_locus=locus,
            patch_class="reopen-capped", changed_entries=changed, patch_direction=direction,
            outcome="REOPEN_CAPPED", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table,
            reopen_price=reopen_price, phase_verdict=phase_verdict))
        return MetaResult(diagnosis, "reopen-capped", [], None, [],
                          lever_prices=lever_table, phase_close=phase_close)
    if (patch_class in ("structural", "split") and _reopen_gate_on(ctx)
            and (why := _verify_reopen_price(reopen_price, probe_readings, known_levers, ctx,
                                             probes_run, best_ref=_priced_best(priced)))):
        # The diagnosis was not R1 (that gate ran above) but the patch reopens the
        # closure anyway: same rule. The landed patch is withdrawn — an unpriced
        # reopen never reaches the student — and the step is recorded as such.
        _write_milestone(ctx, at_entry)
        _log(f"meta: {patch_class} patch @ {locus or '?'} REFUSED — a reopen without a "
             f"measured price ({why}); milestone restored, nothing patched")
        _record(MetaRecord(
            meta_step=meta_step, cause_class=cause, deficient_locus=locus,
            patch_class="unpriced-reopen", changed_entries=changed, patch_direction=direction,
            outcome="UNPRICED_REOPEN", probe_rounds=probe_rounds,
            probe_readings=probe_readings, lever_prices=lever_table,
            reopen_price=reopen_price, phase_verdict=phase_verdict))
        return MetaResult(diagnosis, "unpriced-reopen", [], None, [],
                          lever_prices=lever_table, phase_close=phase_close)
    gate_contract = _parse_gate_contract(patch_out)
    if gate_contract:
        _log("meta: patch declares gate contract "
             + ", ".join(f"{g} ≤ {v * 100:+.0f}%" for g, v in gate_contract.items()))
    resume_after, invalidated = _compute_resume_plan(ctx, changed)
    reopened, kept = ((_compute_reopen_plan(ctx, changed))
                      if patch_class in ("structural", "split") else ([], []))
    if violations:
        _log(f"meta: WARNING patch still invalid after {ctx.max_patch_rounds} "
             f"round(s): {violations[:3]}")
    after = _read_milestone(ctx)
    _record(MetaRecord(
        meta_step=meta_step, cause_class=cause, deficient_locus=locus,
        patch_class=patch_class, changed_entries=changed, patch_direction=direction,
        milestone_before=milestone_before, resume_after=resume_after,
        best_before=best_before, outcome="PENDING", milestone_violations=violations,
        patch_landed=after != at_entry, milestone_delta=_line_delta(at_entry, after),
        probe_rounds=probe_rounds, probe_readings=probe_readings,
        lever_prices=lever_table, reopen_price=reopen_price,
        gate_contract=gate_contract, phase_verdict=phase_verdict))
    return MetaResult(diagnosis, patch_class, changed, resume_after, invalidated,
                      gate_contract=gate_contract,
                      certified=None, milestone_violations=violations,
                      reopened=reopened, kept=kept,
                      lever_prices=lever_table, phase_close=phase_close)


def _apply_phase_close(ctx: LoopContext, meta: "MetaResult") -> None:
    """Carry a meta step's phase verdict to the student's turns. A structural /
    split patch replaces the premise the verdict was read on (the phase's
    decisions are being redone), so it retires any standing notice instead."""
    if meta.patch_class in ("structural", "split"):
        if ctx.phase_close:
            _log(f"meta: phase-close notice for {ctx.phase_close['phase']} retired — "
                 f"a {meta.patch_class} patch reopens the phase's premise")
        ctx.phase_close = None
        return
    if meta.phase_close:
        ctx.phase_close = dict(meta.phase_close)
        _log(f"meta: phase-close notice for {meta.phase_close['phase']} rides the next "
             f"student turns until {meta.phase_close['gate']} is submitted")


# a milestone entry id the patch agent can claim to have touched; §-sections and prose
# labels ("Worked Example") are not checkable this way, so only these are.
_DECISION_ID = re.compile(r"^D\d+\.\d+$")


def _patch_landing_violations(before: str, after: str, declared: list[str]) -> list[str]:
    """Did the patch reach the file, and does the file contain what the reply claims?

    Nothing else writes the milestone: the driver never applies the `patch:` text from
    the agent's reply, so an agent that *reports* its patch instead of *making* it leaves
    the milestone untouched. Every downstream signal still looks healthy — the milestone
    is schema-clean (it is unchanged), the student correctly answers that nothing is left
    to do, and the loop charges a whole inner segment to a patch that never existed. This
    happened twice in one run (meta steps 3 and 4, ~4h) before anything noticed.
    """
    if after == before:
        return ["The patch never reached the file: the milestone file is byte-identical to "
                "before the patch. You must rewrite the milestone file itself with Edit/Write — the "
                "`patch:` text in your reply is never applied by anyone."]
    # An id in `before` but not `after` was deleted, which a structural patch may do.
    # An id in neither was invented for the report: the student can never act on it.
    return [f"`changed_entries` declares {e}, but the milestone file does not contain it "
            f"(neither a new nor an existing entry)"
            for e in (d.strip() for d in declared)
            if _DECISION_ID.match(e) and e not in after and e not in before]


def _line_delta(before: str, after: str) -> str:
    """"+38/-9 lines" — enough to audit a patch step later without storing both texts."""
    import difflib
    diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm=""))
    add = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    rem = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
    return f"+{add}/-{rem} lines"


def _validate_milestone_file(milestone, authoring: bool = False,
                             baseline: str | None = None) -> list[str]:
    """Violations in the milestone file — a missing or unreadable file is itself one.

    Absence is the pre-distillation state, not a conformant milestone. Reporting it
    as clean would tell _needs_authoring the milestone is already written, so the
    loop would start with nothing for the student to read; it would equally let a
    patch that deleted the file pass the post-patch gate.

    ``authoring`` selects the validator's strict mode (orchestration tags
    mandatory). Resume gates and patch validation stay in the default mode:
    a schema addition must not make every pre-existing milestone read as "not
    distilled yet" — that path re-authors v1 over the accumulated meta patches
    and re-measures baselines the resume contract promises to reuse — and a
    patch round must spend its budget on the diagnosed problem, not on
    retrofitting tags.
    """
    from forgekernel.milestone import validate_milestone
    p = Path(milestone)
    try:
        return validate_milestone(p.read_text(encoding="utf-8"),
                                  authoring=authoring, baseline=baseline)
    except OSError as e:
        return [f"milestone file unreadable ({p}): {type(e).__name__}"]


def _dependency_warnings_file(milestone: Path) -> list[str]:
    from forgekernel.milestone import dependency_warnings
    try:
        return dependency_warnings(Path(milestone).read_text(encoding="utf-8"))
    except OSError:
        return []


def _seed_target_ms(ctx: LoopContext) -> None:
    """If the authored milestone §0 lacks a machine-readable target_ms, seed it from
    the CLI --target-ms so the target SSOT lives in the milestone from the start and
    the meta agent has a concrete number to re-baseline.

    Seeded together with ``target_window:`` — the number alone cannot say which
    timing window it is anchored in, and _resolve_target only trusts a §0 target
    whose annotation matches the run's gate window."""
    from forgekernel.milestone import parse_target_ms
    try:
        p = Path(ctx.milestone)
        if not p.exists():
            return
        text = p.read_text(encoding="utf-8")
        if parse_target_ms(text) is not None:
            return  # meta already wrote one
        seeded = re.sub(r"(## 0\.[^\n]*\n)",
                        rf"\1target_ms: {ctx.target_ms}\ntarget_window: {ctx.gate_window}\n",
                        text, count=1)
        if seeded != text:
            p.write_text(seeded, encoding="utf-8")
            _log(f"meta/distill: seeded target_ms: {ctx.target_ms} "
                 f"(target_window: {ctx.gate_window}) into §0")
    except OSError:
        pass


def _resolve_target(ctx: LoopContext) -> float:
    """Stop threshold: the milestone's machine-readable target_ms (SSOT) if present
    AND anchored in this run's gate window, else the CLI-derived seed.

    Reading it from the milestone is what makes the meta agent's §0 target
    re-baseline actually move the bar (it was inert before). The window check is
    what keeps that SSOT honest: a teacher following the e2e-era prompts writes
    ``arch_ref_ms/0.95`` — an e2e number — into §0, and a device-gated loop
    comparing device readings against it gates ~one host tax loose, so the first
    mediocre KEEP stops the run as "target". An unannotated target counts as e2e
    (every pre-annotation milestone is e2e-anchored by construction); a missing or
    mismatched annotation on a device-gated run falls back to ctx.target_ms, which the
    CLI derived from the ref's device reading."""
    from forgekernel.milestone import parse_target_ms, parse_target_window
    try:
        p = Path(ctx.milestone)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            t = parse_target_ms(text)
            if t is not None:
                window = parse_target_window(text) or "e2e"
                if window == ctx.gate_window:
                    return t
                _log(f"milestone §0 target_ms {t} is anchored in the {window} "
                     f"window but this run gates on {ctx.gate_window} — using the "
                     f"CLI-derived {ctx.gate_window} threshold {ctx.target_ms} "
                     f"instead (re-baseline §0 with a matching target_window: line)")
    except OSError:
        pass
    return ctx.target_ms


def _persist_reanchor(ctx: LoopContext, record: dict) -> None:
    """Append the re-anchor account to <op_dir>/reanchor.jsonl — the audit trail.

    A run that stops as "target" off a re-anchor, or a diagnosis that reasoned
    from a re-anchored residual, must be auditable after the fact the way a
    finish confirmation is (best_confirmation.json): which samples, which
    cluster, which estimator, which verdict. Failure to write is logged, never
    raised — the account also travels in the run summary / prompt note."""
    try:
        path = Path(ctx.milestone).parent / "reanchor.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        _log(f"meta/re-anchor: could not persist the account ({e})")


def _write_infeasible_notice(ctx: LoopContext, *, meta_step: int, diagnosis: str,
                             reachability: str, best: float | None) -> Path | None:
    """Write <op_dir>/infeasible_stop.md — the human-facing record of an INFEASIBLE stop.

    The run ends on the teacher's word alone, and that word is the pipeline's
    own verdict — right or wrong, it is evidence about the system under test.
    The notice carries what a human needs to read it without opening the
    transcripts: where the run stands against its stop line, the teacher's own
    reachability line (an OPEN there contradicts the stop request and is worth
    reading first), and the diagnosis verbatim.

    What it must never do is invite the human to overrule the verdict and hand
    the teacher a new direction on this run: choosing the residual owner or the
    reopen is exactly the meta decision under test, so a human doing it turns
    the system into "pipeline + human meta" and voids the run's numbers. The
    notice names the legitimate routes instead (evidence / a harness mechanism
    that applies to every run, tested on a fresh run / a separately labelled
    human-guided run).
    Failure to write is logged, never raised — the stop itself must not depend
    on the disk.
    """
    eq = ctx.equivalence_ratio
    try:
        target = _resolve_target(ctx)
    except Exception:  # noqa: BLE001 — a notice must not fail on a malformed §0
        target = None
    lines = [
        "# Loop stopped: the teacher declared INFEASIBLE (the end of this run)",
        "",
        f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ctx.now()))}",
        f"- meta step: {meta_step}",
        f"- best: {best if best is not None else '?'} ms; target: {target or '?'} ms"
        f" (stop line = reference/student ≥ {eq:g})",
    ]
    memo = ctx.paired_residual_memo
    if memo and best is not None and memo[0] == best:
        lines.append(f"- Latest same-card paired re-measure: +{memo[1]:.1%} from the stop line")
    lines += [
        f"- The teacher's reachability line: {reachability or '(not given)'}",
        "",
        "## The teacher's diagnosis, verbatim",
        "",
        "```text",
        diagnosis.strip(),
        "```",
        "",
        "## What this stop is",
        "",
        "- This run finishes here; the result is recorded at the current best (the final "
        "re-measure is in best_confirmation.json). "
        "`stop=infeasible` is a ruling: the run is not restarted automatically.",
        "- This is the verdict of the pipeline under test itself; right or wrong, it is recorded "
        "as evidence about this run.",
        "- If you disagree with the verdict, **do not** change direction on this run and continue "
        "it: overruling the teacher's final ruling, picking the residual owner or reopening a "
        "direction is exactly the meta decision under test; once a human makes it, the system "
        "under test becomes \"pipeline + human meta\" and this run's numbers are void. "
        "Neither this notice, the diagnosis, nor your judgement goes into the milestone / oracle / "
        "KB, nor into the inputs of the next run.",
        "",
        "## Legitimate routes",
        "",
        "1. Record it as evidence of the pipeline's verdict (run records / the evidence trail "
        "of a change).",
        "2. If you judge it a harness defect, fix it as a mechanism that treats every run alike, "
        "and test it on a new run.",
        "3. If you only want to know whether the target is reachable in engineering terms: open a "
        "separate experiment clearly labelled \"human-guided\"; its results do not enter "
        "ForgeKernel's evaluation numbers.",
        "",
    ]
    try:
        path = Path(ctx.milestone).parent / "infeasible_stop.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
    except OSError as e:
        _log(f"meta: could not write the INFEASIBLE notice ({e})")
        return None


# How long a same-period target check stays good for: a target transported under
# one period's machine conditions says nothing about a period hours later.
_FALSE_WIN_MEMO_TTL_S = 4 * 3600.0


def _false_win_replay(ctx: LoopContext, best: float) -> None:
    """Re-thread the standing false-win note for the next meta entry.

    run_two_layer consumes ``ctx.reanchor_note`` once per meta entry, and
    ``_phantom_gap_reanchor`` returns on ``gap <= 0`` before its own memo replay
    — so after a false win the SECOND meta entry would diagnose against the
    stale target again. The memo is keyed by best, like the phantom-gap one.
    """
    memo = ctx.reanchor_memo
    if memo and memo[0] == best and not ctx.reanchor_note:
        ctx.reanchor_note = memo[1]


def _fresh_ref_target(ctx: LoopContext, iter_no: int, anchor: float,
                      target: float) -> "tuple | None":
    """This period's ref through ``ctx.reanchor_ref``, transported onto the
    target: (samples, cluster, spread, fresh_min, drift, fresh_target), or None
    when no single number represents the period (too few samples, samples that
    straddle node modes, a cluster the anchor was not drawn on)."""
    try:
        samples, fresh_cluster = ctx.reanchor_ref()
    except Exception as e:  # noqa: BLE001 — a starved queue must not kill the run
        _log(f"iter {iter_no}: target check — fresh ref measurement failed ({e})")
        return None
    samples = [s for s in samples if s and s > 0]
    if len(samples) < 2:
        _log(f"iter {iter_no}: target check — {len(samples)} usable ref sample(s), "
             f"need >= 2 (a lone draw must not certify a win)")
        return None
    spread = (max(samples) - min(samples)) / min(samples)
    if spread > 0.10:
        _log(f"iter {iter_no}: target check — ref spread {spread:.1%} > 10% "
             f"(samples straddle node modes; no single number represents this period)")
        return None
    if fresh_cluster and ctx.target_anchor_cluster and \
            fresh_cluster != ctx.target_anchor_cluster:
        _log(f"iter {iter_no}: target check — fresh ref ran on {fresh_cluster!r} but "
             f"the anchor is from {ctx.target_anchor_cluster!r}; absolute readings "
             f"do not transport across clusters")
        return None
    fresh_min = min(samples)
    drift = fresh_min / anchor          # min vs min: the archive keeps the fastest
    return samples, fresh_cluster, spread, fresh_min, drift, target * drift


def _confirm_apparent_target(ctx: LoopContext, best: float,
                             iter_no: int) -> bool | None:
    """An apparent target hit, verified same-period before it may end the run.

    The phantom-gap instrument (``_phantom_gap_reanchor``) only ever guarded the
    FALSE-GAP direction — it returns early on ``gap <= 0`` because "already met:
    the inner loop stops on its own". That left the FALSE-WIN direction
    unguarded, and on a bimodal cluster a false win is the more expensive error:
    a self-ref run stopped as `target` at iteration 0 having changed no
    executable code, because its launch anchor was drawn on a slower node than
    the student's job happened to land on. The anchor and the student reading
    were 5.1% apart on byte-identical code, against a 5% goal.

    So an absolute hit buys a measurement instead of a stop. Which measurement
    depends on what the run can pay in:

    * The in-job comparator IS the declared reference (``ref_rides_job`` /
      ``ref_in_job`` — the only runs on which ``_enforce_single_ref`` keeps the
      in-job readings): the student is re-measured 2 jobs and the verdict is the
      median SAME-JOB ratio against ``equivalence_ratio``. Ref and student ran on
      one card in one minute; no ref jobs are bought. (``ref_in_job`` runs do not
      reach here from run_inner_loop at all — their absolute hit defers to
      ``_paired_parity_stop`` — but a direct caller gets the same rule.)
    * Otherwise (blackbox: the ref is measured through its own tree): re-measure
      the ref this period, transport the target by the drift, re-measure the
      student, compare absolute ms. The two sides run as separate jobs, so this
      is the weakest currency the run has — it catches a launch anchor that
      drifted, not a student job that happens to see a faster period than the
      ref's. The record names the currency so a reader can weigh the verdict.

    Returns True to stop, False when the hit was a node draw (the run keeps
    climbing, the residual threaded into the next diagnosis), and None when it
    could not be checked at all — the caller stops on the absolute reading then,
    loudly, because refusing to stop on an un-checkable run would mean never
    stopping.

    A false win is memoised by the TARGET it transported, not only by the best
    that triggered it: every later KEEP that is still short of that fresh target
    (within _FALSE_WIN_MEMO_TTL_S) re-uses the verdict instead of re-buying jobs.
    """
    target = _resolve_target(ctx)
    if not target or target <= 0:
        return None
    paired = bool(ctx.ref_rides_job or ctx.ref_in_job)
    anchor = ctx.target_anchor_ms
    if not paired and (ctx.reanchor_ref is None or not anchor):
        return None
    memo = getattr(ctx, "_false_win_memo", None)
    if isinstance(memo, dict):
        if memo["best"] == best:
            _false_win_replay(ctx, best)
            _log(f"iter {iter_no}: best {best} unchanged since the last same-period "
                 f"target check said node-draw — not re-buying it; continuing")
            return False
        age = ctx.now() - memo["ts"]
        if best > memo["fresh_target"] and 0 <= age < _FALSE_WIN_MEMO_TTL_S:
            # A new best that is still short of the target this period's check
            # transported cannot have changed the answer: re-use it, otherwise
            # every KEEP under a 5%-stale target re-buys the same jobs.
            residual = (best - memo["fresh_target"]) / memo["fresh_target"]
            note = (
                f"\n## False win (carried over from the contemporaneous conversion "
                f"{age / 60:.0f} min ago)\n"
                f"The new best {best} ms looks like it meets the target under the launch-time "
                f"anchor, but it is still above the target converted from the contemporaneous "
                f"re-measure, {memo['fresh_target']:.4f} ms ({memo['currency']}). "
                f"**True residual = +{residual:.1%}**; the target is not met yet; no re-measure "
                f"was bought again.\n"
                f"Base the diagnosis and the prescription on this residual; do not write the "
                f"converted target back into §0.\n")
            ctx.reanchor_note = note
            ctx.reanchor_memo = (best, note)
            _log(f"iter {iter_no}: best {best} meets the launch-anchor target but is "
                 f"still +{residual:.1%} over the same-period target "
                 f"{memo['fresh_target']:.4f} ms transported {age / 60:.0f} min ago — "
                 f"not re-buying; continuing")
            return False
    _log(f"iter {iter_no}: best {best} meets target {target:.4f} on the launch "
         f"anchor — re-measuring "
         + ("the student same-card (the ref rides the job)"
            if paired else "ref AND student same-period")
         + " before stopping")
    fresh = None
    if not paired:
        fresh = _fresh_ref_target(ctx, iter_no, anchor, target)
        if fresh is None:
            return None
    try:
        conf = run_best_confirmation(ctx, jobs=2, max_extra=1)
    except Exception as e:  # noqa: BLE001
        conf = None
        _log(f"iter {iter_no}: target check — student re-measure failed ({e})")
    if not conf:
        return None
    student_fresh = conf["median_ms"]
    record = {"ts": ctx.now(), "iter": iter_no, "best_ms": best,
              "stale_target_ms": target, "anchor_ms": anchor,
              "student_fresh_ms": student_fresh,
              "student_jobs": [s.get("job") for s in conf.get("jobs", [])],
              "check": "apparent-target"}
    med_ratio = None
    if paired:
        med_ratio, why = _admit_paired(conf)
        if med_ratio is None:
            # The flag promised an in-job reference the jobs did not deliver (a
            # misjudged ref_rides_job, a bench without the comparator): fall back
            # to the fresh-ref channel rather than certify on nothing.
            _log(f"iter {iter_no}: target check — no admissible same-job ratio ({why}); "
                 f"falling back to a fresh ref measurement")
            if ctx.reanchor_ref is None or not anchor:
                return None
            fresh = _fresh_ref_target(ctx, iter_no, anchor, target)
            if fresh is None:
                return None
    if med_ratio is not None:
        eq = ctx.equivalence_ratio
        confirmed = med_ratio >= eq
        # this period's parity line in the student's own ms: ratio = ref / student,
        # so parity sits at student * ratio / eq
        fresh_target = student_fresh * med_ratio / eq
        residual = eq / med_ratio - 1.0
        record.update({"currency": "same-job", "median_ratio": round(med_ratio, 4),
                       "equivalence_ratio": eq, "fresh_target_ms": fresh_target})
        how = (f"same-job median ratio {med_ratio:.4f} over "
               f"{len(conf.get('jobs', []))} job(s) vs equivalence {eq}")
    else:
        samples, fresh_cluster, spread, fresh_min, drift, fresh_target = fresh
        confirmed = student_fresh <= fresh_target
        residual = (student_fresh - fresh_target) / fresh_target
        record.update({"currency": "cross-job-absolute", "samples_ms": samples,
                       "fresh_cluster": fresh_cluster, "estimator": "min-vs-min",
                       "drift": round(drift, 4), "fresh_target_ms": fresh_target})
        how = (f"student {student_fresh} ms vs {fresh_target:.4f} ms (ref this period "
               f"{fresh_min} ms over {len(samples)} job(s), drift x{drift:.4f}; each side "
               f"on its own node — the weakest currency this run has)")
    if confirmed:
        record["verdict"] = "target"
        _persist_reanchor(ctx, record)
        _log(f"iter {iter_no}: target CONFIRMED same-period — {how}")
        return True
    record["verdict"] = "false_win"
    ctx._false_win_memo = {"best": best, "fresh_target": fresh_target,
                           "ts": ctx.now(), "currency": record["currency"]}
    if med_ratio is not None:
        note = (
            f"\n## False win intercepted (same-card re-measure)\n"
            f"The launch-time anchor made best {best} ms look like it meets the target, but the "
            f"same-job re-measure ({len(conf.get('jobs', []))} job(s), "
            f"reference and student run back to back on the same card) has a median ratio of "
            f"{med_ratio:.4f} < the equivalence ratio {eq}. "
            f"**True residual = +{residual:.1%}** (by the same-job ratio); the target is not met "
            f"yet.\n"
            f"Base the diagnosis and the prescription on this residual; do not change the "
            f"absolute-millisecond target.\n")
    else:
        note = (
            f"\n## False win intercepted (contemporaneous re-measure)\n"
            f"The launch-time anchor made best {best} ms look like it meets the target, but a "
            f"contemporaneous re-measure shows that was a node draw: "
            f"the ref's fastest this period is {fresh_min} ms ({len(samples)} runs, spread "
            f"{spread:.1%}; "
            f"launch-time anchor {anchor} ms, drift x{drift:.4f}) → converted target "
            f"{fresh_target:.4f} ms; "
            f"student same-period re-measure {student_fresh} ms. **True residual = "
            f"+{residual:.1%}**; the target is not met yet.\n"
            f"Base the diagnosis and the prescription on this residual; do not write the target "
            f"converted here back into §0 (it contains the drift factor; "
            f"writing it back would count the drift twice).\n")
    ctx.reanchor_note = note
    ctx.reanchor_memo = (best, note)
    _persist_reanchor(ctx, record)
    _log(f"iter {iter_no}: target NOT confirmed — {how} (residual +{residual:.1%}); "
         f"the absolute hit was a node draw, continuing")
    return False


def _phantom_gap_reanchor(ctx: LoopContext, best: float | None) -> dict | None:
    """Re-anchor the ref before a gap-driven diagnosis whose residual sits inside
    the node-noise band — the one place a stale anchor changes DECISIONS.

    In-loop readings compare fresh student jobs against a target derived from a
    ref measured once at launch, hours and many node draws ago. For the loop's
    day-to-day choices that staleness is harmless: best-tracking is
    student-vs-student, and the error direction on the stop check is conservative
    (a fast-node launch anchor makes the loop under-claim, never over-claim). But
    the meta layer reasons from the ABSOLUTE residual: it prices loci against it,
    frames the reachability disproof with it, and sizes prescriptions by it — and
    a residual inside the bimodality band can be entirely the anchor. Live case:
    a bwd-GQA run's meta read "+3.61%" at the M3 gate and opened a second
    structural rebuild (15 iters, ~20h, erased); the finish confirmation — the
    run's only same-period measurement — put the very sha it started from at
    1.0271x. The gap never existed.

    The transport is estimator- and provenance-matched, and two-sided before it
    may stop the run:

    * drift = min(fresh samples) / target_anchor_ms — min against min: the
      launch archive keeps the FASTEST of its samples (baseline_cli), so a
      median on the fresh side would read the median-vs-min offset (~2% on
      quiet nodes) as drift and hand out stop=target for free;
    * the fresh samples must pass admission (>= 2 usable, spread <= 10% — a
      lost job leaving one slow sample, or samples straddling the node
      modes, must not rewrite the bar the run stops on) and come from the SAME
      machine as the launch anchor (absolute readings do not transport across
      machines at all);
    * transport is refused entirely when the target is not the CLI seed paired
      with the anchor — a §0 re-baseline already reflects the conditions of ITS
      own measurement, so multiplying it by drift-since-launch counts the drift
      twice — and when the target was never derived from a measured baseline
      (--target-ms, toml constants: target_anchor_ms is None);
    * a transported target that the standing best appears to meet is still only
      half a claim — the best is itself an hours-old draw — so the student is
      re-measured same-period (run_best_confirmation, 2 jobs) before the run may
      stop, independent of whether the finish confirmation is enabled
      (confirm_jobs=0 runs included).

    Returns the account dict when the run should stop as "target". On a
    CONFIRMED gap it stashes ctx.reanchor_note — run_two_layer threads it
    through every diagnosis of this meta entry, malformed-patch retries included
    — plus a memo so an unchanged best does not re-buy ref jobs at every later
    plateau. Every measured check is persisted to reanchor.jsonl. Any
    measurement failure degrades to the old behavior, loudly.
    """
    band = ctx.reanchor_band
    if best is None or band <= 0 or ctx.reanchor_ref is None:
        return None
    cluster = _active_cluster(ctx)
    if cluster and ctx.home_cluster and cluster != ctx.home_cluster:
        # Another machine: the best is this machine's, the target is restated
        # from a reference measured on it (_cluster_scale), and the launch
        # anchor is the first machine's — there is no launch-period drift to
        # transport.
        _log(f"meta/re-anchor: run is on cluster {cluster!r} (home "
             f"{ctx.home_cluster!r}) — the phantom-gap check is a home-cluster "
             f"instrument; skipping")
        return None
    target = _resolve_target(ctx)
    if not target or target <= 0:
        return None
    if wl_why := _target_workloads_mismatch(ctx):
        _say_target_suspended(ctx, "meta/re-anchor", wl_why)
        if ctx.ref_in_job:
            # the absolute gap cannot say whether a residual exists; the
            # same-card ratio can, and it is the currency this run stops on
            return _paired_reanchor(ctx, best, None)
        return None
    gap = (best - target) / target
    if gap <= 0:
        return None  # already met: the inner loop stops on its own
    if ctx.ref_in_job:
        # The reference rides the student's own jobs: there is no cross-channel
        # drift to transport, only a same-card ratio to re-measure. The absolute
        # band does not gate this — the absolute gap is exactly the cross-node
        # number this path exists to stop reasoning from.
        return _paired_reanchor(ctx, best, gap)
    # beyond the band, drift cannot explain the residual and the diagnosis is
    # legitimately gap-driven.
    if gap > band:
        return None
    if target != ctx.target_ms:
        _log(f"meta/re-anchor: residual +{gap:.1%} is inside the {band:.0%} band "
             f"but §0 target {target} is not the launch-derived seed "
             f"{ctx.target_ms} — a re-baselined target already reflects its own "
             f"measurement period, and transporting launch drift onto it would "
             f"count the drift twice. Skipping (residual is suspect).")
        return None
    anchor = ctx.target_anchor_ms
    if not anchor:
        _log(f"meta/re-anchor: residual +{gap:.1%} is inside the {band:.0%} band "
             f"but the target was not derived from a measured baseline "
             f"(--target-ms / toml constant) — no launch-period anchor to "
             f"transport drift from. Skipping (residual is suspect).")
        return None
    if ctx.reanchor_memo and ctx.reanchor_memo[0] == best:
        # nothing moved since the last confirmed-gap check; the answer cannot
        # have changed except by further drift — re-issue the note, spend nothing.
        ctx.reanchor_note = ctx.reanchor_memo[1]
        _log("meta/re-anchor: best unchanged since the last confirmed-gap check "
             "— reusing its verdict (no new ref jobs)")
        return None
    try:
        samples, fresh_cluster = ctx.reanchor_ref()
    except Exception as e:  # a starved job must not kill the run; degrade loudly
        _log(f"meta/re-anchor: fresh ref measurement failed ({e}) — diagnosing "
             f"against the stale anchor (residual is suspect)")
        return None
    samples = [s for s in samples if s and s > 0]
    if len(samples) < 2:
        _log(f"meta/re-anchor: admission failed — {len(samples)} usable "
             f"sample(s), need >= 2 (a lone draw must not rewrite the bar the "
             f"run stops on); diagnosing against the stale anchor")
        return None
    spread = (max(samples) - min(samples)) / min(samples)
    if spread > 0.10:
        _log(f"meta/re-anchor: admission failed — spread {spread:.1%} > 10% "
             f"(samples straddle node modes; no single number represents this "
             f"period); diagnosing against the stale anchor")
        return None
    if fresh_cluster and ctx.target_anchor_cluster and \
            fresh_cluster != ctx.target_anchor_cluster:
        _log(f"meta/re-anchor: fresh samples ran on cluster {fresh_cluster!r} "
             f"but the launch anchor is from {ctx.target_anchor_cluster!r} "
             f"— absolute readings do not transport across machines; "
             f"diagnosing against the stale anchor")
        return None
    fresh_min = min(samples)
    drift = fresh_min / anchor  # min vs min: the launch archive keeps the fastest
    fresh_target = target * drift
    _log(f"meta/re-anchor: residual +{gap:.1%} <= band {band:.0%} — fresh ref "
         f"min {fresh_min} ms over {len(samples)} job(s) (spread {spread:.1%}; "
         f"launch anchor {anchor} ms, drift x{drift:.4f}) -> target "
         f"{fresh_target:.4f} ms (was {target})")
    record = {"ts": ctx.now(), "best_ms": best, "stale_target_ms": target,
              "gap_vs_stale": round(gap, 4), "anchor_ms": anchor,
              "anchor_cluster": ctx.target_anchor_cluster,
              "samples_ms": samples, "fresh_cluster": fresh_cluster,
              "estimator": "min-vs-min", "drift": round(drift, 4),
              "fresh_target_ms": fresh_target}

    def _confirmed(residual: float, how: str) -> None:
        note = (
            f"\n## Contemporaneous ref re-anchor (read before judging — the residual has been "
            f"recomputed against the new anchor)\n"
            f"The launch-time anchor is stale (nodes are bimodal, ~±6%); the ref was just "
            f"measured contemporaneously {len(samples)} times: "
            f"fastest {fresh_min} ms (spread {spread:.1%}; launch-time anchor {anchor} ms, both "
            f"read as the fastest value) → drift-converted target {fresh_target:.4f} ms. {how}has "
            f"a true residual against the new anchor of **+{residual:.1%}** (on paper against the "
            f"old anchor: +{gap:.1%}).\n"
            f"The diagnosis and the prescription must go by the recomputed residual: the "
            f"on-paper prices of the defect loci you name must add up to explain it; do not "
            f"propose a structural reopen (R1) whose expected gain is below it.\n"
            f"**You must not** write the target converted here back into §0 — it contains the "
            f"drift factor since launch, and once written back the next period would multiply "
            f"the drift in again (counting it twice). A re-baseline may only be based on raw ref "
            f"readings you measured contemporaneously yourself.\n")
        ctx.reanchor_note = note
        ctx.reanchor_memo = (best, note)
        _persist_reanchor(ctx, record)

    if best > fresh_target:
        record["verdict"] = "confirmed_gap"
        _confirmed((best - fresh_target) / fresh_target, f"best {best} ")
        return None
    # The stale-anchor account says the target is already met — but `best` is
    # itself an hours-old node draw. Measure the student in the SAME period
    # before ending the run (the finish confirmation may be disabled).
    _log(f"meta/re-anchor: best {best} meets the re-anchored target on paper — "
         f"re-measuring the student same-period before stopping")
    try:
        conf = run_best_confirmation(ctx, jobs=2, max_extra=1)
    except Exception as e:
        conf = None
        _log(f"meta/re-anchor: student re-measure failed ({e})")
    if not conf:
        record["verdict"] = "student_remeasure_unavailable"
        _log("meta/re-anchor: no same-period student reading — NOT stopping on "
             "half a claim; diagnosing with the re-anchored residual instead")
        _confirmed(0.0, f"best {best} (an old reading; no same-period student re-measure "
                        f"was available, so the residual is doubtful) ")
        return None
    student_fresh = conf["median_ms"]
    record["student_fresh_ms"] = student_fresh
    record["student_jobs"] = [s.get("job") for s in conf.get("jobs", [])]
    if student_fresh <= fresh_target:
        record["verdict"] = "target"
        _persist_reanchor(ctx, record)
        _log(f"meta/re-anchor: student same-period {student_fresh} ms <= "
             f"re-anchored target {fresh_target:.4f} — the residual was the "
             f"stale anchor, not the kernel; stopping as target")
        return record
    # the OLD best reading was the lucky draw; the honest residual uses the fresh one
    record["verdict"] = "confirmed_gap_after_student_remeasure"
    _confirmed((student_fresh - fresh_target) / fresh_target,
               f"student same-period re-measure {student_fresh} ms (the old best reading "
               f"{best} was a fast-node draw) ")
    return None


def _paired_reanchor(ctx: LoopContext, best: float, gap: float | None) -> dict | None:
    """The ref_in_job form of the phantom-gap check: re-measure the best sha
    (2 jobs) and read the same-job ratio.

    Median ratio at or above the equivalence ratio: the residual was the
    cross-node anchor, not the kernel — stop as target. Below it: the honest
    residual for the diagnosis is the paired one, measured to the bar the run
    stops on (eq/ratio - 1, as the apparent-target path reads it), threaded into
    the meta entry the same way the absolute re-anchor's note is; memoised per
    best so an unchanged best re-buys nothing.

    The residual used to be measured to parity (1/ratio - 1) whatever eq was: an
    run with a 1.05x bar at ratio 0.985 was told its residual was +1.5% — inside
    the same-job floor, "prefer parity, no reopen" — when the run still stood
    +6.6% short of its stop line. At eq = 1.0 the two forms coincide.

    ``gap`` None: the absolute target is read over other workloads than the
    gate reading (_target_workloads_mismatch), so there is no absolute residual
    to quote — the same-card ratio is the whole answer.
    """
    eq = ctx.equivalence_ratio
    sha = _best_good_ref_here(load_records(ctx.status_path), ctx)
    if ctx.reanchor_memo and ctx.reanchor_memo[0] == best:
        ctx.reanchor_note = ctx.reanchor_memo[1]
        _log("meta/re-anchor: best unchanged since the last same-job check — "
             "reusing its verdict (no new jobs)")
        return None
    prior = ctx.paired_memo.get(sha) if sha else None
    if prior and prior.get("cluster", "") == _active_cluster(ctx) \
            and prior.get("median_ratio") is not None:
        # the inner loop already re-measured this very sha (parity on paper,
        # unconfirmed) on this cluster: those two jobs answer this question
        med, conf, reused = prior["median_ratio"], None, True
        _log(f"meta/re-anchor: best {sha[:10]} was re-measured same-card at iter "
             f"{prior.get('iter')} (median ratio {med:.4f}) — reusing, no new jobs")
    else:
        reused = False
        what = (f"absolute residual +{gap:.1%} is a cross-node number" if gap is not None
                else "the absolute target covers other workloads than the gate reading")
        _log(f"meta/re-anchor: {what} (ref_in_job) — re-measuring the best "
             f"{(sha or '?')[:10]} same-card (2 jobs)")
        try:
            conf = run_best_confirmation(ctx, jobs=2, max_extra=1, ref=sha)
        except Exception as e:  # noqa: BLE001
            conf = None
            _log(f"meta/re-anchor: same-card re-measure failed ({e}) — diagnosing "
                 f"against the absolute residual (suspect)")
        med, why = _admit_paired(conf)
    record = {"ts": ctx.now(), "best_ms": best, "best_sha": sha,
              "gap_vs_absolute_target": round(gap, 4) if gap is not None else None,
              "estimator": "paired-in-job",
              "equivalence_ratio": eq, "median_ratio": med,
              "cluster": _active_cluster(ctx),
              "student_jobs": (prior["jobs"] if reused
                               else [s.get("job") for s in (conf or {}).get("jobs", [])])}
    if reused:
        record["reused_from_iter"] = prior.get("iter")
    if med is None:
        record["verdict"] = "student_remeasure_unavailable"
        record["why"] = why
        _persist_reanchor(ctx, record)
        _log(f"meta/re-anchor: same-card re-measure inadmissible ({why}) — "
             f"diagnosing against the absolute residual (suspect)")
        return None
    if med >= eq:
        record["verdict"] = "target"
        if not reused:
            ctx.paired_stop = {"sha": sha, "samples": list(conf.get("jobs", [])),
                               "median_ratio": med, "jobs": record["student_jobs"]}
        _persist_reanchor(ctx, record)
        _log(f"meta/re-anchor: same-job median ratio {med:.4f} >= {eq} over "
             f"{len(record['student_jobs'])} job(s) — the residual was the "
             f"cross-node anchor, not the kernel; stopping as target")
        return record
    residual = eq / med - 1.0          # to the stop line, not to parity
    parity = 1.0 / med - 1.0
    above_parity = eq > 1.0 + 1e-9
    record["verdict"] = "confirmed_gap"
    record["paired_residual"] = round(residual, 4)
    if above_parity:
        record["parity_residual"] = round(parity, 4)
    note = (
        f"\n## Same-card paired re-measure (read before judging — the residual has been "
        f"recomputed from the same-job ratio)\n"
        f"This run's reference runs back to back inside the student's job (ref_in_job); the "
        f"absolute milliseconds compare two different sets of nodes"
        + (f" (the old on-paper residual +{gap:.1%} is a cross-node number)" if gap is not None else
           " (the absolute target and the gate reading cover different workloads, so the "
           "on-paper residual cannot be read)")
        + f". The best was just re-measured on the same card "
        f"{len(record['student_jobs'])} times: median same-job ratio **{med:.4f}** "
        f"(reference/student; ≥ {eq:g} reaches this run's stop line) → true residual to the "
        f"stop line = **+{residual:.1%}**.\n"
        f"The diagnosis and the prescription must go by this residual: the on-paper prices of "
        f"the defect loci you name must add up to explain it; "
        f"do not propose a structural reopen (R1) whose expected gain is below it. ")
    if above_parity:
        note += (
            f"This run's target is above parity: a ratio of 1.0 only ties the reference, the "
            f"stop line is {eq:g}× — getting close to or reaching parity does not mean the work "
            f"is done; do not call it parity on the strength of the noise floor.\n")
    else:
        note += ("When the residual falls inside the same-job noise floor (1–2%), prefer calling "
                 "it parity over continuing to hunt for a structural defect.\n")
    ctx.reanchor_note = note
    ctx.reanchor_memo = (best, note)
    ctx.paired_residual_memo = (best, residual)
    _persist_reanchor(ctx, record)
    _log(f"meta/re-anchor: same-job median ratio {med:.4f} < {eq} — paired "
         f"residual +{residual:.1%} to the {eq:g} line"
         + (f" (parity {parity:+.1%})" if above_parity else "")
         + f" (absolute account said +{gap:.1%})")
    return None


def _milestone_dag(ctx: LoopContext) -> list[Decision]:
    """Decision DAG for prefix reuse: an explicit ctx.decisions_dag wins, else parse
    the milestone §4 dependencies (the SSOT once cli stopped passing a fixed empty dag)."""
    if ctx.decisions_dag:
        return ctx.decisions_dag
    from forgekernel.milestone import parse_milestone_dag
    try:
        p = Path(ctx.milestone)
        if p.exists():
            return parse_milestone_dag(p.read_text(encoding="utf-8"))
    except OSError:
        pass
    return []


def _compute_resume_plan(ctx: LoopContext, changed: list[str]) -> tuple[str | None, list[str]]:
    dag = _milestone_dag(ctx)
    if dag and changed:
        plan = compute_resume(dag, changed)
        return plan.resume_after, plan.invalidated
    return None, []


def _compute_reopen_plan(ctx: LoopContext, changed: list[str]) -> tuple[list[str], list[str]]:
    """(reopen, kept): the dependency closure of the changed entries over the
    milestone DAG, and the decisions outside it. Empty when there is no DAG or
    nothing changed."""
    dag = _milestone_dag(ctx)
    if dag and changed:
        plan = compute_reopen(dag, changed)
        return plan.reopen, plan.kept
    return [], []


def _ref_for_decision(records, decision_id: str) -> str | None:
    """The commit sha of the last landed iteration implementing this decision.

    Delivery granularity is the student's choice (template A.6), so a decision may
    have landed inside a bundled commit whose id joins its members with `+`
    (D2.3+D2.4). An exact match would miss it, and the caller's fallback is a full
    restart to the baseline ref — silently discarding every kept iteration because
    the checkpoint was filed under the bundle's name. Membership is the match:
    that sha contains the decision.
    """
    for r in reversed(records):
        if r.sha and decision_id in (p.strip() for p in r.decision_id.split("+")):
            return r.sha
    return None


# inner-loop stop reasons that hand control to the meta loop instead of ending the run.
_META_ACTIONABLE = ("plateau", "stuck", "exhausted", "gate_regression")

# Same-stage gate check for a post-patch segment. A rebuild's M0-gate reading
# is measured before M1-M3 are redone, so comparing it with the checkpoint
# best is predetermined; comparing it with the PREVIOUS segment's SAME gate is
# not — that is the reading the driver had on record and never consulted
# (observed on an ixbwd run: M0-gate 14.43 vs the prior segment's 12.31, −17%, four
# more iterations spent walking M1 before the plateau judged it). The
# tolerance is the harness's own re-measurement noise; a teacher who expects
# a gate to regress on purpose (an exchange-face rewrite that pays at M1)
# declares it per gate in the patch output (`gate_contract:`), and the check
# holds that gate to the declared band instead — and kills at the first gate
# the declaration misses.
_GATE_ID = re.compile(r"^M\d+-gate$")
_GATE_REGRESSION_TOLERANCE = 0.015
_GATE_CONTRACT_RE = re.compile(r"(M\d+-gate)\s*[≤<=]+\s*([+-]?\d+(?:\.\d+)?)\s*%")


def _parse_gate_contract(text: str) -> dict[str, float]:
    """``gate_contract: M0-gate ≤ +15%; M1-gate ≤ 0%`` -> {"M0-gate": 0.15, ...}.

    Values are the allowed regression of that gate against the previous
    segment's same gate (a fraction; negative = must improve by that much)."""
    m = re.search(r"gate_contract\s*[:：]\s*(.+)", text)
    if not m:
        return {}
    return {gate: float(val) / 100.0 for gate, val in _GATE_CONTRACT_RE.findall(m.group(1))}


def _previous_gate_record(records, gate_id: str, before: int, gate_window: str,
                          voided: "list[tuple[int, int]] | None" = None):
    """The last measured KEEP row of ``gate_id`` in the pre-segment history that
    still describes a standing tree — rows inside a reverted segment
    (``voided``: [lo, hi) index ranges) are skipped, so the reading comes from
    the lineage the current tree was actually reset to."""
    voided = voided or []
    for i in range(min(before, len(records)) - 1, -1, -1):
        if any(lo <= i < hi for lo, hi in voided):
            continue
        r = records[i]
        if (r.decision_id == gate_id and r.verdict == "KEEP"
                and gate_ms(r, gate_window) is not None):
            return r
    return None


# Same-stage tolerance on absolute ms across two jobs. The same-job ratio
# (_GATE_REGRESSION_TOLERANCE above) is one card, one minute; an absolute
# reading carries the node population on top — one unchanged commit moves ~6%
# between jobs (IterationRecord.vs_ref) — so a cross-job absolute comparison is
# only read same-cluster and only outside that band.
_GATE_REGRESSION_TOLERANCE_MS = 0.06


def _gate_regression(records, rec, before: int, contract: dict, gate_window: str,
                     voided: "list[tuple[int, int]] | None" = None) -> str | None:
    """A message when ``rec`` (a gate KEEP) regressed past its band, else None.

    Read on the same-job ref/student ratio when both gate rows carry one
    (node-insensitive); else on absolute ms, same-cluster rows only, inside the
    cross-job band — two clusters' absolute readings are not a comparison."""
    gate = rec.decision_id or ""
    if rec.verdict != "KEEP" or not _GATE_ID.match(gate):
        return None
    prev = _previous_gate_record(records, gate, before, gate_window, voided)
    if prev is None:
        return None
    allowed = float(contract.get(gate, 0.0))
    band = (f"declared band {allowed * 100:+.0f}%" if gate in contract
            else "same-stage parity")
    cur_r, prev_r = _paired_ratio(rec, gate_window), _paired_ratio(prev, gate_window)
    if cur_r is not None and prev_r is not None and prev_r > 0:
        # > 1 = student faster: a regression is the ratio dropping past the band
        if cur_r >= prev_r * (1.0 - allowed - _GATE_REGRESSION_TOLERANCE):
            return None
        return (f"{gate} same-job ratio {cur_r:.4f} vs previous segment's {gate} "
                f"{prev_r:.4f} ({(cur_r / prev_r - 1) * 100:+.1f}%; {band} + "
                f"{_GATE_REGRESSION_TOLERANCE * 100:.1f}% noise) — the rebuild is behind "
                f"at the same stage; judging the patch now")
    if rec.cluster and prev.cluster and rec.cluster != prev.cluster:
        return None
    cur, prev_ms = gate_ms(rec, gate_window), gate_ms(prev, gate_window)
    if cur is None or prev_ms is None or prev_ms <= 0:
        return None
    if cur <= prev_ms * (1.0 + allowed + _GATE_REGRESSION_TOLERANCE_MS):
        return None
    return (f"{gate} {cur:.4f} ms vs previous segment's {gate} {prev_ms:.4f} ms "
            f"({(cur / prev_ms - 1) * 100:+.1f}%; {band} + "
            f"{_GATE_REGRESSION_TOLERANCE_MS * 100:.0f}% cross-job noise) — the rebuild is "
            f"behind at the same stage; judging the patch now")


def _resolve_resume(ctx: LoopContext, meta: "MetaResult", records, baseline_ref: str) -> str:
    """Map a meta patch to the git ref the workspace resumes from (replaces reset("HEAD")).

    - additive / kb-example: nothing invalidated -> keep all student state (current HEAD);
    - structural / split with a reusable prefix: reset to that decision's checkpoint sha;
    - foundation patch (earliest affected decision is the first -> resume_after None) or a
      resume point with no recorded sha: full restart to the loop's baseline ref.
    """
    if meta.patch_class in ("additive", "kb-example", "unpriced-reopen", "reopen-capped",
                            "infeasible-refused"):
        return ctx.backend.head()
    if meta.resume_after is None:
        return baseline_ref
    return _ref_for_decision(records, meta.resume_after) or baseline_ref


def _resolve_resume_in_place(ctx: LoopContext, meta: "MetaResult", records,
                             baseline_ref: str) -> "tuple[str, list[str]]":
    """(resume ref, reopened decisions): in-place reopening when it applies,
    the prefix-reuse reset otherwise.

    In place = the best green KEEP tree stays; the student redoes the
    dependency closure of the changed entries on it (announced as a REOPEN
    row). Applies to structural / split patches whose closure is non-empty,
    when a green best tree exists to reopen on and the run has not opted out.
    Pre-green (no measured KEEP) there is nothing to keep, so the old reset
    stands; additive / kb-example patches invalidate nothing either way.
    """
    if (not ctx.reopen_in_place or meta.patch_class not in ("structural", "split")
            or not meta.reopened):
        return _resolve_resume(ctx, meta, records, baseline_ref), []
    best = _best_good_ref(records, ctx.gate_window)
    if best is None:
        return _resolve_resume(ctx, meta, records, baseline_ref), []
    return best, list(meta.reopened)


def _reopen_audit_gate(reopened: list) -> str | None:
    """The gate whose next KEEP closes a reopened foundation: ``M0-gate`` when
    the closure holds any M0 decision (the segment enters green, so the
    green-flip audit trigger cannot fire), else None — the audit reads the
    foundation only, and a reopen above M0 leaves it as built."""
    return "M0-gate" if any(re.match(r"D0\.", d or "") for d in reopened) else None


def _note_reopen(ctx: LoopContext, meta: "MetaResult", ref: str) -> None:
    """Announce an in-place reopen: one NEUTRAL status row the student reads
    (its earlier KEEP rows for the closure are void from here on) and the
    driver-context note for the segment's student turns."""
    reopened = ", ".join(meta.reopened)
    kept = ", ".join(meta.kept) or "(none)"
    changed = ", ".join(meta.changed_entries) or "?"
    change = (f"REOPEN on {ref[:10]}: milestone patch changed {changed}; the dependency "
              f"closure [{reopened}] is reopened — earlier KEEP rows for these decisions "
              f"are void, redo them on this tree per the patched milestone. Kept as built "
              f"(do not rebuild): [{kept}].")
    records = load_records(ctx.status_path)
    append_record(ctx.status_path, IterationRecord(
        iter=len(records), decision_id="REOPEN", change=change,
        correctness=None, verdict="NEUTRAL", sha=""))
    ctx.reopen_note = (
        f"- **Reopened in place (REOPEN)**: the previous meta step's patch changed `{changed}`. "
        f"The workspace was **not** rolled back; it is still the best KEEP tree so far "
        f"`{ref[:10]}`. The newest `decision_id: REOPEN` row in status lists the reopened "
        f"decisions [{reopened}]: their earlier KEEP records are **void**; **redo** them on the "
        f"current tree per the patched milestone (commit under each one's own decision_id as "
        f"usual; bundling is allowed; gate rows are still committed separately). "
        f"The decisions not listed [{kept}] **remain valid; do not rebuild them**. Do not touch "
        f"code outside the reopened set.\n")
    _log(f"meta: reopen in place on {ref[:10]} — redo [{reopened}], keep [{kept}]")
def _set_resume_note(ctx: LoopContext, meta: "MetaResult", resume_ref: str,
                     baseline_ref: str) -> None:
    """Tell the student its workspace was rolled back, and to what.

    Without this the student's only delivery ledger is the trajectory, whose old
    records survive the reset — so after a foundation patch it reads every decision
    as already delivered and reports EXHAUSTED into an empty segment. The note is
    the reset's visible half: which checkpoint the code is at, which decisions the
    patch invalidated, and that old records for those rows describe a rolled-back
    implementation, not a delivery.

    The note describes the WORKSPACE, not this patch: a patch class that resets
    nothing leaves the standing note alone (clearing it would restore the failure
    mode), and only a reset onto a tree where everything is delivered again — the
    two REVERT paths — clears it.
    """
    if meta.patch_class in ("additive", "kb-example"):
        # These do NOT reset the workspace (_resolve_resume returns head()), so an
        # earlier rollback is still in force and its note still describes the tree.
        # Clearing here would reinstate the very failure mode the note exists to fix:
        # a structural rollback the student exhausted out of without re-delivering,
        # then an additive step, and the survived old records read as deliveries
        # again. The note describes the workspace, not this patch — leave it.
        return
    if resume_ref == baseline_ref:
        # Both a foundation patch (resume_after None, whether or not the DAG parsed)
        # and a resume point whose sha is not on record land here: _resolve_resume
        # falls back to baseline for either. The tree is the stub, so naming a
        # checkpoint it is not at — with an invalidation list that omits everything
        # before that checkpoint — would tell the student D0.1..D0.13 are still
        # delivered and send it to write D0.14 on top of a stub.
        ctx.resume_note = (
            "- **The outer meta loop rolled the workspace back to the baseline (stub kernel)** "
            "(the milestone was just patched). **Every decision** of the milestone must be "
            "delivered again per the patched milestone; the old records in the trajectory "
            "belong to the rolled-back implementation and are not evidence of delivery — "
            "do not report EXHAUSTED on their strength. Start from the earliest decision.\n")
        return
    if not meta.invalidated:
        # compute_resume matched no decision: ResumePlan(tip, []) means nothing was
        # invalidated and the reset is a no-op (a patch touching only prose entries
        # gets this). Manufacturing a rollback here would spend the segment
        # re-delivering rows that are already delivered; leave any standing note,
        # which still describes the tree, alone.
        return
    inv = ", ".join(meta.invalidated)
    ctx.resume_note = (
        f"- **The outer meta loop rolled the workspace back to the checkpoint of decision "
        f"{meta.resume_after}** (the milestone was just patched). "
        f"Rolled-back / invalidated decisions: {inv} — these must be **delivered again** per "
        f"the patched milestone. Their old records in the trajectory belong to the rolled-back "
        f"implementation and are not evidence of delivery; do not report EXHAUSTED on their "
        f"strength. Start from the earliest undelivered decision on the invalidated list.\n")


# a milestone patch counts as helpful only if the post-patch inner segment set a new
# best beyond this relative margin; otherwise it is rolled back (transactional).
# Aligned with detect_plateau's noise_floor: the harness's re-measurement span is
# ~1.37% (four samples of one kernel), and a margin below it lets one
# lucky draw certify a useless patch — KEEP_PATCH has no later rollback, so the
# milestone edit would be permanent. Sub-noise gains are unmeasurable and therefore
# unattributable; a patch whose true gain hides under the noise cannot be credited.
_PATCH_IMPROVE_EPS = 0.015

# Runs (by status path) already told their gate covers fewer than PERF_WORKLOADS
# workloads — the warning is said once per run, not on every KEEP.
_SINGLE_WORKLOAD_WARNED: set[str] = set()


def _too_few_workloads(ctx: LoopContext, ev: EvalResult) -> bool:
    """A plain bench's gate below PERF_WORKLOADS workloads. A board or task card times
    what its suite says (a task: every input group of every shape, one workload block
    per shape), so the count is not its contract."""
    return ev.n_workloads < PERF_WORKLOADS and \
        not precision_gate.is_leaderboard_card(ctx.workspace)

# A ⊗ companion group whose members only pay off together cannot be priced one
# member at a time: mounted alone each member is cost without gain, so per-iter
# pricing plus the transactional rollback below erased every incremental attempt
# at one (a bwd-GQA rebuild died mid-package to exactly this). The student
# prompt already lets a declared group ride slower intermediates ("companion-group
# intermediate states") and the ledger already carries the declaration — the
# judgment just never read it. A KEEP whose change opens with a `part k of n` declaration is such an
# intermediate: while one is the segment's latest delivery, the patch is under
# construction, not refuted — HOLD, like a metric-less segment.
# The declaration is a bounded plea, not a pardon: n is capped (a coupled group
# larger than this should be re-cut per the split discipline), the group forfeits
# the exemption once its KEEP budget (n rows) is spent or once a later KEEP moves
# on without declaring, and a trailing run of failed attempts longer than the
# split discipline tolerates goes back to normal pricing.
_PKG_DECL = re.compile(r"\bpart\s+(\d+)\s*(?:of|/)\s*(\d+)\b", re.IGNORECASE)
_PKG_MAX_STEPS = 4
_PKG_TRAIL_SLACK = 3
# A declaration must OPEN the change (the prompt's contract: "the first sentence
# of change") — a passing mention deeper in the prose ("mirrors the ref's part 2
# of 4 split") must not turn an ordinary KEEP into an in-flight package. The opening = the
# first line's first sentence.
_PKG_SENT_END = re.compile(r"(?<=[.!?。！？])\s")


def _pkg_decl(change: str) -> re.Match | None:
    head = (change or "").split("\n", 1)[0]
    return _PKG_DECL.search(_PKG_SENT_END.split(head, 1)[0])


def _declared_construction(change: str) -> str | None:
    """What the delivery's opening declares about the tree being mid-build, or
    None: an unfinished package step (``part k of n``, k < n, the prompt's
    multi-step build package contract) or an opened placeholder (``[PLACEHOLDER: …]``). Read
    by run_inner_iteration to scaffold such a row while the lineage's green state
    is still unknown, instead of buying a full eval that can only say
    "under construction"."""
    m = _pkg_decl(change)
    if m and int(m.group(1)) < int(m.group(2)):
        return f"part {m.group(1)} of {m.group(2)}"
    head = (change or "").split("\n", 1)[0]
    p = _PLACEHOLDER_TAG.search(head)
    if p:
        return f"[PLACEHOLDER: {p.group(1).strip()[:60]}]"
    return None


def _green_on_resume(records, head: str) -> bool | None:
    """The lineage's correctness state when a loop process starts over an existing
    trajectory (a budget / cert-expiry resume): what ``run_two_layer`` seeds the
    first inner segment with instead of "unknown".

    The checkpoint's own record decides first (``_green_at_ref``: a KEEP sha is
    green, a scaffold NEUTRAL sha is mid-construction). When no record owns HEAD
    (a driver seed commit, a kept probe), the trajectory's latest correctness
    verdict stands in: a PASS means green; a FAIL on a lineage that has never
    been green means still under construction (a GDN run resumed after an
    iteration-0 REVERT + five scaffold rows and paid a full eval of build step 6 —
    "non-finite output", REVERT — because this read None); a FAIL after an
    earlier KEEP is a post-green break whose reset target is unknown here, so it
    stays None and the next full eval decides, as before."""
    g = _green_at_ref(records, head)
    if g is not None:
        return g
    ever_green = any(r.verdict == "KEEP" and r.correctness for r in records)
    for r in reversed(records):
        if r.correctness is True:
            # green only with a verified tree to fall back on: HEAD itself was never
            # evaluated (it may be the very delivery whose eval the process died in),
            # so it must not become the rollback anchor — _resume_anchor names the
            # last KEEP instead, and the first full eval decides HEAD as before
            return True if _resume_anchor(records, head) else None
        if r.correctness is False:
            return None if ever_green else False
    if any(r.verdict == "NEUTRAL" and r.correctness is None and r.sha for r in records):
        return False      # scaffold rows only: construction, never evaluated green
    return None


def _resume_anchor(records, head: str) -> str | None:
    """The rollback anchor for a lineage that resumes green (``_green_on_resume``
    True), or None when there is no verified tree to fall back on.

    ``head`` when a KEEP record owns it, else the latest KEEP sha — never an unowned
    HEAD. A HEAD no record owns is at best the last KEEP plus commits no eval
    verified (a delivery the process died evaluating, a kept probe); entering a green
    segment on it would make every later correctness break reset onto a tree that may
    itself be the broken one, and keep resetting there (observed on an MLA run: a delivery
    committed at eval time became the anchor, then four consecutive REVERTs to it)."""
    for r in reversed(records):
        if r.sha and r.sha == head:
            return head if (r.verdict == "KEEP" and r.correctness) else None
    for r in reversed(records):
        if r.verdict == "KEEP" and r.correctness and r.sha:
            return r.sha
    return None


def _open_package_tail(segment) -> tuple[int, int] | None:
    """(k, n) when the segment's latest delivery is a declared, still-unfinished,
    within-budget multi-iter package step; None otherwise."""
    # The budget counts every declared KEEP in the segment, package identity
    # unexamined: a re-cut mid-segment (abandon pkg A, open pkg B with no plain
    # KEEP between) burns A's rows against B. Deliberately conservative — the
    # error direction is "pricing resumes early", never "plea extended".
    pkg_keeps = sum(1 for r in segment
                    if r.verdict == "KEEP" and _pkg_decl(r.change))
    for offset, r in enumerate(reversed(segment)):
        if r.verdict != "KEEP":
            continue
        m = _pkg_decl(r.change)
        if not m:
            return None        # the latest delivery moved on: no package in flight
        k, n = int(m.group(1)), int(m.group(2))
        if not (1 <= k < n <= _PKG_MAX_STEPS):
            return None        # finished (k >= n), malformed, or over the cap
        if pkg_keeps > n:
            return None        # declared budget spent: n rows of grace, no more
        if offset > _PKG_TRAIL_SLACK:
            return None        # stale: too many failed attempts since the step
        return (k, n)
    return None


@dataclass
class _PendingPatch:
    best_before: float | None
    milestone_before: str
    good_ref: str
    # status-record count when the patch landed: rows appended after this index are
    # the post-patch segment, the only evidence the judgment may read.
    records_before: int = 0
    # The cluster best_before was read on. A segment that ran elsewhere is
    # priced against that cluster's own pre-patch best, or held if it has none.
    cluster: str = ""
    # per-gate allowed regression vs the previous segment (see _gate_regression)
    gate_contract: dict = field(default_factory=dict)
    # decisions reopened in place by this patch (empty for a prefix-reuse reset);
    # a REVERT_PATCH voids the REOPEN row so the student reads the old KEEPs as
    # standing again.
    reopened: list = field(default_factory=list)


def _read_milestone(ctx: LoopContext) -> str:
    try:
        p = Path(ctx.milestone)
        return p.read_text(encoding="utf-8") if p.exists() else ""
    except OSError:
        return ""


def _write_milestone(ctx: LoopContext, text: str) -> None:
    if text:  # never clobber with an empty snapshot (no prior milestone existed)
        Path(ctx.milestone).write_text(text, encoding="utf-8")


def _best_good_ref(records, gate_window: str = GATE_WINDOW_DEFAULT) -> str | None:
    """Commit sha of the best (min gate-window ms) KEEP iteration — the checkpoint
    a rolled-back patch returns the workspace to."""
    kept = [r for r in records
            if r.verdict == "KEEP" and gate_ms(r, gate_window) is not None and r.sha]
    return min(kept, key=lambda r: gate_ms(r, gate_window)).sha if kept else None


def _green_at_ref(records, ref: str) -> bool | None:
    """The correctness state of the workspace once reset to ``ref``.

    A sha owned by a KEEP record was a passing kernel (green). One owned by a
    scaffold NEUTRAL record is mid-construction (not green) — resuming there must
    re-enter scaffold mode, not evaluate half-built sub-steps. A ref no record owns
    (the baseline, a probe/kb commit) is unknown: the next full eval decides.
    """
    for r in reversed(records):
        if r.sha and r.sha == ref:
            return bool(r.correctness) if r.correctness is not None else False
    return None


def _judge_patch(ctx: LoopContext, pending: _PendingPatch) -> str:
    """Attribute the last milestone patch against its post-patch inner segment.

    Helpful (a new best beyond EPS — in same-job ratio when both sides carry one,
    else in gate-window ms) -> KEEP_PATCH. Unhelpful with metric evidence ->
    REVERT_PATCH: restore the milestone snapshot AND reset the workspace to the
    pre-patch best-good checkpoint, so milestone and kernel stay consistent (as if
    this meta step never happened).

    A segment with NO metric reading at all -> HOLD_PATCH: the patch is
    unattributable, not refuted. Pre-green construction (scaffold rows, failed gate
    attempts, certified probes) is invisible to the metric, and rolling it back
    converts "ran out of runway" into "never happened" — three runs lost whole
    construction ladders to exactly that (a rewrite whose repair cycle outlives the
    segment can then never survive a wall). Nothing is restored; the transaction
    stays OPEN so a later patch joins it and one joint judgment — with metric
    evidence — settles both. The first-green path above makes the same call for
    rebuild patches cut short at M0; this is that rule extended to every stop with
    a metric-less segment behind it.

    A measured segment whose LATEST delivery declares itself an unfinished
    multi-iter package step (`part k of n`, see _open_package_tail) is also held,
    not refuted: a ⊗ companion group's members pay off only together, so its
    measured intermediates are construction rows that happen to carry a reading.
    The declaration is bounded — n capped, KEEP budget of n rows, forfeited the
    moment a later KEEP moves on — so it is a plea the ledger can audit, not a
    shield against pricing.
    """
    records = load_records(ctx.status_path)
    cluster = _active_cluster(ctx)
    best_before = pending.best_before
    if cluster != pending.cluster:
        # The segment ran on a different cluster than the one best_before was
        # read on. Price it against THAT cluster's pre-patch best; with none,
        # the patch is unattributable (not refuted) — the same call a
        # metric-less segment gets.
        best_before = _best_ms(_on_cluster(records[:pending.records_before], ctx, cluster),
                               ctx.gate_window)
        if best_before is None and pending.best_before is not None:
            if ctx.meta_status_path:
                update_last_outcome(ctx.meta_status_path, outcome="HOLD_PATCH",
                                    best_after=None)
            _log(f"meta: patch HOLD_PATCH (segment ran on cluster {cluster!r}, the "
                 f"pre-patch best {pending.best_before} was read on "
                 f"{pending.cluster!r}, and {cluster!r} has no pre-patch reading to "
                 f"price against — transaction open)")
            return "HOLD_PATCH"
    best_after = _best_ms(_on_cluster(records, ctx, cluster), ctx.gate_window)
    # The verdict currency: the same-job ratio when both sides carry one. Absolute
    # ms is one node draw per row, and best_before is the minimum of every draw so
    # far — a lucky one stands as the bar. Live case: a best read 1.5% fast, a
    # rebuild that lifted the same-job ratio 1.040 -> 1.072 read 0.14% faster in
    # ms than that draw, and was rolled back as "no gain". The ratio cancels the
    # node; before-rows inside a segment an earlier REVERT_PATCH voided describe
    # trees that are gone and do not set the bar.
    voided = [i for lo, hi in ctx.voided_ranges for i in range(lo, hi)]
    before_rows = [r for i, r in enumerate(records[:pending.records_before]) if i not in voided]
    ratio_before = _best_ratio(_on_cluster(before_rows, ctx, cluster), ctx.gate_window)
    ratio_after = _best_ratio(_on_cluster(records[pending.records_before:], ctx, cluster),
                              ctx.gate_window)
    if ratio_before and ratio_after:
        improved = ratio_after > ratio_before * (1 + _PATCH_IMPROVE_EPS)
        reading = (f"same-job ratio {ratio_before:.4f} -> {ratio_after:.4f}; "
                   f"best {best_before} -> {best_after} ms")
    else:
        improved = best_after is not None and (
            best_before is None or best_after < best_before * (1 - _PATCH_IMPROVE_EPS))
        reading = f"best {best_before} -> {best_after}"
    if improved:
        if ctx.meta_status_path:
            update_last_outcome(ctx.meta_status_path, outcome="KEEP_PATCH", best_after=best_after)
        _log(f"meta: patch KEEP_PATCH ({reading}"
             + (f" on cluster {cluster}" if cluster and cluster != ctx.home_cluster else "")
             + ")")
        return "KEEP_PATCH"
    segment = records[pending.records_before:]
    # Attributable evidence = at least one measured KEEP in the segment (either
    # timing window: a KEEP whose gate-window reading is missing is a bench-contract
    # violation, not a construction row, and must not turn a refuted patch into a
    # held one). Scaffold NEUTRALs, correctness-FAIL gate attempts and probe-blocked
    # rows measure nothing about the patch's worth.
    measured = any(r.verdict == "KEEP" and
                   (r.device_median_ms is not None or r.bwd_median_ms is not None)
                   for r in segment)
    if not measured:
        if ctx.meta_status_path:
            update_last_outcome(ctx.meta_status_path, outcome="HOLD_PATCH", best_after=best_after)
        _log(f"meta: patch HOLD_PATCH (no measured KEEP in the {len(segment)}-row "
             f"post-patch segment — pre-green construction preserved, transaction open)")
        return "HOLD_PATCH"
    pkg = _open_package_tail(segment)
    if pkg is not None:
        k, n = pkg
        if ctx.meta_status_path:
            update_last_outcome(ctx.meta_status_path, outcome="HOLD_PATCH", best_after=best_after)
        _log(f"meta: patch HOLD_PATCH (declared multi-iter package in flight — "
             f"part {k} of {n} is the segment's latest delivery; pricing deferred "
             f"to package completion, transaction open)")
        return "HOLD_PATCH"
    _write_milestone(ctx, pending.milestone_before)
    ctx.backend.reset(pending.good_ref)
    # every row of the rolled-back segment describes a tree that is gone
    ctx.voided_ranges.append((pending.records_before, len(records)))
    if pending.reopened:
        # The REOPEN row told the student its KEEPs for the closure were void;
        # with the patch reverted and the tree back at the pre-patch best, they
        # stand again — say so in the same ledger the student reads.
        append_record(ctx.status_path, IterationRecord(
            iter=len(records), decision_id="REOPEN-REVERT",
            change=(f"the milestone patch behind the last REOPEN row was reverted; "
                    f"tree restored to {pending.good_ref[:10]}; the earlier KEEP rows for "
                    f"[{', '.join(pending.reopened)}] stand again, the reopened segment's "
                    f"rows are void."),
            correctness=None, verdict="NEUTRAL", sha=""))
        ctx.reopen_note = ""
    # The workspace is back on the pre-patch good tree: nothing the reverted patch
    # invalidated is rolled back any more, so its standing note must not survive into
    # a student turn that runs on the restored tree.
    ctx.resume_note = ""
    if ctx.meta_status_path:
        update_last_outcome(ctx.meta_status_path, outcome="REVERT_PATCH", best_after=best_after)
    _log(f"meta: patch REVERT_PATCH (no gain: "
         + (f"same-job ratio {ratio_before:.4f} -> {ratio_after:.4f} within "
            f"{_PATCH_IMPROVE_EPS:.1%}; " if ratio_before and ratio_after else "")
         + f"best_before={best_before} best_after={best_after}); "
         f"milestone + workspace rolled back")
    return "REVERT_PATCH"


def _needs_authoring(ctx: LoopContext) -> bool:
    """True when the milestone has not been distilled yet and the teacher has an input
    to distill it from. This is the entry to the distillation loop: meta authors
    milestone v1 before the inner loop starts.

    The input is oracle source under oracle_mode="source" and black-box reconnaissance
    of the reference under "blackbox"; "none" has none and skips distillation. Asking
    only "is there an oracle path" conflated "the teacher may not read source" with
    "there is nothing to distill", so the mode is what is checked.

    "Not distilled yet" is anything that fails the §9 gate — including the file not
    existing at all, which is how a freshly scaffolded operator starts (nothing
    writes a placeholder milestone; run_distill_step creates the file)."""
    if ctx.oracle_mode == "none":
        return False
    if ctx.oracle_mode == "source" and not ctx.oracle_path:
        return False
    return bool(_validate_milestone_file(ctx.milestone))


@dataclass
class ReconResult:
    """What the reconnaissance step produced."""
    rounds: int
    report: Path | None = None
    finished: bool = False


_RECON_STATE_NAME = "recon_complete.json"


def _recon_state_path(ctx: LoopContext) -> Path | None:
    """Where "reconnaissance is finished" is remembered across process restarts.

    Lives next to status.jsonl — the loop's own state directory — and NOT in the
    reconnaissance tree: the recon backend commits with `git add -A`, so anything
    parked there would ride the next probe commit into the tree's history and from
    there into every future factory clone. student/ is gitignored in the operator
    repo, so the marker can never reach a student clone or a GPU tarball either.
    """
    if ctx.recon_path is None:
        return None
    return ctx.status_path.parent / _RECON_STATE_NAME


def _save_recon_state(ctx: LoopContext, rounds: int) -> None:
    """Persist recon completion so a restart does not re-litigate a finished phase.

    Observed on a bwd-MHA run: a watchdog restart put the new process's
    session_start after the previous session's finished report, the freshness gate
    classified that report as stale residue, and the teacher — convinced the phase
    was over — yielded turn after turn until all 12 rounds had been spent on
    re-affirmation theater. The completion is a fact about the experiment, not
    about the process, so it must live on disk. Written only when the write-up
    stands complete (artifacts fresh, evidence delivered or waived).
    """
    state_path = _recon_state_path(ctx)
    if state_path is None:
        return
    candidates = ctx.recon_path.parent / "candidates.md"
    try:
        state = {
            "report": str(ctx.recon_path),
            "report_mtime": ctx.recon_path.stat().st_mtime,
            "candidates_mtime": candidates.stat().st_mtime,
            "rounds": rounds,
        }
    except OSError:
        return  # artifacts vanished between the completeness check and here
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _log(f"meta/recon: completion persisted to {state_path}")
    _persist_gap_queue(ctx)


# --- out-of-run gap queue ------------------------------------------------------
#
# The recon candidates table tags every decision with a VERIFY channel; rows
# tagged `none` — and rows whose remark guesses at expressibility ("may not be
# expressible in the DSL") without an [EXPR:] ladder verdict — are claims nobody
# in the run will
# ever check. On one sparse-attention run such a row (the cp.async cache bits)
# was worth 22 % and sat unread for the whole run. The queue makes them
# visible: banked beside the evidence at recon completion, surfaced in the
# loop summary, and archived for human review under docs/kb-provenance/gaps/.
#
# It never reaches hw_kb. The rows are the reference's structure as the recon
# read it (a DEPBAR depth at a buffer release, a barrier handshake, cache
# bits) — narrated provenance, which the KB discipline keeps out of every
# agent read set precisely because hw_kb is read in full by every later
# student: a "[pending certification]" entry would hand the next same-operator run an
# unverified reference-level structural prior through a door kb_lint cannot
# see (no reference token, no line number — just structure). The only carrier
# into hw_kb is a probe-certified recipe.
_GAP_QUEUE_NAME = "gap_queue.md"
_GAP_ARCHIVE_SUBDIR = "gaps"           # docs/kb-provenance/gaps/<op>.md
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
_EXPR_TAG_RE = re.compile(r"\[EXPR:")
_EXPR_GUESS_RE = re.compile(
    r"(?:not|n't)\s+(?:be\s+)?expressible"
    r"|(?:cannot|can't|may\s+not|might\s+not)\s+be\s+expressed"
    r"|inexpressible|unexpressible", re.I)
# Both spellings the recon writes: a bare VERIFY column (`none`) and the
# inline tag the prompt itself prescribes (`[VERIFY: none]`, parked in a
# `tag` / `source` column). Real tables are mostly the latter: of five runs'
# candidates.md only one carried the column; the other four (sparse,
# fp8, bwd-GQA, fwd-MHA) would have parsed to zero rows.
_VERIFY_TAG_RE = re.compile(r"\[VERIFY:\s*([A-Za-z-]+)\s*\]", re.I)


@dataclass
class GapRow:
    decision: str
    value: str
    verify: str
    remark: str
    reason: str     # "verify-none" | "expressibility-guess" | "inferred"

    _WHY = {"expressibility-guess": "unverified expressibility guess, no [EXPR:] verdict",
            "inferred": ("inferred: the discriminating observation gave no positive reading "
                         "(readings a byte/count identity cannot separate)"),
            "verify-none": "no machine-read verification in this run"}

    def line(self) -> str:
        return (f"- **{self.decision}** = {self.value} — VERIFY `{self.verify}`"
                f"{'; notes: ' + self.remark if self.remark else ''}"
                f" ({self._WHY.get(self.reason, self.reason)})")


def _row_verify(header: list[str], cols: dict, text: str) -> str:
    """The row's VERIFY channel: the `[VERIFY: x]` tag wherever it sits in the
    row, else the bare value of a VERIFY column (`none`, `` `none` ``,
    `none（...）`), else ""."""
    tm = _VERIFY_TAG_RE.search(text)
    if tm:
        return tm.group(1).lower()
    vcol = next((h for h in header if h.upper().startswith("VERIFY")), None)
    if vcol is None:
        return ""
    bare = cols[vcol].strip("`").strip()
    return re.split(r"[\s（(`]", bare, maxsplit=1)[0].lower() if bare else ""


def parse_gap_queue(candidates_text: str) -> list[GapRow]:
    """Rows of every candidates table that are either VERIFY `none` (a bare
    column or the inline `[VERIFY: none]` tag) or guess at expressibility
    without an [EXPR:] ladder verdict. Duplicates (the same decision in
    several candidate sections) collapse on (decision, value)."""
    out: list[GapRow] = []
    seen: set = set()
    header: list[str] | None = None
    for raw in candidates_text.splitlines():
        m = _TABLE_ROW_RE.match(raw.strip())
        if m is None:
            header = None
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if header is None:
            header = cells
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue                                   # the |---|---| rule
        cols = {h: (cells[i] if i < len(cells) else "") for i, h in enumerate(header)}
        text = " ".join(cells)
        verify = _row_verify(header, cols, text)
        decision = cols.get(header[0], "").strip("*").strip()
        value = cols.get(header[1], "") if len(header) > 1 else ""
        remark = next((cols[h] for h in header if h.strip().lower() in
                       ("note", "notes", "remark", "remarks", "comment", "comments")), "")
        if verify == "none":
            reason = "verify-none"
        elif verify == "inferred":
            # Observed in a dsa_sparse recon: the KV-ring stage axis had two readings
            # with identical bytes, the discriminator it wrote printed nothing,
            # and the row shipped as [VERIFY: sass-derived] "no ambiguity". Under the
            # rule an empty discriminator leaves the row `inferred`; it comes
            # here so distill sees BOTH readings as candidates, never a verdict.
            reason = "inferred"
        elif _EXPR_GUESS_RE.search(text) and not _EXPR_TAG_RE.search(text):
            reason = "expressibility-guess"
        else:
            continue
        key = (decision, value)
        if key in seen:
            continue
        seen.add(key)
        out.append(GapRow(decision, value, verify, remark, reason))
    return out


def _gap_queue_text(rows: list[GapRow], op: str = "") -> str:
    body = "\n".join(r.line() for r in rows)
    return (f"# Out-of-run gap queue (recon-table rows this run did not verify"
            f"{': ' + op if op else ''})\n\n"
            "Rows with VERIFY `none` have no machine-read acceptance; rows that say \"may not "
            "be expressible\" without an `[EXPR:]` ladder verdict are unverified guesses; rows "
            "with VERIFY `inferred` are structural readings whose discriminating observation "
            "gave no positive reading (the two readings are byte-identical). None of the three "
            "may be used as a conclusion. These rows are the recon's reading of the reference's "
            "structure (narrated provenance) and **do not enter hw_kb**: only a probe-certified "
            "recipe carries knowledge across runs; to verify one, start a separate probe, with "
            "the evidence under `docs/kb-provenance/`.\n\n"
            f"{body}\n")


def load_gap_queue(ctx: LoopContext) -> list[GapRow]:
    """The queue as parsed from the candidates table now (not the banked
    file): the table is the SSOT and a re-affirmed recon may have closed rows."""
    if ctx.recon_path is None:
        return []
    candidates = ctx.recon_path.parent / "candidates.md"
    try:
        return parse_gap_queue(candidates.read_text(encoding="utf-8"))
    except OSError:
        return []


def _gap_archive_path(ctx: LoopContext) -> Path | None:
    """docs/kb-provenance/gaps/<op>.md — the human-review copy, outside every
    agent read set (the provenance archive's discipline), reviewed with the KB
    changes."""
    op = getattr(ctx, "op", "") or ""
    if not op or ctx.kb_dir is None:
        return None
    # Anchored on the KB's repo root (hw_kb/ and docs/kb-provenance/ are
    # siblings — a reviewer commits them together).
    return lang_mode.kb_repo_root(ctx.kb_dir) / "docs" / "kb-provenance" / _GAP_ARCHIVE_SUBDIR / f"{op}.md"


def _persist_gap_queue(ctx: LoopContext) -> list[GapRow]:
    rows = load_gap_queue(ctx)
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return rows
    targets = [ev / _GAP_QUEUE_NAME]
    archive = _gap_archive_path(ctx)
    if archive is not None and archive.parent.parent.is_dir():   # docs/kb-provenance exists
        targets.append(archive)
    touched = False
    for dst in targets:
        try:
            if rows:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(_gap_queue_text(rows, getattr(ctx, "op", "") or ""),
                               encoding="utf-8")
                touched = True
            elif dst.exists():
                dst.unlink()
                touched = True
        except OSError as e:
            _log(f"meta/recon: could not persist {dst}: {e}")
    if touched:
        _bank_evidence(ctx, "gap queue")
    if rows:
        _log(f"meta/recon: gap queue — {len(rows)} unverified candidates row(s) "
             f"({sum(r.reason == 'expressibility-guess' for r in rows)} expressibility "
             f"guess(es) without an [EXPR:] verdict) -> "
             + ", ".join(str(t) for t in targets) + " (never hw_kb)")
    return rows


def _load_recon_state(ctx: LoopContext) -> ReconResult | None:
    """A prior session's completion marker, if it still describes what is on disk.

    Trust is conditional: the marker must name this run's report path, and both
    artifacts must still exist, be non-empty, and be no older than when the marker
    was written (a later mtime is a legitimate re-affirmation or edit; an earlier
    one means the file was swapped out from under the marker). Anything off —
    missing files, unparsable JSON, a foreign path — and the marker is ignored,
    which simply re-runs reconnaissance: the pre-fix behavior.
    """
    state_path = _recon_state_path(ctx)
    if state_path is None or not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        report = Path(state["report"])
        mtimes = {report: float(state["report_mtime"]),
                  report.parent / "candidates.md": float(state["candidates_mtime"])}
        rounds = int(state["rounds"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if report != ctx.recon_path:
        return None
    for path, recorded in mtimes.items():
        try:
            st = path.stat()
        except OSError:
            return None
        if st.st_size == 0 or st.st_mtime + 1.0 < recorded:  # +1s: fs granularity
            return None
    return ReconResult(rounds=rounds, report=ctx.recon_path, finished=True)


# A truncated stream from the gateway ends in a well-formed-looking tiny final
# message and rc=0 — indistinguishable, at the exit-code level, from a deliberate
# turn. Charged as ordinary rounds, one bad gateway hour spent 8 of a recon's 12
# GPU rounds on nothing. The tell is the combination: an output
# too short to be real work AND no legal trace of the turn on disk. Deliberate
# no-op turns write artifacts or emit requests; deliberate rambling is long.
TRUNCATION_MAX_CHARS = 400
TRUNCATION_BACKOFF_S: tuple[float, ...] = (60.0, 120.0, 300.0)
TRUNCATION_STORM_BUDGET = 8  # per phase; past this the channel is sick, not unlucky


def _read_or_none(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _agent_turn(be, prompt: str, *, left_a_trace, label: str,
                storm_budget: list[int]) -> str:
    """run_agent with bounded FREE retries for truncated turns.

    ``left_a_trace(out)`` answers "did this turn end legally?" (a parseable
    request in the output, or the artifact it was asked to write changed on
    disk). A turn that is both traceless and shorter than TRUNCATION_MAX_CHARS
    is retried after a backoff — truncation correlates with gateway congestion,
    so immediate re-fire mostly re-truncates — WITHOUT charging the caller's
    round budget: budget rounds are measurement opportunities, and spending
    them on turns that never happened destroyed 8/12 of a reconnaissance.

    ``storm_budget`` is a mutable one-element list shared across a phase. When
    it runs dry the channel is systematically sick; AgentError converts the run
    to stop=agent_failed — the transient class the watchdog already restarts —
    with a diagnosis in the log instead of a silently spent budget.
    """
    be = _teacher(be)  # every caller is a teacher phase (recon / distill / lever design)
    out = be.run_agent(prompt)
    for pause in TRUNCATION_BACKOFF_S:
        if len(out.strip()) >= TRUNCATION_MAX_CHARS or left_a_trace(out):
            return out
        if storm_budget[0] <= 0:
            raise AgentError(
                f"{label}: truncation storm — {TRUNCATION_STORM_BUDGET} free "
                "retries consumed this phase and turns still come back tiny and "
                "traceless; the channel is failing systematically (see "
                "TRUNCATION_* in loop/driver.py)")
        storm_budget[0] -= 1
        _log(f"{label}: turn looks truncated ({len(out.strip())} chars, no legal "
             f"trace) -> free retry in {pause:.0f}s, not charged to the round "
             f"budget ({storm_budget[0]} storm-budget left)")
        time.sleep(pause)
        out = be.run_agent(prompt)
    return out


def _recon_prompt(ctx: LoopContext, extra: str = "") -> str:
    """The reconnaissance brief, with the source-mode addendum when the teacher
    also holds the oracle.

    recon_prompt.md is written for the blackbox teacher ("you will not be given
    the source"). A source-mode run that reconnoitres gets the same brief — the
    evidence gates, the tool contracts and the write-up are identical — plus a
    block saying the source IS available and what reconnaissance is for when it
    is: measuring what the source states only as a template (the values the
    binary actually took, the SASS-level schedule, the sync topology) rather
    than re-deriving what the source states plainly. Appended, not branched
    into the file, so the blackbox rendering stays byte-identical.
    """
    addendum = ""
    if ctx.oracle_mode == "source" and ctx.oracle_path:
        addendum = (
            "\n\n## Source-mode addendum: you also hold the reference's source\n\n"
            f"This is a **source run**; the passage above about not being given the source "
            f"does not apply: `ORACLE` = `{ctx.oracle_path}`"
            " (comma-separated, all readable). Reconnaissance therefore has a different purpose "
            "— not to replace the source but to **measure the facts the source gives only as "
            "templates / parameters and that are fixed only in the binary**: the tile / stage / "
            "warp-role values this shape actually takes, the real smem and register usage, the "
            "SASS-level issue order and commit/wait choreography of the K loop, the real "
            "topology and arrival counts of the mbarriers / named barriers, whether the supply "
            "is serial or prefetched, the DEPBAR depth, the rasterization group width.\n\n"
            "- **Read the source first, then decide the probes**: the source tells you what to "
            "ask, the probe tells you how the binary answers; when the two disagree, **the "
            "binary's reading is the fact and the source only explains the intent** — write "
            "both in the report and mark the disagreement.\n"
            "- The evidence gates, the tool contracts and the two deliverables (report + "
            "candidates table) are exactly the same as for a blackbox run — the candidates table "
            "may hold a single candidate (the machine in the source), but every value must "
            "still carry a job number or be marked \"not measured\".\n"
            "- Every fact in the report must still trace to a job number; \"the source says "
            "so\" is not a measurement source.\n"
        )
    return render_prompt(_meta_prompt(ctx, "recon_prompt.md"), _bindings(ctx),
                         extra=addendum + extra)


def run_recon_step(ctx: LoopContext) -> ReconResult:
    """Black-box reconnaissance: the teacher measures the reference it may not read.

    Under oracle_mode="blackbox" the teacher has no source, so the input it distils
    from has to be produced rather than opened. It writes a script, names it, and the
    driver runs it against the reference and hands back the output — the same shape as
    the student's probe sub-loop and the teacher's KB certification, for the same
    reason: the agent decides *what* to ask, and the driver owns submission, polling,
    retry of lost jobs and log recovery, none of which an agent should reimplement.

    Two properties are load-bearing:

    * It runs in ``recon_backend``, a separate submission tree. Reconnaissance output
      is precisely what a student must not see, and the student's job receives its
      operator tree as a tarball.
    * The script is committed before submission. Delivery is ``git archive``, which
      ships committed files only — an uncommitted script reaches the node as absence,
      and the failure looks like a broken script rather than a missing one.

    * SASS evidence is gated, not hoped for. The driver seeds the vendored
      ``sass_tools`` into ``notes/`` so no probe hand-rolls a parser, and a finish
      is only accepted once a job log carried the tool's ``SASS-OK`` verdict or the
      report owns the absence with a ``SASS-WAIVED:`` line (a report once shipped
      without the opcode histogram and no mechanism objected).

    The loop ends when the teacher stops asking — accepted only with both write-up
    artifacts on disk, since a truncated turn is indistinguishable from a deliberate
    stop — or when the round budget runs out; a teacher that has learned nothing
    still has to write its report, and saying so honestly is more useful than a
    report padded to look complete.
    """
    be, rbe = ctx.backend, ctx.recon_backend
    if rbe is None:
        _log("meta/recon: no reconnaissance backend configured — skipping")
        return ReconResult(rounds=0, finished=False)
    if (prior := _load_recon_state(ctx)) is not None:
        _log(f"meta/recon: completed in a previous session ({_RECON_STATE_NAME} "
             "validates against the on-disk write-up) — skipping straight to distillation")
        return prior
    _seed_sass_tools(ctx, rbe)
    # Freshness floor for the write-up gate: artifacts must have been (re)written
    # after this moment. Without it, whatever report.md/candidates.md happen to sit
    # at the recon path — a previous run's, or a previous OPERATOR's (the default
    # recon path ships with a populated GEMM report in-tree) — satisfies the
    # existence check, and a truncated teacher turn finishes reconnaissance
    # against wrong data. Reusing a prior tree stays legal: the teacher just has
    # to re-affirm the write-up by touching it this session.
    session_start = time.time()
    sass_ok = False  # flipped by the first job log carrying a SASS-OK verdict line
    sched_ok = False  # ditto, SCHED-OK (schedule transcription delivered)
    live_ok = False   # ditto, LIVE-OK (register-liveness ledger delivered)
    stall_ok = False  # ditto, STALL-OK (per-instruction stall ledger delivered)
    struct_ok = False  # ditto, STRUCT-OK (per-segment construction table delivered)
    near_miss: dict = {}  # channel -> a "*-OK" line that failed the full-field check
    launch_facts: dict = {}  # cluster shape read off a job log (see _launch_cluster_from_log)
    storm_budget = [TRUNCATION_STORM_BUDGET]
    prompt = _recon_prompt(ctx)
    for r in range(ctx.max_recon_rounds):
        out = _agent_turn(
            be, prompt, label="meta/recon", storm_budget=storm_budget,
            left_a_trace=lambda o: (
                isinstance(parse_student_turn_safe(o, allow_recon=True), ReconRequest)
                or _recon_artifacts_written(ctx, since=session_start)))
        turn = parse_student_turn_safe(out, allow_recon=True)
        if not isinstance(turn, ReconRequest):
            # "No further request" is how a teacher signals it is done — but a
            # truncated turn looks exactly the same. Observed verbatim: a teacher
            # ended a turn with "Writing probe3:" and nothing else; the loop read
            # that as a deliberate finish, 10 of 12 budget rounds were forfeited,
            # and four architecture quantities shipped as "not measured". A finish is
            # therefore only accepted once both write-up artifacts exist on disk;
            # any other request-less turn is re-prompted, and the round budget is
            # charged to it — a teacher that keeps stalling converges on the same
            # write-up guard the budget path uses.
            if _recon_artifacts_written(ctx, since=session_start):
                # The overlap channels (ROLE with tensor, PMTL) are delivered by
                # the driver's own frozen pricing recipe on the ref tree, not by
                # the teacher's improvised probes — it runs here, once, before
                # the teacher is held to them.
                _ensure_ref_pricing(ctx, label="meta/recon")
                missing = _missing_recon_evidence(ctx, sass_ok, sched_ok, live_ok,
                                                  stall_ok, struct_ok, near_miss=near_miss)
                if missing:
                    # Rounds remain, so "deliver or waive" is a real choice here —
                    # a teacher once knew the one round that would close four gaps
                    # and was allowed to finish without running it.
                    _log(f"meta/recon: write-up exists but {'/'.join(missing)} "
                         "evidence is neither delivered nor waived — re-prompting")
                    prompt = _recon_prompt(ctx, extra=_evidence_gate_msg(
                                               missing, near_miss, ctx))
                    continue
                contras = _launch_contradictions_open(ctx, launch_facts)
                if contras:
                    _log(f"meta/recon: RECON-CONTRADICTION cluster: NCU read "
                         f"cluster={launch_facts['cluster']} but the write-up denies it "
                         f"({contras[0][:100]}) — re-prompting")
                    prompt = _recon_prompt(ctx, extra=_contradiction_msg(launch_facts,
                                                                          contras))
                    continue
                _log(f"meta/recon: teacher finished reconnaissance after {r} run(s)")
                _require_recon_artifacts(ctx, be, sass_ok, sched_ok, live_ok,
                                         stall_ok, struct_ok, since=session_start,
                                         near_miss=near_miss, launch=launch_facts)
                _save_recon_state(ctx, rounds=r)
                return ReconResult(rounds=r, report=ctx.recon_path, finished=True)
            _log("meta/recon: turn ended with neither a RECON_REQUEST nor the "
                 "write-up artifacts (written this session) — re-prompting")
            prompt = _recon_prompt(ctx, extra=(
                "\n\n## The previous turn did not end in a legal way\n\n"
                "Your previous turn neither sent `RECON_REQUEST: notes/<name>.py` "
                "nor wrote the two deliverables to disk. There are only two legal endings:\n"
                "1. **More reconnaissance** → write the script into the recon tree and end "
                "with `RECON_REQUEST: <real path>`;\n"
                "2. **Reconnaissance done** → first write both deliverables, the recon report "
                "and the candidate architecture table, then end the turn.\n"
                "If you meant to continue last time but the turn was cut off, pick up where "
                "you left off."))
            continue
        # A request naming a file that is not on disk would still cost a full GPU job
        # to learn "can't open file" — observed verbatim: a teacher echoed the
        # prompt's own `notes/<name>.py` placeholder and a job was burned on it.
        # Round budget is charged to the teacher's mistake, not the cluster.
        # Only checkable when the reconnaissance tree's path is known.
        if ctx.recon_dir is not None and not (ctx.recon_dir / turn.path).is_file():
            _log(f"meta/recon: requested script {turn.path} does not exist — re-prompting")
            prompt = _recon_prompt(ctx, extra=(
                f"\n\n## The previous RECON_REQUEST was rejected\n\n"
                f"You asked to run `{turn.path}`, but there is no such file under "
                f"`{ctx.recon_dir}`. **Write the script into the recon tree first, then send "
                f"the request with its real path** (the `<name>` in "
                f"`RECON_REQUEST: notes/<name>.py` is a placeholder; replace it with the file "
                f"name you actually wrote)."))
            continue
        rbe.commit(f"recon: {turn.path}")
        try:
            log = rbe.submit_gpu("main", f"cmd:python3 {turn.path}")
        except BackendError as e:
            # A starved queue or a dead job must not end the whole run (a probe once
            # sat Queued past the poll bound and the uncaught raise threw away 3.5h of
            # reconnaissance). The round is charged and the teacher is told — the same
            # trust model as a probe that failed to build.
            log = (f"[loop] GPU job failed to return: {e}\n"
                   f"The round is charged to the budget and its result is unavailable. "
                   f"Either schedule that measurement again in the next round, or wrap up "
                   f"from the notes you have — marking the facts you could not measure "
                   f"honestly as \"not measured\".")
            _log(f"meta/recon: submit_gpu failed ({e}) — round charged, teacher informed")
        else:
            from forgekernel.skills.stall_price_run import expand_bundle
            log = expand_bundle(log)   # job log archives clip the head; see the tool
            sass_ok = sass_ok or _SASS_OK.search(log) is not None
            sched_ok = (sched_ok or _SCHED_OK.search(log) is not None
                        or _SCHED_NOT_APPLICABLE.search(log) is not None)
            live_ok = live_ok or _LIVE_OK.search(log) is not None
            _collect_verdict_near_misses(log, near_miss)
            cluster = _launch_cluster_from_log(log)
            if cluster is not None:
                # A later round that profiled a helper / setup kernel must not demote
                # the write-up's launch facts to that kernel's shape: the reading a
                # clustered launch produced stands (compared by CTA count per cluster,
                # so a (2,8,1) reading beats a (16,1,1) one on size, not on dim0).
                # The failure case was a report whose §1 denied cluster while a round
                # carried a helper's (0,0,0) — that round used to overwrite the
                # reference's. A genuine contradiction is still caught: whichever
                # reading wins, cluster > 1 contradicts the denial, and
                # LAUNCH-DISPUTED owns the disagreement from the report's side.
                prev = launch_facts.get("cluster") or (0, 0, 0)
                if cluster[0] * cluster[1] * cluster[2] >= prev[0] * prev[1] * prev[2]:
                    launch_facts.update(cluster=cluster, round=r, source=(
                        "LAUNCH-OK" if _LAUNCH_OK.search(log) else
                        "launch__cluster_dim_* metric in the job log"))
                    _log(f"meta/recon: launch cluster={cluster} read from round {r}")
            if _STALL_OK.search(log) is not None:
                stall_ok = True
            if _STRUCT_OK.search(log) is not None:
                struct_ok = True
            _save_ref_stall_ledger(ctx, log)   # banks what self-checked, per table
            if _STRUCT_OK.search(log) is not None:
                _save_ref_struct_table(ctx, log)
            if _ORCH_OK.search(log) is not None:
                _save_ref_orch_dossier(ctx, log)
            if _ROLE_ANY.search(log) is not None:
                _save_ref_role_exposure(ctx, log)
            _save_ref_pm_timeline(ctx, log)
            _save_recon_log(ctx, turn.path, r, log)
            _log(f"meta/recon: ran {turn.path} ({len(log)} chars back)")
            # The next round's `rbe.commit` would carry these, but the LAST round
            # has no next one — and a job ships a commit, so an unbanked ledger
            # never reaches the node it is meant to inform.
            _bank_evidence(ctx, f"round {r} ledgers")
        prompt = _recon_prompt(ctx, extra=(
            f"\n\n## Output of the previous recon run ({turn.path})\n\n"
            f"```\n{log[-6000:]}\n```\n\n"
            "Read it, then either send the next RECON_REQUEST or write the two deliverables "
            "and finish.\n"
            "If this run measured nothing (the script errored, a metric name was misspelled, "
            "no kernel matched), **fix the script before going on** — one failed run does not "
            "cost you a conclusion, but treating it as \"this fact is unavailable\" does."))
    _log(f"meta/recon: round budget ({ctx.max_recon_rounds}) exhausted")
    _ensure_ref_pricing(ctx, label="meta/recon")
    _require_recon_artifacts(ctx, be, sass_ok, sched_ok, live_ok, stall_ok,
                             struct_ok, since=session_start, near_miss=near_miss,
                             launch=launch_facts)
    # The exhausted path can end in two states the log's WARNING branch already
    # distinguishes: write-up complete (persist — a restart must not re-litigate
    # it) or still incomplete (don't — a fresh session's recon turns are the only
    # remaining chance to finish the write-up).
    if (_recon_artifacts_written(ctx, since=session_start)
            and not _missing_recon_evidence(ctx, sass_ok, sched_ok, live_ok,
                                            stall_ok, struct_ok, near_miss=near_miss)
            and not _launch_contradictions_open(ctx, launch_facts)):
        _save_recon_state(ctx, rounds=ctx.max_recon_rounds)
    return ReconResult(rounds=ctx.max_recon_rounds, report=ctx.recon_path, finished=False)


def _recon_artifacts_written(ctx: LoopContext, since: float | None = None) -> bool:
    """Both write-up artifacts exist — the only acceptable evidence of a finish.

    ``since`` (a POSIX timestamp) additionally requires both files to have been
    written after that moment. Existence alone once let a run finish against a
    stale report already on disk — including the wrong operator's, since the
    default recon path ships populated in-tree.

    When the report path is unknown there is nothing to hold a finish against, so
    the request-less turn is taken at its word (the pre-fix behavior).
    """
    if ctx.recon_path is None:
        return True
    candidates = ctx.recon_path.parent / "candidates.md"
    for p in (ctx.recon_path, candidates):
        try:
            mtime = p.stat().st_mtime
        except OSError:  # missing (FileNotFoundError) or unreadable
            return False
        if since is not None and mtime < since:
            return False
    return True


# The verdict line only sass_tools.analyze prints, and only after its self-checks
# (right kernel, instruction floor, tensor families, no swallowed profiler errors)
# all pass. The full field pattern is required so an incidental "SASS-OK" in prose
# does not open the gate; a teacher printing the line by hand without the tool is
# fabricating evidence, and the transcript audit — not this regex — is the
# backstop for that (same trust model as probe verdicts).
#
# The kernel field is `.+?`, NOT `\S+`: a demangled C++ template name carries
# spaces ("void sparse_attn_fwd_kernel<KernelTemplate<512, 1>, ..."), and \S+
# stops at the first one so the trailing full-field anchor never matches — a
# false NEGATIVE on the tool's own genuine output (a sparse recon burned its
# write-up turns against exactly this; the teacher rightly refused to waive a
# channel it had delivered). The anti-cheat force lives in the line anchor plus
# the trailing numeric fields, which `.+?` preserves.
_SASS_OK = re.compile(r"^SASS-OK kernel=.+? instructions=\d+ opcodes=\d+", re.M)
# The explicit escape hatch: one report line owning the absence and its reason.
# Machine-checkable, and it surfaces on the report's first screen instead of in a
# late section.
_SASS_WAIVED = re.compile(r"^SASS-WAIVED:\s*\S", re.M)

# The schedule-transcription and liveness-ledger verdicts, printed only by
# sched_live_tools.report after its own self-checks. Same trust model and same
# full-field patterns as SASS-OK: prose cannot open the gate. These two became
# mandatory after a bwd run where every counting fact was delivered and the
# milestone still could not say which chain flies over which elementwise segment
# (the schedule) or what register peak that overlap costs (the liveness) — the
# pair of facts its M1 died without.
#
# Three forms, one per MMA ISA the tool transcribes (sched_live_tools picks the
# grammar by census): Hopper wgmma keeps its original fields; Blackwell tcgen05
# and per-warp mma.sync carry `isa=` and their own counts. A dsa_sparse
# B200 teacher's hand-written HMMA tool printed the mma_sync form and this regex
# (then HGMMA-only) refused it for a day — the report had to double-write a
# DELIVERED-MISMATCH line to close a channel that WAS delivered.
_SCHED_OK = re.compile(
    r"^SCHED-OK (?:isa=\S+ )?loop=\S+ (?:hgmma_runs=\d+ depbars=\d+"
    r"|mma_runs=\d+ barriers=\d+ ldsm=\d+"
    r"|utcmma_runs=\d+ commits=\d+ mbar_waits=\d+)", re.M)
_SCHED_WAIVED = re.compile(r"^SCHED-WAIVED:\s*\S", re.M)
# sched_live_tools models the Hopper warpgroup-MMA schedule (chains by accumulator
# register, WARPGROUP.DEPBAR.LE wait depths). A tcgen05 kernel has neither — the
# accumulator lives in TMEM and the waits are mbarrier — and an mma.sync (HMMA)
# kernel has no warpgroup chain to wait on at all; on either the tool refuses
# rather than reporting zeros, and says so on its own verdict line. That is a
# delivered answer about the instrument, not a missing channel: it opens the SCHED
# gate the way a waiver does, so a run is not held hostage by a tool that cannot
# see its ISA (a gdn_prefill B200 run's whole 12-round recon budget went
# to the tcgen05 case; a dsa_sparse B200 run hit the HMMA case six times). The tool only prints the line when the page HAS an MMA of another
# ISA — a page with no MMA at all is a wrong page, not a wrong instrument, and
# still fails. The teacher still owes the schedule through another channel; the
# recon prompt and the tool's own output both say which.
_SCHED_NOT_APPLICABLE = re.compile(
    r"^SCHED-NOTAPPLICABLE isa=\S+ tcgen05_mma=\d+ wgmma=\d+ mma_sync=\d+ reason=\S+",
    re.M)
_LIVE_OK = re.compile(r"^LIVE-OK peak=\d+ guessed_frac=", re.M)
_LIVE_WAIVED = re.compile(r"^LIVE-WAIVED:\s*\S", re.M)

# §1 launch configuration: the cluster shape is a launch ATTRIBUTE, not a grid
# dimension, and the report used to infer it from the grid. The dsa_sparse
# B200 reference launches a flat 128-CTA grid with launch__cluster_dim_x=16; the
# report's §1 said "no cluster", the candidates table reverse-engineered a dense
# GEMM tile with cluster=(1,1,1), the first milestone copied it and the student's
# first kernel was 16x slower than the reference. The driver now reads the
# cluster off every job log — the seeded launch_tools verdict line, or the raw
# NCU metric a probe printed — and holds the write-up against it.
_LAUNCH_OK = re.compile(
    r"^LAUNCH-OK kernel=.+? grid=\(\d+,\d+,\d+\) block=\(\d+,\d+,\d+\) ctas=\d+ "
    r"cluster=\((\d+),(\d+),(\d+)\) cluster_max_active=(\d+|n/a)", re.M)
_CLUSTER_METRIC = {d: re.compile(rf"\blaunch__cluster_dim_{d}\b[^\d\n-]{{0,40}}?(\d+)")
                   for d in "xyz"}
# What a report / candidates table says when it did NOT read the metric. The gate
# reads the DECLARATION, not any sentence containing the word: the brief requires
# §1 and each candidate's first line to write the shape, so the denial is a cluster
# field whose value is 1 / none / n/a, plus the explicit "no cluster" phrase when it
# is a claim about the launch (a correction or an alternative candidate that says it
# is not the measured launch is not a denial — a review measured the old
# pattern bouncing "§1 correction: the earlier 'no cluster' reading was wrong",
# "Candidate B — no-cluster variant" and even the tool's own LAUNCH-PARTIAL text, three
# distinct ways of losing a write-up turn for saying the right thing).
_ONE = r"(?:1|one|single|none|n/?a|no)"
# One value of a shape field: `1`, `none`, `1x1x1`, `(1,1,1)`, `1, 1, 1`, `single CTA`.
# The tail anchor is what separates a declaration from a sentence that merely starts
# with a 1-shaped number (`cluster=16, one group per 16 CTAs` is not a declaration of
# one); a parenthetical aside after the value is allowed (`none (each CTA independent)`).
_ONE_VALUE = (r"\s*[*`]*\s*(?:\(?\s*" + _ONE + r"(?:\s*[,x×]\s*" + _ONE + r"){0,2}\s*\)?)"
              r"\s*(?:CTAs?|ctas?)?\s*")
_FIELD_TAIL = (_ONE_VALUE
               + r"(?:\s*[（(][^）)]{0,40}[）)])?\s*[*`]*"
               # a trailing aside: `none — see the LAUNCH-OK line` (ASCII hyphens stay legal in it)
               + r"(?:\s*[—–]\s*[^—–\n]{0,60}?)?"
               + r"\s*[*`]*\s*[。.；;，,]?\s*$")
# The cluster field: the key itself — not `cluster_dim_x`, not `clusterDim.x` — and
# whatever separator the writer used (ASCII or full-width colon / equals, a
# parenthesis, a markdown cell). `cluster = 1` and `cluster shape: none` both match;
# the value side is decided in `_cluster_field_value`.
_CLUSTER_KEY = re.compile(
    r"(?:^|[\s|*`()])cluster[_ ]?(?:shape|size|dims?)?(?![_A-Za-z0-9])"
    r"\s*[*`]*\s*[:=(]?\s*[*`]*\s*", re.I)
_NO_CLUSTER_CLAIM = re.compile(
    r"\b(?:no|without|non)[- ]?cluster\b(?!\s*(?:barrier|wait|arrive|broadcast|sync))"
    r"|\bnot\s+clustere?d\b"
    r"|\b(?:does\s+not|doesn't)\s+use\s+(?:a\s+)?cluster\b"
    r"(?!\s*(?:barrier|wait|arrive|broadcast|sync))", re.I)


def _a_one_shaped_cluster_field(line: str) -> bool:
    """``line`` declares the launch's cluster as one CTA / none: the cluster field
    followed by a value that is only 1 / one / none — ``cluster=(1,1,1)``,
    ``cluster shape: none``, ``| cluster | none |``.

    The gate reads the DECLARATION rather than any sentence containing "cluster":
    the brief requires §1 and each candidate's first line to write the shape, so a
    line that says what the cluster IS is the claim to hold the report to. A
    sentence about something else — a ``no cluster barrier`` in the mainloop, an
    alternative candidate that is not the measured launch, a correction, a quoted
    tool warning — is not a denial (a review measured the old line-wide
    pattern bouncing all four of those, burning a write-up turn each)."""
    value = _cluster_field_value(line or "")
    if value is None:
        return False
    return bool(re.match(_FIELD_TAIL, value.strip(), re.I))


def _cluster_field_value(line: str) -> str | None:
    """The value the line gives the cluster field, or None when the line has no
    cluster field. `| cluster | none | other |` yields `none` (the first cell)."""
    m = _CLUSTER_KEY.search(line)
    if not m:
        return None
    rest = line[m.end():].strip()
    if rest.startswith("|"):
        rest = rest[1:]
    rest = rest.strip()
    if "|" in rest:
        rest = rest.split("|")[0]
    if not rest:
        return None
    # The field's value must be an ENTIRE value, not the start of a sentence about
    # something else: `cluster=16, one group per 16 CTAs` is not a 1-shaped declaration.
    return rest if re.match(_FIELD_TAIL, rest, re.I) else None


# An aside that states the shape is being corrected, or that the shape belongs to
# something other than the measured launch (an alternative candidate, another
# kernel, a quoted tool warning), is not a claim to hold the report to. It cannot
# excuse a shape DECLARATION (see _launch_contradictions): a correction that still
# declares `cluster=(1,1,1)` gets bounced until it declares what was measured.
_NO_CLUSTER_EXEMPT = re.compile(
    r"correction|corrected|earlier|previous(?:ly)?|wrong|incorrect|mistaken|erroneous|untrue"
    r"|absence is not evidence|does(?:\s+not|n't)\s+(?:mean|imply|show)|is\s+not\s+evidence"
    r"|LAUNCH-PARTIAL",
    re.I)
# An alternative candidate that states what IT is not. The brief builds every
# candidate on the measured launch, so this is a statement about the candidate, not
# a denial of the measurement.
_ALTERNATIVE_CANDIDATE = re.compile(
    r"\bcandidate\s*[A-Za-z0-9]?\s*[—\-–:][^\n|]{0,80}?\b(?:no|without|non)[- ]?cluster", re.I)


def _launch_cluster_from_log(log: str) -> "tuple[int, int, int] | None":
    """The cluster shape a job log carries: the launch_tools verdict line, or the
    raw ``launch__cluster_dim_*`` metric a probe printed (a page with only ``_x``
    reads as (x,1,1)). None when the log says nothing about it.

    A log with several kernels is read through ``launch_tools.kernel_section``, so
    the metrics come from the LAST profiled kernel rather than whichever helper
    appeared first: a dsa_sparse page carried ``cluster_dim_x=16`` for the
    reference and a helper's ``(0,0,0)`` above it, and taking the first turned the
    gate off (a "no cluster" write-up finished cleanly against a clustered
    reference). The chosen section is used for the LAUNCH-OK line too — a verdict
    line printed for a section that is not this one must not certify it."""
    try:
        from forgekernel.skills.launch_tools import kernel_section
        section = kernel_section(log or "")
    except Exception:  # noqa: BLE001 — a reader must not become the failure
        section = log or ""
    m = _LAUNCH_OK.search(section)
    if m is not None:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    dims = {}
    for d, rx in _CLUSTER_METRIC.items():
        mm = rx.search(section)
        if mm is not None:
            dims[d] = int(mm.group(1))
    if "x" not in dims:
        return None
    return (dims["x"], dims.get("y", 1), dims.get("z", 1))


def _launch_contradictions(ctx: LoopContext, launch: "dict | None") -> list:
    """Report / candidates lines that deny a cluster the NCU launch metrics read
    (cluster > 1 in ``launch``). Each entry quotes the line and its file.

    Two kinds of denial, with different exemptions:

    * a shape DECLARATION (``cluster=(1,1,1)``, ``cluster shape: none``): the line
      says what the launch's cluster IS, contradicted by the measurement. Nothing
      excuses it — a "correction" that still declares one CTA is still the wrong
      declaration, and the fix is to write the measured shape;
    * a prose claim (``no cluster``, ``without cluster``): excused when the line is
      correcting the old reading, is about something else (a barrier), or is an
      alternative candidate stating what it is not.
    """
    if not launch or not launch.get("cluster") or max(launch["cluster"]) <= 1:
        return []
    if ctx.recon_path is None:
        return []
    found = []
    for p in (ctx.recon_path, ctx.recon_path.parent / "candidates.md"):
        try:
            text = p.read_text()
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            declared = _a_one_shaped_cluster_field(line)
            prose = bool(_NO_CLUSTER_CLAIM.search(line)) and not (
                _NO_CLUSTER_EXEMPT.search(line) or _ALTERNATIVE_CANDIDATE.search(line))
            if declared or prose:
                found.append(f"{p.name}:{n}: `{line.strip()[:160]}`")
    return found


def _contradiction_msg(launch: dict, found: list) -> str:
    c = launch["cluster"]
    src = launch.get("source", "job log")
    return (
        "\n\n## Finish refused: RECON-CONTRADICTION cluster\n\n"
        f"The NCU launch metrics read **cluster=({c[0]},{c[1]},{c[2]})** ({src}, round "
        f"{launch.get('round', '?')}), but the report / candidates table has these lines "
        "denying a cluster:\n\n"
        + "\n".join(f"- {f}" for f in found) +
        "\n\nThe cluster shape is a **launch attribute** (`launch__cluster_dim_x/y/z`), not "
        "a grid dimension; a flat grid can just as well be a cluster launch. Rewrite §1 and "
        "the candidates table to the measured cluster (the first line of every candidate "
        "states the cluster shape and CTA count first, and the tile derivation takes it as "
        "its premise), or, if you believe the metric was misread, write a line "
        "`LAUNCH-DISPUTED: <the log line carrying the reading + your criterion>` in the "
        "report."
    )


_LAUNCH_DISPUTED = re.compile(r"^LAUNCH-DISPUTED:\s*\S", re.M)


def _launch_contradictions_open(ctx: LoopContext, launch: "dict | None") -> list:
    """``_launch_contradictions`` unless the report owns the disagreement with a
    ``LAUNCH-DISPUTED:`` line — the teacher may have a reason to distrust the
    metric (a wrong kernel profiled), and that reason belongs on the record,
    not in an unresolvable re-prompt loop."""
    found = _launch_contradictions(ctx, launch)
    if not found or ctx.recon_path is None:
        return found
    try:
        if _LAUNCH_DISPUTED.search(ctx.recon_path.read_text()) is not None:
            return []
    except OSError:
        pass
    return found

# The per-instruction stall-pricing verdict, printed only by stall_tools after its
# self-checks (samples floor, mis-aim duration floor, ledger coverage >= 90%).
# Sedimented from a forensic analysis that used to be a manual between-run step: the
# static channels above transcribe structure, this one prices measured time — the
# only channel that can see a gap dispersed through mainloop interleave quality.
# kernel=.+? for the same demangled-name-with-spaces reason as _SASS_OK.
_STALL_OK = re.compile(r"^STALL-OK kernel=.+? samples=\d+ rows=\d+ coverage=\d", re.M)
_STALL_WAIVED = re.compile(r"^STALL-WAIVED:\s*\S", re.M)

# The per-segment construction-table verdict, printed only by struct_tools
# after its self-checks (barrier-fenced region, decisive MMA/LDSM/STSM family,
# integral warp attribution). The channel that says how a phase is BUILT — the
# SASS histogram says what opcodes exist, this says which warps do a phase on
# what tile granule in what precision with what hand-off. Sedimented from a
# gdn_prefill_fwd blackbox run: the whole gap sat in the solver's
# triangular inverse, the recon inferred its construction ("four 16x16 blocks")
# from the instruction mix, and the student built the wrong shape. Mandatory on
# blackbox runs only, where no source channel can fill it.
_STRUCT_OK = re.compile(
    r"^STRUCT-OK kernel=\S+ regions=\d+ segments=\d+ warps=\S+ precision=\S+ granule=\S+", re.M)
_STRUCT_WAIVED = re.compile(r"^STRUCT-WAIVED:\s*\S", re.M)

# Full-field verdict patterns per channel, for the near-miss audit below.
_VERDICT_PATTERNS = {"SASS": _SASS_OK, "SCHED": _SCHED_OK,
                     "LIVE": _LIVE_OK, "STALL": _STALL_OK, "STRUCT": _STRUCT_OK}

# The third legal evidence state, next to delivered and waived: delivered but
# unrecognized by the acceptance pattern. The teacher quotes the tool's actual
# verdict line after this marker; the report line is machine-checkable, shows on
# the first screen, and reads as an acceptance-format defect for the human audit
# — NOT as a missing channel. Without it, a checker false-negative plus a
# teacher honestly refusing to waive a delivered channel is an unresolvable
# re-prompt loop (observed in a sparse recon).
_DELIVERED_MISMATCH = {n: re.compile(rf"^{n}-DELIVERED-MISMATCH:\s*\S", re.M)
                       for n in ("SASS", "SCHED", "LIVE", "STALL", "ROLE", "PMTL", "STRUCT")}


def _collect_verdict_near_misses(log: str, near: dict) -> None:
    """Record verdict-shaped lines that fail their full-field pattern.

    A line that STARTS like a tool verdict ("SASS-OK ...") but does not satisfy
    the full-field regex is almost never fabrication — it is an acceptance-format
    mismatch between the tool's real output and this file's pattern (observed in
    a sparse recon: a demangled kernel name with spaces). Without this audit the
    driver re-prompts "evidence missing", the teacher truthfully insists it was
    delivered, and the deadlock burns every write-up turn. Recording the nearest
    miss lets the gate message show BOTH sides of the disagreement on the first
    turn instead of never."""
    # Only the tool verdicts whose acceptance regex can lag the tool's real
    # print. ROLE / PMTL are deliberately NOT audited here: a `ROLE-OK` line
    # without `tensor=` is not a format mismatch, it is the exposure column
    # the narrowing rejects on purpose (the seeded tool prints ROLE-PARTIAL for
    # it; only an older copy prints ROLE-OK), and a DELIVERED-MISMATCH row
    # must never close those channels — the driver's own pricing or a waiver do.
    for name, pat in _VERDICT_PATTERNS.items():
        if pat.search(log) is not None:
            near.pop(name, None)
            continue
        m = re.search(rf"^{name}-OK\b.*", log, re.M)
        if m is not None:
            # Last observation wins: a round-1 garbage line (partial echo,
            # truncation) must not shadow the real tool verdict a later round
            # prints — the gate note quotes this line verbatim, and preserving
            # the WRONG side of the disagreement would instruct the teacher to
            # enshrine garbage in the DELIVERED-MISMATCH record.
            near[name] = m.group(0)[:240]
# The ledger body stall_tools brackets for verbatim lifting into prompts.
_STALL_LEDGER_RE = re.compile(
    r"=== STALL LEDGER BEGIN ===\n(.*?)\n=== STALL LEDGER END ===", re.S)
# The per-segment account (PC-keyed join of stall cost onto the schedule): the
# row ledger keys on source lines and the student's scheduling rows (HGMMA /
# DEPBAR / barrier — the @cute.jit decorator has no line) pile into one
# unresolved bucket, invisible to a diagnosis. Segment rows carry `SEG` machine
# lines the aligned comparison table parses.
_SEGMENT_LEDGER_RE = re.compile(
    r"=== SEGMENT LEDGER BEGIN ===\n(.*?)\n=== SEGMENT LEDGER END ===", re.S)

# The PM-sampling phase-timeline verdict (pm_timeline_tools): the stall ledger
# prices WHERE time leaks per instruction; the timeline reads WHEN — phase
# structure a per-PC sum disperses (each phase touches the same PCs). Optional
# channel: probes call it for phase-shaped hypotheses; nothing gates on it.
_PMTL_OK = re.compile(
    r"^PMTL-OK kernel=\S+ series=\d+ stall_series=\d+ stall_samples=\d+", re.M)
_PMTL_WAIVED = re.compile(r"^PMTL-WAIVED:\s*\S", re.M)
_PM_TIMELINE_RE = re.compile(
    r"=== PM TIMELINE BEGIN ===\n(.*?)\n=== PM TIMELINE END ===", re.S)
# The ref's timeline is banked beside its role table: together they are the
# ref's overlap account (WHEN each role waits, and how much of that waiting the
# tensor pipe hides). Static SASS cannot read either — see the ROLE comment.
_REF_PM_TIMELINE_NAME = "ref_pm_timeline.txt"

# The orchestration-dossier verdict (orch_tools): the machine form of §2g —
# role-partitioned regions, barrier identity (named ids + mbarrier objects via
# address def-chains), arrive→wait edges, and the remaining §2g bullets.
# Advisory this generation: recognised and banked, but not in the finish gate —
# a brand-new acceptance pattern hard-gating recon is how the near-miss deadlock
# started, and the near-miss machinery only softens that, it does not unprice
# it. The dossier body is banked for the distill/diagnose steps the same way
# the ref stall ledger is.
_ORCH_OK = re.compile(
    r"^ORCH-OK regions=\d+ loops=\d+ nbar_ids=\d+ mbar_objs=\d+ "
    r"resolved=\d+ edges=\d+ itc_candidates=\d+", re.M)
_ORCH_DOSSIER_RE = re.compile(
    r"=== ORCH DOSSIER BEGIN ===\n(.*?)\n=== ORCH DOSSIER END ===", re.S)
_ORCH_SUMMARY_RE = re.compile(r"^ORCH-SUMMARY-JSON: (\{.*\})$", re.M)
_REF_ORCH_DOSSIER_NAME = "ref_orch_dossier.txt"

# The role exposure ledger (role_exposure_tools): the dossier's role-labelled
# regions x the PC-sampled stall ledger x the PM tensor plateau, joined into
# one per-role table — the teacher had been re-deriving it by hand in a
# verdict probe every meta step. Banked like the dossier (ref side at recon,
# student side off the pricing job) and diffed driver-side from the ROLE-JSON
# core.
#
# ROLE-OK carries the tensor term (the exposure column is real); ROLE-PARTIAL
# is the same table without it (shares only, exposed_ns null). Both are banked
# — the shares are evidence — but only ROLE-OK is the overlap account, and
# only ROLE-OK on the REF closes the recon channel: a dsa_sparse run
# reconnoitred with a bare --set source collection, its ref table came back
# tensor=unavailable, the old single pattern accepted it, and every teacher
# entry of a 17h run diagnosed against a ref exposure column of nulls. The
# blackbox teachers of four runs guessed the gather/MMA overlap wrong at v1;
# this is the measurement they never had.
_ROLE_OK = re.compile(
    r"^ROLE-OK roles=\d+ samples=\d+ resolved=[\d.]+ tensor=[\d.]+", re.M)
# Pre-split tools (a recon tree seeded before this change) print ROLE-OK with
# no tensor field; that line is a partial table, not the account.
_ROLE_ANY = re.compile(r"^ROLE-(?:OK|PARTIAL) roles=\d+ samples=\d+ resolved=[\d.]+", re.M)
_ROLE_WAIVED = re.compile(r"^ROLE-WAIVED:\s*\S", re.M)
# The two overlap channels (ROLE / PMTL) are checked against the BANKED
# evidence rather than a per-log flag: the driver's own pricing job delivers
# them, and a later ROLE-PARTIAL must not shadow an earlier ROLE-OK. They have
# no near-miss / DELIVERED-MISMATCH state (see _collect_verdict_near_misses):
# a "ROLE-OK ..." line without the tensor field is the pre-split tool's output,
# i.e. the exposure column the narrowed pattern rejects on purpose.
_ROLE_BLOCK_RE = re.compile(
    r"=== ROLE EXPOSURE BEGIN ===\n(.*?)\n=== ROLE EXPOSURE END ===", re.S)
_REF_ROLE_EXPOSURE_NAME = "ref_role_exposure.txt"
_STUDENT_ROLE_EXPOSURE_NAME = "student_role_exposure.txt"
_REF_PRICE_JOB_LOG_NAME = "ref_price_job.log"   # the ref pricing attempts: see _REF_PRICE_MARKER_NAMES
_STUDENT_ORCH_DOSSIER_NAME = "student_orch_dossier.txt"        # tree under judgment (HEAD)
_STUDENT_BEST_ORCH_DOSSIER_NAME = "student_orch_dossier_best.txt"  # best-so-far KEEP


def _orch_summary(dossier_text: str | None) -> dict | None:
    """The comparable core a banked dossier carries as its ORCH-SUMMARY-JSON
    line; None when the text predates the line or is unparsable."""
    if not dossier_text:
        return None
    m = _ORCH_SUMMARY_RE.search(dossier_text)
    if m is None:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


@dataclass
class PricedTrees:
    """What one teacher entry priced: the tree under judgment (HEAD) and, when
    it is a different commit, the best-so-far KEEP.

    ``ledger``/``segments`` come from the best tree when one exists (the
    ledger rules in the diagnose prompt speak of "student best"), else from
    HEAD. ``head_dossier``/``best_dossier`` are the orchestration dossiers of
    the respective trees; when HEAD is the best, both name the same text."""
    head_sha: str | None = None
    best_sha: str | None = None
    ledger: str | None = None
    segments: str | None = None
    head_dossier: str | None = None
    best_dossier: str | None = None
    roles: str | None = None          # role exposure block of the ledger-source tree
    # Which tree the ledger/segments/roles were priced on ("best" or "HEAD"):
    # the diagnose labels say "student best" and must not when there is no best.
    ledger_tree: str = "best"
    # Pricing jobs that failed or came back without their tables, one line each.
    # Two jobs can half-fail; the surviving reading then rides into the prompt
    # under labels written for the both-succeeded case unless the gap is named.
    missing: list = field(default_factory=list)

    @property
    def head_is_best(self) -> bool:
        return bool(self.best_sha) and self.best_sha == self.head_sha


def _price_student_trees(ctx: LoopContext, records) -> PricedTrees:
    """Price the trees a teacher entry must read: best (ledger source) and the
    tree under judgment (HEAD) — one job when they coincide, two when not.

    HEAD is the tree the stop was decided on: at first_green it is the M0 that
    just closed, at plateau the segment's last KEEP. Its dossier is what an
    audit needs (did the rebuild carry the contract the previous patch wrote?)
    and the best-so-far dossier is what the plateau ledger already priced; the
    diff between the two says what the construction actually changed
    structurally. A missing best (pre-green plateau) prices HEAD alone, as
    before."""
    be = ctx.backend
    # the same tree the rollback point and the stall pricing use: the best
    # KEEP on the active machine, else anywhere (a best from another machine is
    # a verified kernel the student is not standing on)
    out = PricedTrees(head_sha=be.head(),
                      best_sha=_best_good_ref_here(records, ctx))
    primary = out.best_sha or out.head_sha
    banks = [_STUDENT_BEST_ORCH_DOSSIER_NAME] if out.best_sha else []
    if not out.best_sha or out.head_is_best:
        banks.append(_STUDENT_ORCH_DOSSIER_NAME)
    out.ledger_tree = "best" if out.best_sha else "HEAD"
    out.ledger, out.segments, dossier, out.roles = _run_stall_pricing(
        ctx, ref=primary, bank_as=tuple(banks),
        label="best" if out.best_sha else "head")

    def short(sha):
        return (sha or "?")[:10]

    if out.ledger is None and dossier is None:
        out.missing.append(f"{out.ledger_tree} {short(primary)}: pricing job failed or its "
                           "self-check did not pass — no ledger, no segment ledger, no dossier, "
                           "no role ledger")
    elif dossier is None:
        out.missing.append(f"{out.ledger_tree} {short(primary)}: dossier unavailable "
                           "(ledger present)")
    elif out.ledger is None:
        out.missing.append(f"{out.ledger_tree} {short(primary)}: ledger self-check failed, "
                           "no ledger (dossier present)")
    if out.best_sha:
        out.best_dossier = dossier
    if not out.best_sha or out.head_is_best:
        out.head_dossier = dossier
    else:
        _, _, out.head_dossier, _ = _run_stall_pricing(
            ctx, ref=out.head_sha, bank_as=(_STUDENT_ORCH_DOSSIER_NAME,),
            label="head")
        if out.head_dossier is None:
            out.missing.append(f"HEAD {short(out.head_sha)}: pricing job failed or its "
                               "self-check did not pass — the tree judged this time has no "
                               "dossier; the ref vs HEAD / best → HEAD faces are absent")
    return out


def _meta_workload_readings(ctx: LoopContext, priced: PricedTrees, label: str) -> str:
    """A group card's other meta workloads at a teacher entry (loop/meta_shapes.py): the
    student's stall / SASS pricing, the reference's (once per run) and the timeline probe
    on each, on the tree the representative's readings were taken on, one job each --
    rendered beside the representative's. Every machine reading (the dossier diff gates,
    the segment table, the timeline decisions) stays on the representative. A dossier
    whose summary matches the representative's on every axis is the same SASS structure
    and is said so, not listed again. '' for a one-workload card."""
    meta = meta_shapes.card_meta_workloads(ctx.workspace)
    if len(meta) < 2 or not ctx.stall_pricing:
        return ""
    sha = priced.best_sha or priced.head_sha
    rep_summary = _orch_summary(priced.best_dossier if priced.best_sha else priced.head_dossier)
    rows = []
    for wl in meta[1:]:
        row: dict = {"wl": wl}
        rows.append(row)
        if ctx.now() > ctx.deadline_epoch - 1800:
            row["skipped"] = ("less than 30 minutes of budget left; no job was bought for "
                              "this workload")
            continue
        ledger, _segs, dossier, roles = _run_stall_pricing(
            ctx, ref=sha, bank_as=(), label=f"{priced.ledger_tree} {wl}", workload=wl)
        summary = _orch_summary(dossier)
        row.update(ledger=ledger, roles=roles)
        if ledger is None and dossier is None:
            row["missing"] = "stall / SASS pricing job failed or its self-check did not pass"
        if summary is not None and rep_summary is not None:
            if summary == rep_summary:
                row["same_as_rep"] = True
            else:
                row["diff_rep"] = _orch_diff_text(rep_summary, summary, names=(meta[0], wl))
        if (ref_log := _ref_workload_pricing(ctx, wl, label)):
            rm = _ROLE_BLOCK_RE.search(ref_log)
            row["ref_roles"] = rm.group(1) if rm else None
            dm = _ORCH_DOSSIER_RE.search(ref_log)
            ref_summary = _orch_summary(dm.group(1)) if dm else None
            if ref_summary is not None and summary is not None:
                row["diff_ref"] = _orch_diff_text(ref_summary, summary)
        if _timeline_on(ctx):
            tl = _run_timeline(ctx, sha, f"{label} {wl}", workload=wl)
            if tl and _timeline_renderable(tl):
                row["timeline"] = _timeline_tables(tl)
    return meta_shapes.workload_readings_block(
        meta[0], rows, meta_shapes.card_workload_axes(ctx.workspace), str(sha or "")[:10],
        rep_roles=priced.roles)


def _ref_workload_pricing(ctx: LoopContext, wl: str, label: str) -> str | None:
    """The reference tree's stall / SASS pricing on one of a group card's other meta
    workloads (``--workload``, before ``--app``): one job per workload on the tree
    _ensure_ref_pricing prices, its log banked as evidence ``ref_price_<wl>.log`` and read
    back at every later entry. None when the run prices no reference tree, the tree's card
    copy does not list the workload, or no usable reading came back.

    Only a log that carries a reading (_ref_workload_usable: a role table or a dossier) is
    banked as the workload's; one that came back without (ROLE-FAIL, a truncated or lost
    archive, a job that failed after launch) is an attempt, kept as
    ``ref_price_<wl>.attempt<n>.log`` under the key it ran on (_ref_pricing_key), and the
    workload is priced again at a later entry -- the rule _ensure_ref_pricing keeps for the
    representative: a log with a verdict that was not a reading is retried only once the
    tree, the launch target or the app spelling changed (the same tree reproduces it); one
    with no verdict at all (the archive lost it) also on an unchanged tree; at most
    _REF_WL_PRICE_ATTEMPTS attempts per workload per run. A job that never ran (NoRunError,
    the job id unmoved) is no attempt, bounded per session
    (_REF_PRICE_MAX_SUBMIT_FAILURES per workload)."""
    if ctx.ref_price_dir is None or ctx.ref_price_backend is None:
        return None
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return None
    path = ev / f"ref_price_{wl}.log"
    if path.is_file():
        try:
            banked = path.read_text()
        except OSError:
            return None
        if _ref_workload_usable(banked):
            return banked
        # banked before this rule (every finished job was): an attempt on an unknown key
        _ref_workload_attempt(ctx, ev, wl, "legacy", banked, label)
        path.unlink(missing_ok=True)
    entry, _named = _ref_price_entry(ctx)
    if (entry is None or not entry.is_file()
            or wl not in meta_shapes.card_meta_workloads(ctx.ref_price_dir)):
        return None
    attempts = _ref_workload_attempts(ev, wl)
    if len(attempts) >= _REF_WL_PRICE_ATTEMPTS:
        return None
    key = _ref_pricing_key(ctx)
    if attempts:
        head = re.match(r"\[ref-price\] key=(\S+) verdict=(\S+)",
                        attempts[-1].read_text(errors="replace"))
        if head and head.group(1) == key and head.group(2) != "none":
            return None   # same tree, a verdict that was no reading: it would repeat itself
    if ctx.ref_wl_price_submit_failures.get(wl, 0) >= _REF_PRICE_MAX_SUBMIT_FAILURES:
        return None
    be = ctx.ref_price_backend
    cmd = f"{_stall_price_cmd(ctx)} --workload {wl}"
    if ctx.ref_price_app:
        cmd += " --app " + " ".join(shlex.quote(a) for a in ctx.ref_price_app)
    head_before = getattr(be, "last_job_id", "")
    try:
        _seed_pricing_toolkits(ctx.ref_price_dir, be)
        be.commit("ref-price: snapshot the tree the pricing job archives")
        branch = getattr(be, "current_branch", lambda: "main")()
        _log(f"{label}/ref-price[{wl}]: pricing the reference tree on workload {wl} "
             f"(attempt {len(attempts) + 1}/{_REF_WL_PRICE_ATTEMPTS})")
        log = be.submit_gpu(branch, cmd)
    except BackendError as e:
        if (getattr(be, "last_job_id", "") not in ("", head_before)
                and not isinstance(e, NoRunError)):
            # it ran and did not come back (a hung reference kernel's poll timeout):
            # the same tree would hang again
            _ref_workload_attempt(ctx, ev, wl, key, f"[ref-price] job failed after launch: {e}\n",
                                  label, verdict="job-failed")
        else:
            ctx.ref_wl_price_submit_failures[wl] = ctx.ref_wl_price_submit_failures.get(wl, 0) + 1
            _log(f"{label}/ref-price[{wl}]: nothing ran ({e}) — not an attempt "
                 f"({ctx.ref_wl_price_submit_failures[wl]}/{_REF_PRICE_MAX_SUBMIT_FAILURES} "
                 f"this session); no reference column for {wl} this entry")
        return None
    from forgekernel.skills.stall_price_run import expand_bundle
    log = expand_bundle(log)
    if not _ref_workload_usable(log):
        _ref_workload_attempt(ctx, ev, wl, key, log, label)
        return None
    try:
        path.write_text(log)
    except OSError as e:
        _log(f"{label}/ref-price[{wl}]: could not persist {path.name}: {e}")
    _bank_evidence(ctx, f"ref-price {wl}")
    return log


# attempts at one non-representative workload's reference pricing per run
# (== len(_REF_PRICE_MARKER_NAMES), the representative's)
_REF_WL_PRICE_ATTEMPTS = 3
_REF_WL_VERDICT_RE = re.compile(r"^(?:STALL|ROLE|ORCH|PMTL)-(?:OK|PARTIAL|FAIL)\b.*$", re.M)


def _ref_workload_usable(log: str) -> bool:
    """A reference pricing log the other-workload section can read: a role table
    (ROLE-OK / ROLE-PARTIAL) or a dossier with a parsable summary."""
    if _ROLE_ANY.search(log) is not None and _ROLE_BLOCK_RE.search(log) is not None:
        return True
    dm = _ORCH_DOSSIER_RE.search(log)
    return dm is not None and _orch_summary(dm.group(1)) is not None


def _ref_workload_attempts(ev: Path, wl: str) -> list:
    """The workload's failed attempts (ref_price_<wl>.attempt<n>.log), oldest first."""
    found = []
    for q in ev.glob(f"ref_price_{wl}.attempt*.log"):
        m = re.fullmatch(rf"ref_price_{re.escape(wl)}\.attempt(\d+)\.log", q.name)
        if m:
            found.append((int(m.group(1)), q))
    return [q for _, q in sorted(found)]


def _ref_workload_attempt(ctx: LoopContext, ev: Path, wl: str, key: str, log: str,
                          label: str, verdict: str | None = None) -> None:
    """Bank one reference pricing attempt on ``wl`` that left no reading. The header
    names the key it ran on and its verdict (``none`` = the log carries no verdict
    line at all: an archive that lost it, retried even on an unchanged tree)."""
    if verdict is None:
        lines = _REF_WL_VERDICT_RE.findall(log)
        verdict = "none" if not lines else _tok_reason(" / ".join(lines))
    n = len(_ref_workload_attempts(ev, wl)) + 1
    dst = ev / f"ref_price_{wl}.attempt{n}.log"
    try:
        dst.write_text(f"[ref-price] key={key} verdict={verdict}\n" + log)
    except OSError as e:
        _log(f"{label}/ref-price[{wl}]: could not persist {dst.name}: {e}")
    _log(f"{label}/ref-price[{wl}]: no reading came back ({verdict[:160]}) — attempt "
         f"{n}/{_REF_WL_PRICE_ATTEMPTS}; no reference column for {wl} this entry"
         + ("" if verdict == "none" else ", retried once the tree changes"))
    _bank_evidence(ctx, f"ref-price {wl} attempt marker")


def _tok_reason(text: str) -> str:
    return re.sub(r"\s+", "_", text.strip())[:300] or "none"


_ORCH_DIFF_RULES = (
    "An axis-by-axis comparison of the machine-read summaries of both sides' SASS dossiers: "
    "loop roles, mbarrier object count / tiers / per-role distribution, ring shape, "
    "NamedBarrier verdict and rendezvous width, cross-role arrive→wait edges, DEPBAR "
    "in-flight depth, reduction placement, supply issue order, cache-policy bits. Reading "
    "it: a `!` row = a structural difference of the blueprint or the build from the "
    "reference (a contract the milestone declares that mismatches here ⇒ the build did not "
    "deliver it, which takes precedence over any intra-segment tuning); a `=` row = the "
    "axis is already isomorphic — do not prescribe anything more on that axis.\n"
    "**Routing rule (hard)**: a `!` row in ref vs HEAD that falls on a contract axis the "
    "milestone already declares is this diagnosis's first candidate locus — an undelivered "
    "structure takes precedence over revising acceptance wording / gate phrasing / A-B "
    "forms; changing the wording will not turn a `!` into a `=`. Only when every declared "
    "axis is `=`, or a `!` has been judged by evidence to be \"structure the reference does "
    "not explain / an environment difference\", may the locus land on the wording layer. "
    "The best→HEAD diff answers \"did this build actually touch the structure the previous "
    "patch asked for\": no change on that axis ⇒ the student did not build it (a "
    "student-side defect: add a KB recipe rather than rewording the contract again); "
    "changed but ref vs HEAD is still `!` ⇒ it was built in the wrong direction (cite the "
    "same axis in both diffs and state verbatim where they differ).\n")


def _orch_diff_text(a: dict, b: dict, names=("ref", "student")) -> str:
    from forgekernel.skills.orch_tools import diff_summaries
    lines: list = []
    diff_summaries(a, b, out=lines.append, names=names)
    return "\n".join(lines)


def _orch_diff_views(ctx: LoopContext, priced: PricedTrees) -> str | None:
    """The dossier diff block for a teacher prompt, or None when no pair of
    summaries is available.

    Up to three faces: ref vs HEAD (what the judged tree still lacks), ref vs
    best (what the best tree lacks; only when best is a different commit), and
    best -> HEAD (what this construction changed structurally). Every side
    must carry a summary line: the ref's from its banked recon-time dossier,
    the student's from the pricing jobs that just ran. Diffs are computed
    driver-side (orch_tools.diff_summaries) — no SASS page needed."""
    ref = _orch_summary(_read_ref_orch_dossier(ctx))
    head = _orch_summary(priced.head_dossier)
    best = (_orch_summary(priced.best_dossier)
            if priced.best_sha and not priced.head_is_best else None)

    def short(sha):
        return (sha or "?")[:10]

    def verdict_note(side: str, summ: dict | None) -> str:
        v = (summ or {}).get("verdict", "ORCH-OK")
        if v == "ORCH-OK":
            return ""
        return (f"**{side} page verdict `{v}`**: the page has no sync sites / no loop region "
                "— none of the barrier topology the blueprint declares was built; that by "
                "itself is the heaviest `!`, ahead of every row below.\n")

    faces: list = []
    if ref is not None and head is not None:
        faces.append((f"ref (recon time) vs HEAD {short(priced.head_sha)} (the tree judged "
                      "this time)",
                      verdict_note("HEAD", head) + _orch_diff_text(ref, head)))
    if ref is not None and best is not None:
        faces.append((f"ref (recon time) vs student best {short(priced.best_sha)}",
                      verdict_note("best", best) + _orch_diff_text(ref, best)))
    if best is not None and head is not None:
        faces.append((f"student best {short(priced.best_sha)} → HEAD "
                      f"{short(priced.head_sha)} (what structure this segment's build / "
                      "rebuild changed; `!` = changed, `=` = untouched)",
                      _orch_diff_text(best, head, names=("best", "HEAD"))))
    gaps = list(priced.missing)
    if ref is None and (head is not None or best is not None):
        gaps.append("ref: recon left no comparable dossier summary — the ref vs HEAD / "
                    "ref vs best faces are absent")
    if not faces and not gaps:
        return None
    block = ("\n## Orchestration dossier diff (static structure side by side — read it "
             "before you judge)\n" + _ORCH_DIFF_RULES)
    if gaps:
        # Named, not silent: a half-failed pricing pair used to hand the teacher
        # one surviving face under the both-succeeded labels.
        block += ("\n### Faces absent this time (a pricing job half-failed — only the "
                  "readable faces remain below; an absent tree must not be treated as read)\n"
                  + "\n".join(f"- {g}" for g in gaps) + "\n")
    for title, text in faces:
        block += f"\n### {title}\n```\n{text[:5000]}\n```\n"
    return block


def _read_ref_orch_dossier(ctx: LoopContext) -> str | None:
    ev = _evidence_dir(ctx)
    if ev is None:
        return None
    try:
        return (ev / _REF_ORCH_DOSSIER_NAME).read_text()
    except OSError:
        return None


def _verdict_evidence(ctx: LoopContext, ok: bool, waiver: "re.Pattern[str]",
                      mismatch: "re.Pattern[str] | None" = None,
                      mismatch_observed: bool = False) -> bool:
    """Delivered (a tool verdict came back from a job), explicitly waived in the
    report, or explicitly owned as delivered-but-unrecognized — silence is not
    one of the options, for any channel.

    The mismatch line only counts when the driver ITSELF recorded a verdict-
    shaped near-miss for the channel this session. Without that condition the
    line is a free bypass weaker than a waiver: a teacher could write
    "SASS-DELIVERED-MISMATCH: whatever" for a channel no job ever produced a
    verdict-shaped line for, and the human audit would read it as a checker
    defect instead of a missing delivery."""
    if ok:
        return True
    if ctx.recon_path is None or ctx.recon_dir is None:
        return True  # nothing configured to hold a finish against
    try:
        report = ctx.recon_path.read_text()
    except OSError:
        return False
    if waiver.search(report) is not None:
        return True
    return (mismatch is not None and mismatch_observed
            and mismatch.search(report) is not None)


def _struct_recon(ctx: LoopContext) -> bool:
    """The construction-table channel is mandatory on blackbox runs: no source,
    no other channel, so the teacher must decode how the reference BUILDS its
    hot phase from the binary. A source-mode run reads the source for that, so the
    channel stays optional there (waived only if the SASS really has no
    barrier-fenced construction)."""
    return ctx.oracle_mode == "blackbox"


def _missing_recon_evidence(ctx: LoopContext, sass_ok: bool, sched_ok: bool,
                            live_ok: bool, stall_ok: bool = True,
                            struct_ok: bool = True,
                            near_miss: dict | None = None) -> list:
    """Channel names whose evidence is neither delivered nor waived."""
    near = near_miss or {}
    missing = []
    if not _verdict_evidence(ctx, sass_ok, _SASS_WAIVED,
                             _DELIVERED_MISMATCH["SASS"], "SASS" in near):
        missing.append("SASS")
    if not _verdict_evidence(ctx, sched_ok, _SCHED_WAIVED,
                             _DELIVERED_MISMATCH["SCHED"], "SCHED" in near):
        missing.append("SCHED")
    if not _verdict_evidence(ctx, live_ok, _LIVE_WAIVED,
                             _DELIVERED_MISMATCH["LIVE"], "LIVE" in near):
        missing.append("LIVE")
    if ctx.stall_pricing and not _verdict_evidence(
            ctx, stall_ok or _ref_ledger_banked(ctx), _STALL_WAIVED,
            _DELIVERED_MISMATCH["STALL"], "STALL" in near):
        missing.append("STALL")
    if _struct_recon(ctx) and not _verdict_evidence(
            ctx, struct_ok, _STRUCT_WAIVED,
            _DELIVERED_MISMATCH["STRUCT"], "STRUCT" in near):
        missing.append("STRUCT")
    # The ref's overlap account: role table WITH tensor, and the phase timeline.
    # Both come off one pricing job (stall_price_run.py --pm) the driver runs on
    # the ref tree itself; the teacher only ever sees them as banked files.
    # the overlap channels have no mismatch state: banked by the driver's own
    # pricing, or waived (see _collect_verdict_near_misses)
    if ctx.stall_pricing and not _verdict_evidence(
            ctx, _ref_overlap_banked(ctx), _ROLE_WAIVED):
        missing.append("ROLE")
    if ctx.stall_pricing and not _verdict_evidence(
            ctx, _ref_timeline_banked(ctx), _PMTL_WAIVED):
        missing.append("PMTL")
    return missing


def _sass_evidence(ctx: LoopContext, sass_ok: bool,
                   near_miss: dict | None = None) -> bool:
    """SASS was either delivered (a SASS-OK verdict came back from a job) or
    explicitly waived in the report — silence is not one of the options.

    A blackbox attention-fwd run shipped with the opcode histogram missing after two
    probes failed on complementary halves (wrong parser format, wrong kernel), and
    its own report names the single round that would have closed four open gaps.
    Prose discipline in the prompt did not force that round; this check does.
    """
    return _verdict_evidence(ctx, sass_ok, _SASS_WAIVED,
                             _DELIVERED_MISMATCH["SASS"],
                             "SASS" in (near_miss or {}))


_SASS_GATE_MSG = (
    "\n\n## Finish refused: SASS evidence missing\n\n"
    "The recon report and the candidates table both exist, but this reconnaissance had no "
    "successful SASS extraction (no `SASS-OK` verdict line in any job output), and the "
    "report has no `SASS-WAIVED:` waiver line either. The opcode histogram is the only "
    "channel that can settle the (bN,stages) factorization, the register split and the "
    "consumer organization — **a silent absence is not allowed**. Pick one:\n\n"
    "1. **Deliver**: write a probe that does `import sass_tools` (the driver has seeded it "
    "into `notes/`) and runs `sass_tools.analyze(sass_text, profiled_kernel, launch_pairs)` "
    "through kernel selection → parsing → self-check; when everything passes it prints the "
    "`SASS-OK` line. End with `RECON_REQUEST: notes/<name>.py`.\n"
    "2. **Waive**: if SASS is genuinely unavailable (permission denied, both the runtime "
    "and the static channel tried), add a line `SASS-WAIVED: <the concrete reason and the "
    "channels tried>` to the recon report, then end the turn."
)

_SCHED_LIVE_GATE_MSG = (
    "\n\n## Finish refused: {names} evidence missing\n\n"
    "What is missing is the mainloop's **schedule-order transcription** (SCHED) and/or its "
    "**register live-range bill** (LIVE) — measurement showed this pair is indispensable: "
    "with every count-type footprint delivered, the milestone still could not say \"which "
    "chain is in flight across which elementwise segment\" or \"how many registers this "
    "coexistence costs\", and M1 died on exactly those two questions; while delivering the "
    "schedule order without the live ranges induced a port that measured 6% slower with "
    "10.8M spills (an A/B measurement of a schedule-order port). Pick one:\n\n"
    "1. **Deliver**: write a probe that does `import sched_live_tools` (the driver has "
    "seeded it into `notes/` together with sass_tools) and, once it has the SASS page text, "
    "calls `sched_live_tools.report(sass_text)` — it produces both deliverables in one go "
    "and self-checks them, printing the two lines `SCHED-OK` and `LIVE-OK` when everything "
    "passes. It can share one collection with the SASS extraction. "
    "End with `RECON_REQUEST: notes/<name>.py`.\n"
    "2. **Waive**: when genuinely unavailable, add `SCHED-WAIVED: <reason>` / "
    "`LIVE-WAIVED: <reason>` (whichever is missing) to the recon report, then end the "
    "turn.\n\n"
    "**The SCHED channel has three syntaxes by MMA instruction family; the tool picks one "
    "after its own census**: Hopper wgmma (HGMMA chains + `WARPGROUP.DEPBAR.LE` depth, "
    "verdict line `SCHED-OK loop=… hgmma_runs=N depbars=N`); Blackwell tcgen05 (chains "
    "grouped by TMEM accumulator operand, hand-off objects = `UTCBAR` commits and mbarrier "
    "waits, verdict line `SCHED-OK isa=tcgen05 loop=… utcmma_runs=N commits=N "
    "mbar_waits=N …`, with the page's HMMA side path transcribed too); mma.sync/HMMA "
    "(chains grouped by accumulator base, hand-off objects = `DEPBAR.LE SB`, named/cluster "
    "barriers, mbarriers, verdict line `SCHED-OK isa=mma_sync loop=… mma_runs=N "
    "barriers=N ldsm=N …`). The driver accepts all three verdict lines; the hand-off "
    "object in force before each elementwise segment is in `handoff_before_each_ew=[...]`, "
    "and the report must read it per segment as \"this segment runs after which commit / "
    "wait\". If your notes/ still has the old tool (printing only `SCHED-NOTAPPLICABLE`), "
    "that also counts as delivery, but write the architecture's own sync topology into "
    "`SCHED-WAIVED: <...>`. The LIVE half is architecture-independent; deliver it as "
    "usual.\n"
)


_STALL_GATE_MSG = (
    "\n\n## Finish refused: STALL evidence missing\n\n"
    "What is missing is the reference kernel's **per-instruction stall pricing ledger** — "
    "aggregate signatures cannot locate a gap dispersed across the quality of the "
    "mainloop's interleaving: an observed failure mode is meta picking two loci in a row, "
    "each worth only ~1pp, and 10 hours of building all rolled back; whereas one "
    "per-instruction pricing pass takes tens of seconds to pin 30% of the stalls on a "
    "few wait_group sites. Pick one:\n\n"
    "1. **Deliver**: write a probe that collects a `.ncu-rep` with PC sampling using "
    "`--set source` (or `--set full`) (it can share one collection with the SASS "
    "extraction), then does `import stall_tools` (the driver has seeded it into `notes/`) "
    "and calls `stall_tools.price_report(rep_path)` — when the self-checks (sample count, "
    "main-kernel calibration, ledger coverage ≥90%) all pass it prints the `STALL-OK` "
    "line; right after, **in the same probe, also call "
    "`stall_tools.segment_ledger(rep_path)`** — the segment ledger (the PC-keyed segment "
    "table) is printed only by it, and without that call the reference side of "
    "diagnose's same-position segment comparison is always absent. "
    "End with `RECON_REQUEST: notes/<name>.py`.\n"
    "2. **Waive**: when genuinely unavailable (e.g. PC sampling is empty on this image and "
    "that was verified by measurement), add `STALL-WAIVED: <the concrete reason and the "
    "channels tried>` to the recon report, then end the turn.\n"
)


def _near_miss_notes(missing: list, near_miss: dict | None) -> list:
    """One reconciliation note per missing channel whose logs carried a
    verdict-shaped line the full-field pattern rejected — both sides of the
    disagreement, verbatim, plus the third legal closing state."""
    notes = []
    for ch in missing:
        if near_miss and ch in near_miss:
            notes.append(
                f"⚠ {ch} reconciliation: the job log has a line that starts with "
                f"`{ch}-OK` but fails the driver's full-field check:\n"
                f"    `{near_miss[ch]}`\n"
                f"    If this is the tool's real output, the gap is in **reconciling the "
                f"acceptance check with the print format**, not in the delivery — do not "
                f"write a waiver line; instead add a line to the recon report\n"
                f"    `{ch}-DELIVERED-MISMATCH: <paste the line above verbatim>`\n"
                f"    It is a legal finishing state parallel to a waiver; human review "
                f"treats it as a defect of the acceptance check.")
    return notes


_STRUCT_GATE_MSG = (
    "\n\n## Finish refused: the reference's per-segment construction table is missing\n\n"
    "What is missing is the **per-segment construction table of the reference's hot "
    "region** — how many warps are working, how large a block each handles, what internal "
    "precision, how adjacent segments hand off. The histogram / schedule order / stall "
    "ledger cannot say \"how this segment is built\": an observed failure mode is a gap "
    "sitting entirely inside one warp-cooperative construction (e.g. a triangular "
    "inverse), the recon **inferring** the block shape from the instruction order, the "
    "milestone and the certified KB copying the inference, and the student building the "
    "whole run to the wrong shape; whereas the same SASS + per-PC execution counts "
    "**mechanically read out** the real block granule, number of doubling levels, internal "
    "precision and reduction method. Pick one:\n\n"
    "1. **Deliver**: write a probe (it can use a `.ncu-rep` collected with "
    "`--set source`, sharing one collection with the SASS extraction) that passes the "
    "source page as CSV to `struct_tools.source_csv_to_sass_and_counts()` to get the SASS "
    "text + per-instruction execution counts, then does `import struct_tools` and calls "
    "`struct_tools.report(sass_text, counts, kernel=..., ncta=<grid.x>)` — when the "
    "self-checks (named-barrier segmentation, determinate families, integral warp "
    "reductions) all pass it prints the `STRUCT-OK` line; the report must include the "
    "construction table in full and spell out **which structural decision each segment "
    "of the hot region is building**. End with `RECON_REQUEST: notes/<name>.py`.\n"
    "2. **Waive**: the only legal condition is that the SASS really has no construction "
    "segment fenced by named barriers (no `BAR.SYNC <id>, <threads>`) — add "
    "`STRUCT-WAIVED: <the concrete reason>` to the recon report, then end the turn. "
    "Without that line the finish is bounced."
)


def _evidence_gate_msg(missing: list, near_miss: dict | None = None,
                       ctx: "LoopContext | None" = None) -> str:
    parts = []
    if "SASS" in missing:
        parts.append(_SASS_GATE_MSG)
    rest = [m for m in missing if m not in ("SASS", "STALL", "ROLE", "PMTL", "STRUCT")]
    if rest:
        parts.append(_SCHED_LIVE_GATE_MSG.format(names="/".join(rest)))
    if "STALL" in missing:
        parts.append(_STALL_GATE_MSG)
    if "STRUCT" in missing:
        parts.append(_STRUCT_GATE_MSG)
    overlap = [m for m in missing if m in ("ROLE", "PMTL")]
    if overlap:
        note, option1 = (_ref_price_status(ctx, overlap) if ctx is not None
                         else ("(no ctx)", "waive"))
        parts.append(_OVERLAP_GATE_MSG.format(names="/".join(overlap),
                                              note=note, option1=option1))
    notes = _near_miss_notes(missing, near_miss)
    if notes:
        parts.append("\n\n" + "\n\n".join(notes))
    return "".join(parts)


_OVERLAP_GATE_MSG = (
    "\n\n## Finish refused: the reference overlap ledger is missing ({names})\n\n"
    "The reference's role exposure ledger must carry the tensor duty cycle "
    "(`ROLE-OK … tensor=<pct>`), and the phase timeline must be signed off with "
    "`PMTL-OK` — these two are the **only** channels that show \"how the reference hides "
    "its supply behind the MMA\"; static SASS cannot read it (an observed failure mode: "
    "with every exposed value on the reference side null, several consecutive blackbox "
    "runs guessed wrong at the same spot). The driver produces them by running the frozen "
    "recipe `python3 notes/stall_price_run.py --pm` on the recon tree (the PM sampling "
    "interval is chosen automatically from the reference's duration, 8192 / 2048 cycles, "
    "and can be set with loop.toml `[loop] ref_price_pm_interval`); you do not send a "
    "probe for them.\n\n**Status**: {note}.\n\nPick one:\n\n"
    "1. **Deliver**: {option1}.\n"
    "2. **Waive**: add `{names}-WAIVED: <the concrete reason and the channels tried>` to "
    "the recon report (one line per channel), then end the turn. After a waiver the "
    "driver no longer runs pricing for that channel; the distillation step is told "
    "explicitly that the reference overlap ledger does not exist, and the milestone's "
    "[SYNC:]/[GATHER:] rulings must be written as candidates, not as settled."
)


# Every module the driver seeds into a tree's notes/ — the reconnaissance tree
# (_seed_sass_tools) and the student / ref-pricing trees (_seed_pricing_toolkits).
# On a closed-source run scripts/blackbox-guard.sh scans notes/ before EVERY job,
# so a forbidden token in any one of these refuses every job of that tree, not
# just the probe that imports it: timeline_probe.py's stdlib byte-compile import
# did exactly that to two Blackwell reference trees, cutting a
# reconnaissance short. tests/test_blackbox_guard.py runs the guard over
# all of them.
RECON_SEEDED_MODULES = (
    "forgekernel.skills.sass_tools", "forgekernel.skills.sched_live_tools",
    "forgekernel.skills.stall_tools", "forgekernel.skills.sass_census",
    "forgekernel.skills.pm_timeline_tools", "forgekernel.skills.pm_timeline_run",
    "forgekernel.skills.orch_tools", "forgekernel.skills.role_exposure_tools",
    "forgekernel.skills.stall_price_run", "forgekernel.skills.struct_tools",
    "forgekernel.skills.launch_tools",
)
PRICING_SEEDED_MODULES = (
    "forgekernel.skills.stall_tools", "forgekernel.skills.stall_price_run",
    "forgekernel.skills.sass_tools", "forgekernel.skills.sched_live_tools",
    "forgekernel.skills.pm_timeline_tools", "forgekernel.skills.pm_timeline_run",
    "forgekernel.skills.orch_tools", "forgekernel.skills.role_exposure_tools",
    "forgekernel.skills.orch_census_run", "forgekernel.skills.lever_pricing_run",
    "forgekernel.skills.sass_census", "forgekernel.skills.timeline_probe",
    "forgekernel.skills.job_env_check",
)


def _seed_sass_tools(ctx: LoopContext, rbe) -> None:
    """Copy the vendored SASS toolkit into the reconnaissance tree and commit it.

    Probe delivery is ``git archive`` of the committed tree, so the toolkit must be
    a committed file *inside* that tree; and it must be the driver that puts it
    there, because teacher-written one-shot parsers in two probes each
    got a different half wrong. ``python3 notes/<probe>.py`` puts ``notes/`` on
    ``sys.path``, so probes reach it with a bare ``import sass_tools``.
    """
    if ctx.recon_dir is None:
        return
    import importlib
    changed = []
    # role_exposure_tools / stall_price_run ride along: the ref's overlap
    # account is priced in this tree with the frozen recipe, and recon_prompt
    # §2f/§2g tell the teacher both are seeded (a teacher once had to
    # copy them in by hand). launch_tools: §1's cluster shape off the NCU
    # launch metrics, never off the grid shape.
    for vendor in map(importlib.import_module, RECON_SEEDED_MODULES):
        text = Path(vendor.__file__).read_text()
        name = Path(vendor.__file__).name
        dst = ctx.recon_dir / "notes" / name
        if dst.exists() and dst.read_text() == text:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        changed.append(name)
    # The closed-source run's in-job tripwire rides the same way, for the same
    # reason: the ref tree's gpu_entry.sh is what runs on the node, and the node
    # only ever sees `git archive` of that tree — the harness repo's scripts/ is
    # not there. Hand-copying the guard's body into each gpu_entry.sh let the
    # copies inherit each other's bugs. Seeded, gpu_entry.sh
    # sources `$W/notes/blackbox-guard.sh` and carries no logic of its own.
    if ctx.oracle_mode == "blackbox":
        src = _blackbox_guard_source()
        if src is not None:
            text = src.read_text()
            dst = ctx.recon_dir / "notes" / BLACKBOX_GUARD_NAME
            if not (dst.exists() and dst.read_text() == text):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(text)
                changed.append(BLACKBOX_GUARD_NAME)
        else:
            _log(f"meta/recon: {BLACKBOX_GUARD_NAME} not found beside this checkout "
                 f"— the ref tree's gpu_entry.sh must carry its own copy")
    if changed:
        rbe.commit(f"recon: vendor {', '.join(changed)} (driver-seeded)")


BLACKBOX_GUARD_NAME = "blackbox-guard.sh"


def _blackbox_guard_source() -> "Path | None":
    """scripts/blackbox-guard.sh of the checkout this driver runs from. The harness
    runs from its source tree (paths.repo_root is the same resolution); under a
    wheel install there is no scripts/ and the seeding logs and skips."""
    p = Path(__file__).resolve().parents[3] / "scripts" / BLACKBOX_GUARD_NAME
    return p if p.is_file() else None


def _save_recon_log(ctx: LoopContext, script_path: str, round_no: int,
                    log: str) -> None:
    """Persist each probe's fed-back output next to the probe that produced it.

    Findings once had to be re-fished from the job logs at write-up time
    because no job's output had been written to any file.
    Summarising into the report stays the teacher's job; keeping the raw evidence
    is now the driver's.
    """
    if ctx.recon_dir is None:
        return
    dst = ctx.recon_dir / "notes" / f"{Path(script_path).stem}.run{round_no}.log"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(log)
    except OSError as e:
        _log(f"meta/recon: could not persist job log to {dst}: {e}")


_REF_STALL_LEDGER_NAME = "ref_stall_ledger.txt"
_REF_SEGMENT_LEDGER_NAME = "ref_segment_ledger.txt"
_REF_STRUCT_TABLE_NAME = "ref_struct_table.txt"
_STRUCT_BEGIN = "=== STRUCT TABLE BEGIN ==="
_STRUCT_END = "=== STRUCT TABLE END ==="


def _save_ref_struct_table(ctx: LoopContext, log: str) -> None:
    """Keep the ref's construction table where the distill step can read it.

    The STRUCT channel's table is the only place the reference's per-phase
    construction (warps / tile granule / internal precision / hand-off) is
    written down on blackbox runs. Banked verbatim off the job log like the
    stall ledger, so the milestone can cite the ref's build rather than the
    teacher's paraphrase of it.
    """
    if ctx.recon_path is None:
        return
    m = re.search(r"=== STRUCT TABLE BEGIN ===\n(.*?)\n=== STRUCT TABLE END ===",
                  log, re.S)
    if m is None:
        return
    try:
        (ctx.recon_path.parent / _REF_STRUCT_TABLE_NAME).write_text(m.group(1))
    except OSError as e:
        _log(f"meta/recon: could not persist {_REF_STRUCT_TABLE_NAME} to {e}")


def _read_ref_struct_table(ctx: LoopContext) -> str | None:
    if ctx.recon_path is None:
        return None
    try:
        return (ctx.recon_path.parent / _REF_STRUCT_TABLE_NAME).read_text()
    except OSError:
        return None


EVIDENCE_FILE_NAMES = (
    _REF_STALL_LEDGER_NAME, _REF_SEGMENT_LEDGER_NAME, _REF_ORCH_DOSSIER_NAME,
    _REF_ROLE_EXPOSURE_NAME, _REF_PM_TIMELINE_NAME, _STUDENT_ROLE_EXPOSURE_NAME,
    _STUDENT_ORCH_DOSSIER_NAME, _STUDENT_BEST_ORCH_DOSSIER_NAME,
    # the attempt markers: inherited, they would silence a fresh run's ref pricing
    "ref_price_job.log", "ref_price_job.2.log", "ref_price_job.3.log")


def _evidence_dir(ctx: LoopContext, create: bool = False) -> Path | None:
    """The directory pricing evidence is banked in and read back from: the
    configured one, else (older callers / tests) the recon report's directory,
    else None — a ctx with neither has no channel for it."""
    ev = ctx.evidence_dir or (ctx.recon_path.parent if ctx.recon_path else None)
    if ev is not None and create:
        try:
            ev.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _log(f"meta: could not create evidence dir {ev}: {e}")
            return None
    return ev


def _bank_evidence(ctx: LoopContext, why: str) -> str:
    """Commit what the loop just wrote into the reconnaissance tree.

    Everything the loop banks there — ref ledgers, dossiers, timelines, lever
    prices — is written by the driver and committed by nobody. Two things break
    when it stays uncommitted: a GPU job ships a COMMIT, so the next job does not
    carry the evidence the teacher just paid for; and every run's preflight
    refuses to start on a dirty reference tree, so the loop wedges its own
    restart and needs a human to commit before it can resume (mla_paged_decode
    and attn_fwd runs both lost hours to exactly this).

    Bookkeeping never ends a run: any failure is logged and swallowed. Returns
    the reason it did nothing ("" when it committed).
    """
    rbe = getattr(ctx, "recon_backend", None)
    if rbe is None or _evidence_dir(ctx) is None:
        return "no reconnaissance tree"
    try:
        # `dirty` is optional: a backend without it (test fakes, older backends) still
        # banks, it just cannot skip a no-op commit first.
        probe = getattr(rbe, "dirty", None)
        if callable(probe) and not probe():
            return "nothing to bank"
        rbe.commit(f"recon: bank {why}")
    except Exception as e:     # noqa: BLE001 — bookkeeping must not end a run
        _log(f"meta/recon: could not bank evidence ({why}): {e}")
        return str(e)
    _log(f"meta/recon: banked evidence in the reconnaissance tree ({why})")
    return ""


def _save_ref_stall_ledger(ctx: LoopContext, log: str) -> None:
    """Keep the ref's priced ledger where the diagnose step can re-read it.

    The recon logs already persist next to their probes, but the diagnose step
    needs THIS table specifically and must not go fishing through probe logs for
    it — the same reason the recon report exists at a fixed path. The segment
    account is banked the same way when the log carries one: it is the ref side
    of the diagnose step's aligned comparison table.
    """
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    # The two tables have separate self-checks and are banked separately: the
    # row ledger keys on source lines and needs STALL-OK (a library kernel
    # without LINEINFO fails its coverage floor — deep_gemm ref: 85.6%
    # < 90%), the segment account keys on PCs and needs only its own JOIN-OK.
    # Tying both to STALL-OK threw away a clean ref segment account.
    for pattern, name, ok in (
            (_STALL_LEDGER_RE, _REF_STALL_LEDGER_NAME, _STALL_OK.search(log) is not None),
            (_SEGMENT_LEDGER_RE, _REF_SEGMENT_LEDGER_NAME, True)):
        m = pattern.search(log)
        if m is None or not ok or (name == _REF_SEGMENT_LEDGER_NAME
                                   and "JOIN-OK" not in m.group(1)):
            continue
        dst = ev / name
        try:
            dst.write_text(m.group(1))
        except OSError as e:
            _log(f"meta/recon: could not persist {name} to {dst}: {e}")


def _save_ref_orch_dossier(ctx: LoopContext, log: str) -> None:
    """Bank the ref's orchestration dossier next to the stall ledger.

    Same contract as _save_ref_stall_ledger: the distill step's [SYNC:]/
    [GATHER:] adjudications and a later student-vs-ref dossier diff need THIS
    table at a fixed path, not a fishing trip through probe logs."""
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    pick = _pick_ref_dossier(log)
    if pick is None:
        return
    text, note = pick
    if note:
        _log(f"meta/recon: ref dossier {note}")
    dst = ev / _REF_ORCH_DOSSIER_NAME
    # A later probe log carrying only bystander kernels' blocks must not
    # replace a banked bigger kernel: a gdn_prefill_bwd run banked the
    # main kernel (9112 instrs) off one probe, silently, and the next probe's
    # aux-kernel blocks (prepare_h, 6232 instrs) overwrote it — the blueprint
    # for the whole run would have been a 20% side kernel. Same preference as
    # _pick_ref_dossier within one log: an OK verdict, then the largest parsed
    # kernel, wins across logs too; an equal or larger kernel still supersedes.
    kept = _banked_dossier_header(dst)
    new = _banked_dossier_header_text(text)
    if (kept is not None and new is not None and kept[2] == "OK"
            and kept[1] > new[1]):
        _log(f"meta/recon: ref dossier kept banked kernel={kept[0]} "
             f"instrs={kept[1]} over the smaller kernel={new[0]} "
             f"instrs={new[1]} from this log")
        return
    try:
        dst.write_text(text)
    except OSError as e:
        _log(f"meta/recon: could not persist {_REF_ORCH_DOSSIER_NAME} "
             f"to {dst}: {e}")


_BANKED_HEADER_RE = re.compile(
    r"^# banked: block \d+/\d+ kernel=(.+?) instrs=(\d+) verdict=(\S+)")


def _banked_dossier_header_text(text: str) -> tuple[str, int, str] | None:
    """(kernel, instrs, verdict) from a banked dossier's ``# banked:`` line.

    Read from the FIRST LINE only. The kernel name in that line is ncu's fully
    demangled signature, and a CUTLASS or Triton kernel's runs to hundreds of
    characters — a fixed character window would push ``instrs=``/``verdict=`` out of
    range, both sides would parse as None, and the guard against overwriting a bigger
    kernel's dossier would silently fall back to overwriting unconditionally. The
    writer always emits this line first, so the first line is the whole question.
    """
    m = _BANKED_HEADER_RE.match(text.split("\n", 1)[0])
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def _banked_dossier_header(path: Path) -> tuple[str, int, str] | None:
    try:
        return _banked_dossier_header_text(path.read_text())
    except OSError:
        return None


_DOSSIER_INSTRS_RE = re.compile(r"^\[parse\] (\d+) instructions", re.M)
_DOMINANT_LAUNCH_RE = re.compile(
    r"^\[stall_price\] dominant launch index=(\S+) name=(.+)$", re.M)


def _pick_ref_dossier(log: str) -> "tuple[str, str] | None":
    """The dossier block a reference log should be banked from, with its
    kernel identity stamped on top.

    A log can carry several blocks — a chain replay prices several kernels,
    a retried collection prints twice — and the FIRST one is not "the" ref:
    banking it made a bystander kernel's contract axes the blueprint for the
    whole run (every later `!` a phantom). Preference: ORCH-OK verdict, then
    the largest parsed kernel; ties keep the later block (a retry supersedes).
    The banked text is prefixed with one ``# banked:`` line naming the block,
    its instruction count, its verdict and the dominant-launch kernel name the
    pricing run printed before it (or the STALL-OK kernel), so a later reader
    can check the blueprint is the kernel under judgment. Returns
    (text, note) — note is "" when the log carried exactly one block."""
    blocks = list(_ORCH_DOSSIER_RE.finditer(log))
    if not blocks:
        return None
    stall = _STALL_OK.search(log)
    stall_kernel = None
    if stall is not None:
        km = re.match(r"STALL-OK kernel=(.+?) samples=", stall.group(0))
        stall_kernel = km.group(1) if km else None

    def describe(m):
        body = m.group(1)
        im = _DOSSIER_INSTRS_RE.search(body)
        instrs = int(im.group(1)) if im else 0
        summ = _orch_summary(body) or {}
        verdict = summ.get("verdict") or ("ORCH-OK" if _ORCH_OK.search(
            log[m.end():m.end() + 400]) else "?")
        name = None
        for dm in _DOMINANT_LAUNCH_RE.finditer(log, 0, m.start()):
            name = dm.group(2).strip()      # the nearest one before the block
        return instrs, verdict, name or stall_kernel or "?"

    described = [(m, *describe(m)) for m in blocks]
    best = max(range(len(described)),
               key=lambda i: (described[i][2] == "ORCH-OK", described[i][1], i))
    m, instrs, verdict, name = described[best]
    # The verdict is stamped without its ORCH- prefix: the banked text is the
    # dossier BODY, and readers assert the trailing verdict line stays outside.
    header = (f"# banked: block {best + 1}/{len(blocks)} kernel={name} "
              f"instrs={instrs} verdict={verdict.replace('ORCH-', '')}\n")
    note = ""
    if len(blocks) > 1:
        others = ", ".join(f"block {i + 1}: {d[3]} instrs={d[1]} {d[2]}"
                           for i, d in enumerate(described) if i != best)
        note = (f"banked block {best + 1}/{len(blocks)} (kernel={name}, "
                f"instrs={instrs}, {verdict}); not banked: {others}")
    return header + m.group(1), note


def _save_ref_role_exposure(ctx: LoopContext, log: str) -> None:
    """Bank the ref's role exposure ledger beside its dossier and segment
    account — the ref side of the diagnose step's per-role comparison.

    A partial table (no tensor term) is banked only where no full one is: a
    later --set-source probe must not downgrade the account a pricing job
    delivered."""
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    pick = _pick_role_block(log)
    if pick is None:
        return
    block, full = pick
    if not full and _ref_overlap_banked(ctx):
        return
    dst = ev / _REF_ROLE_EXPOSURE_NAME
    try:
        dst.write_text(block)
    except OSError as e:
        _log(f"meta/recon: could not persist {_REF_ROLE_EXPOSURE_NAME} "
             f"to {dst}: {e}")


def _pick_role_block(log: str) -> "tuple[str, bool] | None":
    """(body, carries-tensor) of the LAST role block in a log, judged by the
    verdict line that follows that block — a recon probe may print several
    (a rig round-trip, then the real table) and a whole-log ROLE-OK search
    would let the first block ride on a later block's verdict."""
    blocks = list(_ROLE_BLOCK_RE.finditer(log))
    if not blocks:
        return None
    m = blocks[-1]
    tail = log[m.end():m.end() + 400]
    full = _ROLE_OK.search(tail) is not None
    if not full and _ROLE_ANY.search(tail) is None:
        # no verdict after the block at all (clipped log): trust the core
        from forgekernel.skills.role_exposure_tools import parse_core
        core = parse_core(m.group(1))
        full = bool(core) and core.get("tensor_active") is not None
    return m.group(1), full


def _save_ref_pm_timeline(ctx: LoopContext, log: str) -> None:
    """Bank the ref's PM phase timeline (only on PMTL-OK — a FAIL block is a
    diagnostic, not a timeline)."""
    ev = _evidence_dir(ctx, create=True)
    if ev is None or _PMTL_OK.search(log) is None:
        return
    m = _PM_TIMELINE_RE.search(log)
    if m is None:
        return
    dst = ev / _REF_PM_TIMELINE_NAME
    try:
        dst.write_text(m.group(1))
    except OSError as e:
        _log(f"meta/recon: could not persist {_REF_PM_TIMELINE_NAME} to {dst}: {e}")


def _read_ref_pm_timeline(ctx: LoopContext) -> str | None:
    ev = _evidence_dir(ctx)
    if ev is None:
        return None
    try:
        return (ev / _REF_PM_TIMELINE_NAME).read_text()
    except OSError:
        return None


def _ref_overlap_banked(ctx: LoopContext) -> bool:
    """True iff the banked ref role table carries the tensor term — i.e. the
    ref's overlap account exists, not just its stall shares."""
    from forgekernel.skills.role_exposure_tools import parse_core
    core = parse_core(_read_ref_role_exposure(ctx))
    return bool(core) and core.get("tensor_active") is not None


def _ref_timeline_banked(ctx: LoopContext) -> bool:
    return _read_ref_pm_timeline(ctx) is not None


def _ref_ledger_banked(ctx: LoopContext) -> bool:
    """The driver's own pricing job delivers STALL-OK too; the recon loop's
    per-log flag never sees that job, so the gate reads the banked file."""
    return _read_ref_stall_ledger(ctx) is not None


def _read_ref_role_exposure(ctx: LoopContext) -> str | None:
    ev = _evidence_dir(ctx)
    if ev is None:
        return None
    try:
        return (ev / _REF_ROLE_EXPOSURE_NAME).read_text()
    except OSError:
        return None


_ROLE_RULES = (
    "A join of three tables: the dossier's role partition cut along back-edge regions × "
    "the PC-sampling stall ledger × the tensor-pipe duty cycle of the PM timeline. share = "
    "that role's fraction of all stall samples (measured); exposed = share × "
    "(1 − tensor_active), an **estimate** that apportions the time not hidden by the "
    "tensor pipe to roles by stall share (a role waiting while the tensor pipe is busy is "
    "hidden; the sampler cannot tell them apart); Δexposed ns = student − ref. "
    "Reading it: ① first see how far tensor_active differs between the sides — that is "
    "the total of all non-tensor exposure; ② then see which role has the largest "
    "Δexposed — that is the **owner** of the exposure, and `deficient_locus` should land on "
    "that role's decisions (copy-producer ⇒ supply organization / issue order; "
    "mma-consumer ⇒ exchange surface, bookkeeping, wait→use distance; straight ⇒ per-tile "
    "scheduling / prologue / epilogue); ③ the difference in the same role's dominant stall "
    "class (student wait/barrier-dominated vs ref dispatch/selection-dominated) tells a G5 "
    "from an R1 shape. With tensor_source=ncu-aggregate, exposed is apportioned over the "
    "whole kernel rather than the mainloop — reference only; when the two sides' sources "
    "differ, align the measurement basis before comparing.\n")


def _role_exposure_view(ctx: LoopContext, priced: "PricedTrees") -> str | None:
    """The per-role exposure comparison block for a teacher prompt, or None
    when either side lacks a ROLE-JSON core."""
    from forgekernel.skills.role_exposure_tools import align_roles, parse_core
    ref = parse_core(_read_ref_role_exposure(ctx))
    stu = parse_core(priced.roles)
    if ref is None and stu is None:
        return None
    block = ("\n## Role exposure ledger (dossier × stall ledger × tensor timeline — read it "
             "before you judge)\n" + _ROLE_RULES)
    partial = [n for n, c in (("ref", ref), ("student", stu))
               if c is not None and c.get("tensor_active") is None]
    if partial:
        block += (f"\n**Note: the {', '.join(partial)} side's ledger has no tensor duty "
                  f"cycle (ROLE-PARTIAL) — the exposed column is empty; only stall shares can "
                  f"be compared, not exposed ns; any conclusion of the form \"ref exposes X ns "
                  f"/ student exposes Y ns\" does not hold on that side.**\n")
    if ref is not None and stu is not None:
        block += ("\n### ref (recon time) vs student "
                  f"{(priced.best_sha or priced.head_sha or '?')[:10]}\n"
                  f"```\n{align_roles(ref, stu)}\n```\n")
    else:
        side = "student" if stu is not None else "ref"
        core = stu if stu is not None else ref
        block += (f"\n### Only the {side} side has a ledger (the other side's job printed no "
                  f"ROLE-OK/ROLE-PARTIAL)\n"
                  f"```\n{align_roles(core, core, names=(side, side))}\n```\n")
    if priced.roles:
        block += f"\n### student ledger, verbatim\n```\n{priced.roles[:3000]}\n```\n"
    return block


def _read_ref_stall_ledger(ctx: LoopContext) -> str | None:
    ev = _evidence_dir(ctx)
    if ev is None:
        return None
    try:
        return (ev / _REF_STALL_LEDGER_NAME).read_text()
    except OSError:
        return None


def _read_ref_segment_ledger(ctx: LoopContext) -> str | None:
    ev = _evidence_dir(ctx)
    if ev is None:
        return None
    try:
        return (ev / _REF_SEGMENT_LEDGER_NAME).read_text()
    except OSError:
        return None


_SEG_LINE_RE = re.compile(
    r"^SEG idx=\d+ kind=(\S+) depth=(\S+)(?: imm=(\S+))? pct=([\d.]+) stalls=(\S*)",
    re.M)
# Skeleton anchors, canonical order. Kinds that wait at the loop top before the
# first HGMMA run collapse into one anchor; EW segments aggregate by the DEPBAR
# depth in force; DEPBARs aggregate by their own immediate.
_SEG_WAIT_KINDS = ("SYNCS", "BAR", "SLEEP", "FENCE", "WG")
_SEG_ANCHOR_ORDER = ("loop-top wait", "first HG run", "DEPBAR 0x1", "EW@0x1",
                     "DEPBAR 0x0", "EW@0x0", "EW@none", "HG rest", "BRA", "EXIT", "other")


def _anchor_shares(seg_text: str) -> dict:
    """SEG machine lines -> {anchor: (pct_sum, dominant stalls of biggest row)}."""
    out: dict = {}

    def add(name: str, pct: float, stalls: str) -> None:
        cur = out.setdefault(name, [0.0, "", -1.0])
        cur[0] += pct
        if pct > cur[2]:
            cur[1], cur[2] = stalls, pct

    seen_hg = False
    for m in _SEG_LINE_RE.finditer(seg_text):
        kind, depth, imm, pct, stalls = (m.group(1), m.group(2), m.group(3) or "",
                                         float(m.group(4)), m.group(5))
        if kind == "HG":
            add("first HG run" if not seen_hg else "HG rest", pct, stalls)
            seen_hg = True
        elif not seen_hg and kind in _SEG_WAIT_KINDS:
            add("loop-top wait", pct, stalls)
        elif kind == "DEPBAR":
            add(f"DEPBAR {imm or '?'}", pct, stalls)
        elif kind == "EW":
            add(f"EW@{depth}", pct, stalls)
        elif kind in ("BRA", "EXIT"):
            add(kind, pct, stalls)
        else:
            add("other", pct, stalls)
    return {k: (v[0], v[1]) for k, v in out.items()}


def align_segment_table(ref_seg: str, student_seg: str) -> str | None:
    """Same-position comparison of the two segment accounts, anchor by anchor.

    The diagnose hard rule holds the locus against the ledger's big rows — which
    presupposes the two sides can be held NEXT to each other. Two raw ledgers
    leave the alignment to the reader, and the reading that matters (same
    segment, share X vs share Y, wait-dominated vs dispatch-dominated) only
    exists after alignment. Returns None when either side has no SEG lines.
    """
    ref = _anchor_shares(ref_seg)
    stu = _anchor_shares(student_seg)
    if not ref or not stu:
        return None
    keys = [k for k in _SEG_ANCHOR_ORDER if k in ref or k in stu]
    keys += [k for k in sorted(set(ref) | set(stu)) if k not in keys]
    lines = ["anchor          | ref %body | student %body | ref dominant stall | student dominant stall"]
    for k in keys:
        r, s = ref.get(k), stu.get(k)
        lines.append(f"{k:<15} | {(f'{r[0]:.1f}' if r else '—'):>9} | "
                     f"{(f'{s[0]:.1f}' if s else '—'):>12} | "
                     f"{(r[1] if r else '—'):<16} | {(s[1] if s else '—')}")
    return "\n".join(lines)


def _seed_ws_stall_tools(ctx: LoopContext) -> None:
    """Vendor the pricing toolkit + frozen collection runner into the student ws.

    Committed once at loop start so EVERY subsequent commit — including whichever
    becomes the best — carries the recipe: pricing runs against the best sha via
    `git archive`, which ships committed files only. Same driver-not-agent rule
    as the recon seeding: teacher-written parsers each got a half wrong.
    """
    if not ctx.stall_pricing:
        return
    _seed_pricing_toolkits(ctx.workspace, ctx.backend)


_JOB_ENV_CHECK_NAME = "job_env_check.py"
_JOB_ENV_CHECK_CMD = f"cmd:python3 notes/{_JOB_ENV_CHECK_NAME}"
JOB_ENV_PREFLIGHT_POLICIES = ("refuse", "warn", "off")


def job_env_preflight(ctx: LoopContext, policy: str = "refuse") -> str | None:
    """Before any iteration: can the harness's own channels run on this job image?

    Two short jobs on the workspace as it stands: the seeded job_env_check.py (can the
    job's bare ``python3`` — what the driver's tools re-spawn — import torch?) and the
    tree's NCU job exactly as every evaluated iteration will submit it. Only a
    job-ENVIRONMENT failure counts (loop/job_env.py): a scaffold kernel that raises is
    the tree's business, a launch target that cannot import its profiler binding is
    not. A fused-norm-rope run would have failed both checks before its first
    iteration, and nothing said so for 2.5 days — its reference tree had a gate like
    this (no NCU signature, no distillation), its student tree had none.

    Returns the failures as one line, or None (all clear, not judged, or off).
    """
    if policy == "off":
        return None
    be = ctx.backend
    branch = f"{ctx.branch_prefix}-jobenv"
    problems, seen = [], []
    try:
        _seed_ws_stall_tools(ctx)   # idempotent; the check script rides the pricing toolkit
        checks = []
        if (Path(ctx.workspace) / "notes" / _JOB_ENV_CHECK_NAME).is_file():
            # a CUDA-mode card also needs the toolchain its kernel is built with
            checks.append(("tools", f"{_JOB_ENV_CHECK_CMD} --cuda" if _is_cuda(ctx)
                           else _JOB_ENV_CHECK_CMD))
        checks.append(("ncu", _ncu_mode(ctx)))
        be.push_branch(branch)
        for channel, mode in checks:
            out = be.submit_gpu(branch, mode) or ""
            why = env_failure(out)
            if why is None and channel == "tools" and "JOB-ENV-FAIL" in out:
                why = next(ln.strip() for ln in out.splitlines() if "JOB-ENV-FAIL" in ln)
            if why is not None:
                problems.append(f"{channel}: {why}")
                _note_env_failure(ctx, channel, why)
            elif channel == "tools" and (m := re.search(r"^JOB-ENV-OK .*$", out, re.M)):
                seen.append(m.group(0))
    except BackendError as e:
        _log(f"job-env preflight: job failed ({e}) — the environment was not judged")
        return None
    if problems:
        _log(f"job-env preflight: FAIL — {'; '.join(problems)} "
             f"(policy={policy}{': refusing to start' if policy == 'refuse' else ''})")
        return "; ".join(problems)
    _log("job-env preflight: OK" + (f" ({seen[0]})" if seen else "")
         + "; the NCU launch target starts on this image")
    return None


def _seed_pricing_toolkits(root: Path, backend) -> None:
    """Vendor the pricing toolkit into ``root/notes`` and commit it there.

    sass/sched tools ride along: the segment ledger joins stall samples onto
    the parsed schedule inside the pricing job, and vendored-flat imports
    (notes/ on sys.path) are how every probe already consumes them.
    orch_tools rides along for the student side of the dossier diff: the
    blueprint's sync/supply adjudications came from the ref's dossier, and
    whether the built SASS actually carries them is checkable in-job,
    statically, before any perf number. The same seeding serves the reference
    tree a source-mode run prices (_ensure_ref_pricing).
    """
    import importlib
    changed = []
    for vendor in map(importlib.import_module, PRICING_SEEDED_MODULES):
        text = Path(vendor.__file__).read_text()
        name = Path(vendor.__file__).name
        dst = root / "notes" / name
        if dst.exists() and dst.read_text() == text:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        changed.append(name)
    if changed:
        backend.commit(f"notes: vendor {', '.join(changed)} (driver-seeded "
                       f"recon toolkits)")


def _ref_price_markers(ev: Path) -> list:
    return [ev / n for n in _REF_PRICE_MARKER_NAMES if (ev / n).is_file()]


def _ref_price_entry(ctx: LoopContext) -> "tuple[Path | None, str]":
    """(the launch-target file the pricing recipe will run, how it was named)
    — ref_price_app's script when configured, else eval/ncu_run.py."""
    d = ctx.ref_price_dir
    if ctx.ref_price_app:
        script = next((a for a in ctx.ref_price_app if a.endswith(".py")), None)
        return (d / script if script else None), f"ref_price_app = {ctx.ref_price_app}"
    return d / "eval" / "ncu_run.py", "eval/ncu_run.py"


def _ref_pricing_key(ctx: LoopContext) -> str:
    """What a retry must see changed: the tree's head, the launch target's
    text, the app spelling. The failures a pricing job can come back with
    (no LINEINFO, PM sections crashing, the sample floor) are deterministic
    on one tree — a retry on the same key reproduces the first job."""
    import hashlib
    h = hashlib.sha1()
    # Not the head sha: the snapshot commit before every job is --allow-empty
    # and moves the head whether or not anything changed. The launch target
    # and its eval/ siblings are what a teacher edits to fix a pricing job.
    entry, _ = _ref_price_entry(ctx)
    files = []
    if entry is not None:
        files.append(entry)
    ev_dir = ctx.ref_price_dir / "eval"
    if ev_dir.is_dir():
        files.extend(sorted(q for q in ev_dir.iterdir() if q.is_file()))
    for q in files:
        try:
            h.update(str(q.relative_to(ctx.ref_price_dir)).encode())
            h.update(q.read_bytes())
        except (OSError, ValueError):
            pass
    h.update(repr(ctx.ref_price_app).encode())
    h.update(str(_pm_interval(ctx)).encode())
    return h.hexdigest()[:16]


def _report_waives(ctx: LoopContext, *patterns) -> bool:
    if ctx.recon_path is None:
        return False
    try:
        text = ctx.recon_path.read_text()
    except OSError:
        return False
    return all(p.search(text) is not None for p in patterns)


def _ref_price_missing(ctx: LoopContext) -> list:
    return [n for n, ok in (("ROLE", _ref_overlap_banked(ctx)),
                            ("PMTL", _ref_timeline_banked(ctx))) if not ok]


def _ensure_ref_pricing(ctx: LoopContext, label: str = "meta") -> None:
    """Price the reference tree with the frozen recipe — the ref's overlap account.

    The role-exposure and dossier tables have two sides; the ref side used to
    come from wherever the teacher happened to produce it. A source-mode run has no
    recon step, so its tables were one-sided (an offline check on grouped
    gemm showed the teacher reading a "ref" column that was the student column
    copied verbatim). A blackbox run's teacher priced the ref inside its own
    recon probes — improvised collections that, on one dsa_sparse run,
    fought ncu for two rounds and banked a ROLE table with no tensor term, so
    the ref's exposure column was null for the whole 17h run and the one
    measurement that separates gather/MMA overlap from its absence never
    reached a teacher. Four blackbox runs guessed that overlap wrong at v1.

    So the driver owns it: ``ctx.ref_price_dir`` (loop.toml ``ref_price_op``;
    a blackbox run defaults it to the recon tree) gets the toolkit seeded, the
    tree committed (the job archives the branch, not the working tree), and
    ``stall_price_run.py --pm`` run there — one job yields the stall ledger,
    segment account, dossier, role table WITH tensor and the phase timeline —
    and whatever verdict blocks come back are banked. Called at the end of
    reconnaissance (so the account exists before v1 is authored) and at every
    teacher entry; a no-op once the account is banked, once the teacher has
    waived what is missing, and on an unchanged tree after a job that came
    back without it (a retry re-runs only when the tree, the launch target or
    the app spelling changed; at most three attempts per run).
    """
    if not ctx.stall_pricing or ctx.ref_price_dir is None \
            or ctx.ref_price_backend is None:
        return
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    missing = _ref_price_missing(ctx)
    if not missing:
        return
    if _report_waives(ctx, *[_ROLE_WAIVED if m == "ROLE" else _PMTL_WAIVED for m in missing]):
        return  # the teacher owned the absence; no job behind its back
    if ctx.now() > ctx.deadline_epoch - 1800:
        _log(f"{label}/ref-price: < 30 min of budget left — not pricing the reference tree")
        return
    entry, named = _ref_price_entry(ctx)
    if entry is None or not entry.is_file():
        # Re-checked on every call, not once at preflight: a teacher that adds
        # the launch target during reconnaissance is priced at the next finish.
        _log(f"{label}/ref-price: launch target {named} not found in "
             f"{ctx.ref_price_dir} — the pricing recipe has no profiler-window entry; "
             f"the ref's overlap account stays missing")
        return
    markers = _ref_price_markers(ev)
    key = _ref_pricing_key(ctx)
    if markers:
        if markers[-1].read_text().startswith(f"[ref-price] key={key}"):
            return  # same tree, same target, same spelling: the job would repeat itself
        if len(markers) >= len(_REF_PRICE_MARKER_NAMES):
            return
        if "ROLE" not in missing and _marker_pm_interval(markers[-1]) == _pm_interval(ctx):
            # only the timeline is missing and the sampling interval is what the
            # last attempt ran with — the same collection would starve again
            return
    if ctx.ref_price_submit_failures >= _REF_PRICE_MAX_SUBMIT_FAILURES:
        return
    be = ctx.ref_price_backend
    cmd = _stall_price_cmd(ctx)
    if ctx.ref_price_app:
        cmd += " --app " + " ".join(shlex.quote(a) for a in ctx.ref_price_app)
    marker = ev / _REF_PRICE_MARKER_NAMES[len(markers)]
    head_before = getattr(be, "last_job_id", "")
    try:
        _seed_pricing_toolkits(ctx.ref_price_dir, be)
        # The job archives the BRANCH: a launch target the teacher just wrote
        # into the working tree is not in it until committed (--allow-empty,
        # so an unchanged tree costs nothing).
        be.commit("ref-price: snapshot the tree the pricing job archives")
        branch = getattr(be, "current_branch", lambda: "main")()
        _log(f"{label}/ref-price: pricing the reference tree {ctx.ref_price_dir.name} "
             f"@ {branch} (attempt {len(markers) + 1}/{len(_REF_PRICE_MARKER_NAMES)}, "
             f"{named})")
        log = be.submit_gpu(branch, cmd)
    except BackendError as e:
        # A job that launched and produced nothing usable is an attempt; a
        # verdict that nothing ran (NoRunError) never is, whatever it says —
        # a submit that is retried also moves the job id, so the id
        # moving proves launch, not a run.
        ran = (getattr(be, "last_job_id", "") not in ("", head_before)
               and not isinstance(e, NoRunError))
        if ran:
            # The job ran and did not come back usable (poll timeout on a hung
            # reference kernel, log lost after success): that is an attempt —
            # the same tree would hang again.
            ctx.ref_price_note = ""
            try:
                marker.write_text(f"[ref-price] key={key} app={ctx.ref_price_app!r} "
                                  f"pm_interval={_pm_interval(ctx)}\n"
                                  f"[ref-price] job failed after launch: {e}\n")
            except OSError:
                pass
            _log(f"{label}/ref-price: job failed after launch ({e}) — counted as an "
                 f"attempt; role/dossier tables stay one-sided")
            _bank_evidence(ctx, "ref-price attempt marker")
        else:
            # Nothing ran — never submitted (git push failed), or submitted and
            # never produced output (queue starved, job lost, never
            # got a node): not an attempt, bounded per session.
            ctx.ref_price_submit_failures += 1
            ctx.ref_price_note = f"the pricing job did not run: {e}"
            _log(f"{label}/ref-price: nothing ran ({e}) — not an attempt "
                 f"({ctx.ref_price_submit_failures}/{_REF_PRICE_MAX_SUBMIT_FAILURES} "
                 f"this session)")
        return
    ctx.ref_price_note = ""
    from forgekernel.skills.stall_price_run import expand_bundle
    log = expand_bundle(log)
    banked = []
    _save_ref_stall_ledger(ctx, log)
    if _STALL_OK.search(log) is not None:
        banked.append("ledger")
    if (ev / _REF_SEGMENT_LEDGER_NAME).is_file():
        banked.append("segments")
    if _ORCH_OK.search(log) is not None:
        _save_ref_orch_dossier(ctx, log)
        banked.append("dossier")
    if _ROLE_ANY.search(log) is not None:
        _save_ref_role_exposure(ctx, log)
        banked.append("roles+tensor" if _ROLE_OK.search(log) is not None
                      else "roles (PARTIAL: no tensor term)")
    _save_ref_pm_timeline(ctx, log)
    if _PMTL_OK.search(log) is not None:
        banked.append("timeline")
    try:
        marker.write_text(f"[ref-price] key={key} app={ctx.ref_price_app!r} "
                          f"pm_interval={_pm_interval(ctx)}\n" + log)
    except OSError as e:
        _log(f"{label}/ref-price: could not persist {marker.name}: {e} — "
             f"the reference will be priced again at the next teacher entry")
    # The snapshot commit above ran BEFORE the job; the ledgers, dossier, role
    # table, timeline and marker were written after it and committed by nobody —
    # the exact four files that once held an mla_paged_decode resume at the
    # dirty-tree preflight.
    _bank_evidence(ctx, f"ref-price {', '.join(banked) or 'attempt marker'}")
    _log(f"{label}/ref-price: banked {', '.join(banked) or 'nothing (no OK verdict line)'}"
         f" (job {getattr(be, 'last_job_id', '') or '?'})")
    if _ref_price_missing(ctx):
        _log(f"{label}/ref-price: still missing {'/'.join(_ref_price_missing(ctx))} — "
             f"no retry on an unchanged tree")


def _ref_price_status(ctx: LoopContext, missing: list) -> "tuple[str, str]":
    """(what happened, what the teacher can do) for the overlap gate message,
    derived from what is on disk — the markers and the banked files — so it
    is right after a resume, after a job that banked half the account, and
    after the attempts are spent. ``missing`` is the gate's ROLE/PMTL list."""
    if not ctx.stall_pricing or ctx.ref_price_dir is None or ctx.ref_price_backend is None:
        return ("this run has no reference pricing tree configured (`[loop] ref_price_op`); "
                "the driver has not run pricing",
                "configure `ref_price_op` in loop.toml (default for a blackbox run = the recon "
                "tree) and restart, or waive")
    entry, named = _ref_price_entry(ctx)
    ev = _evidence_dir(ctx)
    markers = _ref_price_markers(ev) if ev is not None else []
    if ctx.ref_price_note:
        return (ctx.ref_price_note, "wait for the driver to retry the submission at the next "
                f"finish attempt (at most {_REF_PRICE_MAX_SUBMIT_FAILURES} times this "
                "session), or waive")
    if entry is None or not entry.is_file():
        if ctx.ref_price_app:
            return (f"the entry file `{named}` points to is not in the recon tree; the driver "
                    f"has not run pricing",
                    f"fix `ref_price_app` in loop.toml (the driver uses it and does **not** "
                    f"fall back to eval/ncu_run.py), or write that entry file into the recon "
                    f"tree; then simply end the turn again")
        return (f"the recon tree has no `eval/ncu_run.py` and loop.toml has no "
                f"`ref_price_app`; the driver has not run pricing",
                "add `eval/ncu_run.py` to the recon tree (a few lines: build the inputs, "
                "`torch.cuda.cudart().cudaProfilerStart()`, call the reference kernel once, "
                "`cudaProfilerStop()`; open the profiler only around the kernel call), then "
                "**simply end the turn again** — the driver submits and re-prices at the "
                "finish check; you do not need to send a probe")
    if not markers:
        return ("the driver has not run pricing yet (budget too low, or this session's "
                "submission failures hit the cap)", "waive, or wait for the next finish attempt")
    last = markers[-1].read_text()
    verdicts = " / ".join(ln for ln in last.splitlines()
                          if re.match(r"^(ROLE|PMTL|STALL)-(OK|PARTIAL|FAIL)\b|\[ref-price\] job failed", ln))
    what = (f"the driver has run pricing {len(markers)} time(s) (entry {named}); last "
            f"verdict lines: {verdicts[:300] or 'none'}; still missing {'/'.join(missing)}")
    if missing == ["PMTL"]:
        n = _pm_interval(ctx)
        return (what, f"the role ledger is in; only the phase timeline is missing (PM sampling "
                      f"interval this time: {n} cycles): with a reference under 0.5 ms, 8192 "
                      f"collects too few samples — write `[loop] ref_price_pm_interval = 2048` "
                      f"in loop.toml and restart, and the driver retries once with the new "
                      f"interval; if the interval is already small enough and it still collects "
                      f"nothing, waive with `PMTL-WAIVED:`")
    same_key = last.startswith(f"[ref-price] key={_ref_pricing_key(ctx)}")
    if len(markers) >= len(_REF_PRICE_MARKER_NAMES):
        return (what, "pricing attempts are used up and the driver will not retry: decide by "
                      "the reason in the verdict lines — if it cannot be fixed, waive")
    if same_key:
        return (what, "rerunning the same tree with the same entry would only reproduce the "
                      "same verdict: the driver will not retry automatically. Ending the turn "
                      "again after changing the entry file or `ref_price_app` triggers a "
                      "retry; otherwise waive")
    return (what, "the recon tree has changed; simply end the turn again and the driver "
                  "re-prices")


_STALL_PRICE_CMD = "cmd:python3 notes/stall_price_run.py --pm"
_PM_INTERVAL_DEFAULT = 8192      # == stall_price_run.PM_INTERVAL
_PM_INTERVAL_FAST = 2048
_PM_FAST_KERNEL_MS = 0.5


def _pm_interval(ctx: LoopContext) -> int:
    """The PM-sampling interval the pricing recipe runs with: the configured
    one; else 2048 when the run's target reading says its kernels are sub-0.5 ms
    (the recipe's own note: fast kernels starve at 8192); else 8192."""
    if ctx.ref_price_pm_interval and int(ctx.ref_price_pm_interval) > 0:
        return int(ctx.ref_price_pm_interval)
    try:
        target = _resolve_target(ctx)
    except Exception:  # noqa: BLE001 — no milestone / target seed yet
        target = None
    if target and target < _PM_FAST_KERNEL_MS:
        return _PM_INTERVAL_FAST
    return _PM_INTERVAL_DEFAULT


def _stall_price_cmd(ctx: LoopContext) -> str:
    """The frozen recipe with the interval this run needs (the student's and
    the reference's tree run the same problem in the same duration class); the
    flag is only spelled when it departs from the script's default. `--app`
    (the ref pricing appends it) must stay LAST: the script takes everything
    after it as the launch target."""
    n = _pm_interval(ctx)
    return _STALL_PRICE_CMD if n == _PM_INTERVAL_DEFAULT else f"{_STALL_PRICE_CMD} --pm-interval {n}"


def _marker_pm_interval(marker: Path) -> int:
    """The PM interval a pricing attempt ran with (its marker's first line);
    attempts from before the field existed ran the script's default."""
    try:
        m = re.search(r"pm_interval=(\d+)", marker.read_text().splitlines()[0])
    except (OSError, IndexError):
        m = None
    return int(m.group(1)) if m else _PM_INTERVAL_DEFAULT


# STALL-PRICE REUSE: every teacher entry prices the best KEEP (and HEAD when it
# differs), but the best rarely moves between entries — in one large batch of
# runs stall pricing was ~6% of loop wall time and 553 of its 1026
# whole readings were a sha the run had already priced (one run: 19 of 22
# jobs on one best). The ledger,
# segment account and dossier are readings of the tree, so one whole reading
# per (run, sha, recipe) serves every later entry of this process; parsing and
# banking still run on each use, only the job is skipped. Keyed by the run's
# status path so tests and runs sharing a process never see each other's.
_STALL_PRICE_LOGS: "dict[tuple[str, str, str], tuple[str, str]]" = {}

_REF_PRICE_MAX_SUBMIT_FAILURES = 3
_REF_PRICE_MARKER_NAMES = ("ref_price_job.log", "ref_price_job.2.log", "ref_price_job.3.log")


def _run_stall_pricing(ctx: LoopContext, ref: str | None = None, *,
                       bank_as: "tuple[str, ...]" = (_STUDENT_ORCH_DOSSIER_NAME,),
                       label: str = "best", workload: str | None = None) \
        -> "tuple[str | None, str | None, str | None, str | None]":
    """Price one student tree (``ref``; the best-so-far KEEP when None);
    return (row ledger, segment account, orchestration dossier, role exposure).
    ``workload``: one of a group card's other meta workloads (``--workload``;
    loop/meta_shapes.py), read beside the representative's and banked as nothing
    (``bank_as`` = () from its caller) -- the banked dossiers stay the card's own.

    The collection rides the PM-sampling sections (``--pm``): the role
    exposure ledger scales stall shares by the mainloop tensor plateau, which
    only the timeline reads; stall_price_run falls back to the plain recipe
    when the combined collection crashes, and the role ledger then scales by
    the whole-kernel aggregate instead (labelled in its own line).

    One GPU job per plateau diagnosis (~1-3 min) buys the diagnose step the one
    evidence face the aggregate digest cannot provide: where the time measurably
    leaks, per instruction, summed. The segment account (PC-keyed, schedule-
    joined) rides the same job — the row ledger's line keying loses the
    scheduling rows to the no-line bucket. The student's orchestration dossier
    (stall_price_run renders the SASS page off the same collection) rides too:
    its ORCH-SUMMARY-JSON line is what the dossier diff against the ref's
    recon-time dossier is computed from, and the body is banked beside the ref's
    for later audits. A job failure degrades to the pre-ledger behavior instead
    of blocking the diagnosis — the ledger is evidence, not a precondition.
    """
    if not ctx.stall_pricing:
        return None, None, None, None
    be = ctx.backend
    if ref is None:
        records = load_records(ctx.status_path)
        ref = _best_good_ref_here(records, ctx)
    branch = f"{ctx.branch_prefix}-stallprice"
    cmd = _stall_price_cmd(ctx) + (f" --workload {workload}" if workload else "")
    push_ref = getattr(be, "push_ref", None)
    key = (str(ctx.status_path), ref, cmd) if ref and push_ref is not None else None
    hit = _STALL_PRICE_LOGS.get(key) if key else None
    if hit is not None:
        log, job = hit
        _log(f"meta/stall-price[{label}]: reusing the pricing of {ref[:10]} (job {job}) — "
             "same tree, same recipe")
    else:
        try:
            if key:
                push_ref(ref, branch)
            else:
                # no best sha yet (pre-green plateau) or a backend without push_ref:
                # price the workspace HEAD — worse than best is still measured truth.
                be.push_branch(branch)
                ref = be.head()
            log = be.submit_gpu(branch, cmd)
        except BackendError as e:
            _log(f"meta/stall-price[{label}]: pricing job failed ({e}) — "
                 "diagnosing without a ledger")
            return None, None, None, None
        # job log archives keep ~300 lines; the bundle at the tail carries the
        # whole analysis (see stall_price_run.expand_bundle).
        from forgekernel.skills.stall_price_run import expand_bundle
        log = expand_bundle(log)
        job = getattr(be, "last_job_id", "") or "?"
        if key and _STALL_LEDGER_RE.search(log) and _STALL_OK.search(log) \
                and (dm := _ORCH_DOSSIER_RE.search(log)) is not None \
                and _orch_summary(dm.group(1)) is not None:
            # only a whole reading is kept: a job that came back without its
            # ledger, self-check or dossier is priced afresh next entry
            _STALL_PRICE_LOGS[key] = (log, job)
    roles = None
    rm = _ROLE_BLOCK_RE.search(log)
    if rm is not None and _ROLE_ANY.search(log) is not None:
        roles = rm.group(1)
        full = _ROLE_OK.search(log) is not None
        _log(f"meta/stall-price[{label}]: role exposure "
             f"{'OK' if full else 'PARTIAL (no tensor term — PM sections missing)'} @ "
             f"{ref[:10] if ref else '?'}")
        if (ev := _evidence_dir(ctx, create=True)) is not None and (
                _STUDENT_ORCH_DOSSIER_NAME in bank_as
                or _STUDENT_BEST_ORCH_DOSSIER_NAME in bank_as):
            try:
                (ev / _STUDENT_ROLE_EXPOSURE_NAME).write_text(roles)
            except OSError as e:
                _log(f"meta/stall-price[{label}]: could not persist "
                     f"{_STUDENT_ROLE_EXPOSURE_NAME}: {e}")
    dossier = None
    dm = _ORCH_DOSSIER_RE.search(log)
    # Banked on a parsable summary, NOT on the trailing ORCH-OK line: the
    # summary carries its own verdict, and a dossier that fails for
    # no-sync-sites is the "blueprint declared, nothing built" reading this
    # channel exists to deliver — requiring ORCH-OK dropped exactly that case.
    if dm is not None and (summ := _orch_summary(dm.group(1))) is not None:
        dossier = dm.group(1)
        _log(f"meta/stall-price[{label}]: orch dossier "
             f"{summ.get('verdict', 'ORCH-OK')} @ {ref[:10] if ref else '?'}")
        if (ev := _evidence_dir(ctx, create=True)) is not None:
            for name in bank_as:
                dst = ev / name
                try:
                    dst.write_text(dossier)
                except OSError as e:
                    _log(f"meta/stall-price[{label}]: could not persist {name}: {e}")
    if roles is not None or dossier is not None:
        _bank_evidence(ctx, f"stall-price[{label}] dossier")
    m = _STALL_LEDGER_RE.search(log)
    ok = _STALL_OK.search(log) is not None
    if m is None:
        why = env_failure(log)
        if why is not None:
            _note_env_failure(ctx, "stall-price", why)
        _log(f"meta/stall-price[{label}]: job returned no ledger"
             + (f" (job environment: {why})" if why else "") + " — diagnosing without one")
        return None, None, dossier, roles
    _note_env_ok(ctx, "stall-price")
    _log(f"meta/stall-price[{label}]: ledger {'OK' if ok else 'SELF-CHECK-FAIL'} "
         f"@ {ref[:10] if ref else '?'} (job {job})")
    # HEAD, not tail: ledger rows print most-expensive first, and the diagnose hard
    # rule holds the locus against the TOP of the ledger. A tail slice would drop
    # exactly those rows, and only in the flat-distribution (dispersed-gap) case
    # this mechanism exists for — the ledger would fail first where it matters most.
    # The opcode-family table at the tail is the acceptable loss.
    body = m.group(1)[:9000]
    if not ok:
        body = ("(WARNING: ledger self-check failed — `STALL-OK` not printed; sample count / "
                "calibration / coverage fell short. Use it only as supporting evidence; "
                "conclusions must be downgraded.)\n" + body)
    seg = _SEGMENT_LEDGER_RE.search(log)
    seg_body = None
    if seg is not None:
        # Verdict before slice: JOIN-OK is segment_ledger's LAST line and the SEG
        # rows above it are unbounded, so a many-segment kernel's block overruns
        # the head-slice — checking after slicing would read every successful join
        # as unsafe on exactly the kernels this table exists for. An unsafe join
        # is still not evidence: no JOIN-OK in the full block drops the table.
        if "JOIN-OK" in seg.group(1):
            seg_body = seg.group(1)[:6000]
    return body, seg_body, dossier, roles


def _require_recon_artifacts(ctx: LoopContext, be, sass_ok: bool = True,
                             sched_ok: bool = True, live_ok: bool = True,
                             stall_ok: bool = True, struct_ok: bool = True,
                             since: float | None = None,
                             near_miss: dict | None = None,
                             launch: dict | None = None) -> None:
    """The reconnaissance only exists as what it wrote down — enforce that it did.

    Observed live: a teacher spent all 12 rounds probing and the loop moved on to
    distillation with {RECON_REPORT} pointing at a file nobody had written, so the
    entire reconnaissance would have reached the milestone as a dangling path. The
    findings were all on disk — in the probe docstrings — but the distill prompt
    reads the report, not the notes. Probing turns are over when this runs: the
    teacher gets write-up turns, and a further RECON_REQUEST is answered with this
    same instruction rather than a job.
    """
    if ctx.recon_path is None:
        return
    candidates = ctx.recon_path.parent / "candidates.md"
    writeup_storm = [TRUNCATION_STORM_BUDGET]
    for _ in range(3):
        lacking = _missing_recon_evidence(ctx, sass_ok, sched_ok, live_ok,
                                          stall_ok, struct_ok, near_miss=near_miss)
        contras = _launch_contradictions_open(ctx, launch)
        if _recon_artifacts_written(ctx, since=since) and not lacking and not contras:
            return
        missing = " and ".join(f"`{p}`" for p in (ctx.recon_path, candidates)
                              if not p.exists())
        gaps = []
        if contras:
            c = launch["cluster"]
            _log(f"meta/recon: RECON-CONTRADICTION cluster: NCU read cluster={c} "
                 f"but the write-up denies it ({contras[0][:100]})")
            gaps.append(f"RECON-CONTRADICTION cluster: the NCU launch metrics read cluster="
                        f"({c[0]},{c[1]},{c[2]}), but the report / candidates table still has "
                        f"lines denying a cluster ({'; '.join(contras[:3])}) — rewrite §1 and "
                        f"the candidates table to the measured cluster (the first line of each "
                        f"candidate states the cluster shape and CTA count first), or write "
                        f"`LAUNCH-DISPUTED: <the log line carrying the reading + criterion>`.")
        if missing:
            gaps.append(f"These deliverables are not written yet: {missing}.")
        elif not _recon_artifacts_written(ctx, since=since):
            # both files exist but predate this session — stale residue from a
            # previous run (possibly a different operator's, the default recon
            # path ships populated). The teacher must re-affirm them.
            gaps.append("Both deliverables are on disk but **were not written by this "
                        "reconnaissance** (stale residue) — check that their content belongs "
                        "to this operator, and update them in this session.")
        if lacking:
            # GPU rounds are gone, so delivery is off the table — the only remaining
            # legal state is an owned waiver, not a silent absence.
            waivers = " / ".join(f"`{n}-WAIVED: <the concrete reason and the channels tried>`"
                                 for n in lacking)
            gaps.append(f"This reconnaissance lacks {'/'.join(lacking)} evidence, and the "
                        f"report has no matching waiver line — the GPU budget is spent and "
                        f"delivery is no longer possible, so you **must** add {waivers} to "
                        f"the recon report.")
            # Both sides of any checker/printer disagreement, verbatim — a
            # deadlock (checker false-negative × honest refusal to
            # waive) is unfindable without this.
            gaps.extend(_near_miss_notes(lacking, near_miss))
        _log(f"meta/recon: write-up incomplete ({'; '.join(gaps)[:120]}) — write-up turn")
        _agent_turn(be, _recon_prompt(ctx, extra=(
            f"\n\n## Recon rounds are over — now only write the deliverables\n\n"
            f"The GPU budget is used up; **you can no longer send RECON_REQUEST** (one sent "
            f"will not be executed).\n"
            + "\n".join(gaps) + "\n"
            f"All your findings are in the recon tree's `notes/probe*.py` docstrings and "
            f"`notes/*.log` — read them now and complete the deliverables. Write as much as "
            f"was measured, and mark what was not honestly as \"not measured\"; an honest "
            f"hole is more useful than a fabricated whole.")),
            label="meta/recon-writeup", storm_budget=writeup_storm,
            left_a_trace=lambda o: _recon_artifacts_written(ctx, since=since))
    if not (_recon_artifacts_written(ctx, since=since)
            and _sass_evidence(ctx, sass_ok, near_miss)):
        _log("meta/recon: WARNING — write-up turns exhausted and artifacts are still "
             "incomplete; distillation will see whatever exists")


def parse_student_turn_safe(out: str, *, allow_recon: bool = False):
    """parse_student_turn, but prose with no marker is a finished turn, not an error.

    The student is required to end in one of three shapes and a violation is a
    protocol error worth re-prompting. The teacher's reconnaissance turn is different:
    "no further request" is how it signals it is done, and that is most naturally
    written as prose — but the caller only accepts the signal once the write-up
    artifacts exist, because a truncated turn produces the same absence.
    """
    try:
        return parse_student_turn(out, allow_recon=allow_recon)
    except ValueError:
        return None


def _source_recon_distill_addendum(ctx: LoopContext) -> str:
    """What a source-mode run's distill brief gains when reconnaissance ran.

    distill_prompt.md is the control run's brief and stays untouched; the
    addendum is appended only when a report exists on disk, so a source-mode run
    without a recon tree renders exactly what it always did. The brief's own
    text tells the teacher to transcribe the issue order, the exchange-surface
    atoms and the segment budget "from the source at cost ≈ 0" — the addendum
    redirects those transcriptions to the measured page wherever the report
    carries one, and keeps the source for what only the source says (why, and
    the parts no probe covered).
    """
    if ctx.oracle_mode != "source" or ctx.recon_path is None:
        return ""
    if not ctx.recon_path.is_file():
        return ""
    candidates = ctx.recon_path.parent / "candidates.md"
    return (
        "\n## Recon report (supplementary input in source mode)\n\n"
        f"Before distillation this run reconnoitred the reference binary: `RECON_REPORT` = "
        f"`{ctx.recon_path}` (candidates table `{candidates}`, recon tree "
        f"`{ctx.recon_path.parent}`, probe logs in its `notes/`). "
        "**Read it first, then the ORACLE.**\n\n"
        "- Every item above that says \"transcribe from the ORACLE source\" — tile / stage / "
        "warp-role values, exchange-surface access atoms, issue order and the segmented "
        "live-set budget, wait-target relations, barrier topology, inter-CTA dispatch order — "
        "**where the recon report has a measured reading, transcribe the measurement**, "
        "tagged `[MEASURED, job N]`; use the source only to explain intent and to fill what "
        "the recon did not cover. When the source's template parameters disagree with the "
        "binary's reading, write the binary's and record the disagreement in the decision.\n"
        "- The two places above that say \"without a recon report, do it yourself / "
        "transcribe directly from the source\" **do not apply** to this run: the load-bearing "
        "ruling on inter-CTA dispatch order follows the report's **§3b residency ruling**; "
        "the intermediate-buffer ledger of a multi-kernel operator (including the \"who "
        "allocates, who zeroes\" column) follows the report's **§3c measured ledger**, and "
        "the source only fills the ledger's empty columns.\n"
        "- For quantities the recon report marks \"not measured\", go back to the source; "
        "for those the source does not have either, mark `[ESTIMATE]` + derivation as the "
        "rules say.\n"
        "- The reference's NCU signature, SASS footprint, stall ledger and orchestration "
        "dossier are all in the recon tree; anchor mechanism-verification lines on them "
        "first.\n"
        "- The red line is unchanged: do not transcribe the ORACLE's implementation body; "
        "the recon deliverables are likewise readable only by you — the student cannot see "
        "them.\n"
    )


def _distill_prompt_path(ctx: LoopContext) -> Path:
    """Which distillation brief the teacher gets.

    The source-mode prompt is left untouched rather than branched inside, because the
    control run has to use the same instructions it would have used without this
    experiment existing. A single prompt carrying "if you have source ... otherwise ..."
    would differ from the status quo for both modes, and then the comparison is against
    something nobody ever ran.
    """
    if ctx.oracle_mode == "blackbox":
        p = _meta_prompt(ctx, "distill_blackbox_prompt.md")
        if p.exists():
            return p
        _log("meta/distill: blackbox mode but no distill_blackbox_prompt.md — falling "
             "back to the source prompt, which references an oracle this run lacks")
    return _meta_prompt(ctx, "distill_prompt.md")


def _ref_overlap_block_for_distill(ctx: LoopContext) -> str:
    """The ref's measured overlap account for the distill prompt: role table
    with tensor + phase timeline when banked, an explicit absence note when
    not. The dossier says WHAT the ref's producer/consumer code is; this says
    how much of each role's waiting the tensor pipe hides — the quantity every
    blackbox v1 on dsa_sparse guessed, four runs in a row, because no teacher
    had it on the page."""
    from forgekernel.skills.role_exposure_tools import align_roles, parse_core
    roles = _read_ref_role_exposure(ctx)
    core = parse_core(roles)
    timeline = _read_ref_pm_timeline(ctx)
    banked = core is not None and core.get("tensor_active") is not None
    if not banked and not (ctx.stall_pricing and (ctx.oracle_mode == "blackbox"
                                                 or ctx.ref_price_dir is not None)):
        # A source-mode run that neither reconnoitres nor prices its ref reads the
        # overlap off the source; the absence note is a blackbox-mode rule
        # (and the control run's distill prompt must stay byte-identical).
        return ""
    if not banked:
        return (
            "\n## Reference overlap ledger: **absent**\n"
            "The reference's role exposure ledger has no tensor duty cycle (or does not exist "
            "at all), and "
            + ("there is no phase timeline either" if timeline is None
               else "there is a phase timeline") +
            ". This means \"how the reference hides its supply behind the MMA\" is "
            "**unmeasured**: every [SYNC:]/[GATHER:]/[PHASE:] ruling in the milestone may only "
            "be written as a candidate carrying [ESTIMATE], never as settled; the M0 gate must "
            "not turn the overlap shape (supply issue order, tight-chain, spin share) into a "
            "hard line.\n")
    block = ("\n## Reference overlap ledger (measured by the recon-time pricing job: role × "
             "stall × tensor duty cycle)\n"
             "exposed = that role's stall share × (1 − tensor_active), an estimate of how much "
             "waiting the reference leaves **exposed** outside tensor work. Read this table "
             "first when ruling on the supply structure: which role exposes how many ns, how "
             "large the spin share is; if the student's v1 writes these roles' exposure an "
             "order of magnitude larger than the reference's, it is guessing.\n"
             f"```\n{align_roles(core, core, names=('ref', 'ref'))}\n```\n")
    if timeline is not None:
        block += ("\n### Reference phase timeline (PM sampling: WHEN, not WHERE)\n"
                  f"```\n{timeline[:4000]}\n```\n")
    return block


def _inferred_rows_block_for_distill(ctx: LoopContext) -> str:
    """Candidates rows the recon left `[VERIFY: inferred]`: structural readings
    whose discriminating observation printed nothing (dsa_sparse: the
    KV-ring stage axis, 32 key x 512 d vs 64 key x 256 d, byte-identical). The
    distill step must carry both readings as [ESTIMATE] candidates — the v1
    that treated the byte identity as a verdict cost a 13-decision reopen."""
    rows = [r for r in load_gap_queue(ctx) if r.reason == "inferred"]
    if not rows:
        return ""
    return ("\n## Inferred rows of the recon table (`[VERIFY: inferred]`, the discriminating "
            "observation gave no positive reading)\n"
            "The structural quantities below had more than one reading in the recon with "
            "identical counts/bytes, and the discriminating experiment printed no reading that "
            "points to one of them. The corresponding decisions **must** list every reading as "
            "a candidate with the numbers tagged `[ESTIMATE]`, and state in the five-tuple "
            "\"the student's first probe decides the axis\"; **they must not be written as "
            "settled and must not enter the M0 Gate**.\n"
            + "\n".join(r.line() for r in rows) + "\n")


def run_distill_step(ctx: LoopContext) -> "MetaResult":
    """Initial distillation: meta reads the ORACLE (open-source code) + POLICY (§9) and
    writes milestone v1 from scratch. Schema-gated (re-prompt until §9-conformant), same
    harness as a patch. Not transactional — there is no prior milestone to roll back to.

    Two hardenings from a dsa_indexer_fwd run:
    - each round's draft is snapshotted, so a revision round that leaves the file
      missing restores the last draft instead of validating a hole;
    - a round that ends with the milestone still nonexistent did no schema work at
      all — the agent session died before its Write (API timeout, truncated stream;
      the CLI exits 0 with the error as its final text, so the backend's rc-based
      retry never fires). Re-prompting such a round with "file unreadable" adds no
      information, so two of them in a row raise AgentError: the run stops as
      agent_failed (transient, watchdog-restartable), not distill_malformed."""
    be = ctx.backend
    bindings = _bindings(ctx)
    meta_step = len(load_meta_records(ctx.meta_status_path)) if ctx.meta_status_path else 0
    out, violations, draft, dead_rounds = "", [], None, 0
    storm_budget = [TRUNCATION_STORM_BUDGET]
    base_extra = (_source_recon_distill_addendum(ctx) + _ref_overlap_block_for_distill(ctx)
                  + _inferred_rows_block_for_distill(ctx))
    if ref_dossier := _read_ref_orch_dossier(ctx):
        # The recon-time orchestration dossier travels INTO the distill prompt
        # rather than waiting in the recon tree: the [SYNC:]/[GATHER:]/[PHASE:]
        # adjudications are exactly what it machine-reads, and a blueprint that
        # guesses them with this on disk repeats the failure §2g exists to end
        # (topology guessed symmetric, supply guessed inline — the recon report
        # prose kept losing the fine grain the dossier keeps).
        base_extra += (
            "\n## Reference orchestration dossier (machine-read at recon time, static SASS "
            "transcription)\n"
            "The milestone's [SYNC:]/[GATHER:]/[PHASE:] rulings must match this dossier one "
            "by one: sync topology (for each barrier object, who arrives and who waits, the "
            "granularity, the release chain), supply issue order (bulk prefetch vs per-row "
            "inline, tight-chain count), in-flight depth (the DEPBAR immediate table), "
            "cache-policy bits. Where an axis of the dossier is UNRESOLVED or absent, rule "
            "from the recon report as before — but on every axis where the dossier has a "
            "clear reading, the ruling cites it instead of guessing.\n"
            f"```\n{ref_dossier[:6000]}\n```\n")
    if ref_struct := _read_ref_struct_table(ctx):
        # The ref's per-phase construction table — the only place a blackbox run
        # has the reference's BUILD (warps / tile granule / precision / hand-off).
        # Not a [SYNC:]/[GATHER:] axis, so it travels alongside the dossier: the
        # phase contracts the milestone writes (esp. an internal phase like a
        # triangular solver) must match this construction, not an inference from
        # the instruction mix.
        base_extra += (
            "\n## Reference per-segment construction table (decoded mechanically at recon "
            "time; a blackbox run's only per-segment construction facts)\n"
            "This is **how** the reference's hot region **is built**, segment by segment — "
            "how many warps per segment, what block size it addresses, what internal "
            "precision, how segments hand off. The milestone's rulings on **a segment's "
            "internal construction** (block granule / doubling levels / reduction placement) "
            "must follow this table; the recon report's prose only paraphrases it — when the "
            "two conflict, the table wins and the disagreement is marked.\n"
            f"```\n{ref_struct[:REF_STRUCT_PROMPT_CHARS]}\n```\n")
    for _ in range(max(1, ctx.max_patch_rounds)):
        extra = base_extra
        if violations:
            extra += ("\n## SCHEMA violations (the milestone you wrote does not conform to "
                      "the §9 skeleton; you must fix it and write it again)\n"
                      + "\n".join(f"- {v}" for v in violations) + "\n")
        pre = _read_or_none(ctx.milestone)
        out = _agent_turn(
            be, render_prompt(_distill_prompt_path(ctx), bindings, extra),
            label="meta/distill", storm_budget=storm_budget,
            # the only legal trace of a distill turn is the milestone changing on
            # disk — a tiny output that left the draft byte-identical is a
            # truncated stream, not a revision, and re-prompting it with the same
            # violation list would burn one of only max_patch_rounds rounds
            left_a_trace=lambda o: _read_or_none(ctx.milestone) != pre)
        if not Path(ctx.milestone).exists():
            if draft is not None:
                Path(ctx.milestone).write_text(draft, encoding="utf-8")
                _log("meta/distill: milestone.md missing after a revision round -> "
                     "restored previous draft and re-prompting")
            else:
                dead_rounds += 1
                if dead_rounds >= 2:
                    raise AgentError(
                        "two distill rounds in a row ended with no milestone file — "
                        "the agent sessions are dying before they write "
                        f"(last output tail: {out[-160:]!r})")
        else:
            dead_rounds = 0
        violations = _validate_milestone_file(ctx.milestone, authoring=True)
        try:
            draft = Path(ctx.milestone).read_text(encoding="utf-8")
        except OSError:
            pass  # nothing written yet this round; keep the last good draft
        if not violations:
            break
        _log(f"meta/distill: milestone still has {len(violations)} violation(s) -> re-prompt")
    _log(f"meta/distill: authored milestone v1 (violations={violations[:3] or 'none'})")
    _seed_target_ms(ctx)
    if ctx.meta_status_path:
        append_meta_record(ctx.meta_status_path, MetaRecord(
            meta_step=meta_step, cause_class="DISTILL", deficient_locus="(initial)",
            patch_class="author", changed_entries=[], patch_direction="author milestone v1 from oracle",
            outcome="AUTHORED", milestone_violations=violations))
    return MetaResult(out, "author", [], None, [], certified=None, milestone_violations=violations)


def run_two_layer(ctx: LoopContext, max_meta: int = 3) -> dict:
    """Inner loop -> (plateau / stuck / exhausted) -> meta patch -> resume, bounded by max_meta.

    Milestone patches are transactional: after a patch's inner segment runs, the patch
    is attributed (best-before vs best-after). A patch that produced no gain is rolled
    back — milestone snapshot restored AND workspace reset — and recorded REVERT_PATCH,
    which the next diagnosis reflects on so the meta agent does not re-propose it.
    A patch whose segment produced no metric reading at all is HELD instead (see
    _judge_patch): pre-green construction survives the wall, and the transaction stays
    open for a joint judgment once a metric exists.
    """
    meta_steps = 0
    start_iter = len(load_records(ctx.status_path))
    # Seed the stall-pricing toolkit BEFORE capturing baseline_ref: the seed is a
    # workspace commit, and foundation patches reset the workspace to baseline_ref —
    # a seed committed after that capture is discarded by the first such reset, and
    # every later pricing job dies file-not-found (exactly the blindness this
    # mechanism exists to prevent). Seeding first puts the recipe below the reset
    # floor, so every lineage commit (the future best included) carries it.
    _seed_ws_stall_tools(ctx)
    baseline_ref = ctx.backend.head()  # foundation patches restart from here, not HEAD
    if not ctx.harness_anchor:
        ctx.harness_anchor = baseline_ref or ""
    pending: _PendingPatch | None = None
    # Correctness state of the segment about to run: None = unknown (a fresh workspace —
    # the first full eval decides; a scaffolded operator enters scaffold mode on its
    # first FAIL), then re-derived from whatever checkpoint each meta reset lands on.
    seg_green: bool | None = None
    resume_anchor: str | None = None   # first segment only; see _resume_anchor
    if start_iter:
        # A process starting over an existing trajectory (budget / certificate
        # resume) is not a fresh workspace: the records say whether the tree it
        # resumes on has ever been green. Seeding "unknown" here made the first
        # post-resume construction row a paid REVERT (observed on a GDN run).
        resumed = load_records(ctx.status_path)
        seg_green = _green_on_resume(resumed, baseline_ref)
        if seg_green:
            # The anchor is the last KEEP, not HEAD: a HEAD no record owns carries
            # commits no eval verified (see _resume_anchor).
            resume_anchor = _resume_anchor(resumed, baseline_ref)
        if seg_green is not None:
            _log(f"resume: lineage at {str(baseline_ref)[:10]} reads "
                 f"{'green' if seg_green else 'under construction'} from the "
                 f"{start_iter} recorded iteration(s)"
                 + ("" if seg_green else " -> scaffold mode until the gate row")
                 + (f"; rollback anchor = last KEEP {str(resume_anchor)[:10]} (HEAD "
                    f"carries commits no record verified)"
                    if seg_green and resume_anchor != baseline_ref else ""))
    # An in-place reopen whose closure touches M0 enters green on the best tree
    # and rebuilds the foundation there; its M0 close is the next M0-gate KEEP,
    # and the audit is keyed on that row (see run_inner_loop audit_on_gate).
    audit_gate: str | None = None
    # distillation entry: if the milestone is an empty skeleton, meta authors v1 from
    # the oracle BEFORE the first inner loop (the loop then revises it on plateau).
    if _needs_authoring(ctx):
        # Distillation without a measured ref writes gates it cannot anchor: every
        # number would be an [ESTIMATE], including target_ms, which is the stop
        # threshold. The loop measures the reference first (loop/cli.py) — that is the
        # whole point of doing this before v1 rather than backfilling after M0.x.
        if ctx.ref_baseline is None:
            _log("meta/distill: no ref baseline and the operator does not declare one "
                 "-> refusing to distill. The loop measures the reference loop.toml names "
                 "(ref_mode / ref_cmd) before distillation; none is declared here")
            return {"stop": "no_ref_baseline", "meta_steps": 0}
        _log(f"meta/distill: ref baseline {ctx.ref_baseline.ms} ms "
             f"(job {ctx.ref_baseline.job or '?'}, ncu={'yes' if ctx.ref_baseline.has_ncu else 'no'})"
             " -> authoring v1 anchored to it")
        # Reconnaissance first. Under blackbox it is the input that stands in for
        # the source the teacher is not allowed to read. A source-mode run reconnoitres
        # too whenever a recon tree is wired (the default; loop.toml recon_op = ""
        # declares a source-only control run): the oracle was assumed to answer
        # the same questions at no cost, and for templated references it does
        # not — the SASS-level issue order, the sync topology and the real
        # resource numbers are what the teacher had to guess, and guessed wrong
        # (grouped GEMM: -30% at M0).
        if ctx.oracle_mode == "blackbox" or ctx.recon_backend is not None:
            recon = run_recon_step(ctx)
            _log(f"meta/recon: {recon.rounds} run(s), "
                 f"{'finished' if recon.finished else 'budget exhausted'}"
                 f"{f', report {recon.report}' if recon.report else ''}")
        distilled = run_distill_step(ctx)
        # v1 is the one write with no rollback behind it: a patch that will not conform
        # is restored from the pre-patch snapshot, but there is no earlier milestone to
        # fall back to here. And v1 shapes everything downstream — later patches edit it
        # in place, so its defects propagate into every version. Refuse to start rather
        # than spend a GPU budget optimizing against a milestone that failed its gate.
        if distilled.milestone_violations:
            _log(f"meta/distill: v1 still non-conformant after {ctx.max_patch_rounds} "
                 f"round(s) -> refusing to start: {distilled.milestone_violations[:3]}")
            return {"stop": "distill_malformed", "meta_steps": 0,
                    "violations": distilled.milestone_violations}
    while True:
        # Force the M0 audit whenever the segment starts non-green (fresh workspace
        # or a reset into scaffolding): its first green IS an M0 close, including
        # the rebuilds after foundation patches.
        reason = run_inner_loop(
            ctx, start_iter, initial_green=seg_green, initial_good_ref=resume_anchor,
            audit_on_green=ctx.m0_audit and seg_green is not True,
            audit_on_gate=audit_gate if ctx.m0_audit else None,
            segment_records_before=(pending.records_before if pending else None),
            gate_contract=(pending.gate_contract if pending else None))
        _log(f"inner loop stopped: {reason} (meta_steps={meta_steps})")
        audit_gate = None   # one audit per reopen, whatever the stop was
        resume_anchor = None   # later segments enter on a checkpoint a record owns
        if reason == "first_green":
            # The M0 gate verified the student against the milestone's own declared
            # values; nothing yet verified the MILESTONE against the hardware. Audit
            # now, before a whole segment is built on the foundation. The open
            # pending patch (if any) is deliberately NOT judged here: its segment
            # was cut short at first green, and judging a rebuild patch at M0 is
            # predetermined (best_after < best_before) — it would revert correct
            # foundations for running out of runway, not for being wrong.
            if meta_steps >= max_meta:
                _log("meta/m0-audit: meta budget exhausted -> skipping audit")
                seg_green = True
                start_iter = len(load_records(ctx.status_path))
                continue
            records = load_records(ctx.status_path)
            best_before = _best_ms(_on_cluster(records, ctx), ctx.gate_window)
            milestone_before = _read_milestone(ctx)
            meta = run_meta_step(ctx, best_before=best_before,
                                 milestone_before=milestone_before, audit=True)
            _apply_phase_close(ctx, meta)
            if meta.patch_class == "audit-clean":
                _log("meta/m0-audit: AUDIT_CLEAN — milestone consistent with the "
                     "hardware truth -> resuming the segment")
                seg_green = True
                start_iter = len(load_records(ctx.status_path))
                continue
            if meta.patch_class == "kb-example":
                # student-side defect: recipe certified, same workspace resumes green
                meta_steps += 1
                ctx.fresh_recipe = meta.recipe
                ctx.g5_note = meta.student_directive or ""
                seg_green = True
                start_iter = len(load_records(ctx.status_path))
                continue
            if meta.patch_class in ("unpriced-reopen", "reopen-capped"):
                # the audit named a reopen it could not price (or the reopen budget
                # is spent): nothing landed, the refusal is in the meta memory, the
                # segment resumes as built — a priced version of the same reopen
                # can come at the next stop
                meta_steps += 1
                _log("meta/m0-audit: UNPRICED_REOPEN — reopen refused for want of a "
                     "measured price -> resuming the segment unpatched"
                     if meta.patch_class == "unpriced-reopen" else
                     "meta/m0-audit: REOPEN_CAPPED — reopen budget spent -> resuming the "
                     "segment unpatched")
                seg_green = True
                start_iter = len(load_records(ctx.status_path))
                continue
            if meta.milestone_violations:
                # a malformed audit patch never landed: restore and resume unpatched
                # rather than burning patch retries here — if the defect is real it
                # will resurface at the next actionable stop with a plateau behind it.
                _write_milestone(ctx, milestone_before)
                if ctx.meta_status_path:
                    update_last_outcome(ctx.meta_status_path, outcome="REVERT_PATCH")
                _log("meta/m0-audit: patch REVERT_PATCH (malformed, never landed) "
                     "-> resuming unpatched")
                seg_green = True
                start_iter = len(load_records(ctx.status_path))
                continue
            meta_steps += 1
            resume_ref, reopened = _resolve_resume_in_place(ctx, meta, records, baseline_ref)
            ctx.backend.reset(resume_ref)
            seg_green = _green_at_ref(records, resume_ref)
            ctx.fresh_recipe = None
            if reopened:
                # In place: the tree is the best green KEEP, nothing rolled
                # back — the REOPEN row + reopen_note describe the workspace;
                # a standing rollback note would contradict them.
                ctx.resume_note = ""
                _note_reopen(ctx, meta, resume_ref)
                audit_gate = _reopen_audit_gate(reopened)
            else:
                ctx.reopen_note = ""
                _set_resume_note(ctx, meta, resume_ref, baseline_ref)
            ctx.g5_note = ""
            if pending is None:
                pending = _PendingPatch(best_before, milestone_before,
                                        _best_good_ref_here(records, ctx)
                                        or baseline_ref,
                                        records_before=len(load_records(ctx.status_path)),
                                        cluster=_active_cluster(ctx),
                                        gate_contract=dict(meta.gate_contract),
                                        reopened=list(reopened))
            else:
                # the prior patch's segment never got to prove itself; the audit patch
                # joins its open transaction — one joint judgment at the next
                # actionable stop, one rollback point restoring the state before BOTH.
                _log("meta/m0-audit: patch joins the open milestone transaction")
                pending.gate_contract.update(meta.gate_contract)
                pending.reopened = sorted(set(pending.reopened) | set(reopened))
            start_iter = len(load_records(ctx.status_path))
            continue
        # judge the previous milestone patch now that its inner segment has run.
        # HOLD_PATCH keeps the transaction open: the segment had no metric reading,
        # so the patch stays landed (and on a budget stop, on disk for the next run)
        # and the next patch joins it for one joint judgment.
        if pending is not None:
            if _judge_patch(ctx, pending) != "HOLD_PATCH":
                pending = None
        # inner stop reasons the meta loop can act on. All three are milestone defects:
        # plateau = levers exhausted, stuck = student can't move, exhausted = the decision
        # graph ran out below target. The phase order ends at Mk (meta_milestone.md §3) —
        # the student has no free phase to fall back on, so "the milestone ran out" must
        # route to the agent that can extend it rather than end the run with the meta
        # budget unspent.
        if reason not in _META_ACTIONABLE:
            return {"stop": reason, "meta_steps": meta_steps}
        records = load_records(ctx.status_path)
        best_before = _best_ms(_on_cluster(records, ctx), ctx.gate_window)
        good_ref = _best_good_ref_here(records, ctx) or baseline_ref
        # Before spending the meta budget on a gap-driven diagnosis, make sure the
        # gap exists: stall pricing, the verdict probes and the prescription all
        # reason from the residual, and a residual inside the noise band may be
        # nothing but the launch-time anchor (see _phantom_gap_reanchor). This
        # runs BEFORE the meta-budget return: a run whose target is genuinely
        # met must never be recorded as "meta_budget" for want of budget to
        # check the residual with.
        reanchored = _phantom_gap_reanchor(ctx, best_before)
        if reanchored is not None:
            return {"stop": "target", "meta_steps": meta_steps,
                    "reanchor": reanchored}
        if meta_steps >= max_meta:
            return {"stop": "meta_budget", "meta_steps": meta_steps}
        # The note (if a re-anchor confirmed the gap) rides every diagnosis of
        # this meta entry — the malformed-patch retry below re-runs the diagnosis
        # and must see the same re-anchored residual, not the stale one.
        note, ctx.reanchor_note = ctx.reanchor_note, None
        while True:
            milestone_before = _read_milestone(ctx)
            meta = run_meta_step(ctx, best_before=best_before,
                                 milestone_before=milestone_before,
                                 reanchor_note=note)
            meta_steps += 1
            _apply_phase_close(ctx, meta)
            if meta.patch_class == "infeasible-stop":
                # the teacher's stop request ends the run here, budget unspent:
                # the stop is the run's verdict, the notice its record.
                return {"stop": "infeasible", "meta_steps": meta_steps,
                        "notice": str(Path(ctx.milestone).parent / "infeasible_stop.md")}
            if meta.patch_class in ("unpriced-reopen", "reopen-capped"):
                # nothing landed and nothing to judge: the refusal is now in the
                # reflection, so retry the META step (not the segment — resuming
                # the same plateaued state is predetermined), budget permitting
                if meta_steps >= max_meta:
                    return {"stop": "meta_budget", "meta_steps": meta_steps}
                _log("meta: UNPRICED_REOPEN — reopen refused for want of a measured "
                     "price -> retrying the meta step on the same state"
                     if meta.patch_class == "unpriced-reopen" else
                     "meta: REOPEN_CAPPED — reopen budget spent -> retrying the meta "
                     "step on the same state")
                continue
            if meta.patch_class == "infeasible-refused":
                # same shape: the refused stop request is in the reflection,
                # nothing landed — re-diagnose the same state, budget permitting
                if meta_steps >= max_meta:
                    return {"stop": "meta_budget", "meta_steps": meta_steps}
                _log("meta: INFEASIBLE_REFUSED — stop request refused by the reachability "
                     "check -> retrying the meta step on the same state")
                continue
            if meta.patch_class == "kb-example" or not meta.milestone_violations:
                break
            # a malformed patch that never became §9-conformant never lands: roll back —
            # and retry the META step, not the inner segment. The rollback restores
            # exactly the state whose segment already stalled, so re-running the student
            # on it is predetermined (a re-plateau or an instant EXHAUSTED): GPU evals
            # and agent turns for no information — the redundant turn the landing gate
            # alone could not save. The next diagnose sees this
            # REVERT_PATCH through the reflection and must propose differently.
            _write_milestone(ctx, milestone_before)
            if pending is None:
                ctx.backend.reset(good_ref)
                ctx.resume_note = ""  # restored tree: the never-landed patch invalidated nothing
            # else: an open held transaction means the workspace tip IS the pre-green
            # construction HOLD_PATCH preserved — and good_ref predates the whole chain
            # (a metric-less segment contributes no measured KEEP), so resetting here
            # would wipe exactly that work over a patch that never even landed. Restore
            # the milestone only (same call the m0-audit malformed branch makes) and
            # leave the workspace where the held segment left it.
            if ctx.meta_status_path:
                update_last_outcome(ctx.meta_status_path, outcome="REVERT_PATCH")
            _log("meta: patch REVERT_PATCH (malformed milestone, never landed) "
                 "-> retrying the meta step on the restored state")
            if meta_steps >= max_meta:
                return {"stop": "meta_budget", "meta_steps": meta_steps}
        resume_ref, reopened = _resolve_resume_in_place(ctx, meta, records, baseline_ref)
        ctx.backend.reset(resume_ref)
        seg_green = _green_at_ref(records, resume_ref)
        if reopened:
            ctx.resume_note = ""      # in place: nothing rolled back (see the audit path)
            _note_reopen(ctx, meta, resume_ref)
            audit_gate = _reopen_audit_gate(reopened)
        else:
            ctx.reopen_note = ""
            _set_resume_note(ctx, meta, resume_ref, baseline_ref)
        if meta.patch_class == "kb-example":
            # point the student at the just-certified recipe so the G3 loop actually closes
            # (a passive KB addition was ignored by the student in the live run).
            ctx.fresh_recipe = meta.recipe
            ctx.g5_note = meta.student_directive or ""
        else:  # a milestone patch is transactional; a KB addition is not
            ctx.fresh_recipe = None
            ctx.g5_note = ""
            if pending is None:
                pending = _PendingPatch(best_before, milestone_before, good_ref,
                                        records_before=len(load_records(ctx.status_path)),
                                        cluster=_active_cluster(ctx),
                                        gate_contract=dict(meta.gate_contract),
                                        reopened=list(reopened))
            else:
                # the prior patch was HELD (metric-less segment): this patch joins its
                # open transaction — one joint judgment, one rollback point restoring
                # the state before BOTH (mirrors the m0-audit join above).
                _log("meta: patch joins the open milestone transaction (prior patch held)")
                pending.gate_contract.update(meta.gate_contract)
                pending.reopened = sorted(set(pending.reopened) | set(reopened))
        start_iter = len(load_records(ctx.status_path))


# A confirmation sample this far from the KEEP reading of the SAME sha is not
# noise: nodes differ by ~±6% absolute and a same-job ratio carries a 1-5% floor,
# so two honest draws of one commit sit within ~12% of each other at the very
# worst. Beyond that the MACHINE is the suspect (live case: a probe's orphaned
# launch target held the GPU, three confirmation jobs read 23.3 ms against a KEEP
# of 10.0 ms). The sample is reported with its deviation and the confirmation
# flags the KEEP reading as disputed; a human decides which to trust.
CONFIRM_NODE_BAND_PCT = 15.0


def _keep_reading_of(records, sha: str, window: str) -> float | None:
    """The gate-window reading the trajectory holds for ``sha`` (its latest
    KEEP row) — the number a confirmation sample is checked against."""
    for rec in reversed(list(records)):
        if rec.verdict != "KEEP" or not rec.sha or not sha:
            continue
        if not (rec.sha.startswith(sha) or sha.startswith(rec.sha)):
            continue
        ms = rec.device_median_ms if window == "device" else rec.bwd_median_ms
        if ms:
            return float(ms)
    return None


def _job_node(be, job: str) -> str:
    """Where a confirmation sample ran: the backend's machine when it names one (every
    job on one machine is the SAME node draw), else the job itself."""
    machine = getattr(be, "machine", "")
    if isinstance(machine, str) and machine:
        return machine
    return f"job:{job}" if job and job != "?" else ""


def _pct_off(ms: float | None, keep_ms: float | None) -> float | None:
    if not ms or not keep_ms:
        return None
    return 100.0 * (ms / keep_ms - 1.0)


def _confirm_workload_medians(samples: list[dict], window: str) -> list[dict]:
    """Per-workload medians across confirmation samples (multi-workload cards).

    Workloads are matched by name, in the order the first sample lists them;
    a workload some sample did not report is summarized over the ones that did.
    [] when no sample carried per-workload readings.
    """
    ms_key, ratio_key = (("device_median_ms", "vs_ref_device") if window == "device"
                         else ("bwd_median_ms", "vs_ref"))
    order: list[str] = []
    by: dict[str, list[dict]] = {}
    for s in samples:
        for w in s.get("workloads") or ():
            name = w.get("name")
            if name not in by:
                order.append(name)
                by[name] = []
            by[name].append(w)
    out = []
    for name in order:
        rows = by[name]
        item = {"name": name, "n": len(rows)}
        mss = [r[ms_key] for r in rows if isinstance(r.get(ms_key), (int, float))]
        rts = [r[ratio_key] for r in rows if isinstance(r.get(ratio_key), (int, float))]
        if mss:
            item[ms_key] = statistics.median(mss)
        if rts:
            item[ratio_key] = statistics.median(rts)
        out.append(item)
    return out


def run_best_confirmation(ctx: LoopContext, jobs: int = 3, max_extra: int = 2,
                          ref: str | None = None,
                          seed_samples: list | None = None) -> dict | None:
    """Post-stop confirmation: re-measure the best KEEP sha ``jobs`` times.

    One eval job is one node draw — the cluster's H100 nodes are bimodal (~±6%
    absolute) and even same-job ratios carry a 1-5% noise floor, so a single draw
    cannot separate a real win from a lucky node (live case: one fwd run's
    single-job 0.2585 ms best re-measured at 0.282 ms, three jobs, spread 0.7%).
    Per-iteration verdicts stay single-job for throughput; only the run's final
    claim pays for repetition. Runs after the loop stops (any reason) so it never
    competes with iterations for wall-clock budget. Off per task via loop.toml
    [loop] confirm_jobs = 0.

    When the ref rides the job the claim is the median same-job ratio. When it does
    not (blackbox runs — single-ref contract, no stand-in library in the job), the
    jobs yield an absolute median only; the caller re-anchors the ref through its
    own channel over the same period and forms the ratio median-vs-median.
    """
    if jobs <= 0:
        return None
    # ``ref``: the sha to re-measure; default the standing best. The same-job
    # parity stop names its own sha — the row that read parity need not be the
    # absolute best (a slow-node draw can be at parity while a fast-node draw
    # holds the watermark).
    records = load_records(ctx.status_path)
    ref = ref or _best_good_ref_here(records, ctx)
    if not ref:
        _log("confirm: no green KEEP iteration on record — nothing to confirm")
        return None
    keep_ms = _keep_reading_of(records, ref, ctx.gate_window)
    be = ctx.backend
    push_ref = getattr(be, "push_ref", None)
    if push_ref is None:
        _log("confirm: backend cannot push refs — skipping (HEAD may not be the best sha)")
        return None
    branch = f"{ctx.branch_prefix}-confirm"
    push_ref(ref, branch)
    # ``seed_samples``: readings of THIS sha already bought (the same-job parity
    # stop's re-measure) count toward ``jobs`` instead of being bought again.
    samples: list[dict] = [dict(s) for s in (seed_samples or []) if s.get("ms")]
    if samples:
        _log(f"confirm: {len(samples)} same-period sample(s) of {ref[:10]} already "
             f"in hand — buying {max(0, jobs - len(samples))} more")
    consecutive_errors = 0
    attempts = 0
    while len(samples) < jobs and attempts < jobs + max_extra:
        attempts += 1
        try:
            log = be.submit_gpu(branch, "all")
        except BackendError as e:
            consecutive_errors += 1
            _log(f"confirm: job failed ({e})")
            if consecutive_errors >= 2:  # infra is down, not unlucky — stop burning jobs
                break
            continue
        consecutive_errors = 0
        ev = _enforce_single_ref(ctx, parse_eval(log))
        job = getattr(be, "last_job_id", "") or "?"
        if ctx.gate_window == "device":
            ms, ref_ms, ratio = ev.device_median_ms, ev.ref_device_ms, ev.vs_ref_device
        else:
            ms, ref_ms, ratio = ev.bwd_median_ms, ev.ref_median_ms, ev.vs_ref
        if ratio is None and ms and ref_ms:
            ratio = ref_ms / ms
        if not ev.correctness or ms is None:
            # a correctness break on the best sha is a red flag, not just a bad sample.
            # A missing RATIO is deliberately not a rejection: a run whose bench
            # carries no in-job ref (misjudged ref_rides_job, target-ms-only) would
            # otherwise reject every sample and burn the whole job budget for a None —
            # an absolute median is still the claim repetition exists to firm up.
            why = ev.error or f"no {ctx.gate_window}-window reading in the log"
            _log(f"confirm: job {job} unusable ({why})")
            continue
        sample = {"job": job, "ms": ms, "ref_ms": ref_ms, "ratio": ratio,
                  "node": _job_node(be, job)}
        if ev.workloads:
            sample["workloads"] = ev.workloads
        off = _pct_off(ms, keep_ms)
        if off is not None and abs(off) > CONFIRM_NODE_BAND_PCT:
            sample["off_keep_pct"] = round(off, 2)
            _log(f"confirm: job {job} {ctx.gate_window}={ms} is {off:+.1f}% off the KEEP "
                 f"reading {keep_ms} of {ref[:10]} — outside the ±{CONFIRM_NODE_BAND_PCT:.0f}% "
                 f"node band: the reading stands, and the machine ({sample['node'] or '?'}) "
                 f"is worth a look (another process on the GPU, clocks)")
        samples.append(sample)
        _log(f"confirm: job {job} {ctx.gate_window}={ms}"
             + (f" ref={ref_ms} ratio={ratio:.4f}" if ratio is not None else ""))
    if not samples:
        _log("confirm: no usable sample — the claim stays single-job; check the jobs above")
        return None
    med_ms = statistics.median(s["ms"] for s in samples)
    ratios = [s["ratio"] for s in samples if s["ratio"] is not None]
    short = " (SHORT of requested — treat as provisional)" if len(samples) < jobs else ""
    out = {"sha": ref, "window": ctx.gate_window, "requested": jobs,
           "jobs": samples, "median_ms": med_ms, "keep_ms": keep_ms,
           "node_band_pct": CONFIRM_NODE_BAND_PCT}
    med_off = _pct_off(med_ms, keep_ms)
    out["keep_disputed"] = bool(med_off is not None and abs(med_off) > CONFIRM_NODE_BAND_PCT)
    if out["keep_disputed"]:
        _log(f"confirm: median {med_ms} ms is {med_off:+.1f}% off the KEEP reading {keep_ms} "
             f"of {ref[:10]} — the confirmation DISPUTES the trajectory's reading; the "
             f"number below is what fresh jobs measured, the KEEP row is the one to doubt")
    nodes = sorted({s["node"] for s in samples if s.get("node")})
    if len(samples) > 1 and len(nodes) == 1:
        _log(f"confirm: all {len(samples)} sample(s) ran on one machine ({nodes[0]}) — that "
             f"is ONE node draw, not {len(samples)}; their spread says nothing about "
             f"node-to-node variation")
    out["nodes"] = nodes
    if per_wl := _confirm_workload_medians(samples, ctx.gate_window):
        # the headline ratio is the geomean over the card's workloads; keep each
        # workload's own median beside it, so a win carried by one workload and a
        # loss on another stays visible in the record
        out["workloads"] = per_wl
        _log("confirm: per-workload medians: " + describe_workload_readings(per_wl))
    if ratios:
        out["median_ratio"] = statistics.median(ratios)
        _log(f"confirm: median ratio {out['median_ratio']:.4f} over {len(ratios)} job(s) "
             f"@ {ref[:10]} ({ctx.gate_window} window){short}")
    else:
        if ctx.ref_rides_job:
            _log("confirm: no job carried an in-job ref reading (ref_rides_job "
                 "misconfig or bench defect) — falling back to the absolute median")
        _log(f"confirm: median {med_ms} ms over {len(samples)} job(s) @ {ref[:10]} "
             f"({ctx.gate_window} window; ref measured separately){short}")
    return out


def apply_window_credit(ctx: LoopContext, confirm: dict) -> dict | None:
    """Price the declared contract tax into the confirmation record.

    Reads the confirmed sha's banked per-kernel table (falling back to the
    sitting best KEEP's) and the reference's, the contract's ``price`` regexes
    on each side, and the ref median the confirmation settled on (same-job or
    median-vs-median), and writes ``confirm["window_credit"]``: per-side matched
    shares, the credited readings and the credited ratio next to the raw one.
    The raw ``median_ratio`` is untouched — the target stop, the KEEP ledger and
    every cross-run ranking keep reading it; the credit says how much of the
    residual no kernel-quality lever can remove.

    Before any share is applied the table is reconciled with the timed window
    it is meant to describe (``timing_audit.ncu_coverage``, the mis-aimed-profile
    arithmetic): a card whose ncu task does not confine the profiler to the
    timed call carries the bench's setup launches in its table, and a share read
    off that table is diluted (or a same-named prep launch inflates it). Such a
    credit, and one whose regex matched nothing, is recorded as **unusable**
    with its reason — never as a 0-share credited ratio. Returns the credit
    dict, or None when nothing was priced (the log says why either way).
    """
    from forgekernel.window_contract import credited_ratio, window_credit
    contract = ctx.window_contract
    if contract is None or not contract.declared:
        return None
    if not contract.priced_entries:
        if contract.unpriced_entries:
            _log("confirm: window contract has price-class asymmetries without a "
                 "kernel regex — credited ratio not computed "
                 f"({', '.join(a.name for a in contract.unpriced_entries)})")
        return None
    records = load_records(ctx.status_path)
    sha = confirm.get("sha") or ""
    rec = next((r for r in records if sha and r.sha == sha and r.kernels), None)
    if rec is None:
        best = _best_keep_record(records, ctx.gate_window)
        rec = best if best is not None and best.kernels else None
    ref = _ref_for_window(ctx)
    ref_ms = confirm.get("ref_median_ms")
    if ref_ms is None:
        refs = [s.get("ref_ms") for s in confirm.get("jobs", []) if s.get("ref_ms")]
        ref_ms = statistics.median(refs) if refs else None
    credit = window_credit(
        contract, rec.kernels if rec else None, ref.kernels if ref else None,
        student_profiler=rec.profiler if rec else None,
        # the KEEP row's own reading: the table was profiled on that tree
        student_ms=gate_ms(rec, ctx.gate_window) if rec else None,
        ref_profiler=ref.ncu if ref else None,
        ref_ms=ref.device_anchor_ms if ref else None)
    if not credit:
        return None
    out = {**credit, "source_iter": rec.iter if rec else None}
    if not credit["usable"]:
        _log("confirm: window credit UNUSABLE — " + "; ".join(credit["why"])
             + ". No credited ratio is reported; the raw ratio stands alone.")
        confirm["window_credit"] = out
        return out
    cr = credited_ratio(credit, confirm.get("median_ms"), ref_ms)
    if cr:
        out.update(cr)
        _log(f"confirm: window credit — student side {cr['student_fraction'] * 100:.1f}% "
             f"({', '.join(credit['student']['priced']) or 'none'}), reference side "
             f"{cr['reference_fraction'] * 100:.1f}% "
             f"({', '.join(credit['reference']['priced']) or 'none'}); "
             f"raw ratio {cr['raw_ratio']:.4f} → credited {cr['credited_ratio']:.4f} "
             f"(student {confirm.get('median_ms')} → {cr['credited_student_ms']:.4f} ms, "
             f"ref {ref_ms} → {cr['credited_ref_ms']:.4f} ms). "
             f"The raw ratio remains the verdict.")
    else:
        _log("confirm: window credit — shares priced, but no ref median to form a "
             "credited ratio against")
    if credit.get("reconciliation"):
        _log("confirm: window credit — NCU table vs timed window: "
             + ", ".join(f"{k} {v:.2f}" for k, v in credit["reconciliation"].items()))
    confirm["window_credit"] = out
    return out
