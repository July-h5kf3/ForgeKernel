"""Structured inner-loop trajectory: status.jsonl (machine SSOT) + Layer-1 digest.

Each iteration appends one IterationRecord. The record holds OBJECTIVE facts only
— the optimization technique applied and the measured outcome (correctness, perf,
NCU) — for BOTH kept and reverted iterations. The inner agent's SUBJECTIVE analysis
(why it expected a gain, its bottleneck theory) is deliberately NOT recorded, so
the meta agent diagnoses from measured hardware truth rather than the student's
prose. build_digest() assembles the operator-agnostic Layer-1 digest (design §4.1bis).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class IterationRecord:
    iter: int
    decision_id: str
    change: str            # objective: the optimization technique / code change applied
    correctness: bool | None  # None = not evaluated (a scaffold-phase NEUTRAL row)
    verdict: str           # KEEP | REVERT | NEUTRAL (NEUTRAL = committed, not evaluated)
    bwd_median_ms: float | None = None
    mfu: float | None = None
    # In-job reading of THE declared reference — the single-ref contract: whatever
    # library rides the eval job to be timed back-to-back with the student must BE
    # this run's declared ref, never a stand-in (the retired "instrument"/ruler
    # concept: a second library timed only to calibrate node speed. It made every
    # vs_* number ambiguous — a fwd kernel's 1.0418x read as "beat FA" when the
    # denominator was cuDNN). Runs whose ref cannot ride the job (blackbox: the ref
    # package is readable source) time only the student, and these stay None —
    # ratios for such runs are median-vs-median across jobs, never same-job.
    # When present, the same-job ratio is what cross-round comparisons are read
    # from: absolute latency for one unchanged commit moves ~6% between jobs
    # (node bimodality), wider than most single-round gains.
    ref_median_ms: float | None = None
    vs_ref: float | None = None
    # Second timing window, when the harness measures one (see EvalResult in
    # loop/parse.py for the full account). bwd_median_ms is a per-call window and
    # includes host work; device_median_ms is a pipelined batch window ≈ device time.
    # The two sides of a ratio can pay different host taxes, so vs_ref (e2e) is a
    # call-latency claim while vs_ref_device is the kernel-quality claim. None when
    # the harness only times one window.
    device_median_ms: float | None = None
    ref_device_ms: float | None = None
    vs_ref_device: float | None = None
    # Precondition of the device window (see EvalResult in loop/parse.py): how long
    # the host alone takes to enqueue one call, and that over the device reading.
    # Recorded per iteration so the criterion lives in the trajectory the meta agent
    # and the digest read, not only in a job log nobody re-opens.
    host_enqueue_ms: float | None = None
    host_bound_ratio: float | None = None
    ref_host_bound_ratio: float | None = None
    # The basis device_median_ms was read on (plain_batch / cuda_graph /
    # cupti_kernel_span). host_bound_ratio is None BY DESIGN under a host-free
    # basis, and a reader of the trajectory needs this to tell "unmeasured" from
    # "cannot apply"; None = recorded before the field existed, or a bench that
    # did not declare one.
    device_window_basis: str | None = None
    metric_before: float | None = None
    error: str = ""        # objective failure signature (compile error / correctness diff) for REVERTs
    profiler: dict = field(default_factory=dict)  # NCU key metrics — recorded for ALL iters incl. REVERT
    sha: str = ""          # commit sha of the landed iteration (for prefix-reuse resume); "" if none
    # Which GPU machine the eval job ran on (backend.last_cluster at eval time). One
    # kernel's absolute latency moves more between machines than most
    # single-round gains, so every absolute-ms comparison the driver makes —
    # best, plateau, patch judgment, target — reads same-machine rows only.
    # "" = recorded before the field existed, or a backend that cannot say;
    # the driver takes such rows to be from the run's own machine.
    cluster: str = ""
    # HEAD the student built this iteration on. Lineage, not order: after a
    # reset / meta reopen the next iteration's base is an earlier tree, and the
    # axis harvest pairs a KEEP with its ancestor's census, not its neighbour's.
    base_sha: str = ""
    # The tree's SASS contract axes (orch_tools summary: sync topology, in-flight
    # depth, supply order, cache-policy bits) off the same NCU job, when the
    # per-iteration census ran. Kept OUT of ``profiler`` so the Layer-1 digest
    # stays flat; the harvest reads it to bill a KEEP that moved an axis.
    orch_summary: dict = field(default_factory=dict)
    # Every kernel the student's profiled window launched (inner.ncu.kernel_table:
    # name / launches / time_ns / pct), off the same NCU job as ``profiler``. Kept
    # OUT of ``profiler`` (flat digest); read by the timed-window contract
    # (forgekernel.window_contract) to price the launches that exist only to
    # rebuild a reference-side input, and shown to the teacher next to the
    # reference's table. [] = recorded before the field existed / no NCU.
    kernels: list = field(default_factory=list)
    # The bench's window-outside prep step (eval JSON ``performance.prep``): what
    # it produced (name / shape / dtype, already held to window.toml by the
    # driver) and what it cost. Banked so the outside step is a reading on the
    # teacher's ledger, never a place work can be moved into unseen. {} = no
    # prep step / recorded before the field existed.
    prep: dict = field(default_factory=dict)
    # Multi-workload cards: the readings above are the geometric mean over
    # ``n_workloads`` workloads of one fixed shape; ``workloads`` keeps the
    # per-workload readings (see EvalResult in loop/parse.py). 1 / [] for a
    # single-input bench and for rows recorded before the field existed.
    n_workloads: int = 1
    workloads: list = field(default_factory=list)
    # A row whose tree is the tree of an earlier evaluated row (nothing changed
    # outside notes/, or a gate the delivery itself closed) carries THAT row's
    # reading instead of buying a fresh eval: ``reading_of`` is the iter of the
    # row whose job produced every measured field above. None = this row's own
    # job. A re-measure of an unchanged tree is one more noise draw — kept as a
    # reading it lets the best be set by luck (see loop/driver.py
    # _unchanged_tree_reading).
    reading_of: int | None = None
    # The precision gate's reading of this row's eval (loop/precision_gate.summarize):
    # contract, status (pass / fail / unavailable / unreported), the worst
    # e_cand / e_ref and the stress profiles' statuses. {} = contract off, no eval, or
    # recorded before the field existed.
    precision: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.verdict not in ("KEEP", "REVERT", "NEUTRAL"):
            raise ValueError(f"invalid verdict: {self.verdict!r}")


_FIELDS = {f.name for f in fields(IterationRecord)}

# The two timing windows a bench can measure (see EvalResult in loop/parse.py for
# the full account). "e2e" is the per-call window — host work lands inside it —
# and has been the gate since the series began; "device" is the pipelined batch
# window ≈ device time. Which one drives the loop's verdicts (best, target stop,
# plateau, rollback anchor) is the operator's gate_window, chosen once per run:
# gating a training-shaped kernel on host-inclusive latency optimizes python
# plumbing the deployment would never pay, and gating an inference-shaped one on
# device time hides a host tax the caller pays on every call. Neither window is
# right in general, so the choice is config, not a constant — but the DEFAULT is
# device: the benchmark question is kernel quality, the
# claim every reference comparison and cross-run ranking is read from. The e2e
# window remains available per run (--gate-window / loop.toml) for kernels whose
# deployment pays the host tax on every call. The default is deliberately loud on
# under-equipped runs: a baseline archive or bench that never measured the device
# window refuses to start / stops as gate_unmeasured rather than silently gating
# on the other window.
GATE_WINDOW_E2E = "e2e"
GATE_WINDOW_DEVICE = "device"
GATE_WINDOWS = (GATE_WINDOW_E2E, GATE_WINDOW_DEVICE)
GATE_WINDOW_DEFAULT = GATE_WINDOW_DEVICE


def gate_ms(rec, gate_window: str = GATE_WINDOW_DEFAULT) -> float | None:
    """The record's reading in the gate window; None when that window was not measured.

    Works on anything carrying ``bwd_median_ms``/``device_median_ms`` (IterationRecord,
    EvalResult). None under gate_window="device" is a bench-contract violation for an
    evaluated KEEP — the driver refuses it loudly rather than letting every verdict
    silently degrade to "no reading" (a device-gated loop where the bench only times
    e2e would never hit target and never plateau).
    """
    if gate_window not in GATE_WINDOWS:
        raise ValueError(f"gate_window must be one of {GATE_WINDOWS}, got {gate_window!r}")
    if gate_window == GATE_WINDOW_DEVICE:
        return rec.device_median_ms
    return rec.bwd_median_ms


def append_record(path: str | Path, rec: IterationRecord) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def load_records(path: str | Path) -> list[IterationRecord]:
    """Load records, dropping any unknown keys (tolerates pre-refactor status files)."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            out.append(IterationRecord(**{k: v for k, v in d.items() if k in _FIELDS}))
    return out


# A correctness FAIL whose per-tensor max_diff repeats BIT-IDENTICALLY across
# distinct failing attempts means the scored outputs never changed — the edits
# are not reaching the computation that produces them (outputs still hold their
# init constants: a broken inter-kernel handoff, a writeback that never lands, a
# stale compile cache). Real numeric bugs move when the code moves; four attempts
# in a row landing on 4.504025/7.560892/7.823140 to the last digit is not a
# numeric bug, and no one — student, meta, or log reader — was told (observed on
# an MHA bwd run: four such iterations were only diagnosed by hand afterwards).
_DIFF_SIG = re.compile(r"max_diff=([0-9]+\.[0-9]+|inf|nan)")
CONSTANT_OUTPUT_MIN_REPEATS = 3


def constant_output_flag(records: list[IterationRecord]) -> dict | None:
    """The trailing streak of REVERTs whose diff signatures are bit-identical.

    Only correctness-FAIL records that report max_diff values participate; rows
    without a diff (probe-blocked, build errors, missing eval) neither extend nor
    break the streak — the observed pattern interleaved both. Returns None below
    CONSTANT_OUTPUT_MIN_REPEATS identical signatures.
    """
    streak: list[int] = []
    sig: tuple[str, ...] | None = None
    for r in reversed(records):
        if r.verdict == "KEEP":
            break  # a KEEP means outputs demonstrably respond to the code
        if r.verdict != "REVERT" or not r.error:
            continue
        found = tuple(_DIFF_SIG.findall(r.error))
        if not found:
            continue
        if sig is None:
            sig = found
        if found != sig:
            break
        streak.append(r.iter)
    if sig is None or len(streak) < CONSTANT_OUTPUT_MIN_REPEATS:
        return None
    return {
        "kind": "constant_output",
        "iters": sorted(streak),
        "signature": list(sig),
        "hint": ("{} consecutive correctness FAILs have a bit-identical max_diff — the scored "
                 "output does not respond to these changes at all, so this is most likely not a "
                 "numerical bug: the output is still the initialization constant (a broken "
                 "hand-off of intermediates between kernels / a write-back that never lands / a "
                 "compile cache pointing at old code). First verify that \"the end-to-end output "
                 "is not constant and changes with the computation\", then fix the numerics."
                 .format(len(streak))),
    }


def build_digest(records: list[IterationRecord], target: dict,
                 gate_window: str = GATE_WINDOW_DEFAULT) -> dict:
    """Assemble the Layer-1 digest (target/achieved/curve/decisions/regressions/profiler).

    Objective-only: decisions carry the technique + measured metrics; regressions
    carry the objective failure signature; profiler carries NCU for every iteration
    that has it (KEEP and REVERT), tagged with the verdict.

    ``gate_window`` picks which timing window "best" and the target gap are read
    from — the digest must agree with the driver's verdicts, or the meta agent
    diagnoses a different run than the one that happened. Both windows' readings
    stay in the rows regardless; the window only decides which one ranks.
    """
    kept = [r for r in records
            if r.verdict == "KEEP" and gate_ms(r, gate_window) is not None]
    best = min(kept, key=lambda r: gate_ms(r, gate_window)) if kept else None
    achieved = None
    if best is not None:
        tv = target.get("value")
        best_ms = gate_ms(best, gate_window)
        achieved = {
            "bwd_median_ms": best.bwd_median_ms,
            "device_median_ms": best.device_median_ms,
            "gate_window": gate_window,
            "mfu": best.mfu,
            "iter": best.iter,
            "gap_pct": (best_ms - tv) / tv * 100 if tv else None,
        }
        if best.workloads:
            # the gate reading is a geomean: show which workload carries or drags it
            achieved["aggregate"] = f"geomean over {best.n_workloads} workloads"
            achieved["workloads"] = best.workloads
    return {
        "target": target,
        "achieved": achieved,
        "curve": [{"iter": r.iter, "bwd_median_ms": r.bwd_median_ms,
                   **({"device_median_ms": r.device_median_ms}
                      if r.device_median_ms is not None else {}),
                   **({"cluster": r.cluster} if r.cluster else {}),
                   "verdict": r.verdict}
                  for r in records],
        # metric_before is the gate-window best at entry (the driver writes it that
        # way), so metric_after must read the same window or the delta is fiction.
        "decisions": [{"iter": r.iter, "id": r.decision_id, "change": r.change,
                       "correctness": r.correctness, "metric_before": r.metric_before,
                       "metric_after": gate_ms(r, gate_window), "verdict": r.verdict}
                      for r in records],
        "regressions": [{"id": r.decision_id, "error": r.error or "regression", "iter": r.iter}
                        for r in records if r.verdict == "REVERT"],
        "profiler": [{"iter": r.iter, "verdict": r.verdict, **r.profiler}
                     for r in records if r.profiler],
        # red flags derived from the trajectory shape itself; empty list when clean
        "flags": [f for f in (constant_output_flag(records),) if f],
    }
