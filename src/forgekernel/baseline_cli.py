"""Measure the reference once and archive it (``forgekernel.baseline.RefBaseline``).

A baseline is a property of (operator, shape, hardware), not of a run: the loop measures
it once at its start (``measure_and_archive``) and every gate of the run anchors to that
archive; the finish re-measures the reference without archiving (``measure_ref_fresh``)
so node drift over the run's hours is read as drift, not as kernel change.

The measurement itself is whatever the tree's bench prints; this module only locates
the numbers and files them. It runs no GPU work of its own — it submits the ref task
through the same harness the loop uses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from forgekernel.baseline import (
    BASIS_PLAIN,
    HOST_FREE_BASES,
    ROLE_LIBRARY,
    RefBaseline,
    load_baseline,
    save_baseline,
)

# Perf keys a bench script might print, most specific first. The loop's parse_perf is
# bwd-specific (bwd_median_ms); a ref bench for an arbitrary operator is not, so this
# accepts the common spellings rather than mandating one.
_MS_KEYS = ("median_ms", "ms", "latency_ms", "bwd_median_ms", "fwd_median_ms", "time_ms")
_MS_DEVICE_KEYS = ("device_median_ms", "device_ms", "ms_device")
# The device window is device time only while the host keeps the pipeline fed, so a
# ref script may report the check and the host-free alternative alongside it. All of
# it is archived: a device reading whose basis is not recorded cannot be audited
# later, and an inflated one is indistinguishable from a clean one by value alone.
_MS_DEVICE_GRAPH_KEYS = ("graph_device_median_ms", "device_graph_ms", "ms_device_graph",
                         "cuda_graph_ms")
_MS_DEVICE_PLAIN_KEYS = ("device_plain_ms", "plain_device_ms", "ms_device_plain")
_HOST_ENQUEUE_KEYS = ("host_enqueue_ms", "enqueue_ms", "host_ms")
_HOST_RATIO_KEYS = ("host_bound_ratio",)
_BASIS_KEYS = ("device_window_basis", "device_basis", "window_basis")
_TFLOPS_KEYS = ("tflops", "tflop_s", "tflops_s", "achieved_tflops")
_MFU_KEYS = ("mfu_pct", "mfu", "bwd_mfu_pct", "fwd_mfu_pct")

# Fallback for scripts that print prose instead of JSON, e.g. "median 2.3641 ms".
_MS_PROSE = re.compile(r"(?:median|latency|time)\D{0,12}?([0-9]*\.?[0-9]+)\s*ms",
                       re.IGNORECASE)


def _walk(obj):
    """Every dict nested anywhere in a parsed JSON document."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _json_objects(text: str):
    """Every top-level JSON object in free-form stdout, outermost first."""
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        yield json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        pass
                    break


def _number(d: dict, keys) -> float | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _text(d: dict, keys) -> str:
    """First non-empty STRING value among ``keys`` (window basis and friends)."""
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def parse_ref_stdout(text: str) -> dict:
    """Locate median ms / TFLOPs / MFU in a bench script's output.

    A ref bench commonly times several backends and reports them side by side
    (flash / cudnn / efficient). The ref is the **fastest** of them — that is what
    "reproduce the reference" means, and it is what the operator's own
    measure_target.py picks as best_backend. So the timing block with the smallest
    ms wins, rather than whichever the JSON happens to list first; TFLOPs and MFU
    are then read from that same block so the three numbers describe one backend.

    Returns the keys it found; a caller that gets no ``ms`` should refuse to archive
    rather than file a baseline with a hole in it.
    """
    dicts = [d for obj in _json_objects(text) for d in _walk(obj)]
    timed = [(ms, d) for d in dicts if (ms := _number(d, _MS_KEYS)) is not None]
    # A multi-workload report (timing.workloads_performance) nests one timing
    # block per workload of the SAME reference under its geomean block; those are
    # not rival backends, and "fastest block wins" would archive the reference's
    # lightest workload as its time. The aggregate block is the reading.
    aggregate = [(ms, d) for ms, d in timed if d.get("aggregate") == "geomean"]
    if aggregate:
        timed = aggregate
        # The per-workload blocks are not a fallback source either: a key the
        # aggregate block lacks (a graph reading, the enqueue clock, TFLOPs) read
        # off one workload would sit in the archive beside a geomean. Matched by
        # content — _json_objects also yields each nested block as its own object.
        nested = {json.dumps(sub, sort_keys=True) for _, d in aggregate
                  for w in (d.get("workloads") or ()) for sub in _walk(w)}
        dicts = [d for d in dicts if json.dumps(d, sort_keys=True) not in nested]
    out: dict = {}
    if timed:
        ms, block = min(timed, key=lambda pair: pair[0])
        out["ms"] = ms
        if block.get("aggregate") == "geomean":
            # which workloads the reading covers: a gate reading is comparable to
            # it only over the same ones (baseline.workloads_mismatch)
            names = [str(w["name"]) for w in (block.get("workloads") or ())
                     if isinstance(w, dict) and w.get("name") is not None]
            n = block.get("n_workloads")
            out["n_workloads"] = (int(n) if isinstance(n, int) and not isinstance(n, bool)
                                  and n > 0 else max(1, len(names)))
            out["workloads"] = names
        # second timing window (pipelined batch ≈ device time), when the script
        # reports one — read from the winning block only, so both numbers describe
        # the same backend
        ms_dev = _number(block, _MS_DEVICE_KEYS)
        if ms_dev is not None:
            out["ms_device"] = ms_dev
        # Window provenance + the precondition check. Read from the winning block
        # first, then from anywhere in the document: a script may print the graph
        # reading and the enqueue clock in a sibling object rather than inline.
        for archived, keys in (("ms_device_graph", _MS_DEVICE_GRAPH_KEYS),
                               ("ms_device_plain", _MS_DEVICE_PLAIN_KEYS),
                               ("host_enqueue_ms", _HOST_ENQUEUE_KEYS),
                               ("host_bound_ratio", _HOST_RATIO_KEYS)):
            value = _number(block, keys)
            if value is None:
                value = next((v for d in dicts if (v := _number(d, keys))), None)
            if value is not None:
                out[archived] = value
        basis = _text(block, _BASIS_KEYS) or next(
            (t for d in dicts if (t := _text(d, _BASIS_KEYS))), "")
        if basis:
            out["device_window_basis"] = basis
        tf = _number(block, _TFLOPS_KEYS)
        if tf is None:  # some scripts put throughput a level up from the timing
            tf = next((v for d in dicts if (v := _number(d, _TFLOPS_KEYS))), None)
        if tf is not None:
            out["tflops"] = tf
        mfu = _number(block, _MFU_KEYS)
        if mfu is None:
            mfu = next((v for d in dicts if (v := _number(d, _MFU_KEYS))), None)
        if mfu is not None:
            out["mfu"] = mfu
        return out
    m = _MS_PROSE.search(text)
    if m:
        out["ms"] = float(m.group(1))
    return out


def _submit_cluster(op: str) -> str:
    """The machine this op's jobs run on ("local" or the ssh host), for stamping
    baseline provenance. "" when the submit config is unreadable — an unstamped
    baseline is merely unguarded, not wrong."""
    try:
        from forgekernel.submit.config import load_config
        return load_config(op).cluster
    except Exception:
        return ""


def _measure(op: str, perf_mode: str, ncu_mode: str | None, *, repeat: int = 3,
             attempts: int = 3, submit_op: str | None = None
             ) -> tuple[str, str, str, list[float], list[dict]]:
    """Measure the ref ``repeat`` times and keep the fastest run.

    Two distinct failure modes, handled separately:

    * a job that yields *no numbers* — a run that was killed or crashed still
      reaches a terminal status, so its log simply has no result. Those are
      retried, up to ``attempts`` each.
    * a job that yields *slow numbers* — measured four times on one operator, the
      same cuDNN backward came back 2.3688 / 2.3645 / 2.3555 / 2.8141 ms. The
      outlier is +19%, with the other backend slow in the same run: node
      contention, not a different kernel. Taking a single sample there would put
      target_ms 19% loose and let the student stop while genuinely far from ref.

    The fastest sample is the honest one: the ref is a fixed library kernel, so
    what it does when nothing else is competing for the node is its real number
    and everything slower is an artifact of when we happened to ask. Every sample
    is returned so the archive can record the spread rather than hide it.

    Returns (perf_stdout_of_best, ncu_csv, job_of_best, all_samples, parsed_runs) —
    parsed_runs is every timing-yielding run's parse_ref_stdout dict, in run order,
    so callers needing per-run window readings (the post-stop ref re-anchor) get
    them without re-submitting. Imported
    lazily so ``--show`` and ``--from-json`` work without a GPU machine configured.
    """
    from forgekernel.loop.cli import RealBackend

    # submit_op lets a reference be measured through a *different* submission tree than
    # the operator's own. The architecture reference needs one: it installs the released
    # flash-attn package and runs its own gpu_entry task, and that tree carries oracle
    # material that must never reach a student job.
    be = RealBackend(submit_op or op, Path.cwd(), agent_cmd=None,
                     git_name="forge", git_email="forge@local")
    best_out, best_job, best_ms, samples = "", "", None, []
    parsed_runs: list[dict] = []
    for run in range(1, repeat + 1):
        out, job = "", ""
        for attempt in range(1, attempts + 1):
            out = be.submit_gpu(_branch_placeholder(), perf_mode)
            job = be.last_job_id  # captured at submit; the logs do not carry it
            if "ms" in parse_ref_stdout(out):
                break
            if attempt < attempts:
                print(f"[baseline] job {job or '?'} produced no timing "
                      f"(lost or crashed) — retrying {attempt + 1}/{attempts}")
        parsed = parse_ref_stdout(out)
        ms = parsed.get("ms")
        if ms is None:
            print(f"[baseline] run {run}/{repeat}: no timing after {attempts} attempts")
            continue
        samples.append(ms)
        parsed_runs.append({**parsed, "job": job,
                            "cluster": getattr(be, "last_cluster", "")})
        marker = ""
        if best_ms is None or ms < best_ms:
            best_out, best_job, best_ms, marker = out, job, ms, "  <- best so far"
        print(f"[baseline] run {run}/{repeat}: {ms} ms (job {job or '?'}){marker}")

    if len(samples) > 1:
        spread = (max(samples) - min(samples)) / min(samples) * 100
        print(f"[baseline] {len(samples)} samples, spread {spread:.1f}% "
              f"(min {min(samples)}, max {max(samples)}) — anchoring to the fastest")
        if spread > 10:
            print("[baseline] NOTE: >10% spread suggests node contention; the fastest "
                  "sample is the ref's real number but consider re-running if it looks off")

    ncu_out = ""
    if ncu_mode:
        for attempt in range(1, attempts + 1):
            ncu_out = be.submit_gpu(_branch_placeholder(), ncu_mode)
            if _looks_like_ncu_csv(ncu_out):
                break
            if attempt < attempts:
                print(f"[baseline] NCU pass produced no CSV — retrying {attempt + 1}/{attempts}")
    elif _looks_like_ncu_csv(best_out):
        # One job that times *and* profiles: the reference's own harness may do both in
        # a single run, which is cheaper and guarantees the two halves describe the same
        # build. Costs nothing for the two-command case -- a perf log has no CSV header.
        ncu_out = best_out
    return best_out, ncu_out, best_job, samples, parsed_runs


def _looks_like_ncu_csv(text: str) -> bool:
    """NCU emits a header row naming the metric columns; anything else is a failed run."""
    return '"ID","Process ID"' in text or ('"Metric Name"' in text and '"Metric Value"' in text)


def _branch_placeholder() -> str:
    """The ref runs from whatever the submit config already points at; the baseline
    measures the *reference*, not the student's workspace, so no branch is pushed."""
    return "HEAD"


def _archived_basis(measured: dict) -> str:
    """Which window the archived ``ms_device`` came from.

    Declared by the ref script when it knows; otherwise inferred from the timing
    contract — ``device_median_ms`` is the pipelined batch window by definition
    (``src/forgekernel/card_template/eval/timing.py``), and a graph reading arrives under its
    own key. Left empty when there is no device reading at all, so "unknown basis"
    stays distinguishable from "plain batch".
    """
    declared = str(measured.get("device_window_basis") or "").strip()
    if declared:
        return declared
    if measured.get("ms_device") is not None:
        return BASIS_PLAIN
    return ""


def _report_host_bound(baseline: RefBaseline) -> None:
    """Say it out loud when the archived device reading is really host time.

    Printed at archive time, not only at start time, because this is where the
    number becomes durable: everything downstream reads the file, and an archive
    that records a host-bound reading without saying so is the exact artifact that
    took a run 11.5 hours to disprove.
    """
    from forgekernel.timing_audit import reference_warning
    verdict = baseline.host_bound
    if not verdict.measured:
        print("[baseline] NOTE: the ref script reports no host_enqueue_ms — the device "
              "window's precondition (host enqueues faster than the device drains) is "
              "UNCHECKED for this reference. Have the ref script use "
              "timing.py measure_windows / measure_windows_with_graph.")
        return
    # The host-free reading the archive anchors on, whichever field carries it: a
    # graph replay under its own key, or a span/graph reading declared as the basis
    # of ms_device itself (see baseline.HOST_FREE_BASES) — the warning then says the
    # fix is already in place instead of prescribing it.
    host_free = baseline.ms_device_graph
    if host_free is None and baseline.device_anchor_basis in HOST_FREE_BASES:
        host_free = baseline.device_anchor_ms
    warning = reference_warning(verdict, ref_name=baseline.ref_name,
                                device_ms=baseline.ms_device_plain or baseline.ms_device,
                                graph_ms=host_free)
    if warning:
        print(f"[baseline] {warning}")
    else:
        print(f"[baseline] device-window precondition OK "
              f"(host_bound_ratio={verdict.ratio:.3f}, basis={baseline.device_anchor_basis})")


class RefMeasurementError(RuntimeError):
    """The ref ran but produced nothing usable — refuse rather than archive a hole."""


def measure_and_archive(op: str, op_dir: Path, spec: dict, *,
                        repeat: int = 3, role: str = ROLE_LIBRARY) -> RefBaseline:
    """Measure the ref per ``spec`` and archive it.

    ``spec`` carries the operator facts — ``ref_cmd``, optional ``ref_ncu_cmd``, and
    the labels. Everything about *how* to measure (repeats, keeping the fastest,
    retrying a lost job, parsing) lives here.

    ``role`` selects which reference is being measured, and the two differ in how they
    are invoked. The library baseline is a command (``ref_cmd``) run through the
    operator's own harness. The architecture reference is a *task* (``ref_mode``) in a
    separate submission tree (``submit_op``), because reproducing it means installing
    the released package and running its own entry point, and that tree carries oracle
    material a student job must never see. Both archive through this one function so
    neither can drift into a private notion of what a measurement is -- the
    architecture reference once spent a whole run as a hand-transcribed constant, which is
    exactly the drift this prevents.
    """
    mode = spec.get("ref_mode")
    if not mode and not spec.get("ref_cmd"):
        raise RefMeasurementError("no ref_cmd and no ref_mode: nothing to measure")
    perf_mode = mode or f"cmd:{spec['ref_cmd']}"
    ncu_mode = f"cmd:{spec['ref_ncu_cmd']}" if spec.get("ref_ncu_cmd") else None
    perf_out, ncu_csv, job, samples, _ = _measure(
        op, perf_mode, ncu_mode, repeat=max(1, repeat), submit_op=spec.get("submit_op"))
    measured = parse_ref_stdout(perf_out)
    if "ms" not in measured:
        raise RefMeasurementError(
            "could not find a median latency in the ref output. The bench script should "
            f"print JSON with one of {list(_MS_KEYS)}, or a line like 'median 2.3641 ms'")
    from forgekernel.inner.ncu import kernels_from_ncu, profiler_from_ncu
    baseline = RefBaseline(
        op=op, ms=measured["ms"], role=role, job=job, shape=spec.get("ref_shape", ""),
        hardware=spec.get("ref_hardware", ""), ref_name=spec.get("ref_name", ""),
        tflops=measured.get("tflops"), mfu=measured.get("mfu"),
        ms_device=measured.get("ms_device"),
        device_window_basis=_archived_basis(measured),
        ms_device_plain=measured.get("ms_device_plain"),
        ms_device_graph=measured.get("ms_device_graph"),
        host_enqueue_ms=measured.get("host_enqueue_ms"),
        host_bound_ratio=measured.get("host_bound_ratio"),
        n_workloads=measured.get("n_workloads", 1),
        workloads=measured.get("workloads", []),
        ncu=profiler_from_ncu(ncu_csv) if ncu_csv else {},
        kernels=kernels_from_ncu(ncu_csv) if ncu_csv else [],
        samples_ms=samples, raw_stdout=perf_out[-4000:],
        cluster=_submit_cluster(spec.get("submit_op") or op),
    )
    path = save_baseline(op_dir, baseline)
    print(f"[baseline] archived -> {path}")
    print(baseline.render())
    _report_host_bound(baseline)
    if not baseline.has_ncu:
        print("[baseline] NOTE: no NCU signature — distillation will write no "
              "NCU-anchored gate lines (template A.4).")
    return baseline


def measure_ref_fresh(op: str, spec: dict, *, repeat: int = 3) -> list[dict]:
    """Measure the ref ``repeat`` times WITHOUT archiving — per-run parsed readings.

    The post-stop confirmation re-anchors the ref through its own channel when it
    cannot ride the student job (single-ref contract): comparing fresh student jobs
    against the hours-old frozen loop-start archive would read node drift as kernel
    change, and overwriting that archive here would retroactively move the target
    every gate of the finished run was anchored to. Returns the parsed dict
    (ms / ms_device / tflops / mfu, plus the job id) of every run that yielded a
    timing; the caller picks its window and forms the median.
    """
    mode = spec.get("ref_mode")
    if not mode and not spec.get("ref_cmd"):
        raise RefMeasurementError("no ref_cmd and no ref_mode: nothing to measure")
    perf_mode = mode or f"cmd:{spec['ref_cmd']}"
    _, _, _, _, parsed_runs = _measure(
        op, perf_mode, None, repeat=max(1, repeat), submit_op=spec.get("submit_op"))
    return parsed_runs
