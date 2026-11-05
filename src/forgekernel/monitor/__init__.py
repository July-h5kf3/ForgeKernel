"""Correctness monitor: one check that works on every card as it stands.

Each card judges correctness its own way (its bench, its oracle, its tolerance, its
inputs), and the driver only reads ``correctness.pass``. The monitor does not replace
that judge; it widens what the judge is asked about, from outside the card:

* the card's own ``correctness`` task is re-run under several input profiles (fresh
  seeds, scaled / spiky / zero-blocked / repeated-row normal draws — see
  runtime/hook/fk_monitor_hook.py); the bench hands the same reshaped inputs to the
  kernel and to its reference, so the card's judge still decides;
* every run also watches the kernel's outputs (NaN / inf counts, checksums), so a
  NaN the card's judge lets through, or a result that changes between two identical
  runs, is reported whatever the card checks.

Nothing on the card changes: the runtime rides into the job as an overlay commit
(``.fk_monitor/``) and is loaded through ``sitecustomize``. The verdict is advisory —
recorded in ``operators/<op>/correctness_monitor.jsonl`` and the confirmation
record, never fed to an agent and never turned into a KEEP / REVERT.

Entry point: the loop's finish (``loop.cli._run_finish_monitor``) on the confirmed best
sha, else the best green KEEP.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
TREE_DIR = ".fk_monitor"
MARKER = "=== FK MONITOR JSON ==="
PROFILE_MARKER = "=== FK MONITOR PROFILE ==="
LEDGER = "correctness_monitor.jsonl"
POLICIES = ("off", "best")
DEFAULT_PROFILES = ("base", "seed1", "zero_block", "int_edges", "amp", "spikes", "repeat",
                    "tiny", "const_rows", "seed2")
SEED_PROFILES = ("seed1", "seed2")
STRESS_PROFILES = ("amp", "tiny", "spikes", "zero_block", "const_rows", "int_edges")
KNOWN_PROFILES = ("base", "repeat", *SEED_PROFILES, *STRESS_PROFILES)
PROFILE_DESC = {
    "base": "the card's own inputs",
    "repeat": "the card's own inputs, a second time",
    "seed1": "a fresh seed",
    "seed2": "a second fresh seed",
    "amp": "normal draws x4",
    "tiny": "normal draws x1/16",
    "spikes": "0.1% of normal draws set to +/-12",
    "zero_block": "leading 1/8 of the last dim and 1/16 of the rows zeroed",
    "const_rows": "leading 1/16 of the rows repeating row 0",
    "int_edges": "integer draws at the edges of their range (randint low / high-1, "
                 "full-fill leading 1/16 at 0)",
}
# Two identical runs whose outputs differ by more than this (relative, over the
# finite-only abs / square sums) are a race, not accumulation order.
NONDET_WARN_REL = 1e-3
_SEVERITY = {"ok": 0, "info": 0, "warn": 1, "alarm": 2}


def runtime_files() -> dict[str, str]:
    """Tree path -> text for everything the job needs."""
    rel = {"run.py": "run.py", "hook/sitecustomize.py": "hook/sitecustomize.py",
           "hook/fk_monitor_hook.py": "hook/fk_monitor_hook.py"}
    return {f"{TREE_DIR}/{dst}": (RUNTIME_DIR / src).read_text(encoding="utf-8")
            for dst, src in rel.items()}


def job_mode(profiles=DEFAULT_PROFILES, kernel_file: str = "kernel.py",
             budget_s: float | None = None) -> str:
    """The ``cmd:`` gpu_entry mode that runs the monitor (quote-free, as `cmd`
    modes must be)."""
    bad = [p for p in profiles if p not in KNOWN_PROFILES]
    if bad:
        raise ValueError(f"unknown monitor profile(s): {', '.join(bad)} "
                         f"(known: {', '.join(KNOWN_PROFILES)})")
    for s in (kernel_file,):
        if any(c.isspace() or c in "'\"`$;&|<>" for c in s):
            raise ValueError(f"kernel file {s!r} cannot ride a quote-free cmd mode")
    cmd = f"python3 {TREE_DIR}/run.py --profiles {','.join(profiles)} --kernel {kernel_file}"
    if budget_s:
        cmd += f" --budget-s {int(budget_s)}"
    return "cmd:" + cmd


def overlay(with_file: Callable[[str, str, str, str], str], sha: str) -> str:
    """``sha`` with the runtime added under .fk_monitor/ (one child commit per file
    that differs; the tree under judgment is otherwise the same)."""
    out = sha
    for path, text in sorted(runtime_files().items()):
        out = with_file(out, path, text,
                        f"correctness monitor: {path} (overlay for the monitor job only)") or out
    return out


def _json_after(text: str, at: int, marker: str) -> dict | None:
    for line in text[at + len(marker):].splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except ValueError:
                return None
            return obj if isinstance(obj, dict) else None
    return None


def parse_log(text: str) -> dict | None:
    """The runner's final result; when the job was cut short before it, a partial
    result (``complete`` False) rebuilt from the per-profile blocks."""
    text = text or ""
    at = text.rfind(MARKER)
    if at != -1:
        return _json_after(text, at, MARKER)
    recs, pos = [], text.find(PROFILE_MARKER)
    while pos != -1:
        rec = _json_after(text, pos, PROFILE_MARKER)
        if rec is not None:
            recs.append(rec)
        pos = text.find(PROFILE_MARKER, pos + len(PROFILE_MARKER))
    return {"profiles": recs, "skipped": [], "complete": False} if recs else None


# --- classification ---------------------------------------------------------------
def _calls(rec: dict) -> dict[tuple, dict]:
    return {(c.get("fn"), c.get("i")): c for c in (rec.get("hook") or {}).get("calls") or []}


def _nonfinite(outputs: list[dict], key: str) -> int:
    return sum(int(o.get(key) or 0) for o in outputs)


def _written(*recs: dict | None) -> dict[str, set[int]]:
    """Per kernel function, the argument positions any of its recorded calls wrote
    (destination-passing outputs, in-place updates). A position one call wrote is an
    output of that function in every call: a later call can refill a buffer with
    exactly what it already held (the allocator hands back the block the previous,
    identical result lived in) and then reads as unwritten."""
    out: dict[str, set[int]] = {}
    for rec in recs:
        for c in (rec or {}).get("hook", {}).get("calls") or []:
            out.setdefault(c.get("fn"), set()).update(c.get("written") or ())
    return out


def _bad_inputs(c: dict, written: dict[str, set[int]]) -> int:
    """Non-finite values the call's real inputs held going in (hook records carry them
    per argument, read before the call; older records only the total)."""
    per = c.get("in_nonfinite")
    if per is None:
        return int(c.get("inputs_nonfinite") or 0)
    w = written.get(c.get("fn"), set())
    return sum(int(n or 0) for j, n in enumerate(per) if j not in w)


def _in_sig(rec: dict, written: dict[str, set[int]] | None = None) -> list:
    """What the kernel was handed, call by call — output buffers left out."""
    written = written or {}
    sig = []
    for k, c in sorted(_calls(rec).items(), key=str):
        w = written.get(c.get("fn"), set())
        sums = c.get("in_sums")
        sig.append((k, None if sums is None else
                    [None if j in w else s for j, s in enumerate(sums)]))
    return sig


def _by_slot(outputs: list[dict]) -> dict:
    return {o.get("slot", k): o for k, o in enumerate(outputs)}


def _rel(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    den = max(abs(a), abs(b))
    return 0.0 if den == 0 else abs(a - b) / den


def classify(result: dict | None) -> dict:
    """Findings (alarm / warn / info) and an overall status from the runner's JSON."""
    findings: list[dict] = []

    def add(sev: str, prof: str, what: str) -> None:
        findings.append({"severity": sev, "profile": prof, "what": what})

    if not result:
        add("warn", "-", "the monitor job printed no result (it did not run, or was cut short)")
        return {"status": "warn", "findings": findings, "ran": [], "not_applicable": []}

    recs = {r.get("profile"): r for r in result.get("profiles") or []}
    base = recs.get("base")
    ran, n_a = [], []
    if result.get("complete") is False:
        add("warn", "-", f"the monitor job was cut short after {len(recs)} profile(s) — "
                         f"the rest never ran")
    for s in result.get("skipped") or []:
        add("info", s.get("profile", "?"), f"not run: {s.get('why')}")
    if base is None:
        add("warn", "base", "the base run is missing — nothing to compare the other profiles with")
    else:
        if base.get("card_pass") is not True:
            why = ("no eval verdict" + (" (timed out)" if base.get("timed_out") else
                                        f" (rc={base.get('rc')})")
                   if base.get("card_pass") is None else "the card's judge FAILS")
            add("alarm", "base", f"on the card's own inputs: {why}")
        if not (base.get("hook") or {}).get("calls"):
            add("info", "base", "no kernel call was observed (the bench does not import the "
                                "kernel as a module) — output checks unavailable; the card's "
                                "verdicts under each profile still count")
        for (fn, i), c in _calls(base).items():
            nan = _nonfinite(c.get("outputs") or [], "nan")
            if nan and base.get("card_pass") is True:
                add("warn", "base", f"{fn}[{i}] produces {nan} NaN on the card's own inputs "
                                    f"and the card passes it")

    base_calls = _calls(base) if base else {}
    for prof, rec in recs.items():
        if prof == "base":
            ran.append(prof)
            continue
        hook = rec.get("hook") or {}
        verdict = rec.get("card_pass")
        desc = PROFILE_DESC.get(prof, prof)
        # argument positions the kernel writes, in either run: outputs, not inputs
        written = _written(base, rec)
        base_sig = _in_sig(base, written) if base else None
        if verdict is None:
            if rec.get("timed_out"):
                add("warn", prof, f"timed out under {desc} ({rec.get('seconds')}s)")
            elif base and base.get("card_pass") is not None:
                add("alarm", prof, f"the correctness task died under {desc} "
                                   f"(rc={rec.get('rc')}, no verdict) while the base run "
                                   f"gave one")
            continue
        if prof in SEED_PROFILES or prof in STRESS_PROFILES:
            # the kernel's own inputs are the evidence when calls were seen; the
            # hook's counters only when they were not — or when the only arguments
            # that could carry the change are ones the kernel also writes (an
            # in-place update's input is not in the signature)
            if base_sig is not None and hook.get("calls") and base_calls:
                changed = _in_sig(rec, written) != base_sig
                if not changed and any(written.values()):
                    changed = bool(hook.get("perturbed") or hook.get("reseeded"))
            else:
                changed = bool(hook.get("perturbed") or hook.get("reseeded"))
            if not changed:
                n_a.append(prof)
                add("info", prof, f"not applicable: the kernel's inputs did not change under "
                                  f"{desc} (the bench does not draw them from torch's RNG)")
                continue
        bad_in = [c for c in _calls(rec).values() if _bad_inputs(c, written)
                  and not _bad_inputs(base_calls.get((c.get("fn"), c.get("i"))) or {}, written)]
        if bad_in:
            n_a.append(prof)
            add("info", prof, f"not counted: {desc} made the kernel's inputs non-finite")
            continue
        ran.append(prof)
        # NaN the base run did not have, whatever the card's judge says
        for key, c in _calls(rec).items():
            nan = _nonfinite(c.get("outputs") or [], "nan")
            nan0 = _nonfinite((base_calls.get(key) or {}).get("outputs") or [], "nan")
            if nan and not nan0:
                add("alarm", prof, f"{key[0]}[{key[1]}] produces {nan} NaN under {desc} "
                                   f"(none on the card's own inputs); card verdict: "
                                   f"{'PASS' if verdict else 'FAIL'}")
        if prof == "repeat":
            if base and verdict != base.get("card_pass"):
                add("alarm", prof, f"the card's verdict flips between two identical runs "
                                   f"({base.get('card_pass')} -> {verdict})")
            if base_sig is not None and _in_sig(rec, written) != base_sig:
                add("info", prof, "the bench's inputs are not reproducible between runs — "
                                  "output reproducibility not checked")
                continue
            worst = 0.0
            for key, c in _calls(rec).items():
                b = base_calls.get(key)
                if not b:
                    continue
                mine, theirs = _by_slot(c.get("outputs") or []), _by_slot(b.get("outputs") or [])
                for o, ob in ((mine[s], theirs[s]) for s in mine if s in theirs):
                    worst = max(worst, _rel(o.get("abs_sum"), ob.get("abs_sum")),
                                _rel(o.get("sq_sum"), ob.get("sq_sum")),
                                *(1.0 for k in ("nan", "posinf", "neginf")
                                  if (o.get(k) or 0) != (ob.get(k) or 0)))
            if worst > NONDET_WARN_REL:
                add("warn", prof, f"kernel outputs differ between two identical runs by up "
                                  f"to {worst:.2e} (relative checksum) — a race, not "
                                  f"accumulation order")
            elif worst > 0:
                add("info", prof, f"kernel outputs are not bitwise reproducible "
                                  f"(checksum rel {worst:.1e}; atomics / split order)")
            continue
        if verdict is False:
            failed = [c.get("case") for c in rec.get("cases") or [] if c.get("pass") is False]
            which = f" (case(s): {', '.join(str(x) for x in failed[:6])})" if failed else ""
            if prof in SEED_PROFILES:
                add("alarm", prof, f"FAILS the card's judge on {desc}{which} — the pass on "
                                   f"the card's own seed does not generalise")
            elif rec.get("ref_pass") is True:
                add("alarm", prof, f"FAILS the card's judge under {desc}{which} where the "
                                   f"reference, run through the same bench, passes")
            elif rec.get("ref_pass") is False:
                add("info", prof, f"fails under {desc}{which}, and so does the reference — "
                                  f"the card's tolerance does not carry to this distribution")
            else:
                add("warn", prof, f"FAILS the card's judge under {desc}{which}; the reference "
                                  f"was not run under it (no cardref task, or no budget left)")
    worst = max((_SEVERITY[f["severity"]] for f in findings), default=0)
    status = {0: "ok", 1: "warn", 2: "alarm"}[worst]
    return {"status": status, "findings": findings, "ran": ran, "not_applicable": n_a}


def describe(verdict: dict) -> list[str]:
    head = (f"correctness monitor: {verdict['status'].upper()} — profiles counted: "
            f"{', '.join(verdict.get('ran') or []) or 'none'}")
    if verdict.get("not_applicable"):
        head += f"; not applicable: {', '.join(verdict['not_applicable'])}"
    return [head] + [f"  [{f['severity']}] {f['profile']}: {f['what']}"
                     for f in verdict.get("findings") or []
                     if f["severity"] != "info" or verdict["status"] == "ok"]


# --- running it ------------------------------------------------------------------
def run(backend, sha: str, *, branch: str, op_dir: Path, profiles=DEFAULT_PROFILES,
        kernel_file: str = "kernel.py", budget_s: float | None = None,
        trigger: str = "manual", log: Callable[[str], None] = print) -> dict:
    """Submit one monitor job for ``sha`` through ``backend`` (RealBackend-shaped:
    with_file / push_ref / submit_gpu / last_job_id), classify it, append the record
    to ``op_dir/correctness_monitor.jsonl`` and return it. Raises what the backend
    raises; the caller decides whether that may cost anything."""
    mode = job_mode(profiles, kernel_file, budget_s)
    tree = overlay(backend.with_file, sha)
    backend.push_ref(tree, branch)
    log(f"correctness monitor: {sha[:10]} ({len(profiles)} profile(s)) -> {branch}")
    text = backend.submit_gpu(branch, mode)
    result = parse_log(text)
    verdict = classify(result)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "sha": sha, "tree": tree,
           "job": getattr(backend, "last_job_id", "") or None, "trigger": trigger,
           "profiles": list(profiles), **verdict, "result": result}
    try:
        with open(Path(op_dir) / LEDGER, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError as e:
        log(f"correctness monitor: could not append to {LEDGER} ({e})")
    for line in describe(verdict):
        log(line)
    return rec
