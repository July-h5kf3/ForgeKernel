"""Leaderboard cards: an operator card that IS one FlashInfer-Bench / SOL-ExecBench
problem, judged and timed the way that board judges and times it.

A board scores a submission on its own terms — its element-wise tolerance, its input
draws, one call at a time with L2 flushed — and a kernel tuned under a card's own judge
and its warm-L2 pipelined window can pass here and fail there, or win here and lose
there. So a leaderboard card is generated, not hand-written: ``forge init <op>
--leaderboard <suite> ...`` freezes one (definition, workload) of the suite's dataset
into ``eval/leaderboard/problem.json`` and wires the card to
``src/forgekernel/card_template/eval/leaderboard.py``, which reproduces the suite's evaluation
protocol (pinned upstream commits; the file's docstring lists what is and is not
covered). The student's kernel keeps the definition's signature, so the kept kernel
is what gets submitted: ``forge export`` writes the suite's solution JSON from it
(``export_solution``).

One card = one workload, or one GROUP of workloads (``--workload 3,5,9`` / ``3-7``): a
board averages over all of a definition's workloads, and a size class of them (the shapes
one kernel should serve well together) can be one card. A group card judges every
workload by the board's rule (all must pass) and times each on the board's clock; the
gate reading is the geometric mean over them (the multi-workload contract of
timing.workloads_performance: top-level keys = geomean, per-workload blocks under
``workloads``), so a kernel cannot win the card by fitting one shape.

This module is pure file I/O over the suite's dataset layout (no torch):

  FlashInfer-Bench  a clone of huggingface.co/datasets/flashinfer-ai/flashinfer-trace
                    (definitions/<group>/<name>.json, workloads/<group>/<name>.jsonl,
                    solutions/baseline/<group>/<name>/*.json, blob/...)
  SOL-ExecBench     the tree SOL-ExecBench's scripts/download_solexecbench.py writes
                    (<subset>/<problem>/{definition.json, workload.jsonl}); safetensors
                    blobs of its FlashInfer-Bench subset resolve against --blob-root
                    (a flashinfer-trace clone), like its FLASHINFER_TRACE_DIR.

The one network read is SOL-ExecBench's public board (``fetch_sol_board``): the
per-workload latency of its scoring baseline (T_b) and hardware floor (T_SOL), which
the card freezes at ``forge new`` — they are what the board scores against, so they
are what a SOL card's target derives from (``--sol-board FILE`` / ``--no-sol-board``
for offline use).
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import shutil
import tomllib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SUITE_FIB = "flashinfer-bench"
SUITE_SOL = "sol-execbench"
# reference_impl kinds that are a sealed real kernel, timed in-job beside the student
# (binary: forgekernel.binref; sealed source: forgekernel.sealref).
SEALED_KINDS = ("binary", "sealed")
SUITES = (SUITE_FIB, SUITE_SOL)

# The upstream harness the template reproduces. The pinned commits (flashinfer-bench
# 15618a67, SOL-ExecBench a9fa0804) live in src/forgekernel/card_template/eval/leaderboard.py
# UPSTREAM, which stamps them into every eval JSON, and in docs/leaderboards.md — not
# in these strings, which are rendered into cards the agents read (kb_lint's
# exam-surface rule keeps sha-shaped tokens out of those).
UPSTREAM = {
    SUITE_FIB: "flashinfer-ai/flashinfer-bench",
    SUITE_SOL: "NVIDIA/SOL-ExecBench",
}
DATASETS = {
    SUITE_FIB: "huggingface.co/datasets/flashinfer-ai/flashinfer-trace",
    SUITE_SOL: "huggingface.co/datasets/nvidia/SOL-ExecBench",
}

# FlashInfer-Bench routes these to evaluators the template does not reproduce
# (bench/evaluators/registry.py). Judging them with the default element-wise rule
# would pass kernels the board fails and fail ones it passes.
FIB_UNSUPPORTED = {
    "sampling": "sampling definitions are judged by the TVD of 500k samples "
                "(SamplingEvaluator)",
    "dsa_topk_indexer": "DSA top-k indexer definitions are judged by sorted recomputed "
                        "scores over the returned indices (DsaTopkIndexerEvaluator)",
}

# A dataset cloned with GIT_LFS_SKIP_SMUDGE=1 holds these instead of the blobs.
_LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"

# Blob bytes past which a card's input files are worth a warning: every job sends the
# tree to the GPU machine, so they cost transfer time on every job.
BLOB_WARN_BYTES = 64 * 1024 * 1024

LEADERBOARD_DIR = "eval/leaderboard"
PROBLEM_FILE = f"{LEADERBOARD_DIR}/problem.json"

# SOL-ExecBench's public board. ``kernels`` lists the problems (id, name, subset tag);
# ``kernels/{id}`` carries, per workload, the latency of the board's scoring baseline
# (``baseline_latency_ms`` = T_b, an optimized PyTorch implementation whose source is
# not published) and the SOLAR hardware floor (``sol_ms`` = T_SOL), as the board timed
# them. Its workload list is the dataset's workload.jsonl, in order (checked for all
# 235 problems when this was written; attach_sol_board re-checks count and axes).
SOL_BOARD_API = "https://research.nvidia.com/benchmarks/sol-execbench/api"
# The board's clock (SOL-ExecBench core/bench/config/device_config.py, B200 preset):
# T_b and T_SOL are readings at this lock, so a card's absolute reading is only
# comparable to them under the same one.
SOL_BOARD_CLOCK = {"sm_mhz": 1500, "mem_mhz": 3996}


@dataclass
class Problem:
    """One (definition, workload) of a suite, plus the files the card must carry."""
    suite: str
    definition: dict
    workload: dict
    source: dict                                     # provenance (dataset, paths)
    blobs: dict[str, Path] = field(default_factory=dict)       # card-relative -> src
    reference_impl: dict = field(default_factory=lambda: {"kind": "definition"})
    reference_files: dict[str, str] = field(default_factory=dict)  # card-relative -> text
    notes: list[str] = field(default_factory=list)
    # SOL-ExecBench: the board's T_b / T_SOL for this workload (attach_sol_board), or None
    board: dict | None = None
    # A group card: every workload of the group, in card order -- {"index" (the dataset's
    # 0-based position), "name", "workload", "board"}; ``workload`` / ``board`` above are
    # the first one's (the representative: NCU profiles it). Empty for a one-workload card.
    group: list[dict] = field(default_factory=list)
    # Correctness-only workloads (``forge init --check-workload``), entries shaped like
    # ``group``'s: workloads the kernel is scored on elsewhere (a size group wider than the
    # card times), so it must run them, and run them right, but the gate never times them.
    check_only: list[dict] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.definition["name"])

    @property
    def workloads(self) -> list[dict]:
        """Every workload the card judges and times (one entry for a one-workload card)."""
        return [g["workload"] for g in self.group] if self.group else [self.workload]

    def var_values(self, axis: str) -> list[str]:
        return list(dict.fromkeys(json.dumps((w.get("axes") or {}).get(axis))
                                  for w in self.workloads))

    def check_values(self, axis: str) -> list[str]:
        """``axis``'s values on the correctness-only workloads the timed ones do not take."""
        timed = set(self.var_values(axis))
        return [v for v in dict.fromkeys(json.dumps((g["workload"].get("axes") or {}).get(axis))
                                         for g in self.check_only) if v not in timed]


# --- dataset readers ---------------------------------------------------------------

def _pick_workload(rows: list[dict], selector: str | None, where: str) -> dict:
    """The workload named by uuid (or a unique uuid prefix), or by 0-based index."""
    if not rows:
        raise ValueError(f"{where}: no workloads")
    if selector is None:
        raise ValueError(f"{where}: {len(rows)} workloads — pick one with --workload "
                         f"<uuid|index> (`forge new ... --list-workloads` lists them)")
    if selector.isdigit() and int(selector) < len(rows):
        return rows[int(selector)]
    hits = [w for w in rows if str(w.get("uuid", "")).startswith(selector)]
    if len(hits) != 1:
        raise ValueError(f"{where}: workload {selector!r} matches {len(hits)} of "
                         f"{len(rows)} workloads")
    return hits[0]


def _pick_workloads(rows: list[dict], selector: str | None, where: str) -> list[int]:
    """Dataset indices of the workload(s) ``selector`` names: one uuid / prefix / index, or a
    group -- comma-separated items, each an index, an inclusive ``lo-hi`` index range or a
    uuid prefix (order kept, repeats dropped)."""
    parts = [x.strip() for x in str(selector).split(",")] if selector is not None else [None]
    out: list[int] = []
    for part in parts:
        if part == "":
            raise ValueError(f"{where}: empty item in --workload {selector!r}")
        m = re.fullmatch(r"(\d+)-(\d+)", part or "")
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi or hi >= len(rows):
                raise ValueError(f"{where}: workload range {part!r} is not within "
                                 f"0-{len(rows) - 1}")
            picked = list(range(lo, hi + 1))
        else:
            w = _pick_workload(rows, part, where)
            picked = [next(i for i, r in enumerate(rows) if r is w)]
        out += [i for i in picked if i not in out]
    return out


def _group_entry(index: int, workload: dict) -> dict:
    return {"index": index, "name": f"wl{index}", "workload": workload, "board": None}


def _check_entries(rows: list[dict], selector: str | None, timed: list[dict],
                   where: str) -> list[dict]:
    """The correctness-only entries ``selector`` (``--check-workload``, --workload's syntax)
    names, minus the workloads the card already times."""
    if not selector:
        return []
    have = {g["index"] for g in timed}
    return [_group_entry(i, json.loads(json.dumps(rows[i])))
            for i in _pick_workloads(rows, selector, where) if i not in have]


def workload_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append(obj["workload"] if "workload" in obj and "axes" not in obj else obj)
    return rows


def _blob_target(rel: str) -> str:
    """Card-relative home of a dataset blob: under eval/leaderboard/, keeping the
    dataset-relative path so two workloads' blobs never collide."""
    parts = [p for p in Path(rel).parts if p not in (".", "..")]
    return str(Path("blob", *parts[1:]) if parts and parts[0] == "blob" else Path("blob", *parts))


def _collect_blobs(problem: Problem, roots: list[Path], workload: dict | None = None) -> None:
    """Copy-plan every safetensors input and rewrite its path relative to problem.json."""
    for name, spec in ((workload or problem.workload).get("inputs") or {}).items():
        if spec.get("type") != "safetensors":
            continue
        rel = str(spec["path"])
        src = next((r / rel for r in roots if (r / rel).is_file()), None)
        if src is None:
            raise FileNotFoundError(
                f"safetensors input {name!r}: {rel} not found under "
                f"{', '.join(str(r) for r in roots)} (clone the dataset with its LFS blobs, "
                f"e.g. `git lfs pull --include={rel.lstrip('./')}`, or pass --blob-root)")
        with open(src, "rb") as fh:
            if fh.read(len(_LFS_POINTER)) == _LFS_POINTER:
                clean = rel.lstrip("./")
                raise ValueError(
                    f"safetensors input {name!r}: {src} is a Git LFS pointer, not the "
                    f"blob — fetch it first (in the dataset clone: `git lfs pull "
                    f"--include={clean}`, or download https://{DATASETS[SUITE_FIB]}"
                    f"/resolve/main/{clean} over the pointer)")
        target = _blob_target(rel)
        problem.blobs[target] = src
        spec["path"] = target


def fib_problem(dataset: str | Path, definition: str, workload: str | None,
                *, baseline: str | None = None, check_workloads: str | None = None) -> Problem:
    """One FlashInfer-Bench (definition, workload) from a flashinfer-trace clone."""
    root = Path(dataset).expanduser().resolve()
    found = sorted(root.glob(f"definitions/*/{definition}.json"))
    if len(found) != 1:
        raise FileNotFoundError(f"definition {definition!r}: {len(found)} matches under "
                                f"{root}/definitions/*/")
    dpath = found[0]
    defn = json.loads(dpath.read_text(encoding="utf-8"))
    for mark, why in FIB_UNSUPPORTED.items():
        if defn.get("op_type") == mark or defn["name"].startswith(mark):
            raise ValueError(f"{defn['name']}: {why}; the leaderboard mode does not "
                             f"reproduce that evaluator yet")
    group = dpath.parent.name
    wpath = root / "workloads" / group / f"{definition}.jsonl"
    if not wpath.is_file():
        raise FileNotFoundError(f"no workloads file {wpath}")
    rows = workload_rows(wpath)
    picked = [_group_entry(i, json.loads(json.dumps(rows[i])))
              for i in _pick_workloads(rows, workload, str(wpath))]
    problem = Problem(SUITE_FIB, defn, picked[0]["workload"], source={
        "dataset": DATASETS[SUITE_FIB], "dataset_root": str(root),
        "definition_file": str(dpath.relative_to(root)),
        "workload_file": str(wpath.relative_to(root)), "upstream": UPSTREAM[SUITE_FIB],
        "dataset_commit": _git_head(root)}, group=picked if len(picked) > 1 else [],
        check_only=_check_entries(rows, check_workloads, picked, str(wpath)))
    for g in picked + problem.check_only:
        _collect_blobs(problem, [root], g["workload"])
    _attach_fib_baseline(problem, root, group, baseline)
    return problem


def _attach_fib_baseline(problem: Problem, root: Path, group: str,
                         choice: str | None) -> None:
    """The board ranks against the dataset's ``baseline`` author: that solution is the
    card's in-job reference (timed on the same clock, mirrored)."""
    sdir = root / "solutions" / "baseline" / group / problem.name
    sols = sorted(sdir.glob("*.json")) if sdir.is_dir() else []
    if choice:
        sols = [s for s in sols if json.loads(s.read_text(encoding="utf-8"))["name"] == choice]
        if not sols:
            raise FileNotFoundError(f"baseline solution {choice!r} not under {sdir}")
    if not sols:
        problem.notes.append(f"no baseline solution under {sdir}: the in-job reference is "
                             f"the definition's own reference (the board's ranking "
                             f"baseline is not timed)")
        return
    if len(sols) > 1:
        problem.notes.append(f"{len(sols)} baseline solutions; using "
                             f"{sols[0].stem} (--baseline picks another)")
    sol = json.loads(sols[0].read_text(encoding="utf-8"))
    spec = sol["spec"]
    lang = str(spec.get("language", "python"))
    if lang not in ("python", "triton"):
        problem.notes.append(f"baseline {sol['name']} is a {lang} solution — only python "
                             f"baselines ride the job; the in-job reference is the "
                             f"definition's own reference")
        return
    file_part, entry = spec["entry_point"].split("::", 1)
    base = Path("baseline")
    for src in sol["sources"]:
        problem.reference_files[str(base / src["path"])] = src["content"]
    problem.reference_impl = {
        "kind": "baseline_solution", "name": sol["name"], "author": sol.get("author"),
        "path": str(base / file_part), "entry": entry,
        "dps": bool(spec.get("destination_passing_style", True)),
        "dependencies": list(spec.get("dependencies") or [])}


def sol_problem(dataset: str | Path, problem_name: str, workload: str | None,
                *, blob_roots: list[str | Path] | None = None,
                board=None, board_source: str = "", check_workloads: str | None = None) -> Problem:
    """One SOL-ExecBench (problem, workload) from the download script's tree.

    ``board``: the board's record for the problem (the ``kernels/{id}`` payload, e.g.
    from ``--sol-board FILE``; ``board_source`` says where it came from), or a callable
    ``(name, subset) -> record`` that fetches it (fetch_sol_board), or None. A board
    that cannot be read or does not line up leaves a note, not a failure: the card
    still judges and times the board's way, it only lacks T_b / T_SOL."""
    root = Path(dataset).expanduser().resolve()
    pdir = root / problem_name if (root / problem_name).is_dir() else None
    if pdir is None:
        found = sorted(p for p in root.glob(f"*/{problem_name}") if p.is_dir())
        if len(found) != 1:
            raise FileNotFoundError(f"problem {problem_name!r}: {len(found)} matches under "
                                    f"{root} (give <subset>/<problem>)")
        pdir = found[0]
    defn = json.loads((pdir / "definition.json").read_text(encoding="utf-8"))
    if not defn.get("reference") and (pdir / "reference.py").is_file():
        defn["reference"] = (pdir / "reference.py").read_text(encoding="utf-8")
    wpath = pdir / "workload.jsonl"
    rows = workload_rows(wpath)
    group = [_group_entry(i, json.loads(json.dumps(rows[i])))
             for i in _pick_workloads(rows, workload, str(wpath))]
    problem = Problem(SUITE_SOL, defn, group[0]["workload"], source={
        "dataset": DATASETS[SUITE_SOL], "dataset_root": str(root),
        "problem": str(pdir.relative_to(root)), "upstream": UPSTREAM[SUITE_SOL]},
        group=group if len(group) > 1 else [],
        check_only=_check_entries(rows, check_workloads, group, str(wpath)))
    roots = [pdir, root, *(Path(r).expanduser().resolve() for r in (blob_roots or []))]
    for g in group + problem.check_only:
        _collect_blobs(problem, roots, g["workload"])
    problem.reference_impl = {"kind": "definition", "name": "definition reference (eager)"}
    if board is not None:
        name, subset = str(defn.get("name") or pdir.name), pdir.parent.name
        try:
            record = board(name, subset) if callable(board) else board
            source = board_source or f"{SOL_BOARD_API}/kernels/{record.get('id')}"
        except (OSError, ValueError, LookupError) as e:  # URLError / timeout are OSError
            problem.notes.append(f"SOL board not read ({e}): {_NO_BOARD}")
        else:
            whys = []
            for g in group:
                g["board"], why = sol_board_row(problem.name, g["workload"], record,
                                                g["index"], len(rows), source=source)
                if why:
                    whys.append(why)
            if whys:
                # a group's score and target cover every workload: one gap means none
                for g in group:
                    g["board"] = None
                problem.notes.append(f"SOL board not attached ({'; '.join(whys)}): {_NO_BOARD}")
            problem.board = group[0]["board"]
            # a correctness-only workload's row only orders the meta workloads by size and
            # scores a scoped meta reading of it: a gap there leaves just that row out
            for g in problem.check_only:
                g["board"], _ = sol_board_row(problem.name, g["workload"], record,
                                              g["index"], len(rows), source=source)
    else:
        problem.notes.append(f"SOL board not requested: {_NO_BOARD}")
    return problem


_NO_BOARD = ("no T_b / T_SOL, so no SOL score in the eval JSON and no board-derived "
             "target — pass the loop an explicit --target-ms")


def _http_json(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "forgekernel-leaderboard"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https host
        return json.loads(resp.read().decode("utf-8"))


def fetch_sol_board(name: str, subset: str | None = None, *,
                    api: str = SOL_BOARD_API) -> dict:
    """The board's record for problem ``name`` (``kernels/{id}``'s payload:
    {id, name, tags, workloads: [{axes, baseline_latency_ms, sol_ms}], ...}). Names are
    unique across the board; ``subset`` (the dataset directory: L1 / L2 / Quant /
    FlashInfer-Bench) must also be among its tags, so a renamed problem cannot match
    a namesake."""
    listing = _http_json(f"{api}/kernels")
    kernels = (listing.get("data") or {}).get("kernels") or []
    hits = [k for k in kernels if k.get("name") == name
            and (not subset or subset in (k.get("tags") or []))]
    if len(hits) != 1:
        raise LookupError(f"{name!r} ({subset or 'any subset'}) matches {len(hits)} of the "
                          f"board's {len(kernels)} problems")
    record = _http_json(f"{api}/kernels/{int(hits[0]['id'])}")
    return record.get("data", record) if isinstance(record, dict) else record


def load_sol_board(path: str | Path) -> dict:
    """A saved ``kernels/{id}`` response (with or without its ``data`` wrapper)."""
    obj = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: not a board record")
    return obj.get("data", obj)


def sol_score(t_k: float, t_b: float, t_sol: float) -> float:
    """SOL-ExecBench sol_score.py: S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL)) — 0.5 at
    T_b, 1 at T_SOL (above 1 past it; the board does not clamp)."""
    gap = t_b - t_sol
    if gap <= 0:
        return 1.0 if t_k <= t_sol else 0.0
    return 1.0 / (1.0 + (t_k - t_sol) / gap)


def time_for_score(score: float, t_b: float, t_sol: float) -> float | None:
    """The kernel latency that scores ``score`` (sol_score inverted), or None when the
    board's T_b does not sit above its T_SOL."""
    if not 0 < score or t_b - t_sol <= 0:
        return None
    return t_sol + (t_b - t_sol) * (1.0 / score - 1.0)


def attach_sol_board(problem: Problem, record: dict, index: int, n_rows: int, *,
                     source: str) -> None:
    """Freeze the board's T_b / T_SOL for the card's workload (the ``index``-th of the
    dataset's ``n_rows``) into ``problem.board``, or note why not. The board lists
    the workloads in the dataset's order; count and axes are checked against the
    pick, because several workloads of one problem can share their axes."""
    board, why = sol_board_row(problem.name, problem.workload, record, index, n_rows,
                               source=source)
    if why:
        problem.notes.append(f"SOL board not attached ({why}): {_NO_BOARD}")
        return
    problem.board = board


def sol_board_row(name: str, workload: dict, record: dict, index: int, n_rows: int, *,
                  source: str) -> tuple[dict | None, str | None]:
    """(the board's frozen T_b / T_SOL block for ``workload``, the ``index``-th of the
    dataset's ``n_rows``, or None; why not, or None)."""
    wls = record.get("workloads") if isinstance(record, dict) else None
    why = None
    t_b = t_sol = None
    if not isinstance(wls, list):
        why = "the record has no workloads list"
    elif record.get("name") != name:
        why = f"the record is {record.get('name')!r}, not {name!r}"
    elif len(wls) != n_rows:
        why = f"the board lists {len(wls)} workloads, the dataset {n_rows}"
    else:
        row = wls[index] if isinstance(wls[index], dict) else {}
        board_axes = row.get("axes") or {}
        off = {k: [v, board_axes.get(k)]
               for k, v in (workload.get("axes") or {}).items()
               if board_axes.get(k) != v}
        try:
            t_b, t_sol = float(row["baseline_latency_ms"]), float(row["sol_ms"])
        except (KeyError, TypeError, ValueError):
            why = f"workload {index} carries no baseline_latency_ms / sol_ms"
        if off:
            why = f"workload {index}'s axes differ on the board (card, board): {off}"
        elif why is None and not (t_b > 0 and t_sol > 0):
            why = f"workload {index}: T_b {t_b} / T_SOL {t_sol} are not positive"
    if why:
        return None, why
    return {"kernel_id": record.get("id"), "workload_index": index,
            "t_b_ms": t_b, "t_sol_ms": t_sol, "source": source,
            "clock_mhz": dict(SOL_BOARD_CLOCK)}, None


def _git_head(path: Path) -> str | None:
    head = path / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            return (path / ".git" / ref[5:]).read_text(encoding="utf-8").strip()[:12]
        return ref[:12]
    except OSError:
        return None


def list_workloads(problem_rows: list[dict]) -> list[str]:
    """One line per workload: index, uuid, axes, input kinds."""
    out = []
    for i, w in enumerate(problem_rows):
        kinds = sorted({str(s.get("type", "random")) for s in (w.get("inputs") or {}).values()})
        out.append(f"{i:3d}  {w.get('uuid', '?')}  axes={json.dumps(w.get('axes'))}  "
                   f"inputs={','.join(kinds) or 'random'}")
    return out


# --- card rendering -------------------------------------------------------------------

def problem_json(problem: Problem, *, dps: bool = True, kernel_file: str = "kernel.py") -> dict:
    out = {
        "schema": 1,
        "suite": problem.suite,
        "definition": problem.definition,
        "workload": problem.workload,
        "dps": dps,
        "kernel_file": kernel_file,
        "entry": "run",
        "reference_impl": problem.reference_impl,
        # Whether bench.py times the in-job reference beside the kernel. Not on a SOL
        # card with the definition's reference: that is the eager PyTorch run — many
        # small launches, host-bound, a per-call span that flips between readings —
        # and the board scores against its own T_b instead (sol_board). A sealed
        # binary reference (forgekernel.binref) or a sealed source one (forgekernel.sealref)
        # is a real kernel: timed in-job.
        "time_reference": (problem.suite != SUITE_SOL
                           or problem.reference_impl.get("kind") in SEALED_KINDS),
        "source": problem.source,
    }
    if problem.board:
        out["sol_board"] = problem.board
    if problem.group:
        # read by the card's eval/leaderboard.py workload_views: every workload judged and
        # timed, the gate the geomean over them ("workload" above = the first one)
        out["group"] = [{"index": g["index"], "name": g["name"], "workload": g["workload"],
                         **({"sol_board": g["board"]} if g.get("board") else {})}
                        for g in problem.group]
        # the workloads the loop's meta tools read one at a time (FK_LB_WORKLOAD_SCOPE)
        out["meta_workloads"] = meta_workload_names(problem)
    if problem.check_only:
        # read by the card's eval/leaderboard.py check_views: judged on every run, never timed
        out["check_only"] = [{"index": g["index"], "name": g["name"], "workload": g["workload"],
                              **({"sol_board": g["board"]} if g.get("board") else {})}
                             for g in problem.check_only]
    return out


# How many workloads of a group card the meta tools read: the representative, the group's
# smallest and its largest (priced on the representative alone, a lever's reading
# agreed in direction with the rest of the group only 61% of the time a change was real).
META_WORKLOADS = 3


def meta_workload_names(problem: Problem) -> list[str]:
    """The group workloads the loop's meta tools (lever pricing, stall / SASS pricing, the
    timeline probe) read, one at a time: the representative first, then the group's smallest
    and its largest -- by T_SOL where the board gives every workload one, ties (a weight-bound
    GEMM's small-M rows share one T_SOL) broken by the element count of the definition's
    tensors at the workload's axes -- no repeats. When the representative is itself the
    smallest or the largest (56% of the >= 3-workload group cards in one large batch of board runs),
    the third is the group's middle one in that order instead of a repeat, so a group of 3 or
    more always reads 3 sizes. The sizes span the card's correctness-only workloads too
    (``check_only``: the rest of a size group wider than the card times -- the kernel is
    scored on them, so the group's true smallest / largest may be one of them; a scoped
    meta reading times it alone). [] for a one-workload card."""
    if len(problem.group) < 2:
        return []
    every = problem.group + problem.check_only
    boards = [g.get("board") for g in every]
    tsol = ([float(b["t_sol_ms"]) for b in boards] if all(b and b.get("t_sol_ms") for b in boards)
            else [0.0] * len(boards))
    elems = [_tensor_elements(problem.definition, g["workload"]) for g in every]
    order = sorted(range(len(every)), key=lambda i: (tsol[i], elems[i], i))
    picks = list(dict.fromkeys([0, order[0], order[-1]]))
    mid = (len(order) - 1) / 2
    for pos in sorted(range(len(order)), key=lambda k: (abs(k - mid), k)):
        if len(picks) >= META_WORKLOADS:
            break
        if order[pos] not in picks:
            picks.append(order[pos])
    return [every[i]["name"] for i in picks][:META_WORKLOADS]


_AXIS_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.FloorDiv: operator.floordiv, ast.Div: operator.truediv, ast.Mod: operator.mod,
             ast.Pow: operator.pow}


def _tensor_elements(definition: dict, workload: dict) -> int:
    """Elements of every input and output tensor of ``definition`` at ``workload``'s axes
    (const axes, the workload's var axes, then expr axes; the card's eval/leaderboard.py
    resolved_axes rule); 0 when an axis cannot be resolved -- a size order, not a check."""
    axes_def = definition.get("axes") or {}
    env = {n: a.get("value") for n, a in axes_def.items() if a.get("type") == "const"}
    env.update(workload.get("axes") or {})

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _AXIS_OPS:
            return _AXIS_OPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError(type(node).__name__)

    try:
        for n, a in axes_def.items():
            if a.get("type") == "expr":
                env[n] = ev(ast.parse(str(a["expression"]), mode="eval").body)
        total = 0
        for spec in list((definition.get("inputs") or {}).values()) + \
                list((definition.get("outputs") or {}).values()):
            n = 1
            for d in spec.get("shape") or []:
                n *= int(d) if str(d).isdigit() else int(env[d])
            total += n
        return total
    except (KeyError, ValueError, TypeError, ZeroDivisionError, SyntaxError):
        return 0


def describe_workloads(problem: Problem) -> str:
    """The card's workload: its uuid, or for a group ``wl3 {axes}; wl5 {axes}; ...``."""
    extra = ("; correctness only: " + ", ".join(g["name"] for g in problem.check_only)
             if problem.check_only else "")
    if not problem.group:
        return str(problem.workload.get("uuid")) + extra
    return "; ".join(f"{g['name']} {json.dumps(g['workload'].get('axes'))}"
                     for g in problem.group) + extra


def _signature(problem: Problem, dps: bool) -> list[str]:
    names = list(problem.definition["inputs"])
    return names + list(problem.definition["outputs"]) if dps else names


def _spec_line(name: str, spec: dict) -> str:
    shape = spec.get("shape")
    dims = "python scalar" if shape is None else f"[{', '.join(map(str, shape))}]"
    return f"    {name}: {spec.get('dtype')} {dims}"


def _var_note(problem: Problem, axis: str) -> str:
    more = problem.check_values(axis)
    also = (f"; ALSO {', '.join(more)} -- judged for correctness on every run, never timed: "
            f"the kernel must run them" if more else "")
    if not problem.group:
        return f"var (this card: {problem.workload['axes'].get(axis)}{also})"
    return (f"var (this card's {len(problem.group)} workloads: "
            f"{', '.join(problem.var_values(axis))} -- one kernel serves them all; the gate "
            f"is the geomean over them{also})")


def render_kernel_stub(problem: Problem, *, dps: bool = True) -> str:
    defn = problem.definition
    axes = defn.get("axes") or {}
    ax_lines = [f"    {n}: " + (str(a.get("value")) if a.get("type") == "const"
                                   else a.get("expression") if a.get("type") == "expr"
                                   else _var_note(problem, n))
                for n, a in axes.items()]
    inputs = "\n".join(_spec_line(n, s) for n, s in defn["inputs"].items())
    outputs = "\n".join(_spec_line(n, s) for n, s in defn["outputs"].items())
    style = ("DPS: the outputs arrive pre-allocated after the inputs; write them in "
             "place, return nothing." if dps else "Value-returning: return the outputs "
             "(a tensor, or a tuple in the order below).")
    return f'''"""{defn["name"]} — the student's kernel ({problem.suite} leaderboard card).

Signature frozen by the leaderboard definition: parameter names are the definition's
input names{", then its output names" if dps else ""}, in order ({problem.suite}
checks them). {style}

Axes:
{chr(10).join(ax_lines)}
Inputs:
{inputs}
Outputs:
{outputs}

The mathematical specification is the definition's reference
(eval/leaderboard/problem.json, "reference"); judged and timed by
eval/leaderboard.py exactly as {problem.suite} does.
"""

import torch  # noqa: F401


def run({", ".join(_signature(problem, dps))}):
    raise NotImplementedError("write the kernel")
'''


# A CUDA C++ kernel file (``forge new ... --kernel-file kernel.cu``): the board's cuda_cpp
# solution format — one .cu with its own torch binding. The card builds it as the board does
# (eval/leaderboard.py load_cuda_entry); `leaderboard export` submits it as cuda_cpp.
CUDA_KERNEL_SUFFIXES = (".cu",)
# Mirrors the card's eval/leaderboard.py (CUDA_DEFAULT_CFLAGS / CUDA_GENCODE /
# CUDA_FLAGS_LINE / CUDA_FLAG_OK): the flags the card builds with are the flags exported.
CUDA_DEFAULT_CFLAGS = ["-O3", "--use_fast_math"]
CUDA_GENCODE = "-gencode=arch=compute_100a,code=sm_100a"
CUDA_FLAGS_LINE = re.compile(r"^[ \t]*//[ \t]*fk-cuda-cflags:(.*)$", re.M)


def is_cuda_kernel_file(kernel_file: str) -> bool:
    return Path(kernel_file).suffix in CUDA_KERNEL_SUFFIXES


def cuda_kernel_flags(src: str) -> list[str]:
    """The nvcc flags the card builds a CUDA kernel file with (no allow-list check here: the
    card's fence refuses a disallowed flag before a build)."""
    extra = [f for line in CUDA_FLAGS_LINE.findall(src) for f in line.split()]
    flags = CUDA_DEFAULT_CFLAGS + [f for f in extra if f not in CUDA_DEFAULT_CFLAGS]
    if not any(f.startswith(("-gencode", "-arch")) for f in flags):
        flags.append(CUDA_GENCODE)
    return flags


def _cpp_param(name: str, spec: dict) -> str:
    if spec.get("shape") is not None:
        return f"torch::Tensor {name}"
    dt = str(spec.get("dtype") or "")
    ctype = "bool" if dt == "bool" else "double" if dt.startswith(("float", "bfloat")) else "int64_t"
    return f"{ctype} {name}"


def render_kernel_stub_cuda(problem: Problem, *, dps: bool = True) -> str:
    defn = problem.definition
    axes = defn.get("axes") or {}
    ax_lines = [f"//     {n}: " + (str(a.get("value")) if a.get("type") == "const"
                                     else a.get("expression") if a.get("type") == "expr"
                                     else _var_note(problem, n))
                for n, a in axes.items()]
    specs = {**defn["inputs"], **(defn["outputs"] if dps else {})}
    params = ", ".join(_cpp_param(n, specs[n]) for n in _signature(problem, dps))
    inputs = "\n".join("//" + _spec_line(n, s) for n, s in defn["inputs"].items())
    outputs = "\n".join("//" + _spec_line(n, s) for n, s in defn["outputs"].items())
    ret = "void" if dps else "torch::Tensor"
    style = ("DPS: the outputs arrive pre-allocated after the inputs; write them in place, "
             "return nothing." if dps else "Value-returning: return the output tensor.")
    return f"""// {defn["name"]} — the student's kernel ({problem.suite} leaderboard card), CUDA C++.
//
// The board's cuda_cpp solution format: this one file, its own torch binding
// (PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) below; the build names the module). Built by
// eval/leaderboard.py the way {problem.suite} builds it: torch cpp_extension, nvcc
// {" ".join(CUDA_DEFAULT_CFLAGS)} {CUDA_GENCODE}, -lcuda, CUTLASS / CuTe C++ headers on the
// include path. Extra nvcc flags (code generation / diagnostics only) go on a line
//     // fk-cuda-cflags: -Xptxas=-v
// and are submitted with the solution. No other local file is compiled with it.
//
// Signature frozen by the leaderboard definition: parameter names are the definition's
// input names{", then its output names" if dps else ""}, in order. {style}
//
// Axes:
{chr(10).join(ax_lines)}
// Inputs:
{inputs}
// Outputs:
{outputs}
//
// The mathematical specification is the definition's reference
// (eval/leaderboard/problem.json, "reference"); judged and timed by
// eval/leaderboard.py exactly as {problem.suite} does.

#include <torch/extension.h>

{ret} run({params}) {{
    TORCH_CHECK(false, "write the kernel");
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &run);
}}
"""


BENCH_TEMPLATE = '''"""{name} — leaderboard-mode bench (GENERATED by `forge init --leaderboard {suite}`).

Do not hand-edit the judge or the clock: both are eval/leaderboard.py, which
reproduces {suite}'s evaluation protocol ({upstream}) on the problem frozen in
eval/leaderboard/problem.json (definition {name}, workload {workload}).
{group_note}

    python3 eval/bench.py --mode all|correctness|perf   the student kernel
    python3 eval/bench.py --impl ref                    card self-test: the in-job
                                                        reference as the implementation
    python3 eval/bench.py --mode ref                    [loop] ref_cmd: the reference alone
"""

from leaderboard import run_leaderboard_bench

if __name__ == "__main__":
    raise SystemExit(run_leaderboard_bench())
'''

NCU_TEMPLATE = '''"""NCU target for {name}: the student's calls on one input set, inside the
profiler window (input construction and JIT warm-up stay outside it)."""

from leaderboard import ncu_window

if __name__ == "__main__":
    ncu_window()
'''

GPU_ENTRY_TEMPLATE = r'''#!/bin/bash
# {name} ({suite} leaderboard card) — GPU job entry point
#
# Tasks:
#   correctness  the suite's correctness protocol on the student kernel
#   perf | all   correctness, then the suite's timing (paired with the in-job reference)
#   cardref      card self-test: the in-job reference judged and timed as the kernel
#   ref          the in-job reference alone on the suite's clock ([loop] ref_cmd)
#   ncu          NCU key-metrics CSV of the kernel (driver parses it)
#   cmd <CMD>    run an arbitrary command

set -euo pipefail

# An image that carries the board's own evaluation environment (a node set up for
# SOL-ExecBench keeps it at /opt/sol: its python, torch, CUTLASS headers) runs in it.
SOL_ENV="${{SOL_ENV:-/opt/sol/env.sh}}"
if [ -f "$SOL_ENV" ]; then set +u; source "$SOL_ENV"; set -u; fi
export PATH=/usr/local/cuda/bin:${{PATH}}
WORKSPACE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE:$WORKSPACE/eval:${{PYTHONPATH:-}}"
export PYTHONUNBUFFERED=1
export CUTE_DSL_ARCH="${{CUTE_DSL_ARCH:-{arch}}}"

# Clocks. {clock_note}
# A clock lock is GPU state, not process state: it outlives the job that set it and
# slows every later tenant of this GPU. So every entry first clears whatever an
# earlier job left (a killed container runs no trap), locks only what this card asks
# for, checks that it took, and clears it again on the way out — the EXIT trap also
# fires on a failed task and on a timeout kill. LB_CLOCK_STATE tells bench.py what
# the readings were taken under (the eval JSON's leaderboard.clock).
LB_NVIDIA_SMI="${{LB_NVIDIA_SMI:-nvidia-smi}}"
LB_SM_CLOCK_MHZ="${{LB_SM_CLOCK_MHZ-{clock}}}"
LB_MEM_CLOCK_MHZ="${{LB_MEM_CLOCK_MHZ-{mem_clock}}}"
LB_REQUIRE_CLOCK_LOCK="${{LB_REQUIRE_CLOCK_LOCK:-{require_lock}}}"
LB_CLOCK_SETTLE_S="${{LB_CLOCK_SETTLE_S:-3}}"
LB_CLOCK_TOL_MHZ="${{LB_CLOCK_TOL_MHZ:-50}}"
# Only this job's GPU: a lock and a reset act on every GPU nvidia-smi is not told to skip
# (it does not read CUDA_VISIBLE_DEVICES), and a multi-GPU machine may be timing another
# run on the next GPU.
lb_smi() {{
    if [ -n "${{CUDA_VISIBLE_DEVICES:-}}" ]; then
        "$LB_NVIDIA_SMI" -i "$CUDA_VISIBLE_DEVICES" "$@"
    else
        "$LB_NVIDIA_SMI" "$@"
    fi
}}
lb_clocks_reset() {{
    lb_smi -rgc > /dev/null 2>&1 || true
    lb_smi -rmc > /dev/null 2>&1 || true
}}
lb_clocks_read() {{
    lb_smi --query-gpu=clocks.current.graphics,clocks.current.memory \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd ';' - || true
}}
lb_clocks_reset
LB_CLOCK_STATE="unlocked"
if [ -n "$LB_SM_CLOCK_MHZ" ]; then
    trap lb_clocks_reset EXIT
    if lb_smi -lgc "$LB_SM_CLOCK_MHZ,$LB_SM_CLOCK_MHZ" > /dev/null 2>&1 && \
       {{ [ -z "$LB_MEM_CLOCK_MHZ" ] || \
          lb_smi -lmc "$LB_MEM_CLOCK_MHZ,$LB_MEM_CLOCK_MHZ" > /dev/null 2>&1; }}; then
        sleep "$LB_CLOCK_SETTLE_S"
        LB_CLOCK_READ="$(lb_clocks_read)"
        # every GPU's "<graphics>,<memory>" within the tolerance of what was asked
        if [ -n "$LB_CLOCK_READ" ] && awk -F'[,;]' -v sm="$LB_SM_CLOCK_MHZ" \
                -v mem="$LB_MEM_CLOCK_MHZ" -v tol="$LB_CLOCK_TOL_MHZ" '
                {{ if (NF < 2 || NF % 2) exit 1
                   for (i = 1; i < NF; i += 2) {{
                       d = $i - sm; if (d < 0) d = -d; if (d > tol) exit 1
                       if (mem != "") {{ d = $(i + 1) - mem; if (d < 0) d = -d; if (d > tol) exit 1 }}
                   }} }}' <<< "$LB_CLOCK_READ"; then
            LB_CLOCK_STATE="locked sm=$LB_SM_CLOCK_MHZ mem=${{LB_MEM_CLOCK_MHZ:-free}} (read $LB_CLOCK_READ)"
        else
            LB_CLOCK_STATE="lock not held (asked sm=$LB_SM_CLOCK_MHZ mem=${{LB_MEM_CLOCK_MHZ:-free}}, read ${{LB_CLOCK_READ:-nothing}})"
            lb_clocks_reset
        fi
    else
        LB_CLOCK_STATE="lock refused (nvidia-smi -lgc / -lmc failed)"
        lb_clocks_reset
    fi
fi
export LB_CLOCK_STATE

# The board's clock is CUPTI read with cupti-python; without it eval/leaderboard.py falls
# back to torch.profiler, whose per-op instrumentation reads multi-kernel calls long.
# Best effort, bounded; bake cupti-python into the image to skip it.
if [ "${{LB_INSTALL_CUPTI:-1}}" = "1" ] && ! python3 -c "import cupti" 2>/dev/null; then
    timeout 180 python3 -m pip install -q cupti-python > /dev/null 2>&1 \
        || echo "[gpu_entry.sh] cupti-python not installable — timer falls back to torch.profiler"
fi

# The SOL image's fbtriton wheel ships launch.h as triton/runtime/launch.h, while its NVIDIA
# driver looks only in backends/nvidia/ (and a source-tree path): every Triton kernel — a
# Triton reference's or student's — then dies with "launch.h not found". Put a copy where
# the driver looks (idempotent; the same fix the SOL nodes' bootstrap applies).
TRITON_DIR="$(python3 -c 'import os, triton; print(os.path.dirname(triton.__file__))' 2>/dev/null || true)"
if [ -n "$TRITON_DIR" ] && [ -f "$TRITON_DIR/runtime/launch.h" ] \
        && [ ! -f "$TRITON_DIR/backends/nvidia/launch.h" ] && [ -d "$TRITON_DIR/backends/nvidia" ]; then
    cp "$TRITON_DIR/runtime/launch.h" "$TRITON_DIR/backends/nvidia/launch.h" 2>/dev/null \
        && echo "[gpu_entry.sh] patched triton: backends/nvidia/launch.h (fbtriton wheel layout)"
fi

echo "=== gpu_entry.sh ({name}, {suite}) ==="
echo "  task:      ${{1:-help}}"
echo "  hostname:  $(hostname)"
echo "  date:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GPU:       $(lb_smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  clocks:    $LB_CLOCK_STATE"
# A lock that did not take: what the driver reports about the GPU right after (event
# reasons, the clock it settled at, power, temperature) — the evidence for the node.
if [ -n "$LB_SM_CLOCK_MHZ" ] && [ "${{LB_CLOCK_STATE%% *}}" != "locked" ]; then
    echo "  clockdiag: $(lb_smi --query-gpu=clocks_event_reasons.active,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,pstate --format=csv,noheader 2>/dev/null || echo 'N/A')"
fi
echo "  PyTorch:   $(python3 -c 'import torch; print(f"{{torch.__version__}} CUDA={{torch.version.cuda}}")' 2>/dev/null || echo 'N/A')"
echo "  timer:     FK_LB_TIMER=${{FK_LB_TIMER:-auto}}  cupti-python: $(python3 -c 'import cupti; print("yes")' 2>/dev/null || echo no)"
echo ""

TASK="${{1:-help}}"
shift || true
TASK_TIMEOUT="${{TASK_TIMEOUT:-1800}}"
NCU_TIMEOUT="${{NCU_TIMEOUT:-600}}"

# A card whose readings are only comparable to the board's under its lock does not
# time without it: an unlocked reading is faster than the board's, and a target or
# score derived from the board's numbers would pass a kernel the board does not. Nor
# does it run anything else on a node where the lock was asked for and did not hold:
# a probe's, a pricing tool's or NCU's reading there is just as far off the board's
# clock, so every task but help is refused; the loop reports such a job as not run
# and resubmits it.
# (LB_SM_CLOCK_MHZ="" asks for no lock: then only the timing tasks are refused.)
LB_REFUSE=0
if [ "$LB_REQUIRE_CLOCK_LOCK" = "1" ] && [ "${{LB_CLOCK_STATE%% *}}" != "locked" ]; then
    case "$TASK" in
        help) ;;
        perf|all|cardref|ref) LB_REFUSE=1 ;;
        *) [ "$LB_CLOCK_STATE" = "unlocked" ] || LB_REFUSE=1 ;;
    esac
fi
if [ "$LB_REFUSE" = "1" ]; then
    # ForgeKernel reads the REFUSED line (submit/jobs.py CLOCK_LOCK_REFUSED_MARKER):
    # keep its wording and keep it at the start of the line.
    echo "[gpu_entry.sh] REFUSED: this card times under the board's clock lock, and" \
         "it did not hold here ($LB_CLOCK_STATE). LB_REQUIRE_CLOCK_LOCK=0 times anyway" \
         "(the readings are then not comparable to the board's)."
    exit 3
fi

run_with_timeout() {{
    echo "[gpu_entry.sh] timeout=${{TASK_TIMEOUT}}s"
    set +e
    timeout --signal=KILL "$TASK_TIMEOUT" "$@"
    local rc=$?
    set -e
    if [ $rc -eq 137 ]; then
        echo "[gpu_entry.sh] KILLED: task exceeded ${{TASK_TIMEOUT}}s timeout (likely kernel deadlock)"
        exit 137
    fi
    return $rc
}}

ncu_bounded() {{
    set +e
    timeout --signal=KILL "$NCU_TIMEOUT" "$@"
    local rc=$?
    set -e
    [ $rc -ne 0 ] && echo "[gpu_entry.sh] NCU exited rc=$rc — continuing without a profile"
    return 0
}}

# Must match KEY_METRICS in src/forgekernel/inner/ncu.py.
NCU_METRICS="gpu__time_active.sum,launch__registers_per_thread,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_elapsed,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic"

case "$TASK" in
    correctness) run_with_timeout python3 eval/bench.py --mode correctness ;;
    perf|all)    run_with_timeout python3 eval/bench.py --mode all ;;
    cardref)     run_with_timeout python3 eval/bench.py --mode all --impl ref ;;
    ref)         run_with_timeout python3 eval/bench.py --mode ref ;;
    calibrate)   run_with_timeout python3 eval/bench.py --mode calibrate ;;
    ncu)
        ncu_bounded ncu --csv --target-processes all --profile-from-start off \
            --launch-count 24 --kill yes --metrics "$NCU_METRICS" python3 eval/ncu_run.py
        ;;
    cmd)         run_with_timeout bash -c "$*" ;;
    help|*)
        sed -n '2,9p' "$0"
        exit 1
        ;;
esac

echo ""
echo "[gpu_entry.sh] task=$TASK finished"
'''

# (SM MHz, memory MHz, require the lock for every task but help, note). SOL-ExecBench's board
# locks B200 at its clock_lock preset and verifies it; FlashInfer-Bench runs unlocked.
SUITE_CLOCK = {
    SUITE_SOL: (str(SOL_BOARD_CLOCK["sm_mhz"]), str(SOL_BOARD_CLOCK["mem_mhz"]), "1",
                "SOL-ExecBench's board locks B200 at SM 1500 / memory 3996 MHz and times "
                "T_b and T_SOL there."),
    SUITE_FIB: ("", "", "0", "FlashInfer-Bench runs unlocked; set LB_SM_CLOCK_MHZ (and "
                             "LB_MEM_CLOCK_MHZ) to pin one for steadier readings."),
}


def render_gpu_entry(problem: Problem) -> str:
    clock, mem_clock, require_lock, clock_note = SUITE_CLOCK[problem.suite]
    return GPU_ENTRY_TEMPLATE.format(
        name=problem.name, suite=problem.suite, arch="sm_100a", clock=clock,
        mem_clock=mem_clock, require_lock=require_lock, clock_note=clock_note)


def render_window_toml(problem: Problem) -> str:
    ins = json.dumps(list(problem.definition["inputs"]))
    outs = json.dumps(list(problem.definition["outputs"]))
    ref = problem.reference_impl
    if ref.get("kind") == "binary":
        who = f"{ref.get('name')} (sealed binary reference, ref_bin/)"
    elif ref.get("kind") == "sealed":
        who = f"{ref.get('name')} (sealed source reference, ref_bin/ref_payload.bin)"
    elif ref.get("kind") == "baseline_solution":
        who = f"{ref.get('name')} (dataset baseline solution)"
    elif problem.board:
        who = "the board's scoring baseline T_b (the definition's calling convention)"
    else:
        who = "the definition's reference run"
    return f"""# {problem.name} — timed-window contract (GENERATED by `forge init --leaderboard`).
#
# Both sides take the definition's inputs and produce its outputs: the reference
# ({who}) and the student's run() have the same boundary by construction, so there
# is nothing to declare. The DPS student receives its outputs pre-allocated outside
# the window (the suite's harness does the same); the value-returning reference
# allocates them inside its call — torch's caching allocator, no kernel.

schema_version = 1

[reference]
window  = "{who}, one call per timed sample"
inputs  = {ins}
outputs = {outs}

[student]
window  = "run(), one call per timed sample (DPS)"
inputs  = {ins}
outputs = {outs}
"""


def render_loop_toml_example(problem: Problem, op: str, kernel_file: str = "kernel.py") -> str:
    """The suggested loop.toml. What it must get right is the stop: the loop derives
    target_ms from the reference declared here (the architecture one when present)
    divided by --equivalence-ratio, and stops once a KEEP reads under it. So the
    reference must be the number the board ranks against — never the definition's
    eager PyTorch run, which any fused kernel beats on its first KEEP. The language
    follows the kernel file: a CUDA C++ card runs the loop's CUDA mode (cuda-sol: its
    own prompts and hw_kb/cuda/), any other the CuTe DSL mode."""
    ref = problem.reference_impl
    ref_name = ref.get("name") or "definition reference"
    shape = ", ".join(f"{k}={v}" for k, v in (problem.workload.get("axes") or {}).items())
    ref_shape = f"{problem.name} workload {problem.workload.get('uuid')} ({shape})"
    if problem.group:
        ref_shape = (f"{problem.name} group of {len(problem.group)} workloads "
                     f"({', '.join(g['name'] for g in problem.group)}; geomean)")
    head = "# Suggested loop.toml for this leaderboard card — copy to loop.toml and review.\n"
    language = "cuda-sol" if is_cuda_kernel_file(kernel_file) else "cutedsl"
    common = f"""[loop]
language    = "{language}"{' ' * (17 - len(language))}# what the student writes the kernel in
kb_arch     = "sm100"            # B200
gate_window = "device"           # = the leaderboard reading (per-call span, cold L2)
"""
    if ref.get("kind") in SEALED_KINDS:
        rt = f"_{op}ref"
        if ref.get("kind") == "binary":
            what = (f"# Blackbox run: the reference is a sealed binary (ref_bin/{ref.get('module')}.so, built by\n"
                    f"# the recon tree {rt} from sources the agents never see). bench.py times it in the same job\n")
            task, tool, task_note = "binref", "binref.py", "build if needed, check, bench, ncu"
        else:
            what = (f"# Blackbox run: the reference is sealed SOURCE (ref_bin/ref_payload.bin, keyed; the key is\n"
                    f"# in the gitignored submit.local.toml of this card and of the recon tree {rt}, decoded in\n"
                    f"# memory inside the job only). bench.py times it in the same job\n")
            task, tool, task_note = "sealref", "sealref.py", "check, bench, ncu"
        return head + what + f"""# as the kernel on the board's clock; target_ms = its device reading / --equivalence-ratio
# (1.05 = beat it by 5%). ref_source_paths must list every place its source lives.
""" + common + f"""oracle_mode        = "blackbox"
recon_op           = "{rt}"
ref_mode           = "{task}"{' ' * (13 - len(task))}# {rt} gpu_entry task: {task_note}
ref_submit_op      = "{rt}"
ref_in_job         = true
ref_name           = "{ref_name}"
ref_shape          = "{ref_shape}"
ref_hardware       = "B200"
arch_ref_mode      = "{task}"
arch_ref_submit_op = "{rt}"
arch_ref_name      = "{ref_name}"
ref_price_op       = "{rt}"
ref_price_app      = ["python3", "{tool}", "ncu"]
timeline_probe     = "on"
ref_source_paths   = []             # REQUIRED: fill (forge init --ref-source-path)
ref_source_urls    = []
"""
    if problem.suite == SUITE_FIB:
        caveat = ""
        if ref.get("kind") != "baseline_solution":
            caveat = ("# No dataset baseline solution: the reference below is the definition's\n"
                      "# eager PyTorch run, which a fused kernel beats at once — give the loop\n"
                      "# an explicit --target-ms (or a large --equivalence-ratio).\n")
        return head + caveat + common + f"""
# The in-job reference: bench.py times it mirrored with the kernel in every job, and
# `forge {op} baseline` archives it alone on the same clock. target_ms = its reading /
# --equivalence-ratio (1.0 = match it, 1.2 = beat it by 20%).
ref_cmd      = "python3 eval/bench.py --mode ref"
ref_name     = "{ref_name}"
ref_shape    = "{ref_shape}"
ref_hardware = "B200"
"""
    board = problem.board
    if not board:
        return head + f"""# No SOL-ExecBench board scores are attached to this card (README notes), so
# nothing here says when a kernel is good enough. The definition's eager reference
# is not a target (host-bound, unstable, beaten at once) and is not timed in-job:
# give the loop an explicit --target-ms — e.g. from the board's T_b for this workload
# (`forge new ... --sol-board <saved kernels/<id> JSON>` attaches it).
""" + common + f"""
ref_shape    = "{ref_shape}"
ref_hardware = "B200"
"""
    if problem.group:
        return head + _group_board_example(problem, common, ref_shape)
    t_b, t_sol = board["t_b_ms"], board["t_sol_ms"]
    ladder = "   ".join(f"S {s}: {t:.6g} ms" for s in (0.6, 0.7, 0.8, 0.9)
                        if (t := time_for_score(s, t_b, t_sol)) is not None)
    clock = board.get("clock_mhz") or SOL_BOARD_CLOCK
    return head + f"""# SOL-ExecBench scores S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL)) against its own
# scoring baseline T_b (an optimized PyTorch implementation; source unpublished) and
# the hardware floor T_SOL, both timed by the board at SM {clock['sm_mhz']} / memory {clock['mem_mhz']} MHz,
# the lock this card's gpu_entry.sh holds (and refuses to time without).
# This workload (board problem {board.get('kernel_id')}, workload {board.get('workload_index')}):
#   T_b = {t_b:.6g} ms   T_SOL = {t_sol:.6g} ms
# The reference below IS T_b, declared (the board's number, not re-measured here), so
# target_ms = T_b / --equivalence-ratio: 1.0 stops at S = 0.5. Kernel latency per
# score, for an explicit --target-ms:
#   {ladder or '(T_b does not sit above T_SOL on the board — no ladder)'}
# The definition's eager reference is not timed in-job (problem.json time_reference).
""" + common + f"""
arch_ref_name         = "SOL-ExecBench T_b (the board's scoring baseline)"
arch_ref_ms           = {t_b!r}
arch_ref_device_ms    = {t_b!r}
arch_ref_device_basis = "cupti_kernel_span"   # the board's per-call CUPTI span
ref_shape             = "{ref_shape}"
ref_hardware          = "B200"
"""


def _geomean(values) -> float:
    vals = [float(v) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _group_board_example(problem: Problem, common: str, ref_shape: str) -> str:
    """A group SOL card's board-derived reference: the gate reading is the geomean over the
    group's workloads, so the declared reference is the geomean of their T_b, named over
    the same workloads (arch_ref_workloads)."""
    clock = problem.board.get("clock_mhz") or SOL_BOARD_CLOCK
    rows = "\n".join(f"#   {g['name']:<6} T_b = {g['board']['t_b_ms']:.6g} ms   "
                     f"T_SOL = {g['board']['t_sol_ms']:.6g} ms" for g in problem.group)
    t_b = _geomean(g["board"]["t_b_ms"] for g in problem.group)
    names = json.dumps([g["name"] for g in problem.group])
    return f"""# SOL-ExecBench scores S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL)) per workload against its
# scoring baseline T_b and the hardware floor T_SOL, timed by the board at SM {clock['sm_mhz']} /
# memory {clock['mem_mhz']} MHz (the lock this card's gpu_entry.sh holds). This card is a GROUP of
# {len(problem.group)} workloads of board problem {problem.board.get('kernel_id')}; the gate is the geomean over them:
{rows}
# The reference below is the geomean of their T_b (the board's numbers, not re-measured
# here), so target_ms = geomean T_b / --equivalence-ratio. The eval JSON's
# leaderboard.sol_board carries every workload's S and their mean.
# The definition's eager reference is not timed in-job (problem.json time_reference).
""" + common + f"""
arch_ref_name         = "SOL-ExecBench T_b (the board's scoring baseline), geomean over the group"
arch_ref_ms           = {t_b!r}
arch_ref_device_ms    = {t_b!r}
arch_ref_device_basis = "cupti_kernel_span"   # the board's per-call CUPTI span
arch_ref_workloads    = {names}
ref_shape             = "{ref_shape}"
ref_hardware          = "B200"
"""


def render_readme_section(problem: Problem, kernel_file: str = "kernel.py") -> str:
    ref = problem.reference_impl
    kf = Path(kernel_file).name
    notes = "".join(f"\n- {n}" for n in problem.notes)
    if ref.get("kind") == "binary":
        ref_line = (f"- In-tree reference: the sealed binary reference `ref_bin/{ref.get('module')}.so` "
                    f"({ref.get('name')}), timed in the same job and on the same clock as the student "
                    f"(vs_ref_device); its source is not in the tree, and {kf} touching it or the "
                    f"compiler toolchain is judged FENCE by the fence.")
        selftest = ("- Self-test: `bash gpu_entry.sh cardref` (the binary reference as the "
                    "implementation: correctness should PASS, vs_ref_device ≈1.0).")
    elif ref.get("kind") == "sealed":
        ref_line = (f"- In-tree reference: the sealed source reference `ref_bin/ref_payload.bin` "
                    f"({ref.get('name')}, encrypted with a key the agent cannot read), unpacked in "
                    f"memory inside the job and timed in the same job and on the same clock as the "
                    f"student (vs_ref_device); the source never touches disk, and {kf} touching its "
                    f"loader, payload, key, environment variables, interpreter reflection or JIT "
                    f"cache is judged FENCE by the fence.")
        selftest = ("- Self-test: `bash gpu_entry.sh cardref` (the sealed reference as the "
                    "implementation: correctness should PASS, vs_ref_device ≈1.0).")
    elif problem.suite == SUITE_SOL:
        board = problem.board
        ref_line = ("- The in-tree reference is not timed (the definition's eager PyTorch reference is "
                    "host-limited and its readings are unstable; it is used only to judge correctness).")
        if board and problem.group:
            ref_line += ("\n- Board scoring baseline (frozen from the board at `forge new`, per "
                         "workload): " + "; ".join(
                f"{g['name']} T_b {g['board']['t_b_ms']:.6g} / T_SOL {g['board']['t_sol_ms']:.6g} ms"
                for g in problem.group) + "; the eval JSON's `leaderboard.sol_board` gives each "
                "workload's SOL score and their mean, and the reference in loop.toml.example is the "
                "geomean of the T_b values.")
        elif board:
            ref_line += (f"\n- Board scoring baseline (frozen from the board at `forge new`): "
                         f"T_b = {board['t_b_ms']:.6g} ms, T_SOL = {board['t_sol_ms']:.6g} ms; the "
                         f"eval JSON's `leaderboard.sol_score` is the board's SOL score, and the "
                         f"reference in loop.toml.example is T_b.")
        ref_line += ("\n- Clock lock: gpu_entry.sh locks SM 1500 / memory 3996 MHz as the board does "
                     "and reads it back to verify; if the lock fails it refuses every task except help "
                     "(`LB_REQUIRE_CLOCK_LOCK=0` lets it through); it resets on entry and on exit, so no "
                     "lock is left for the next job.")
        selftest = ("- Self-test: `bash gpu_entry.sh cardref` (the definition reference as the "
                    "implementation: correctness should PASS).")
    else:
        ref_line = (f"- In-tree reference: {ref.get('name') or 'definition reference'} "
                    f"({ref.get('kind')}), on the same clock as the student, mirror-paired.")
        selftest = ("- Self-test: `bash gpu_entry.sh cardref` (the reference as the implementation; "
                    "vs_ref_device should be ≈1.0).")
    cuda_line = ("\n- `{0}` is CUDA C++ (the board's cuda_cpp format: a single file with its own "
                 "PYBIND11_MODULE binding), compiled by eval/leaderboard.py the way the board does "
                 "(torch cpp_extension, -O3 --use_fast_math sm_100a, -lcuda, CUTLASS/CuTe headers); "
                 "extra nvcc options go on a `// fk-cuda-cflags: ...` line (code-generation / "
                 "diagnostic options only), and a compile failure is judged COMPILE_ERROR."
                 .format(kf) if is_cuda_kernel_file(kf) else "")
    if problem.group:
        what = (f"workload group ({len(problem.group)} workloads; one kernel has to be fast on the "
                f"whole group):\n" + "\n".join(
            f"  - `{g['name']}`: axes `{json.dumps(g['workload'].get('axes'))}`" for g in problem.group)
            + "\n- Multi-workload tree: every workload is judged for correctness by the board's rules "
              "(it passes only if all pass) and timed one by one as the board does; the gate reading "
              "(device_median_ms / vs_ref_device) is the geomean over the whole group, with "
              "per-workload readings in `performance.workloads`. Each round's NCU samples only the "
              "first workload (the representative); pricing, stall / SASS analysis and the timeline "
              "probe each read `meta_workloads` (the representative + the group's smallest + "
              "largest, plus the group's middle one when the representative is itself at an end: "
              + ", ".join(f"`{n}`" for n in meta_workload_names(problem))
              + "), one reading per shape, then merged.")
    else:
        what = (f"workload\n  `{problem.workload.get('uuid')}`, axes "
                f"`{json.dumps(problem.workload.get('axes'))}`.")
    if problem.check_only:
        what += ("\n- **Correctness-only workloads, not timed** (" + str(len(problem.check_only))
                 + "; the kernel is scored on them outside the tree too, so it must run on them and "
                 "run correctly):\n" + "\n".join(
                     f"  - `{g['name']}`: axes `{json.dumps(g['workload'].get('axes'))}`"
                     for g in problem.check_only)
                 + "\n- Every judging of the tree first judges these workloads by the board's rules; "
                   "if any one fails, the whole tree fails (the error names it and marks it "
                   "\"correctness-only\"); they are not timed and do not enter the gate geomean. A "
                   "kernel that supports only the timed workloads (e.g. hard-codes M and reports "
                   "unsupported for any other size) is rejected here.")
    return f"""

## Leaderboard mode (generated by `forge init --leaderboard {problem.suite}`)

- Problem: `{problem.name}` ({problem.suite}, dataset {problem.source.get('dataset')}), {what}
- Frozen in `eval/leaderboard/problem.json`; judging and timing all go through `eval/leaderboard.py`
  (a line-by-line replica of {problem.source.get('upstream')}'s evaluation protocol; do not edit
  it). **Do not write your own tolerance or timing anywhere in the tree.**
- Gate reading = board reading: a single call, L2 flushed before the call, CUPTI kernel span
  (`device_window_basis` names the timer actually used).
{ref_line}
- The `run` signature of `{kf}` is frozen by the board definition (parameter names = the
  definition's inputs + outputs, DPS).{cuda_line}
{selftest}
- Submission: `forge export <run>` writes the board's solution JSON.{notes}
"""


def write_card(op_dir: Path, problem: Problem, template_eval: Path, *,
               dps: bool = True, kernel_file: str = "kernel.py") -> list[Path]:
    """Write the leaderboard files into a freshly scaffolded card."""
    op_dir = Path(op_dir)
    lb_dir = op_dir / LEADERBOARD_DIR
    lb_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def put(rel: str, text: str) -> None:
        path = op_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        created.append(path)

    put(PROBLEM_FILE, json.dumps(problem_json(problem, dps=dps, kernel_file=kernel_file), indent=2) + "\n")
    for rel, src in problem.blobs.items():
        dst = lb_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        created.append(dst)
    for rel, text in problem.reference_files.items():
        put(f"{LEADERBOARD_DIR}/{rel}", text)
    for name in ("leaderboard.py", "timing.py", "precision.py"):
        dst = op_dir / "eval" / name
        shutil.copy2(template_eval / name, dst)
        created.append(dst)
    fmt = {"name": problem.name, "suite": problem.suite,
           "upstream": problem.source.get("upstream"),
           "workload": problem.workload.get("uuid") if not problem.group else
           f"group of {len(problem.group)}: {', '.join(g['name'] for g in problem.group)}",
           "group_note": ("" if not problem.group else
                          "Every workload of the group is judged (all must pass) and timed; the gate\n"
                          "reading is the geometric mean over them.\n")
           + ("" if not problem.check_only else
              f"Correctness-only workloads ({', '.join(g['name'] for g in problem.check_only)}) are\n"
              "judged first on every run (all must pass) and never timed.\n")}
    put("eval/bench.py", BENCH_TEMPLATE.format(**fmt))
    put("eval/ncu_run.py", NCU_TEMPLATE.format(**fmt))
    put("gpu_entry.sh", render_gpu_entry(problem))
    (op_dir / "gpu_entry.sh").chmod(0o755)
    put(kernel_file, render_kernel_stub_cuda(problem, dps=dps) if is_cuda_kernel_file(kernel_file)
        else render_kernel_stub(problem, dps=dps))
    put("window.toml", render_window_toml(problem))
    put("loop.toml.example", render_loop_toml_example(problem, op_dir.name, kernel_file))
    readme = op_dir / "README.md"
    if readme.exists():
        readme.write_text(readme.read_text(encoding="utf-8") + render_readme_section(problem, kernel_file),
                          encoding="utf-8")
    return created


def blob_bytes(problem: Problem) -> int:
    return sum(p.stat().st_size for p in problem.blobs.values())


# --- export -------------------------------------------------------------------------

# ForgeKernel language -> (FlashInfer-Bench spec.language, SOL-ExecBench spec.languages,
# SOL dependencies).
_EXPORT_LANG = {
    "cutedsl": ("python", ["cute_dsl"], ["cutlass"]),
    "triton": ("triton", ["triton"], ["triton >= 2.3"]),
    "pytorch": ("python", ["pytorch"], ["torch"]),
    "cuda-sol": ("cuda", ["cuda_cpp"], []),
}


def infer_language(kernel_src: str, loop_language: str | None = None) -> str:
    """cutedsl / triton / pytorch, from loop.toml's language or the kernel's imports."""
    if loop_language in ("cutedsl", "triton"):
        return loop_language
    tree = ast.parse(kernel_src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    if "cutlass" in mods:
        return "cutedsl"
    if "triton" in mods:
        return "triton"
    return "pytorch"


_CPP_FN = r"(?:void|torch::Tensor|at::Tensor)\s+{entry}\s*\(([^)]*)\)"


def cuda_signature_findings(kernel_src: str, problem: dict) -> list[str]:
    """A CUDA kernel file: a binding that takes its module name from the build, ``entry``
    defined and bound, its parameters the definition's names in order."""
    entry = problem.get("entry", "run")
    kf = Path(problem.get("kernel_file", "kernel.cu")).name
    src = re.sub(r"//[^\n]*|/\*.*?\*/", " ", kernel_src, flags=re.S)
    out = []
    if not re.search(r"PYBIND11_MODULE\s*\(\s*TORCH_EXTENSION_NAME\s*,", src):
        out.append(f"{kf} has no PYBIND11_MODULE(TORCH_EXTENSION_NAME, m): the build names the module")
    if not re.search(rf'\.def\s*\(\s*"{re.escape(entry)}"', src):
        out.append(f'{kf} binds no "{entry}" (m.def("{entry}", &{entry}))')
    m = re.search(_CPP_FN.format(entry=re.escape(entry)), src)
    if m is None:
        return out + [f"{kf} defines no `void {entry}(...)` (or `torch::Tensor {entry}(...)`)"]
    got = [re.split(r"[\s&*]+", p.strip())[-1] for p in m.group(1).split(",") if p.strip()]
    defn = problem["definition"]
    want = list(defn["inputs"]) + (list(defn["outputs"]) if problem.get("dps", True) else [])
    if got != want:
        out.append(f"{entry}({', '.join(got)}) — the definition wants {entry}({', '.join(want)})")
    return out


def signature_findings(kernel_src: str, problem: dict) -> list[str]:
    """Does kernel.py's entry take exactly the definition's names, in order?"""
    entry = problem.get("entry", "run")
    if is_cuda_kernel_file(problem.get("kernel_file", "kernel.py")):
        return cuda_signature_findings(kernel_src, problem)
    try:
        tree = ast.parse(kernel_src)
    except SyntaxError as e:
        return [f"kernel.py does not parse: {e}"]
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry),
              None)
    if fn is None:
        return [f"kernel.py defines no top-level {entry}()"]
    defn = problem["definition"]
    want = list(defn["inputs"]) + (list(defn["outputs"]) if problem.get("dps", True) else [])
    got = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    if got != want:
        return [f"{entry}({', '.join(got)}) — the definition wants "
                f"{entry}({', '.join(want)})"]
    return []


def export_solution(op_dir: Path, *, name: str, author: str, language: str | None = None,
                    description: str | None = None, hardware: str = "B200") -> dict:
    """The suite's solution JSON for this card's kernel.py."""
    op_dir = Path(op_dir)
    problem = json.loads((op_dir / PROBLEM_FILE).read_text(encoding="utf-8"))
    kfile = problem.get("kernel_file", "kernel.py")
    src = (op_dir / kfile).read_text(encoding="utf-8")
    if (bad := signature_findings(src, problem)):
        raise ValueError("; ".join(bad))
    cuda = is_cuda_kernel_file(kfile)
    lang = "cuda-sol" if cuda else language or infer_language(src, _loop_language(op_dir))
    if lang not in _EXPORT_LANG:
        raise ValueError(f"language must be one of {sorted(_EXPORT_LANG)}, got {lang!r}")
    fib_lang, sol_langs, sol_deps = _EXPORT_LANG[lang]
    entry = f"{Path(kfile).name}::{problem.get('entry', 'run')}"
    dps = bool(problem.get("dps", True))
    sources = [{"path": Path(kfile).name, "content": src}]
    desc = description or f"ForgeKernel {lang} kernel for {problem['definition']['name']}"
    if problem["suite"] == SUITE_FIB:
        spec = {"language": fib_lang, "target_hardware": [f"NVIDIA {hardware}"],
                "entry_point": entry, "dependencies": [], "destination_passing_style": dps}
    else:
        spec = {"languages": sol_langs, "target_hardware": [hardware],
                "entry_point": entry, "destination_passing_style": dps,
                "dependencies": sol_deps}
        if cuda:  # the flags the card built it with (SOL resolves ld_flags to -lcuda)
            spec["compile_options"] = {"cuda_cflags": cuda_kernel_flags(src)}
    return {"name": name, "definition": problem["definition"]["name"], "author": author,
            "description": desc, "spec": spec, "sources": sources}


def _loop_language(op_dir: Path) -> str | None:
    path = Path(op_dir) / "loop.toml"
    try:
        return tomllib.loads(path.read_text(encoding="utf-8")).get("loop", {}).get("language")
    except (OSError, ValueError):
        return None
