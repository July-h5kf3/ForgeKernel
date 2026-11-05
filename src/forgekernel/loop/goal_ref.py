"""No reference to distil from: generate one with a ``/goal`` session, then run as usual.

A source-mode run distils its milestone from an oracle — the source of a kernel that already
does what the student is asked to reproduce. A card that has none (a leaderboard card
scaffolded without ``--sealed-ref`` / ``--binary-ref``, a hand-made card whose loop.toml
names no ``oracle``) had nothing to distil from: ``_has_distillation_input`` said no, the
teacher never ran, and the student started against whatever milestone was on disk — on a
fresh card, none. This module closes that gap the way a source reference is often written by
hand: an interactive ``claude`` driven by ``/goal`` writes the kernel from scratch on the
card's own harness, the harness (not the agent) picks and re-measures the best correct
version, and that version becomes the run's oracle (source mode). From there the loop is
unchanged: the teacher distils milestone v1 from it, the student starts from the card's
stub as in every source-mode run.

When it runs (``[loop] ref_from_goal``, written by ``forge run`` from the task):

  auto    (default) oracle_mode is "source", no oracle is declared, and no valid
          milestone is on disk. A blackbox run has its sealed reference, and a
          milestone on disk is the run's input.
  off     never (a task with a reference, a leaderboard card).

Whatever the setting, a reference this operator directory already generated is reused
(``goal_ref/state.json`` status "accepted"), a generation the loop was interrupted in is
resumed, and one that failed refuses the launch until the operator removes
``goal_ref/`` or turns the setting off.

How the session measures. The agent never submits GPU work itself — the student does not
either; the driver does. Its evaluation command (``python -m forgekernel.loop.goal_ref
eval``) only snapshots the kernel file into a request directory inside its workspace and
waits; the loop process, outside the agent's sandbox, serves the request: it copies the
snapshot into a submission tree of its own (``operators/_<op>goalref``, a clone of the
card: the card's gpu_entry.sh and eval/, so the verdict is the card's), submits the
card's ``all`` task through ``RealBackend.submit_gpu`` (a lost runner, clock-lock
refusals as for every FK job), parses the eval JSON
with the driver's own ``parse_eval`` and appends the reading to a ledger the agent cannot
write. The answer it writes back ends in a machine line the ``/goal`` condition refers to
(``FK_GOALREF_RESULT`` / ``FK_GOALREF_DONE`` / ``FK_GOALREF_DEADLINE_PASSED``), and the
session's deliverable check (``goal_ref check``) reads the ledger, never the agent's text.

What counts. A candidate is eligible when the card judged it correct (and, with the
precision gate merged, its contract did not fail it) and its source calls no kernel
library for the computation (``library_findings``: a reference that is a flashinfer call
has nothing to distil). The speed bar (``resolve_bar``) is the first of: --target-ms, the
architecture reference the card declares (a SOL card's T_b), the in-job reference the
bench times beside the kernel (same-job ratio >= ``DEFAULT_RATIO``), an archived
library baseline; with none of them there is no bar and the session runs to its
deadline. The session ends at the bar or at the deadline (``ref_from_goal_hours`` after
it started, absolute: a resumed session does not get a fresh budget), plus
``Generation.report_s`` past the deadline to write its report and run the check; then
the eligible candidates are re-measured by the loop, fastest
first, ``ACCEPT_RUNS`` times each, and the first that stays correct in every
run is the reference. No eligible candidate — or none that survives — fails the launch
(EXIT_GOAL_REF_FAILED); a wrong kernel is never made an oracle.

What the loop gets (``apply``): ``oracle`` = the accepted kernel (and the session's
report beside it), ``oracle_mode`` = "source", ``kb_lint_oracle_tokens`` = false unless
loop.toml says otherwise (the generated kernel may use the KB's helper names), and the
acceptance reading declared as the architecture reference (the target) when it is
the fastest known reference: faster than an architecture reference the card declares as
constants (a SOL card's T_b) and not slower than the same-job library reference
(``arch_decision``). A measured architecture reference and --target-ms always win.

Layout, all under ``operators/<op>/goal_ref/`` (git-excluded from the card; denied to the
student as a teacher-only tree):

  state.json      status, deadline, bar, attempts, the accepted candidate and readings
  ledger.jsonl    one row per evaluation (and per acceptance run)
  candidates/NNN  the kernel each evaluation judged, its job log and readings
  acceptance/     the acceptance runs' logs
  reference/      the oracle: the accepted kernel file + NOTES.md (the session's report)
  session/        TASK.md as given, the agent's stdout/stderr per attempt, transcripts
  ws/             the agent's working directory (a clone of the card)
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from forgekernel.paths import operators_dir, repo_root

MODES = ("auto", "off")
DEFAULT_MODE = "auto"
DEFAULT_HOURS = 2.0
# The speed bar's ratio over an in-job reference, and how many acceptance runs a
# candidate must stay correct in: fixed (docs/design/v1-launch.md Appendix A.2).
DEFAULT_RATIO = 1.0
ACCEPT_RUNS = 2
# How many eligible candidates the acceptance tries before it gives up.
ACCEPT_CANDIDATES = 3
# Agent invocations per generation: one that ends early (usage limit, crash) is started
# again while the deadline allows; a limited session resumes (scripts/claude-goal.py).
MAX_ATTEMPTS = 12

DIRNAME = "goal_ref"
STATE = "state.json"
LEDGER = "ledger.jsonl"
SPOOL = ".goalref"
TREE_MARKER = ".fk-goalref-tree"

STATUS_GENERATING = "generating"
STATUS_ACCEPTED = "accepted"
STATUS_FAILED = "failed"

# How the loop measures an accepted reference through the submission tree when the card
# declares no library reference of its own (``apply``): the card's perf task and its NCU
# task in one job, so the archive carries the NCU signature the distillation anchors to.
# (A script committed to the tree: a ``cmd:`` mode is exec'd unquoted by the job entry,
# so `a && b` would end at `a`.)
MEASURE_SCRIPT = "goalref_measure.sh"
MEASURE_SCRIPT_TEXT = ("#!/bin/bash\n# goal-ref: time the reference (perf) and take its NCU "
                       "signature (ncu) in one job\nbash gpu_entry.sh perf\nrc=$?\n"
                       "bash gpu_entry.sh ncu\nexit $rc\n")
REF_TASK = f"cmd:bash {MEASURE_SCRIPT}"
EXEC_SCRIPT = "probes/.fk_exec.sh"

RESULT_LINE = "FK_GOALREF_RESULT"
DONE_LINE = "FK_GOALREF_DONE"
DEADLINE_LINE = "FK_GOALREF_DEADLINE_PASSED"
PENDING_LINE = "FK_GOALREF_PENDING"
# The session's deliverable check looks for these (scripts/claude-goal.py OK_LINE).
OK_LINE = "TURN-DELIVERABLE-OK"
MISSING_LINE = "TURN-DELIVERABLE-MISSING"
CONDITION_MAX = 4000

# Caps on what a request may carry: every job copies the tree, and a probe directory has
# no business being big.
KERNEL_MAX_BYTES = 1 << 20
PROBES_MAX_BYTES = 256 << 10
LOG_TAIL_CHARS = 12000


class GoalRefError(RuntimeError):
    """Reference generation cannot start, or ended without a usable reference."""


# -- configuration ------------------------------------------------------------------------

def resolve_mode(cfg: dict) -> tuple[str, str]:
    """(mode, where it came from): [loop] ref_from_goal, else auto."""
    value = (cfg.get("loop") or {}).get("ref_from_goal")
    if value is None or value == "":
        return DEFAULT_MODE, "default"
    if isinstance(value, bool):          # `ref_from_goal = false` reads as off
        value = "auto" if value else "off"
    value = str(value).strip().lower()
    if value not in MODES:
        raise GoalRefError(f"[loop] ref_from_goal must be one of {list(MODES)}, got {value!r}")
    return value, "[loop] ref_from_goal"


@dataclass(frozen=True)
class Settings:
    hours: float = DEFAULT_HOURS


def settings(cfg: dict) -> Settings:
    lc = cfg.get("loop") or {}
    try:
        hours = float(lc.get("ref_from_goal_hours", DEFAULT_HOURS))
    except (TypeError, ValueError) as e:
        raise GoalRefError(f"[loop] ref_from_goal_hours: {e}") from None
    if hours <= 0:
        raise GoalRefError("[loop] ref_from_goal_hours must be > 0")
    return Settings(hours=hours)


# -- paths & state ------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def of(cls, op_dir: Path) -> "Paths":
        return cls(Path(op_dir) / DIRNAME)

    state = property(lambda s: s.root / STATE)
    ledger = property(lambda s: s.root / LEDGER)
    candidates = property(lambda s: s.root / "candidates")
    acceptance = property(lambda s: s.root / "acceptance")
    reference = property(lambda s: s.root / "reference")
    session = property(lambda s: s.root / "session")
    ws = property(lambda s: s.root / "ws")
    spool = property(lambda s: s.root / "ws" / SPOOL)
    turn = property(lambda s: s.root / "ws" / ".fk-turn" / "goalref")


def tree_op(op: str) -> str:
    """The submission tree's operator name (operators/_<op>goalref)."""
    return f"_{op}goalref"


def teacher_only_paths(root: Path, op: str) -> list[Path]:
    """What the student must never read: the generated reference and its whole history,
    and the submission tree its candidates were committed to."""
    return [operators_dir(root) / op / DIRNAME, operators_dir(root) / tree_op(op)]


def session_read_only_paths(op_dir: Path) -> list[Path]:
    """What the generating session may read but never write (its policy's deny_write):
    the records the harness judges it by — state (deadline, bar), the ledger the check
    reads, the candidates the acceptance re-measures — and the session's spec.json (the
    check command and the condition), which sits inside its own working directory."""
    p = Paths.of(Path(op_dir).resolve())
    return [p.state, p.ledger, p.candidates, p.acceptance, p.reference, p.session,
            p.turn / "spec.json"]


def load_state(op_dir: Path) -> dict | None:
    try:
        d = json.loads(Paths.of(op_dir).state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def save_state(op_dir: Path, state: dict) -> None:
    p = Paths.of(op_dir).state
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def load_ledger(path: Path) -> list[dict]:
    out = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for ln in lines:
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -- the decision -------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    action: str          # "off" | "generate" | "resume" | "reuse"
    why: str
    mode: str = DEFAULT_MODE
    state: dict | None = None

    @property
    def runs_session(self) -> bool:
        return self.action in ("generate", "resume")

    @property
    def gives_oracle(self) -> bool:
        return self.action in ("generate", "resume", "reuse")


def plan(op_dir: Path, *, mode: str, oracle_mode: str, oracle_paths: list[str],
         milestone_valid: bool) -> Plan:
    """What this launch does about a missing reference. Raises GoalRefError when a
    previous generation in this operator directory failed."""
    if mode == "off":
        return Plan("off", "ref_from_goal = off", mode)
    state = load_state(op_dir)
    if state is not None:
        status = state.get("status")
        if status == STATUS_ACCEPTED and _reference_files(op_dir, state):
            return Plan("reuse", f"reusing the reference generated at "
                                 f"{_stamp(state.get('ended'))} (candidate "
                                 f"{(state.get('accepted') or {}).get('candidate')})", mode, state)
        if status == STATUS_GENERATING:
            return Plan("resume", "resuming the interrupted reference generation "
                                  f"(deadline {_stamp(state.get('deadline'))})", mode, state)
        if status == STATUS_FAILED:
            raise GoalRefError(
                f"reference generation already failed in this operator directory "
                f"({state.get('failure') or 'no reason recorded'}) — remove "
                f"{Paths.of(op_dir).root} to try again, or set ref_from_goal = \"off\"")
        raise GoalRefError(f"{Paths.of(op_dir).state}: unreadable status {status!r} "
                           f"(or its reference files are missing) — remove "
                           f"{Paths.of(op_dir).root} to start over")
    if oracle_mode != "source":
        return Plan("off", f"oracle_mode = {oracle_mode} (not a missing reference)", mode)
    if oracle_paths:
        return Plan("off", "an oracle is declared", mode)
    if milestone_valid:
        return Plan("off", "a valid milestone is on disk (the run's input)", mode)
    return Plan("generate", "source run with no oracle and no milestone", mode)


def _reference_files(op_dir: Path, state: dict) -> list[str]:
    files = list((state.get("accepted") or {}).get("reference") or [])
    return files if files and all(Path(f).is_file() for f in files) else []


def _stamp(t) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(t)))
    except (TypeError, ValueError):
        return "?"


# -- the speed bar ------------------------------------------------------------------------

def _problem(op_dir: Path) -> dict | None:
    try:
        d = json.loads((Path(op_dir) / "eval" / "leaderboard" / "problem.json")
                       .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def in_job_reference(cfg: dict, op_dir: Path) -> bool:
    """Whether the card's bench times a reference beside the kernel in the same job."""
    prob = _problem(op_dir)
    if prob is not None:
        return bool(prob.get("time_reference", True))
    lc = cfg.get("loop") or {}
    return bool(lc.get("ref_cmd")) and not lc.get("ref_mode") and not lc.get("ref_submit_op")


def resolve_bar(cfg: dict, op_dir: Path, *, target_ms: float | None, ratio: float,
                gate_window: str) -> dict:
    """The speed a candidate must reach for the session to end early (see module doc)."""
    lc = cfg.get("loop") or {}
    win = "device" if gate_window == "device" else "e2e"
    if target_ms:
        return {"kind": "target_ms", "ms": float(target_ms), "window": win,
                "text": f"{win} reading <= {float(target_ms):.6g} ms (--target-ms)"}
    declared = lc.get("arch_ref_device_ms") if win == "device" else None
    if declared is None:
        declared = lc.get("arch_ref_ms")
    if declared is not None:
        ms = float(declared) / ratio
        name = lc.get("arch_ref_name") or "the declared architecture reference"
        return {"kind": "arch_ref", "ms": ms, "window": win,
                "text": f"{win} reading <= {ms:.6g} ms ({name} {float(declared):.6g} ms"
                        f"{f' / {ratio:g}' if ratio != 1 else ''})"}
    if in_job_reference(cfg, op_dir):
        return {"kind": "in_job_ref", "ratio": ratio, "window": win,
                "text": f"same-job ratio vs the card's in-job reference "
                        f"(vs_ref{'_device' if win == 'device' else ''}) >= {ratio:g} "
                        f"on every evaluation that counts"}
    try:
        from forgekernel.baseline import ROLE_LIBRARY, load_baseline
        lib = load_baseline(Path(op_dir), ROLE_LIBRARY)
    except Exception:  # noqa: BLE001 — no archive is the common case
        lib = None
    lib_ms = (getattr(lib, "ms_device", None) if win == "device" else None) or \
        (getattr(lib, "ms", None) if lib is not None else None)
    if lib_ms:
        ms = float(lib_ms) / ratio
        return {"kind": "library_archive", "ms": ms, "window": win,
                "text": f"{win} reading <= {ms:.6g} ms (archived library baseline "
                        f"{float(lib_ms):.6g} ms{f' / {ratio:g}' if ratio != 1 else ''})"}
    return {"kind": "none", "window": win,
            "text": "no baseline is known for this card, so there is no speed bar and no "
                    "early finish: keep making the kernel faster until the deadline (the "
                    "fastest correct candidate becomes the reference, and its reading the "
                    "student's target)"}


def bar_met(bar: dict, reading: float | None, vs_ref: float | None) -> bool:
    """Whether a reading ends the session early. With no bar (kind "none") nothing does:
    the first correct kernel is usually the naive one, and the accepted reference anchors
    the student's target (``arch_decision``) — the session spends its whole budget."""
    kind = bar.get("kind")
    if kind in ("target_ms", "arch_ref", "library_archive"):
        return reading is not None and reading <= float(bar["ms"])
    if kind == "in_job_ref":
        return vs_ref is not None and vs_ref >= float(bar.get("ratio", 1.0))
    return False


# -- what a candidate may not be ----------------------------------------------------------

_PY_LIBS = ("flashinfer", "sgl_kernel", "vllm", "deep_gemm", "flash_attn", "flash_mla",
            "xformers", "transformer_engine", "apex", "liger_kernel", "torchao",
            "fbgemm_gpu", "tensorrt_llm", "bitsandbytes", "cudnn", "nvmath", "cublas")
_LIB_PATTERNS = (
    (re.compile(r"^\s*(?:from|import)\s+(" + "|".join(_PY_LIBS) + r")\b", re.M),
     "imports the kernel library {0}"),
    (re.compile(r"\btorch\s*\.\s*(matmul|mm|bmm|addmm|baddbmm|einsum|_scaled_mm|softmax|"
                r"layer_norm|rms_norm|group_norm|conv\w*|cdist)\s*\("),
     "computes with torch.{0}"),
    (re.compile(r"\b(?:F|torch\s*\.\s*nn\s*\.\s*functional)\s*\.\s*(linear|scaled_dot_product_attention|"
                r"softmax|log_softmax|layer_norm|rms_norm|group_norm|conv\w*|"
                r"embedding|silu|gelu)\s*\("),
     "computes with torch.nn.functional.{0}"),
    (re.compile(r"\btorch\s*\.\s*ops\s*\.\s*(\w+)"), "calls the registered op torch.ops.{0}"),
    (re.compile(r"[\w\)\]][ \t]*@[ \t]*[\w\(]"), "uses the @ (matmul) operator"),
    (re.compile(r"\b(cublas\w*|cudnn\w*|cublasLt\w*)\s*\("), "calls {0}"),
    (re.compile(r"#\s*include\s*[<\"](cublas\w*|cudnn\w*)"), "includes {0}"),
)


def _strip_comments(src: str, kernel_file: str) -> str:
    """The code without comments and (Python) string literals: the stub's contract text
    names torch functions on purpose, and so may a comment."""
    if str(kernel_file).endswith((".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp")):
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        return re.sub(r"//[^\n]*", "", src)
    import io
    import tokenize
    lines: list[str] = []
    cur: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.NEWLINE, tokenize.NL):
                lines.append(" ".join(cur))
                cur = []
            elif tok.type not in (tokenize.COMMENT, tokenize.STRING, tokenize.INDENT,
                                  tokenize.DEDENT, tokenize.ENDMARKER):
                cur.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        body = re.sub(r'("""|\'\'\').*?\1', "", src, flags=re.S)
        return re.sub(r"#[^\n]*", "", body)
    lines.append(" ".join(cur))
    return "\n".join(lines)


def library_findings(src: str, kernel_file: str) -> list[str]:
    """Why this source is not a kernel of its own (a call into an existing kernel
    library does the computation). A heuristic over the source text: it names the common
    spellings, it cannot prove their absence."""
    body = _strip_comments(src, kernel_file)
    found = []
    for rx, what in _LIB_PATTERNS:
        for m in rx.finditer(body):
            text = what.format(*(m.groups() or ("",)))
            if text not in found:
                found.append(text)
    return found


# -- reading one evaluation ---------------------------------------------------------------

def gate_reading(ev, window: str) -> float | None:
    if window == "device":
        return ev.device_median_ms if ev.device_median_ms is not None else ev.bwd_median_ms
    return ev.bwd_median_ms if ev.bwd_median_ms is not None else ev.device_median_ms


def gate_vs_ref(ev, window: str) -> float | None:
    return ev.vs_ref_device if window == "device" else ev.vs_ref


def precision_failure(cfg: dict, op_dir: Path, ev) -> str | None:
    """The card's precision contract applied to this eval, once the precision gate
    (loop/precision_gate.py) is part of the tree; None before that or when it passes."""
    try:
        from forgekernel.loop import precision_gate  # type: ignore[attr-defined]
    except ImportError:
        return None
    try:
        contract, _ = precision_gate.resolve_contract(cfg, Path(op_dir))
        summary = precision_gate.summarize(contract, getattr(ev, "precision", None))
        return precision_gate.gate_failure(contract, summary)
    except Exception as e:  # noqa: BLE001 — an unreadable block must not pass silently
        return f"precision gate could not be applied: {e}"


# -- the submission tree and the workspace ------------------------------------------------

def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise GoalRefError(f"git {' '.join(args)} in {cwd}: {r.stderr.strip()[-400:]}")
    return r


def _exclude(repo: Path, pattern: str) -> None:
    from forgekernel.loop.agent_policy import _git_exclude
    _git_exclude(Path(repo), pattern)


def _clone(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["git", "clone", "-q", str(src), str(dst)], capture_output=True, text=True)
    if r.returncode != 0:
        raise GoalRefError(f"git clone {src} -> {dst}: {r.stderr.strip()[-400:]}")
    for k, v in (("user.name", "forge-goalref"), ("user.email", "forge-goalref@local")):
        _git(dst, "config", k, v)


def ensure_tree(root: Path, op: str, op_dir: Path, *, fresh: bool) -> Path:
    """operators/_<op>goalref: a clone of the card the candidates are committed to and
    submitted from. ``fresh`` (a new generation) re-clones it; a directory of that name
    this module did not create is never touched."""
    tree = operators_dir(root) / tree_op(op)
    if tree.exists():
        if not (tree / TREE_MARKER).is_file():
            raise GoalRefError(f"{tree} exists but is not a goal-ref submission tree "
                               f"(no {TREE_MARKER}) — move it away")
        if not fresh:
            return tree
        shutil.rmtree(tree)
    _clone(Path(op_dir), tree)
    local = Path(op_dir) / "submit.local.toml"
    if local.is_file():                      # gitignored: does not travel with the clone
        shutil.copy2(local, tree / "submit.local.toml")
    (tree / TREE_MARKER).write_text(f"goal-ref submission tree of {op}\n", encoding="utf-8")
    _exclude(tree, TREE_MARKER)
    return tree


def ensure_ws(paths: Paths, op_dir: Path) -> Path:
    ws = paths.ws
    if not (ws / ".git").exists():
        if ws.exists():
            shutil.rmtree(ws)
        _clone(Path(op_dir), ws)
    for pat in (SPOOL + "/", ".fk-turn/", ".claude/"):
        _exclude(ws, pat)
    for d in ("requests", "responses", "logs"):
        (paths.spool / d).mkdir(parents=True, exist_ok=True)
    return ws


# -- the evaluation service (runs in the loop process) ------------------------------------

@dataclass
class Service:
    """Serves the session's evaluation requests: one GPU job at a time, in order."""

    op: str
    op_dir: Path
    cfg: dict
    paths: Paths
    tree: Path
    kernel_file: str
    bar: dict
    deadline: float
    backend: object                     # RealBackend-like: submit_gpu(branch, mode) -> log
    now: Callable[[], float] = time.time
    log: Callable[[str], None] = lambda s: print(f"[goal-ref] {s}", flush=True)
    base_branch: str = ""

    def __post_init__(self):
        if not self.base_branch:
            self.base_branch = _base_branch(self.tree)

    # -- the queue --------------------------------------------------------------------

    def pending(self) -> list[Path]:
        req = self.paths.spool / "requests"
        done = self.paths.spool / "responses"
        try:
            items = sorted(p for p in req.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return []
        return [p for p in items if not (done / f"{p.name}.txt").exists()]

    def serve_pending(self) -> int:
        n = 0
        for req in self.pending():
            try:
                text = self.serve(req)
            except Exception as e:  # noqa: BLE001 — one bad request must not end the session
                text = f"the harness could not serve this request: {e.__class__.__name__}: {e}\n"
                self.log(f"request {req.name}: {text.strip()}")
            self._respond(req.name, text)
            n += 1
        return n

    def _respond(self, rid: str, text: str) -> None:
        out = self.paths.spool / "responses" / f"{rid}.txt"
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, out)

    def serve(self, req: Path) -> str:
        try:
            meta = json.loads((req / "request.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return f"malformed request ({e})\n"
        if self.now() >= self.deadline:
            return (f"The deadline ({_stamp(self.deadline)}) has passed: no more GPU jobs.\n"
                    f"{self.summary_line()}\n{DEADLINE_LINE}\n")
        kind = meta.get("kind")
        if kind == "eval":
            return self.evaluate(req)
        if kind == "exec":
            return self.execute(req, str(meta.get("cmd") or ""))
        return f"unknown request kind {kind!r}\n"

    # -- committing a candidate --------------------------------------------------------

    def _commit(self, branch: str, kernel: Path | None, probes: Path | None,
                message: str, script: str | None = None) -> None:
        _git(self.tree, "checkout", "-q", "-f", self.base_branch)
        _git(self.tree, "checkout", "-q", "-B", branch)
        if kernel is not None:
            dst = self.tree / self.kernel_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(kernel, dst)
            _git(self.tree, "add", "--", self.kernel_file)
        if probes is not None and probes.is_dir():
            pdst = self.tree / "probes"
            if pdst.exists():
                shutil.rmtree(pdst)
            shutil.copytree(probes, pdst)
            _git(self.tree, "add", "-f", "--", "probes")
        if script is not None:
            sp = self.tree / EXEC_SCRIPT
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text("#!/bin/bash\n" + script + "\n", encoding="utf-8")
            _git(self.tree, "add", "-f", "--", EXEC_SCRIPT)
        _git(self.tree, "commit", "-q", "--allow-empty", "-m", message)

    def _submit(self, branch: str, mode: str) -> tuple[str, str, str]:
        """(log, job, error) — never raises."""
        try:
            log = self.backend.submit_gpu(branch, mode) or ""
            err = ""
        except Exception as e:  # noqa: BLE001 — a node fault is the agent's to retry
            log, err = "", f"{e.__class__.__name__}: {e}"
        finally:
            try:
                _git(self.tree, "checkout", "-q", "-f", self.base_branch)
            except GoalRefError:
                pass
        return log, str(getattr(self.backend, "last_job_id", "") or ""), err

    # -- eval ----------------------------------------------------------------------------

    def next_number(self) -> int:
        return 1 + sum(1 for r in load_ledger(self.paths.ledger) if r.get("kind") == "eval")

    def evaluate(self, req: Path) -> str:
        snap = req / Path(self.kernel_file).name
        if not snap.is_file():
            return f"the request carries no {self.kernel_file}\n"
        if snap.stat().st_size > KERNEL_MAX_BYTES:
            return f"{self.kernel_file} is larger than {KERNEL_MAX_BYTES} bytes\n"
        n = self.next_number()
        cdir = self.paths.candidates / f"{n:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        kernel = cdir / Path(self.kernel_file).name
        shutil.copy2(snap, kernel)
        src = kernel.read_text(encoding="utf-8", errors="replace")
        import hashlib
        sha = hashlib.sha256(kernel.read_bytes()).hexdigest()
        prior = next((r for r in load_ledger(self.paths.ledger)
                      if r.get("kind") == "eval" and r.get("sha256") == sha), None)
        branch = f"cand-{n:03d}"
        self._commit(branch, kernel, None, f"goal-ref candidate {n:03d}")
        self.log(f"candidate {n:03d} ({sha[:10]}): submitting `all`")
        t0 = self.now()
        log, job, err = self._submit(branch, "all")
        rec = self.record(log, job, err, kind="eval", n=n, sha=sha, kernel=kernel,
                          src=src, started=t0)
        (cdir / "job.log").write_text(log or err, encoding="utf-8")
        (self.paths.spool / "logs" / f"cand-{n:03d}.log").write_text(log or err, encoding="utf-8")
        (cdir / "result.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        _append(self.paths.ledger, rec)
        self.log(f"candidate {n:03d}: {_brief(rec)}")
        return self.response(rec, log, repeat_of=prior.get("n") if prior else None)

    def record(self, log: str, job: str, err: str, *, kind: str, n: int, sha: str,
               kernel: Path, src: str, started: float) -> dict:
        from forgekernel.loop.parse import parse_eval
        ev = parse_eval(log) if log else None
        win = self.bar.get("window", "device")
        rec = {"kind": kind, "n": n, "at": self.now(), "wall_s": round(self.now() - started, 1),
               "sha256": sha, "kernel": str(kernel), "job": job,
               "cluster": str(getattr(self.backend, "last_cluster", "") or "")}
        if ev is None:
            rec.update(correct=False, eligible=False, bar_met=False,
                       error=err or "the job returned no log")
            return rec
        prec = precision_failure(self.cfg, self.op_dir, ev)
        libs = library_findings(src, self.kernel_file)
        correct = bool(ev.correctness) and not prec
        reading = gate_reading(ev, win)
        vs = gate_vs_ref(ev, win)
        eligible = correct and not libs and reading is not None
        rec.update(
            correct=correct, error=(ev.error or prec or "") if not correct else "",
            precision_failure=prec or "", library_findings=libs, eligible=eligible,
            reading_ms=reading, device_ms=ev.device_median_ms, e2e_ms=ev.bwd_median_ms,
            ref_device_ms=ev.ref_device_ms, ref_ms=ev.ref_median_ms,
            vs_ref_device=ev.vs_ref_device, vs_ref=ev.vs_ref, vs_ref_gate=vs,
            basis=ev.device_window_basis or "", n_workloads=ev.n_workloads,
            workloads=ev.workloads,
            bar_met=bool(eligible and bar_met(self.bar, reading, vs)))
        return rec

    def best(self) -> dict | None:
        rows = [r for r in load_ledger(self.paths.ledger)
                if r.get("kind") == "eval" and r.get("eligible")]
        return min(rows, key=lambda r: r["reading_ms"]) if rows else None

    def summary_line(self) -> str:
        b = self.best()
        done = any(r.get("bar_met") for r in load_ledger(self.paths.ledger)
                   if r.get("kind") == "eval")
        return (f"best eligible: "
                + (f"candidate {b['n']:03d} {b['reading_ms']:.6g} ms" if b else "none yet")
                + f"; speed bar {'met' if done else 'not met'}")

    def response(self, rec: dict, log: str, repeat_of: int | None = None) -> str:
        win = self.bar.get("window", "device")
        lines = [f"candidate {rec['n']:03d} (sha256 {rec['sha256'][:12]}) — job "
                 f"{rec.get('job') or '?'}{(' on ' + rec['cluster']) if rec.get('cluster') else ''}"
                 f", {rec.get('wall_s')} s"]
        if repeat_of:
            lines.append(f"(byte-identical to candidate {repeat_of:03d}: this is a repeat "
                         f"measurement of the same kernel)")
        if rec.get("error") and not rec.get("correct"):
            lines.append("correctness: FAIL — " + str(rec["error"])[:3000])
        else:
            lines.append("correctness: PASS")
        if rec.get("reading_ms") is not None:
            lines.append(f"{win} reading (the gate window"
                         + (f", geomean over {rec['n_workloads']} workloads" if
                            (rec.get("n_workloads") or 1) > 1 else "")
                         + f"): {rec['reading_ms']:.6g} ms"
                         + (f"  [basis {rec['basis']}]" if rec.get("basis") else ""))
            if rec.get("e2e_ms") is not None and win == "device":
                lines.append(f"e2e (per call + sync) reading: {rec['e2e_ms']:.6g} ms")
        from forgekernel.loop.parse import describe_workload_readings
        if (rec.get("n_workloads") or 1) > 1 and rec.get("workloads"):
            lines.append("per workload: " + describe_workload_readings(rec["workloads"]))
        if rec.get("vs_ref_gate") is not None:
            ref = rec.get("ref_device_ms") if win == "device" else rec.get("ref_ms")
            lines.append(f"in-job reference: {ref} ms; same-job ratio "
                         f"{rec['vs_ref_gate']:.4f}x (>1 = your kernel is faster)")
        for f in rec.get("library_findings") or []:
            lines.append(f"NOT ELIGIBLE: the source {f} — the reference must compute "
                         f"with its own kernel")
        if rec.get("precision_failure"):
            lines.append("precision gate: " + rec["precision_failure"])
        lines.append(f"speed bar: {self.bar.get('text')} -> "
                     f"{'MET' if rec.get('bar_met') else 'not met'}")
        lines.append(self.summary_line())
        left = self.deadline - self.now()
        lines.append(f"deadline: {_stamp(self.deadline)} ({max(0, left) / 60:.0f} min left)")
        if not rec.get("correct") and log:
            tail = log[-LOG_TAIL_CHARS // 2:]
            lines.append(f"job log tail (full log: {self.paths.spool / 'logs' / ('cand-%03d.log' % rec['n'])}):")
            lines.append(tail)
        lines.append(f"{RESULT_LINE} cand={rec['n']} correct={int(bool(rec.get('correct')))} "
                     f"eligible={int(bool(rec.get('eligible')))} "
                     f"bar_met={int(bool(rec.get('bar_met')))} reading_ms="
                     f"{rec.get('reading_ms') if rec.get('reading_ms') is not None else 'na'} "
                     f"deadline_passed={int(left <= 0)}")
        if rec.get("bar_met"):
            lines.append(DONE_LINE)
        if left <= 0:
            lines.append(DEADLINE_LINE)
        return "\n".join(lines) + "\n"

    # -- exec ----------------------------------------------------------------------------

    def execute(self, req: Path, cmd: str) -> str:
        if not cmd.strip():
            return "exec: empty command\n"
        snap = req / Path(self.kernel_file).name
        probes = req / "probes"
        if probes.is_dir():
            size = sum(p.stat().st_size for p in probes.rglob("*") if p.is_file())
            if size > PROBES_MAX_BYTES:
                return f"exec: probes/ is {size} bytes (limit {PROBES_MAX_BYTES})\n"
        n = sum(1 for r in load_ledger(self.paths.ledger) if r.get("kind") == "exec") + 1
        branch = f"exec-{n:03d}"
        self._commit(branch, snap if snap.is_file() else None,
                     probes if probes.is_dir() else None, f"goal-ref exec {n:03d}",
                     script=cmd)
        self.log(f"exec {n:03d}: {cmd[:120]}")
        t0 = self.now()
        # the job entry runs a cmd: mode unquoted; the command itself rides in a script
        log, job, err = self._submit(branch, f"cmd:bash {EXEC_SCRIPT}")
        path = self.paths.spool / "logs" / f"exec-{n:03d}.log"
        path.write_text(log or err, encoding="utf-8")
        _append(self.paths.ledger, {"kind": "exec", "n": n, "at": self.now(), "job": job,
                                    "cmd": cmd[:2000], "error": err,
                                    "wall_s": round(self.now() - t0, 1)})
        body = (log or err)[-LOG_TAIL_CHARS:]
        return (f"exec {n:03d} — job {job or '?'} ({round(self.now() - t0)} s); full log: "
                f"{path}\n{body}\n(exec does not judge or record a candidate)\n")


def _brief(rec: dict) -> str:
    if not rec.get("correct"):
        return "FAIL " + str(rec.get("error") or "")[:160]
    s = f"PASS {rec.get('reading_ms')} ms"
    if rec.get("vs_ref_gate") is not None:
        s += f" x{rec['vs_ref_gate']:.4f} vs in-job ref"
    if rec.get("library_findings"):
        s += " (not eligible: " + "; ".join(rec["library_findings"]) + ")"
    return s + (" BAR MET" if rec.get("bar_met") else "")


# -- the agent-side commands --------------------------------------------------------------

def command(*args: str, python: str | None = None) -> str:
    """A shell command running this module with THIS checkout's package."""
    return ("PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=" + shlex.quote(str(repo_root() / "src")) + " "
            + " ".join(shlex.quote(x) for x in (python or sys.executable, "-W",
                                                 "ignore::RuntimeWarning", "-m",
                                                 "forgekernel.loop.goal_ref", *args)))


def _new_request(ws: Path, kind: str, kernel_file: str, extra: dict) -> tuple[str, Path]:
    spool = ws / SPOOL
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    tmp = spool / "requests" / f".tmp-{rid}"
    tmp.mkdir(parents=True)
    src = ws / kernel_file
    if src.is_file():
        shutil.copy2(src, tmp / Path(kernel_file).name)
    if kind == "exec" and (ws / "probes").is_dir():
        shutil.copytree(ws / "probes", tmp / "probes")
    (tmp / "request.json").write_text(json.dumps({"kind": kind, **extra, "at": time.time()}),
                                      encoding="utf-8")
    final = spool / "requests" / rid
    os.replace(tmp, final)
    return rid, final


def _wait(ws: Path, rid: str, wait_s: float, poll_s: float | None = None) -> int:
    poll_s = float(os.environ.get("FK_GOALREF_POLL_S", 2.0)) if poll_s is None else poll_s
    resp = ws / SPOOL / "responses" / f"{rid}.txt"
    end = time.time() + wait_s
    while time.time() < end:
        if resp.is_file():
            sys.stdout.write(resp.read_text(encoding="utf-8", errors="replace"))
            return 0
        time.sleep(poll_s)
    print(f"{PENDING_LINE} id={rid}: no answer within {wait_s:.0f} s (queue wait or a long "
          f"job). Wait again with:\n  {command('wait', str(ws), rid)}")
    return 3


def _context(ws: Path) -> dict:
    try:
        return json.loads((ws / SPOOL / "context.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def check(goal_dir: Path, now: float | None = None) -> tuple[bool, list[str]]:
    """The deliverable check the /goal condition names (and the session re-runs)."""
    paths = Paths(Path(goal_dir))
    state = load_state(paths.root.parent) or {}
    now = time.time() if now is None else now
    problems = []
    reply = Path(state.get("reply") or paths.turn / "reply.md")
    try:
        if not reply.read_text(encoding="utf-8", errors="replace").strip():
            problems.append(f"{reply} is empty — write your report there")
    except OSError:
        problems.append(f"{reply} does not exist — write your report there")
    rows = [r for r in load_ledger(paths.ledger) if r.get("kind") == "eval"]
    done = any(r.get("bar_met") for r in rows)
    deadline = float(state.get("deadline") or 0)
    if not done and now < deadline:
        el = [r for r in rows if r.get("eligible")]
        best = min(el, key=lambda r: r["reading_ms"]) if el else None
        problems.append(
            "no evaluation has met the speed bar yet ("
            + (f"best eligible so far: candidate {best['n']:03d} at {best['reading_ms']:.6g} ms"
               if best else f"{len(rows)} evaluation(s), none eligible")
            + f"); keep improving the kernel and evaluate it — the deadline is "
              f"{_stamp(deadline)}")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = ("usage: python -m forgekernel.loop.goal_ref eval <ws> [--wait S] | "
             "exec <ws> -- <command> | wait <ws> <id> | status <goal_ref dir> | "
             "check <goal_ref dir>")
    if not argv:
        sys.stderr.write(usage + "\n")
        return 2
    cmd, rest = argv[0], argv[1:]
    wait_s = float(os.environ.get("FK_GOALREF_WAIT_S", 1080))
    if "--wait" in rest:
        i = rest.index("--wait")
        wait_s = float(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]
    if cmd == "eval" and len(rest) == 1:
        ws = Path(rest[0])
        ctx = _context(ws)
        kf = ctx.get("kernel_file") or "kernel.py"
        if not (ws / kf).is_file():
            print(f"{ws / kf} does not exist")
            return 2
        rid, _ = _new_request(ws, "eval", kf, {})
        print(f"submitted evaluation request {rid}; waiting for the GPU verdict ...", flush=True)
        return _wait(ws, rid, wait_s)
    if cmd == "exec" and len(rest) >= 2 and rest[1] == "--":
        ws = Path(rest[0])
        shell = " ".join(rest[2:])
        ctx = _context(ws)
        rid, _ = _new_request(ws, "exec", ctx.get("kernel_file") or "kernel.py", {"cmd": shell})
        print(f"submitted exec request {rid}; waiting ...", flush=True)
        return _wait(ws, rid, wait_s)
    if cmd == "wait" and len(rest) == 2:
        return _wait(Path(rest[0]), rest[1], wait_s)
    if cmd == "status" and len(rest) == 1:
        paths = Paths(Path(rest[0]))
        state = load_state(paths.root.parent) or {}
        rows = load_ledger(paths.ledger)
        print(f"deadline {_stamp(state.get('deadline'))}; speed bar: "
              f"{(state.get('bar') or {}).get('text', '?')}")
        for r in rows:
            if r.get("kind") == "eval":
                print(f"  candidate {r['n']:03d} {r.get('sha256', '')[:10]}: {_brief(r)}")
            elif r.get("kind") == "exec":
                print(f"  exec {r['n']:03d}: {r.get('cmd', '')[:80]}")
        return 0
    if cmd == "check" and len(rest) == 1:
        ok, problems = check(Path(rest[0]))
        if ok:
            print(f"{OK_LINE} (the reference session's deliverable is complete)")
            return 0
        print(f"{MISSING_LINE}: " + " | ".join(problems))
        return 1
    sys.stderr.write(usage + "\n")
    return 2


# -- the task and the condition -----------------------------------------------------------

def _contract_files(op_dir: Path, kernel_file: str) -> list[str]:
    out = [kernel_file]
    for rel in ("eval/leaderboard/problem.json", "README.md", "eval/bench.py"):
        if (Path(op_dir) / rel).is_file():
            out.append(rel)
    return out


def task_text(*, op: str, op_dir: Path, ws: Path, kernel_file: str, language: str,
              bar: dict, deadline: float, hours: float, reply: Path,
              read_dirs: list[str], kb_dirs: list[str], report_s: float = 0.0) -> str:
    from forgekernel.loop import lang_mode
    cuda = lang_mode.is_cuda(language)
    lang_line = ("CUDA C++ (the conventions of `" + kernel_file + "` are in the stub in that file)"
                 if cuda else
                 f"preferably {language} (this run's student writes in it, and a reference in "
                 "the same language is easiest for the teacher to distill); other forms the "
                 "evaluator accepts (Triton, inline PTX, ...) are fine too")
    ev = command("eval", str(ws))
    ex = command("exec", str(ws), "--", "<shell command>")
    st = command("status", str(ws.parent))
    ck = command("check", str(ws.parent))
    files = ", ".join(f"`{f}`" for f in _contract_files(op_dir, kernel_file))
    reads = ""
    if read_dirs or kb_dirs:
        reads = ("\n- Material readable on this machine: "
                 + ", ".join(f"`{d}`" for d in [*kb_dirs, *read_dirs])
                 + " (the KB is GPU-certified hardware knowledge).")
    return f"""# Task: write a reference kernel for the operator `{op}` from scratch

This run has no reference implementation to distill from. The kernel you write becomes this
run's **reference (oracle)**: the teacher reads its source to distill the milestone, and its
measured speed may also become the student's target. So it must be **correct**, and it must be
**fast**.

## Deliverable
- The current directory `{ws}` is a copy of this operator tree. Write the implementation into
  `{kernel_file}`; the entry point, signature, inputs/outputs and the DPS convention are in
  {files} (the reference in the definition is the standard of correctness).
- Language: {lang_line}.
- **The main computation must be done by a kernel you write yourself**: do not call existing
  operator libraries (flashinfer, cuBLAS / cuDNN, torch's matmul / attention / normalization /
  softmax and other compute functions and `@`, torch.ops, vllm, sgl_kernel, deep_gemm,
  flash_attn ...); use torch only for allocation, views and dtype conversion. The evaluator scans
  the source, and a candidate that hits one is not counted (marked NOT ELIGIBLE).
- The evaluator takes only `{kernel_file}`: `eval/`, `gpu_entry.sh` and the other evaluator files
  are always the operator tree's own; changing them has no effect.

## How to measure (the only GPU channel; this machine has no GPU)
- `{ev}`
  Hands the current `{kernel_file}` to this operator tree's official evaluator (correctness +
  timing, exactly what the student is judged by), prints correctness, the reading, whether the
  speed bar is met and the current best, and a last line `{RESULT_LINE} ...`; when the bar is
  met there is an extra line `{DONE_LINE}`, and past the deadline `{DEADLINE_LINE}`. One run
  takes a few minutes (queueing not included). The command waits at most 18 minutes; if the
  result has not arrived it prints `{PENDING_LINE}` and the command to keep waiting. Use it
  serially, and give Bash a generous timeout (e.g. 1200000 ms).
- `{ex}`
  Runs any command (NCU, probe scripts) on a GPU node, in a copy of "this operator tree + your
  current `{kernel_file}` + the `./probes/` directory". It does not judge correctness and does
  not count as a candidate; put probes in `./probes/` (≤256 KiB).
- `{st}`: uses no GPU; lists each candidate's correctness, reading and whether it met the bar.

## Speed bar
{bar['text']}.

## Deadline
**{_stamp(deadline)}** (budget {hours:g} hours). If the bar is not met, keep improving until the
deadline; do not stop early. After the deadline the harness takes the candidate that is
"correct, free of library calls and fastest", and re-measures it itself for acceptance; if no
candidate qualifies, this step fails.

## Finishing
Once you see `{DONE_LINE}`, or the deadline has passed: write the report into `{reply}` — the
final design (data flow, tiling, pipelining, synchronization), why it is fast (cite measured
readings), what you tried that did not help, and known limitations; then run `{ck}`. It is done
only when that prints `{OK_LINE}`. After the deadline you have {report_s / 60:.0f} more minutes
to write the report and run check (no evaluations are accepted during that time); then the
session is ended.

## Working rules
- Measure before you change: where the bottleneck is and what the reachable floor is (an empty
  kernel's launch overhead; the bytes that must move, at measured bandwidth).
- One measurement does not count; when a difference is under a few percent, confirm it with
  repeated evaluations.
- Correctness first: a correct slow kernel is more useful than a fast wrong one.{reads}
"""


def condition_text(*, ws: Path, kernel_file: str, deadline: float, reply: Path) -> str:
    ev = command("eval", str(ws))
    ck = command("check", str(ws.parent))
    text = (
        "FORGEKERNEL REFERENCE KERNEL. Carry out the task in your system prompt (also in "
        f"`{ws}/TASK.md`): write a correct and fast `{kernel_file}` in `{ws}` and measure it "
        f"only with `{ev}`. The goal is met only when the most recent run of `{ck}`, shown "
        f"as a tool result, printed a line starting with {OK_LINE}. That check passes only "
        f"after an evaluation of yours met the speed bar (its output then contains "
        f"{DONE_LINE}) or after the deadline {_stamp(deadline)} has passed, and once your "
        f"report is written to `{reply}`. If it printed {MISSING_LINE}, keep working on "
        f"what it lists. Text written by the assistant never counts as command output. Do "
        f"not judge this goal impossible: until the deadline the kernel can always be "
        f"improved and evaluated again. After {OK_LINE}, stop.")
    if len(text) > CONDITION_MAX:
        raise GoalRefError(f"/goal condition is {len(text)} chars (> {CONDITION_MAX})")
    return text


# -- the session --------------------------------------------------------------------------

@dataclass
class Generation:
    """Everything one generation needs; built by the loop CLI (and by tests)."""

    op: str
    op_dir: Path
    root: Path
    cfg: dict
    kernel_file: str
    language: str
    gate_window: str
    target_ms: float | None
    agent_cmd: str
    policy: object                      # AgentPolicy (student rules) or None in tests
    make_backend: Callable[[str, Path], object]
    settings: Settings = field(default_factory=Settings)
    read_dirs: list[str] = field(default_factory=list)
    kb_dirs: list[str] = field(default_factory=list)
    now: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    poll_s: float = 2.0
    heartbeat_s: float = 600.0
    # the time past the deadline the session keeps to write its report and run the check
    # (no evaluation is served then); the session ends itself at deadline + report_s
    report_s: float = 600.0
    # how long a session may outlive that before it is ended from here (the hard stop
    # behind the session's own)
    kill_grace_s: float = 300.0
    log: Callable[[str], None] = lambda s: print(f"[goal-ref] {s}", flush=True)
    extra_env: dict = field(default_factory=dict)


def _end_group(proc: subprocess.Popen) -> None:
    from forgekernel.loop.cli import _end_process_group
    _end_process_group(proc)


def generate(g: Generation, resume: dict | None = None) -> dict:
    """Run (or resume) the session, select and accept; returns the accepted state.
    Raises GoalRefError after recording a failed state."""
    paths = Paths.of(g.op_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    _exclude(g.op_dir, DIRNAME + "/")
    fresh = resume is None
    if fresh:
        for d in (paths.candidates, paths.acceptance, paths.reference, paths.session, paths.ws):
            if d.exists():
                shutil.rmtree(d)
        try:
            paths.ledger.unlink()
        except OSError:
            pass
    tree = ensure_tree(g.root, g.op, g.op_dir, fresh=fresh)
    ws = ensure_ws(paths, g.op_dir)
    # everything session_read_only_paths names exists before the policy denies writing it
    for d in (paths.session, paths.candidates, paths.acceptance, paths.reference, paths.turn):
        d.mkdir(parents=True, exist_ok=True)
    paths.ledger.touch()
    reply = paths.turn / "reply.md"
    if fresh:
        started = g.now()
        state = {"version": 1, "status": STATUS_GENERATING, "op": g.op,
                 "kernel_file": g.kernel_file, "language": g.language,
                 "gate_window": g.gate_window, "started": started,
                 "deadline": started + g.settings.hours * 3600, "budget_h": g.settings.hours,
                 "bar": resolve_bar(g.cfg, g.op_dir, target_ms=g.target_ms,
                                    ratio=DEFAULT_RATIO, gate_window=g.gate_window),
                 "tree": str(tree), "ws": str(ws), "reply": str(reply),
                 "agent_cmd": g.agent_cmd, "attempts": []}
    else:
        state = dict(resume)
    save_state(g.op_dir, state)
    deadline = float(state["deadline"])
    bar = state["bar"]
    (ws / SPOOL / "context.json").write_text(json.dumps(
        {"kernel_file": g.kernel_file, "op": g.op}), encoding="utf-8")
    task = task_text(op=g.op, op_dir=g.op_dir, ws=ws, kernel_file=g.kernel_file,
                     language=g.language, bar=bar, deadline=deadline,
                     hours=float(state.get("budget_h") or g.settings.hours), reply=reply,
                     read_dirs=g.read_dirs, kb_dirs=g.kb_dirs, report_s=g.report_s)
    (ws / "TASK.md").write_text(task, encoding="utf-8")
    _exclude(ws, "TASK.md")
    (paths.session / "TASK.md").write_text(task, encoding="utf-8")
    spec = {"kind": "goal-ref", "role": "student", "started": state["started"],
            "since": state["started"], "artifacts": [], "allow_ok": False,
            "reply": str(reply), "check": command("check", str(paths.root)),
            "condition": condition_text(ws=ws, kernel_file=g.kernel_file,
                                        deadline=deadline, reply=reply)}
    (paths.turn / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    backend = g.make_backend(tree_op(g.op), tree)
    svc = Service(op=g.op, op_dir=g.op_dir, cfg=g.cfg, paths=paths, tree=tree,
                  kernel_file=g.kernel_file, bar=bar, deadline=deadline, backend=backend,
                  now=g.now, log=g.log)
    g.log(f"session: deadline {_stamp(deadline)}, speed bar: {bar['text']}; workspace {ws}; "
          f"submission tree {tree}")
    try:
        _run_session(g, state, svc, ws, task)
    finally:
        svc.serve_pending()          # answer what is left (after the deadline: refusals)
        state["ended"] = g.now()
        state["wall_s"] = round(state["ended"] - float(state["started"]), 1)
        _collect_transcripts(paths, state)
        save_state(g.op_dir, state)
    return accept(g, state, svc)


def _reap_orphan(g: Generation, state: dict) -> None:
    """A loop that died mid-session (SIGKILL, reboot) leaves its agent running in its own
    process group, still serving nobody. Before this loop starts another session in the
    same workspace, that group is ended — only after checking that its leader is still
    the agent command this generation started (a recycled PID is left alone)."""
    import signal
    for rec in state.get("attempts") or []:
        pgid = rec.get("pgid")
        if not pgid or "end" in rec:
            continue
        r = subprocess.run(["ps", "-ww", "-p", str(pgid), "-o", "command="],
                           capture_output=True, text=True)
        cmdline = r.stdout.strip()
        if r.returncode == 0 and cmdline and g.agent_cmd.strip() in cmdline:
            g.log(f"ending the agent session an interrupted run left behind (pgid {pgid})")
            try:
                os.killpg(int(pgid), signal.SIGTERM)
            except OSError:
                pass
            time.sleep(15)        # timeout-run.py forwards the TERM and waits up to 10 s
            try:
                os.killpg(int(pgid), signal.SIGKILL)
            except OSError:
                pass
        rec.update(end=g.now(), rc=None, outcome="orphaned (previous loop process died)")
    save_state(g.op_dir, state)


def _run_session(g: Generation, state: dict, svc: Service, ws: Path, task: str) -> None:
    paths = svc.paths
    deadline = float(state["deadline"])
    attempts = state.setdefault("attempts", [])
    _reap_orphan(g, state)
    while True:
        now = g.now()
        if now >= deadline:
            g.log("deadline reached")
            return
        if any(r.get("bar_met") for r in load_ledger(paths.ledger) if r.get("kind") == "eval") \
                and check(paths.root, now)[0]:
            g.log("the speed bar is met and the report is written")
            return
        if len(attempts) >= MAX_ATTEMPTS:
            g.log(f"{MAX_ATTEMPTS} session attempts spent — stopping before the deadline")
            return
        prompt = task
        if g.policy is not None:
            g.policy.install("student", ws)
            prompt = g.policy.notice("student") + task
        remaining = int(deadline - now)
        leg_end = deadline + g.report_s
        env = dict(os.environ)
        env.update({"FK_AGENT_DRIVER": "goal", "FK_GOAL_TURN_DIR": str(paths.turn),
                    # the session's clock covers the report window past the deadline
                    "FK_GOAL_DEADLINE": str(int(leg_end)),
                    "FK_GOAL_SESSION_PREFIX": "fkgoalref",
                    "BASH_DEFAULT_TIMEOUT_MS": "1200000", "BASH_MAX_TIMEOUT_MS": "3600000"})
        env.update(g.extra_env)
        k = len(attempts) + 1
        out_p = paths.session / f"attempt-{k:02d}.out"
        err_p = paths.session / f"attempt-{k:02d}.err"
        g.log(f"attempt {k}: starting the /goal session ({remaining // 60} min to the deadline)")
        rec = {"n": k, "start": now}
        attempts.append(rec)
        save_state(g.op_dir, state)
        with open(out_p, "w", encoding="utf-8") as fo, open(err_p, "w", encoding="utf-8") as fe:
            proc = subprocess.Popen(g.agent_cmd, shell=True, stdin=subprocess.PIPE, stdout=fo,
                                    stderr=fe, text=True, cwd=str(ws), env=env,
                                    start_new_session=True)
            rec["pgid"] = proc.pid
            save_state(g.op_dir, state)
            try:
                try:
                    proc.stdin.write(prompt)
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                last_beat = g.now()
                while proc.poll() is None:
                    served = svc.serve_pending()
                    t = g.now()
                    if t > leg_end + g.kill_grace_s:
                        g.log(f"the session outlived its deadline and report window by "
                              f"{g.kill_grace_s:.0f} s — ending it")
                        _end_group(proc)
                        break
                    if t - last_beat >= g.heartbeat_s:
                        g.log(f"session running ({svc.summary_line()}; "
                              f"{max(0, deadline - t) / 60:.0f} min left)")
                        last_beat = t
                    if not served:
                        g.sleep(g.poll_s)
            except BaseException:
                _end_group(proc)
                rec.update(end=g.now(), rc=None, outcome="interrupted")
                save_state(g.op_dir, state)
                raise
        rc = proc.wait()
        svc.serve_pending()
        rec.update(end=g.now(), rc=rc, outcome=_leg_outcome(paths.turn))
        save_state(g.op_dir, state)
        g.log(f"attempt {k} ended rc={rc} ({rec['outcome'] or 'no leg result'}; "
              f"{svc.summary_line()})")
        if rc != 0 and rec["end"] - rec["start"] < 120 and g.now() < deadline:
            g.sleep(min(600.0, 60.0 * k, max(0.0, deadline - g.now())))


def _leg_outcome(turn: Path) -> str:
    try:
        lines = (turn / "result.json").read_text(encoding="utf-8").splitlines()
        return str(json.loads(lines[-1]).get("outcome") or "")
    except (OSError, ValueError, IndexError):
        return ""


def _collect_transcripts(paths: Paths, state: dict) -> None:
    """Copy the session transcripts the agent wrapper recorded into session/ (provenance)."""
    seen = []
    for ln in (paths.turn / "result.json").read_text(encoding="utf-8").splitlines() \
            if (paths.turn / "result.json").is_file() else []:
        try:
            t = json.loads(ln).get("transcript")
        except ValueError:
            continue
        if t and t not in seen and Path(t).is_file():
            seen.append(t)
    out = []
    for i, t in enumerate(seen, 1):
        dst = paths.session / f"transcript-{i:02d}.jsonl"
        try:
            shutil.copy2(t, dst)
            out.append(str(dst))
        except OSError:
            out.append(t)
    for name in ("result.json", "leg.log", "session.json"):
        if (paths.turn / name).is_file():
            shutil.copy2(paths.turn / name, paths.session / f"leg-{name}")
    if out:
        state["transcripts"] = out


# -- selection and acceptance -------------------------------------------------------------

def accept(g: Generation, state: dict, svc: Service) -> dict:
    paths = svc.paths
    rows = [r for r in load_ledger(paths.ledger) if r.get("kind") == "eval"]
    eligible = sorted((r for r in rows if r.get("eligible")), key=lambda r: r["reading_ms"])
    # one row per distinct kernel (a re-measurement of the same bytes is not a new candidate)
    seen, cands = set(), []
    for r in eligible:
        if r["sha256"] not in seen:
            seen.add(r["sha256"])
            cands.append(r)
    if not cands:
        why = (f"no eligible candidate: {len(rows)} evaluation(s), "
               f"{sum(1 for r in rows if r.get('correct'))} correct")
        return _fail(g, state, why)
    svc.deadline = float("inf")      # acceptance runs after the session's deadline
    tried = []
    for cand in cands[:ACCEPT_CANDIDATES]:
        # the bytes the session judged, or this candidate is not the one it judged
        import hashlib
        try:
            now_sha = hashlib.sha256(Path(cand["kernel"]).read_bytes()).hexdigest()
        except OSError:
            now_sha = ""
        if now_sha != cand["sha256"]:
            g.log(f"acceptance: candidate {cand['n']:03d} changed since it was judged — skipped")
            tried.append({"candidate": cand["n"], "ok": False,
                          "error": "the kernel file changed since it was evaluated"})
            continue
        g.log(f"acceptance: candidate {cand['n']:03d} ({cand['reading_ms']:.6g} ms in the "
              f"session), {ACCEPT_RUNS} run(s)")
        runs, ok = [], True
        for k in range(1, ACCEPT_RUNS + 1):
            branch = f"accept-{cand['n']:03d}-{k}"
            kernel = Path(cand["kernel"])
            svc._commit(branch, kernel, None, f"goal-ref acceptance {cand['n']:03d} run {k}")
            t0 = g.now()
            log, job, err = svc._submit(branch, "all")
            rec = svc.record(log, job, err, kind="accept", n=cand["n"], sha=cand["sha256"],
                             kernel=kernel, src=kernel.read_text(encoding="utf-8", errors="replace"),
                             started=t0)
            rec["run"] = k
            paths.acceptance.mkdir(parents=True, exist_ok=True)
            (paths.acceptance / f"{branch}.log").write_text(log or err, encoding="utf-8")
            _append(paths.ledger, rec)
            runs.append(rec)
            g.log(f"acceptance {cand['n']:03d} run {k}: {_brief(rec)}")
            if not rec.get("eligible"):
                ok = False
                break
        tried.append({"candidate": cand["n"], "ok": ok,
                      "error": "" if ok else (runs[-1].get("error") or "not eligible")})
        if ok:
            return _accepted(g, state, svc, cand, runs, tried)
    return _fail(g, state, "no candidate survived acceptance: " + "; ".join(
        f"{t['candidate']:03d}: {t['error'][:160]}" for t in tried), tried)


def _fail(g: Generation, state: dict, why: str, tried: list | None = None) -> dict:
    state["status"] = STATUS_FAILED
    state["failure"] = why
    if tried:
        state["acceptance"] = tried
    save_state(g.op_dir, state)
    g.log(f"FAILED: {why}")
    raise GoalRefError(f"reference generation failed: {why} (records in {Paths.of(g.op_dir).root})")


def _accepted(g: Generation, state: dict, svc: Service, cand: dict, runs: list,
              tried: list) -> dict:
    paths = svc.paths
    paths.reference.mkdir(parents=True, exist_ok=True)
    ref = paths.reference / Path(g.kernel_file).name
    shutil.copy2(cand["kernel"], ref)
    files = [str(ref)]
    reply = Path(state.get("reply") or "")
    if reply.is_file() and reply.read_text(encoding="utf-8", errors="replace").strip():
        notes = paths.reference / "NOTES.md"
        notes.write_text("# Report of the /goal session that wrote this reference\n\n"
                         + reply.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        files.append(str(notes))
    best = min(runs, key=lambda r: r["reading_ms"])
    vs = [r.get("vs_ref_gate") for r in runs if r.get("vs_ref_gate") is not None]
    accepted = {
        "candidate": cand["n"], "sha256": cand["sha256"], "reference": files,
        "session_reading_ms": cand["reading_ms"], "reading_ms": best["reading_ms"],
        "device_ms": best.get("device_ms"), "e2e_ms": best.get("e2e_ms"),
        "basis": best.get("basis"), "n_workloads": best.get("n_workloads"),
        "workloads": [w.get("name") for w in best.get("workloads") or []],
        "vs_ref_gate": min(vs) if vs else None,
        "jobs": [r.get("job") for r in runs], "runs": len(runs),
        "bar_met": all(r.get("bar_met") for r in runs),
    }
    state.update(status=STATUS_ACCEPTED, accepted=accepted, acceptance=tried)
    state["arch"] = arch_decision(g.cfg, g.target_ms, accepted, g.settings)
    save_state(g.op_dir, state)
    g.log(f"ACCEPTED candidate {cand['n']:03d}: {best['reading_ms']:.6g} ms "
          f"({g.gate_window}), oracle = {', '.join(files)}; target: {state['arch']['why']}")
    commit_reference(svc.tree, g.kernel_file, ref, cand["n"], branch=svc.base_branch)
    return state


def _base_branch(tree: Path) -> str:
    """The card's branch the tree was cloned at (origin/HEAD), else the current one."""
    r = _git(tree, "symbolic-ref", "--short", "refs/remotes/origin/HEAD", check=False)
    if r.returncode == 0 and r.stdout.strip().startswith("origin/"):
        return r.stdout.strip()[len("origin/"):]
    return _git(tree, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def commit_reference(tree: Path, kernel_file: str, ref: Path, n, *,
                     branch: str | None = None) -> bool:
    """Put the accepted kernel on the submission tree's checked-out branch, so the tree
    measures the reference whenever a job runs it at HEAD (``apply`` wires it as the
    loop's measured reference when the card has none). True when a commit was made."""
    _git(tree, "checkout", "-q", "-f", branch or _base_branch(tree))
    dst = Path(tree) / kernel_file
    script = Path(tree) / MEASURE_SCRIPT
    try:
        if dst.read_bytes() == Path(ref).read_bytes() and \
                script.read_text(encoding="utf-8") == MEASURE_SCRIPT_TEXT:
            return False
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ref, dst)
    script.write_text(MEASURE_SCRIPT_TEXT, encoding="utf-8")
    _git(tree, "add", "--", kernel_file, MEASURE_SCRIPT)
    _git(tree, "commit", "-q", "-m", f"goal-ref: the accepted reference (candidate {n})")
    return True


def ensure_reference_tree(root: Path, op: str, op_dir: Path, state: dict) -> Path:
    """The submission tree, present and carrying the accepted reference (a reuse after
    the tree was removed, or one accepted before this existed)."""
    tree = ensure_tree(root, op, op_dir, fresh=False)
    acc = state.get("accepted") or {}
    ref = next((Path(f) for f in acc.get("reference") or [] if Path(f).name ==
                Path(state.get("kernel_file") or "kernel.py").name), None)
    if ref is not None and ref.is_file():
        commit_reference(tree, state.get("kernel_file") or "kernel.py", ref,
                         acc.get("candidate"))
    return tree


def arch_decision(cfg: dict, target_ms: float | None, accepted: dict) -> dict:
    """Whether the accepted reading becomes the architecture reference (the target).

    The target anchors to what the run reproduces — the oracle — unless something
    faster is already known: a reference that the student only has to match should not
    stop the loop while a faster one sits in plain sight (observed: a target taken
    from the library baseline stopped a run 26% short of its oracle).
    So the generated reading becomes the architecture reference when it is the fastest
    known: faster than a declared one (a SOL card's T_b is declared this way) and not
    slower than the same-job library reference. A measured architecture reference
    (arch_ref_mode: a real implementation timed through its own tree) and --target-ms
    are never overridden."""
    lc = cfg.get("loop") or {}
    if target_ms:
        return {"declare": False, "why": "--target-ms given"}
    if lc.get("arch_ref_mode"):
        return {"declare": False, "why": "the card measures its own architecture reference "
                                         "(arch_ref_mode)"}
    vs = accepted.get("vs_ref_gate")
    if vs is not None and vs < 1.0:
        return {"declare": False,
                "why": f"the same-job library reference was faster ({vs:.4f}x): it keeps "
                       f"anchoring the target"}
    mine = accepted.get("device_ms") if accepted.get("device_ms") is not None \
        else accepted.get("reading_ms")
    declared = lc.get("arch_ref_device_ms", lc.get("arch_ref_ms"))
    if declared is not None and mine is not None and float(declared) <= float(mine):
        return {"declare": False,
                "why": f"the declared architecture reference ({lc.get('arch_ref_name') or 'declared'}, "
                       f"{float(declared):.6g} ms) is at least as fast: it keeps anchoring the target"}
    if declared is not None:
        return {"declare": True, "replaces": {k: lc[k] for k in (
                    "arch_ref_name", "arch_ref_ms", "arch_ref_device_ms") if k in lc},
                "why": f"the generated reference ({mine:.6g} ms) is faster than the declared "
                       f"{lc.get('arch_ref_name') or 'architecture reference'} "
                       f"({float(declared):.6g} ms): it anchors the target"}
    return {"declare": True, "why": f"the generated reference ({accepted['reading_ms']:.6g} ms) "
                                    f"anchors the target"}


def apply(cfg: dict, state: dict, *, target_ms: float | None = None) -> tuple[dict, dict]:
    """(``cfg`` with the accepted reference wired in as the oracle — and, when
    ``arch_decision`` says so for THIS launch's loop.toml and flags, as the architecture
    reference —, the decision). A deep copy; loop.toml on disk is never rewritten."""
    import copy
    out = copy.deepcopy(cfg)
    lc = out.setdefault("loop", {})
    acc = state.get("accepted") or {}
    arch = arch_decision(cfg, target_ms, acc)
    lc["oracle_mode"] = "source"
    lc["oracle"] = list(acc.get("reference") or [])
    lc.setdefault("kb_lint_oracle_tokens", False)
    if not (lc.get("ref_cmd") or lc.get("ref_mode")) and state.get("op"):
        # No library reference to anchor numbers to (a SOL card without a sealed one):
        # the loop measures the generated reference itself, through the submission tree
        # (whose HEAD carries it), timing and NCU in one job — the shape of a source-mode
        # run, whose measured reference is the very kernel its oracle is.
        lc["ref_mode"] = REF_TASK
        lc["ref_submit_op"] = tree_op(state["op"])
        lc.setdefault("ref_name", f"goal-generated reference (candidate {acc.get('candidate')})")
    if arch.get("declare"):
        for k in ("arch_ref_device_basis", "arch_ref_workloads", "arch_ref_graph_ms",
                  "arch_ref_host_enqueue_ms", "arch_ref_ncu", "arch_ref_job",
                  "arch_ref_provenance"):
            lc.pop(k, None)     # the declaration replaced describes another reading
        e2e = acc.get("e2e_ms") if acc.get("e2e_ms") is not None else acc.get("reading_ms")
        lc["arch_ref_ms"] = float(e2e)
        if acc.get("device_ms") is not None:
            lc["arch_ref_device_ms"] = float(acc["device_ms"])
        if acc.get("basis"):
            lc["arch_ref_device_basis"] = acc["basis"]
        if (acc.get("n_workloads") or 1) > 1 and acc.get("workloads"):
            lc["arch_ref_workloads"] = list(acc["workloads"])
        lc["arch_ref_name"] = f"goal-generated reference (candidate {acc.get('candidate')})"
        lc["arch_ref_job"] = ",".join(j for j in acc.get("jobs") or [] if j)
        lc["arch_ref_provenance"] = (f"forgekernel.loop.goal_ref acceptance: "
                                     f"{acc.get('runs')} run(s), fastest kept")
    return out, arch


if __name__ == "__main__":
    sys.exit(main())
