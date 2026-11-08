"""``forge init | new | run | status | harvest``: the launch flow (docs/design/v1-launch.md).

    forge init <op>                    tasks/<op>/ from tasks/_template/
    forge new tasks/<op> [--dry-run]   a new run of the task: .artifacts/<op>_<id>/
    forge run .artifacts/<op>_<id>     run it; on a run that has started, continue it
    forge status [.artifacts/<op>_<id>] where a run is (every run when omitted)
    forge harvest .artifacts/<op>_<id> the end-of-run knowledge-base write-up of a stopped run

`forge run` exits 0 when the run ends at its target, 2 on any other stop (the loop's stop
reason is in run.json and report.md).

A run, in order (each step is skipped on resume once it is done):

  1. the judge   a teacher turn writes check/{definition.json, judge.py, groups.json} from
                 the task (forgekernel.judge); checked statically, retried with the findings
  2. the trees   operators/<op>/ (the card) and, with a reference, operators/_<op>ref/,
                 the reference sealed into both (forgekernel.taskcard)
  3. calibration with a reference: the reference against the judge on the GPU, on every
                 input group of every shape; a disagreement goes back to step 1 with the
                 numbers, an agreement freezes the tolerance. Without one the tolerance is
                 the dtype floor, and the /goal session that writes the reference is held
                 to it.
  4. the loop    loop.toml from the task, the student's workspace, the agent commands --
                 then the two-layer loop (forgekernel.loop.cli), which resumes itself from
                 its own records
  5. the report  report.md: how the run ended, the best kernel, new knowledge-base entries
                 waiting for review

Nothing on the command line changes the task.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from forgekernel import judge as judge_mod
from forgekernel import runs as runs_mod
from forgekernel import taskcard
from forgekernel import workspace as workspace_mod
from forgekernel.agents import agent_command
from forgekernel.localenv import MachineError, load_machine
from forgekernel.paths import OPERATORS_ENV, repo_root
from forgekernel.runs import RunError, new_run, open_run
from forgekernel.task import (MODE_BLACKBOX, MODE_GENERATE, MODE_LEADERBOARD, MODE_SOURCE,
                              TaskError, load_task)

TASKS_DIR = "tasks"
TEMPLATE_TASK = "_template"
CALIBRATE_MODE = "calibrate"
KB_DIRS = ("hw_kb", "docs/kb-provenance")
BEST_DIR = "best"


def _say(msg: str) -> None:
    print(f"[forge] {msg}", flush=True)


# -- init -----------------------------------------------------------------------------------

BOARD_TASK_TOML = """# {op} -- one {suite} problem, frozen from the board (`forge init --leaderboard`).
# The board decides the judge, the inputs and the clock; the card below carries them.

[task]
language = "{language}"
arch     = "sm100"

[leaderboard]
suite = "{suite}"
card  = "leaderboard/card"{ref_line}

# Everything else takes its value from global_config/; write only what changes.
# [target]
# ratio  = 1.2                       # x the reference the card declares
# [budget]
# hours    = 4
# max_meta = 3
"""


def cmd_init(argv) -> int:
    """tasks/<op>/: a copy of tasks/_template/ to fill in; with --leaderboard, one board
    problem (forgekernel.scaffold)."""
    if "--leaderboard" in argv or any(a.startswith("--leaderboard=") for a in argv):
        return new_board_task(argv)
    p = argparse.ArgumentParser(prog="forge init",
                                description="Create tasks/<op>/ from tasks/_template/.")
    p.add_argument("op")
    args = p.parse_args(argv)
    d = repo_root() / TASKS_DIR / args.op
    if d.exists():
        sys.stderr.write(f"[forge] {d} already exists\n")
        return 2
    shutil.copytree(repo_root() / TASKS_DIR / TEMPLATE_TASK, d)
    _say(f"wrote {d}/: fill in description.md and shapes.json, adjust task.toml, add "
         f"reference/ if you have one, then `forge new {TASKS_DIR}/{args.op}`")
    return 0


def new_board_task(argv) -> int:
    """tasks/<op>/ for one board problem: the card `forge init --leaderboard` generates (the
    board's judge, inputs, clock and, when one is given, its sealed reference with a recon
    tree), frozen under leaderboard/, and a task.toml naming it."""
    from forgekernel import scaffold
    if not argv or argv[0].startswith("-"):
        sys.stderr.write("[forge] usage: forge init <op> --leaderboard <suite> ...\n")
        return 2
    op = argv[0]
    d = repo_root() / TASKS_DIR / op
    if d.exists():
        sys.stderr.write(f"[forge] {d} already exists\n")
        return 2
    with tempfile.TemporaryDirectory(prefix="forge-new-") as tmp:
        rc = scaffold.main([*argv, "--base-dir", tmp])
        card = Path(tmp) / "operators" / op
        if rc != 0 or not card.is_dir():
            return rc          # --list-workloads, or a refusal already explained
        (d / taskcard.BOARD_DIR).mkdir(parents=True)
        shutil.copytree(card, d / taskcard.BOARD_DIR / "card")
        ref = Path(tmp) / "operators" / taskcard.ref_tree(op)
        if ref.is_dir():
            shutil.copytree(ref, d / taskcard.BOARD_DIR / "ref")
    problem = json.loads((d / taskcard.BOARD_DIR / "card" / taskcard.PROBLEM).read_text())
    language = "cuda" if str(problem.get("kernel_file", "")).endswith(".cu") else "cutedsl"
    ref_line = '\nref   = "leaderboard/ref"' if (d / taskcard.BOARD_DIR / "ref").is_dir() else ""
    (d / "task.toml").write_text(BOARD_TASK_TOML.format(
        op=op, suite=problem["suite"], language=language, ref_line=ref_line), encoding="utf-8")
    _say(f"wrote {d}/ ({problem['suite']} {problem['definition']['name']}): adjust task.toml, "
         f"then `forge new {TASKS_DIR}/{op}`")
    return 0


# -- new / run -----------------------------------------------------------------------------


def describe(task, machine) -> str:
    if task.mode == MODE_LEADERBOARD:
        shapes = "; ".join(f"{s.name} {json.dumps(s.axes)}" for s in task.shapes) \
            + " (inputs per the board's protocol)"
    else:
        shapes = "; ".join(f"{s.name} {json.dumps(s.axes)} ({len(s.inputs)} input group(s) given"
                           + (f", {s.groups_to_add} to add" if s.groups_to_add else "") + ")"
                           for s in task.shapes)
    lines = [f"task {task.name}: {task.language} on {task.arch}, mode {task.mode}",
             "  shapes: " + shapes]
    if task.reference is not None:
        lines.append(f"  reference: {task.reference.run}"
                     + (f" (source {task.reference.source})" if task.mode == MODE_SOURCE
                        else " (black box)"))
    target = (f"{task.target.ms} ms" if task.target.ms is not None
              else f"{task.target.ratio}x the reference")
    lines += [f"  target: {target}, {task.target.timing} timing",
              f"  agents: teacher {_agent_line(task.teacher)}, student "
              f"{_agent_line(task.student)}",
              f"  budget: {task.budget.hours:g} h, {task.budget.max_meta} teacher round(s)"
              + (f", {task.budget.reference_hours:g} h to write the reference"
                 if task.mode == MODE_GENERATE else ""),
              f"  GPU: {machine.describe()}"]
    if machine.workspace:
        lines.append(f"  agents: this machine; SSH workspace: {machine.remote_workdir}/{task.name}/<run>")
    lines += [f"  note: {n}" for n in task.notes]
    return "\n".join(lines)


def _agent_line(spec) -> str:
    extra = [x for x in (spec.effort, None if spec.provider == "native" else spec.provider) if x]
    return f"{spec.backend}/{spec.model}" + (f" ({', '.join(extra)})" if extra else "")


def _check_providers(task, machine) -> None:
    """A provider's key must be there before a run is created, not hours into it."""
    from forgekernel import provider as provider_mod
    env = {**os.environ, **{k: v for k, v in machine.env.items() if v}}
    for spec in (task.teacher, task.student):
        provider_mod.grant(spec.backend, spec.provider, env)


def cmd_new(argv) -> int:
    """A new run of a task: the task is checked, the plan printed, and the run directory
    created under .artifacts/ (`forge run` runs it)."""
    from forgekernel.provider import ProviderError
    p = argparse.ArgumentParser(prog="forge new", description="Create a new run of a task.")
    p.add_argument("task", help="the task directory (tasks/<op>)")
    p.add_argument("--dry-run", action="store_true", help="check and print the plan only")
    args = p.parse_args(argv)
    try:
        task = load_task(args.task)
        machine = load_machine()
        _check_providers(task, machine)
    except (TaskError, MachineError, ProviderError) as e:
        sys.stderr.write(f"[forge] {e}\n")
        return 2
    print(describe(task, machine), flush=True)
    if args.dry_run:
        return 0
    run = new_run(task, machine=machine)
    try:
        workspace_mod.sync_run(run, machine)
    except workspace_mod.WorkspaceError as e:
        run.update(error=str(e))
        sys.stderr.write(f"[forge] {e}; local run preserved at {run.dir}\n")
        return 2
    _say(f"run directory {run.dir}: `forge run {_shown(run.dir)}` runs it")
    return 0


def _shown(path: Path) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def cmd_run(argv) -> int:
    """Run a run: from the start when it is new, else from where it stopped (every step
    that is done is skipped)."""
    p = argparse.ArgumentParser(prog="forge run",
                                description="Run a run, or continue one that has started.")
    p.add_argument("run", help="the run directory (.artifacts/<op>_<id>)")
    args = p.parse_args(argv)
    try:
        run = open_run(args.run)
        machine = load_machine()
    except (RunError, MachineError) as e:
        sys.stderr.write(f"[forge] {e}\n")
        return 2
    started = run.record().get("status") != runs_mod.STATUS_CREATED
    try:
        return drive(run, machine, resuming=started)
    except workspace_mod.WorkspaceError as e:
        sys.stderr.write(f"[forge] {e}\n")
        return 2


def cmd_status(argv) -> int:
    p = argparse.ArgumentParser(prog="forge status")
    p.add_argument("run", nargs="?", help="a run directory; omitted: every run")
    args = p.parse_args(argv)
    if args.run is None:
        for r in runs_mod.list_runs():
            rec = r.record()
            print(f"{r.name}  {rec.get('status')}  {rec.get('updated')}")
        return 0
    try:
        run = open_run(args.run)
    except RunError as e:
        sys.stderr.write(f"[forge] {e}\n")
        return 2
    print(status_text(run))
    return 0


def status_text(run) -> str:
    rec = run.record()
    lines = [f"{run.name}: {rec.get('status')} (updated {rec.get('updated')})"]
    for key in ("stage", "loop_rc", "best", "error"):
        if rec.get(key) is not None:
            lines.append(f"  {key}: {rec[key]}")
    status = run.operators_dir / run.task_name / "student" / "status.jsonl"
    if status.is_file():
        rows = [json.loads(x) for x in status.read_text().splitlines() if x.strip()]
        keeps = [r for r in rows if r.get("verdict") == "KEEP"]
        lines.append(f"  iterations: {len(rows)} ({len(keeps)} kept)")
    if run.log.is_file():
        lines.append(f"  log: {run.log}")
    return "\n".join(lines)


def cmd_harvest(argv) -> int:
    p = argparse.ArgumentParser(prog="forge harvest")
    p.add_argument("run")
    args = p.parse_args(argv)
    try:
        run = open_run(args.run)
        machine = load_machine()
    except (RunError, MachineError) as e:
        sys.stderr.write(f"[forge] {e}\n")
        return 2
    return drive(run, machine, resuming=True, harvest_only=True)


# -- the run itself -------------------------------------------------------------------------


class _Tee:
    def __init__(self, stream, path: Path):
        self._stream = stream
        self._file = open(path, "a", encoding="utf-8")

    def write(self, s):
        self._stream.write(s)
        self._file.write(s)
        return len(s)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


@contextlib.contextmanager
def _entered(run, machine):
    """While the run is driven, this process and everything it starts work on the run's
    trees with local.env's credentials, and its output goes to run.log as well; both are
    put back afterwards."""
    saved_env = {k: os.environ.get(k) for k in [OPERATORS_ENV, *machine.env]}
    saved_out, saved_err = sys.stdout, sys.stderr
    os.environ[OPERATORS_ENV] = str(run.operators_dir)
    for k, v in machine.env.items():
        if v:       # an empty KEY= line (local.env.example) must not blank a login
            os.environ.setdefault(k, v)
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout, run.log)
        sys.stderr = _Tee(sys.stderr, run.log)
    token = workspace_mod.ACTIVE_RUN.set((run, machine.workspace) if machine.workspace else None)
    entered = False
    try:
        workspace_mod.sync_run(run, machine)
        entered = True
        yield
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            try:
                if entered:
                    workspace_mod.sync_run(run, machine)
            except workspace_mod.WorkspaceError as e:
                if not failed:
                    raise
                sys.stderr.write(f"[forge] final workspace sync failed: {e}\n")
        finally:
            workspace_mod.ACTIVE_RUN.reset(token)
            sys.stdout, sys.stderr = saved_out, saved_err
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def deny_paths(run, task) -> list[str]:
    """What no student (and, in blackbox mode, no teacher) may read: the reference wherever
    a copy of it lives, every other run, the task's own extras."""
    out = []
    if task.reference is not None:
        out.append(str(task.reference.run_file.parent))
        if task.reference.source is not None:
            out.append(str(task.reference.source))
        src = Path(run.record().get("task_source", ""))
        if src.is_dir() and (src / "reference").exists():
            out.append(str(src / "reference"))
        out += [str(p) for p in task.reference.deny]
    if task.leaderboard is not None:
        # the frozen card carries the sealed reference and its key, the recon tree too
        out.append(str(task.leaderboard.card.parent))
        src = Path(run.record().get("task_source", ""))
        if (src / taskcard.BOARD_DIR).is_dir():
            out.append(str(src / taskcard.BOARD_DIR))
    runs_directory = (run.dir.parent.parent if run.record().get("remote_workspace")
                      else run.dir.parent)
    for other in runs_mod.list_runs(runs_directory):
        if other.dir != run.dir:
            out.append(str(other.dir))
    return list(dict.fromkeys(out))


def drive(run, machine, *, resuming: bool = False, harvest_only: bool = False) -> int:
    with _entered(run, machine):
        task = load_task(run.task_dir, name=run.task_name)
        run.update(status=runs_mod.STATUS_RUNNING)
        try:
            rc = _drive(run, task, machine, resuming=resuming, harvest_only=harvest_only)
        except Exception as e:  # noqa: BLE001 -- the record must say why the run stopped
            run.update(status=runs_mod.STATUS_FAILED, error=f"{type(e).__name__}: {e}")
            raise
        stop = loop_stop(run, task) if rc == 0 else None
        met = rc == 0 and confirmed_at_target(run, task)
        final = (runs_mod.STATUS_DONE if stop == STOP_TARGET or met
                 else runs_mod.STATUS_STOPPED)
        run.update(status=final, loop_rc=rc, stop=stop, target_met=met)
        write_report(run, task, rc)
        return 0 if final == runs_mod.STATUS_DONE else 2


STOP_TARGET = "target"


def loop_stop(run, task) -> str | None:
    """Why the loop ended (its summary's ``stop``: target, meta_budget, infeasible, ...)."""
    from forgekernel.loop.cli import SUMMARY_FILE
    try:
        return json.loads((run.operators_dir / task.name / SUMMARY_FILE).read_text()).get("stop")
    except (OSError, ValueError):
        return None


def confirmed_at_target(run, task) -> bool:
    """Whether the final confirmation of the best kernel meets the task's target, whatever
    stopped the loop (a budget can run out after the target was reached on noisy readings).
    A correctness-monitor alarm on that kernel rules it out."""
    c = confirmation(run, task)
    if c is None or (c.get("correctness_monitor") or {}).get("status") == "alarm":
        return False
    if task.target.ms is not None:
        return float(c["median_ms"]) <= task.target.ms
    return (task.target.ratio is not None and c["speedup"] is not None
            and c["speedup"] >= task.target.ratio)


def _drive(run, task, machine, *, resuming: bool, harvest_only: bool) -> int:
    if task.mode == MODE_LEADERBOARD:
        return _drive_board(run, task, machine, resuming=resuming, harvest_only=harvest_only)
    check_dir = run.dir / judge_mod.CHECK_DIR
    op_dir = run.operators_dir / task.name
    deny = deny_paths(run, task)
    rec = run.record()
    feedback = ""
    for attempt in range(1, judge_mod.ATTEMPTS + 1):
        if not rec.get("judge_written"):
            write_judge(run, task, check_dir, deny, feedback)
            run.update(judge_written=True, stage="judge written")
            rec = run.record()
        check = judge_mod.load_check(check_dir)
        if not op_dir.exists():
            run.update(stage="writing the trees")
            taskcard.write_card(run, task, check, machine)
            if task.reference is not None:
                taskcard.seal_reference(run, task, op_dir)
        else:
            _refresh_machine(run, task, machine)
        if rec.get("tolerance_frozen"):
            break
        if task.reference is None:
            taskcard.set_tolerance(op_dir, {}, "the dtype floor: no reference to calibrate "
                                               "against (a generated one must pass the judge)")
            run.update(tolerance_frozen=True, stage="tolerance frozen")
            rec = run.record()
            break
        run.update(stage=f"calibrating the judge (attempt {attempt})")
        cal = calibrate(task, op_dir)
        if cal.get("match"):
            taskcard.set_tolerance(op_dir, cal["tolerance"],
                                   f"from the reference on every input group (attempt {attempt})")
            run.update(tolerance_frozen=True, stage="tolerance frozen",
                       tolerance=cal["tolerance"])
            rec = run.record()
            break
        feedback = ("The judge and the reference disagree:\n"
                    + "\n".join(f"- {m}" for m in cal.get("problems", [])[:20])
                    + "\nEither the judge computes something else than the reference, or an "
                      "input group is not legal for it. Fix the files.")
        _say(f"calibration: no match (attempt {attempt}/{judge_mod.ATTEMPTS})")
        _discard_trees(run, task)
        run.update(judge_written=False)
        rec = run.record()
    else:
        raise RuntimeError(f"the judge did not agree with the reference after "
                           f"{judge_mod.ATTEMPTS} attempts: {feedback}")
    run.update(stage="probing NCU")
    ncu = probe_ncu(task, op_dir)
    run.update(ncu_available=ncu, stage="the loop")
    if not ncu:
        _say("NCU cannot profile on this machine: the run degrades to timing only (no "
             "per-kernel signatures, no timeline / stall pricing)")
    taskcard.write_loop_toml(run, task, op_dir, deny=deny, ncu=ncu)
    taskcard.clone_workspace(op_dir, run.ws)
    return run_loop(run, task, resuming=resuming, harvest_only=harvest_only)


def _drive_board(run, task, machine, *, resuming: bool, harvest_only: bool) -> int:
    """A board problem: the board's judge and clock come with the card, so there is no
    judge to write and nothing to calibrate."""
    op_dir = run.operators_dir / task.name
    deny = deny_paths(run, task)
    if not op_dir.exists():
        run.update(stage="installing the board's trees")
        taskcard.install_board_trees(run, task, machine)
    else:
        _refresh_machine(run, task, machine)
    run.update(stage="probing NCU")
    ncu = probe_ncu(task, op_dir)
    run.update(ncu_available=ncu, stage="the loop")
    taskcard.write_board_loop_toml(run, task, op_dir, deny=deny, ncu=ncu)
    taskcard.clone_workspace(op_dir, run.ws)
    return run_loop(run, task, resuming=resuming, harvest_only=harvest_only)


def write_judge(run, task, check_dir: Path, deny: list[str], feedback: str = "") -> None:
    """Teacher turns until the judge's files pass the static check."""
    from forgekernel.loop.agent_policy import AgentPolicy
    from forgekernel.loop.cli import RealBackend
    check_dir.mkdir(parents=True, exist_ok=True)
    blackbox = task.mode == MODE_BLACKBOX
    policy = AgentPolicy(oracle_mode="blackbox" if blackbox else "source",
                         repo_root=repo_root(), ref_source_paths=tuple(Path(p) for p in deny),
                         ref_source_urls=(), teacher_trees=(), dsl_source_dirs=(),
                         secret_files=(), language=task.loop_language)
    be = RealBackend(task.name, check_dir, None, "forge-judge", "forge-judge@localhost",
                     teacher_agent_cmd=agent_command(task.teacher, "teacher"),
                     agent_policy=policy)
    be.agent_cmd = be.teacher_agent_cmd
    source = task.reference.source if task.mode == MODE_SOURCE else None
    for attempt in range(1, judge_mod.ATTEMPTS + 1):
        _say(f"the judge: teacher turn {attempt}/{judge_mod.ATTEMPTS}")
        be.run_agent(judge_mod.judge_prompt(task, check_dir, reference_source=source,
                                            feedback=feedback), role="teacher")
        problems = judge_mod.check_problems(check_dir, task)
        if not problems:
            _say("the judge: files pass the static check")
            return
        feedback = "\n".join(f"- {p}" for p in problems)
        _say(f"the judge: {len(problems)} problem(s): {problems[:3]}")
    raise RuntimeError(f"the teacher's judge files still fail the check: {feedback}")


def calibrate(task, op_dir: Path) -> dict:
    """The reference against the judge on the GPU (the card's ``calibrate`` task)."""
    from forgekernel.loop.cli import RealBackend
    be = RealBackend(task.name, op_dir, None, "forge-judge", "forge-judge@localhost")
    log = be.submit_gpu(taskcard.BRANCH, CALIBRATE_MODE)
    return parse_calibration(log)


def probe_ncu(task, op_dir: Path) -> bool:
    """Can NCU read the GPU's counters on this machine? One short job, every launch (the
    machine local.env names may have changed since the last one)."""
    from forgekernel.loop.cli import BackendError, RealBackend
    be = RealBackend(task.name, op_dir, None, "forge-judge", "forge-judge@localhost")
    try:
        log = be.submit_gpu(taskcard.BRANCH, taskcard.NCU_PROBE_MODE)
    except BackendError as e:
        _say(f"the NCU probe job failed ({e}); assuming NCU is unavailable")
        return False
    return ncu_probe_ok(log)


def ncu_probe_ok(log: str) -> bool:
    """A metric row for the probe's kernel, and no counter-permission refusal."""
    if "ERR_NVGPUCTRPERM" in log:
        return False
    return any('"gpu__time_active.sum"' in line and line.rstrip().endswith('"')
               and any(ch.isdigit() for ch in line.rsplit('","', 1)[-1])
               for line in log.splitlines())


def parse_calibration(log: str) -> dict:
    tail = log.rsplit("=== JSON ===", 1)
    if len(tail) != 2:
        return {"match": False, "problems": ["the calibrate job printed no result:\n"
                                              + log[-2000:]]}
    try:
        obj, _ = json.JSONDecoder().raw_decode(tail[1].strip())
    except json.JSONDecodeError as e:
        return {"match": False, "problems": [f"the calibrate job's result does not parse: {e}"]}
    return obj.get("calibration") or {"match": False, "problems": ["no calibration block"]}


def _discard_trees(run, task) -> None:
    for name in (task.name, taskcard.ref_tree(task.name)):
        shutil.rmtree(run.operators_dir / name, ignore_errors=True)


def _refresh_machine(run, task, machine) -> None:
    """A resumed run follows local.env as it is now."""
    for name in (task.name, taskcard.ref_tree(task.name)):
        tree = run.operators_dir / name
        if (tree / "submit.toml").exists():
            timeout = taskcard.JOB_TIMEOUT_S if name == task.name else taskcard.REF_JOB_TIMEOUT_S
            taskcard.write_submit(tree, task, machine, timeout=timeout)
            taskcard._commit(tree, "Chore: follow local.env")


def run_loop(run, task, *, resuming: bool, harvest_only: bool) -> int:
    from forgekernel.loop import cli as loop_cli
    argv = [task.name, "--workspace", str(run.ws),
            "--hours", f"{task.budget.hours:g}", "--max-meta", str(task.budget.max_meta),
            "--agent-cmd", agent_command(task.student, "student"),
            "--teacher-agent-cmd", agent_command(task.teacher, "teacher")]
    if task.target.ms is not None:
        argv += ["--target-ms", repr(task.target.ms)]
    else:
        argv += ["--equivalence-ratio", repr(task.target.ratio)]
    if resuming:
        argv += ["--allow-preexisting-artifacts", "--allow-dirty-kb"]
    if harvest_only:
        argv.append("--harvest-only")
    _say("the loop: " + " ".join(argv[:5]) + " ...")
    return loop_cli.main(argv)


# -- the report -----------------------------------------------------------------------------


def kb_changes() -> list[str]:
    """Knowledge-base entries this machine's runs certified and nobody has committed yet."""
    try:
        out = subprocess.run(["git", "-C", str(repo_root()), "status", "--porcelain", "--",
                              *KB_DIRS], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line[3:] for line in out.splitlines() if line.strip()]


def confirmation(run, task) -> dict | None:
    """best_confirmation.json with "speedup" (x the reference, or None) filled in; None
    before a confirmation exists."""
    op_dir = run.operators_dir / task.name
    try:
        c = json.loads((op_dir / "best_confirmation.json").read_text())
    except (OSError, ValueError):
        return None
    if not c.get("sha") or not c.get("median_ms"):
        return None
    speedup = c.get("median_ratio")
    if speedup is None:
        # no reference rode the jobs (a generated reference): against its archived reading
        try:
            arch = json.loads((op_dir / "arch_baseline.json").read_text())
            key = "ms_device" if c.get("window") == "device" else "ms"
            if arch.get(key):
                speedup = float(arch[key]) / float(c["median_ms"])
        except (OSError, ValueError, TypeError):
            pass
    return {**c, "speedup": speedup}


def best_of(run, task) -> dict | None:
    """The run's confirmed best kernel: {"sha", "window", "median_ms", "speedup" (x the
    reference, or None), "ratios" (each confirmation job's same-job ratio; empty when the
    reference did not ride the jobs), "kernel" (its source)}; None before a confirmation
    exists."""
    c = confirmation(run, task)
    if c is None:
        return None
    kf = taskcard.kernel_file(task)
    src = subprocess.run(["git", "-C", str(run.ws), "show", f"{c['sha']}:{kf}"],
                         capture_output=True, text=True)
    return {"sha": c["sha"], "window": c.get("window"), "median_ms": c["median_ms"],
            "speedup": c["speedup"],
            "ratios": [j["ratio"] for j in c.get("jobs", []) if j.get("ratio") is not None],
            "kernel": src.stdout if src.returncode == 0 else None}


def write_report(run, task, rc: int) -> Path:
    rec = run.record()
    lines = [f"# {run.name}", "", f"- task: {task.name} ({task.mode} mode)",
             f"- status: {rec.get('status')} (stop: {rec.get('stop') or 'unknown'}, "
             f"loop exit {rc})"]
    if rec.get("target_met") and rec.get("stop") != STOP_TARGET:
        lines.append("- the final confirmation meets the target")
    best = best_of(run, task)
    if best is not None:
        lines.append(f"- best: {best['sha'][:12]} -- {best['window']} median {best['median_ms']} ms"
                     + (f", {best['speedup']:.4f}x the reference" if best.get("speedup") else ""))
        if best.get("kernel"):
            kf = run.dir / BEST_DIR / Path(taskcard.kernel_file(task)).name
            kf.parent.mkdir(exist_ok=True)
            kf.write_text(best["kernel"], encoding="utf-8")
            lines.append(f"- kernel: {kf.relative_to(run.dir)}")
    lines.append(f"- workspace: {run.ws}")
    kb = kb_changes()
    if kb:
        lines += ["", "## New knowledge-base entries (review, then commit or discard)", ""]
        lines += [f"- {p}" for p in kb]
        _say(f"{len(kb)} new knowledge-base file(s) wait for review: "
             + ", ".join(kb[:5]) + (" ..." if len(kb) > 5 else ""))
    p = run.dir / "report.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _say(f"report: {p}")
    return p


def cmd_export(argv) -> int:
    """The run's best kernel: a leaderboard task's as the board's solution JSON, any other
    task's as its source."""
    from forgekernel import leaderboard as lbmod
    p = argparse.ArgumentParser(prog="forge export")
    p.add_argument("run")
    p.add_argument("--out", default=None, help="write here (default: stdout)")
    p.add_argument("--name", default=None, help="the solution's name")
    p.add_argument("--author", default="forgekernel")
    p.add_argument("--hardware", default="B200")
    args = p.parse_args(argv)
    try:
        run = open_run(args.run)
        task = load_task(run.task_dir, name=run.task_name)
    except (RunError, TaskError) as e:
        sys.stderr.write(f"[forge] {e}\n")
        return 2
    best = best_of(run, task)
    if best is None or not best.get("kernel"):
        sys.stderr.write("[forge] the run has no confirmed best kernel to export yet\n")
        return 2
    if task.mode != MODE_LEADERBOARD:
        # no board to submit to: the kernel source itself
        if args.out:
            Path(args.out).write_text(best["kernel"], encoding="utf-8")
            _say(f"wrote {args.out} (the best kernel, {best['sha'][:12]})")
        else:
            sys.stdout.write(best["kernel"])
        return 0
    card = run.operators_dir / task.name
    with tempfile.TemporaryDirectory(prefix="forge-export-") as tmp:
        # the card with the best kernel in place of its working copy
        tmp_card = Path(tmp) / task.name
        shutil.copytree(card, tmp_card, ignore=shutil.ignore_patterns(".git", "student",
                                                                     "*.local.toml"))
        kf = json.loads((card / taskcard.PROBLEM).read_text()).get("kernel_file", "kernel.py")
        (tmp_card / kf).write_text(best["kernel"], encoding="utf-8")
        try:
            sol = lbmod.export_solution(tmp_card, name=args.name or f"{task.name}_forgekernel",
                                        author=args.author, hardware=args.hardware)
        except (OSError, ValueError) as e:
            sys.stderr.write(f"[forge] export refused: {e}\n")
            return 2
    text = json.dumps(sol, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        _say(f"wrote {args.out} ({task.leaderboard.suite} solution {sol['name']}, best "
             f"{best['sha'][:12]})")
    else:
        sys.stdout.write(text)
    return 0


COMMANDS = {"init": cmd_init, "new": cmd_new, "run": cmd_run, "status": cmd_status,
            "harvest": cmd_harvest, "export": cmd_export}
