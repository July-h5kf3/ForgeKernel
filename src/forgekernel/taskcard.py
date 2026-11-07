"""The trees a run works on, generated from its task (``<run>/operators/``).

``forge run`` turns ``tasks/<op>/`` into what the loop has always run on:

    <run>/operators/<op>/        the card: the judge and clock (eval/, problem.json), the
                                 kernel stub the student starts from, gpu_entry.sh, the
                                 timed-window contract, submit.toml (local.env's machine),
                                 loop.toml -- a git repository, the student's workspace is a
                                 clone of it
    <run>/operators/_<op>ref/    with a reference: its recon tree (the sealed reference,
                                 sealref.py) -- the teacher's, never the student's

Nothing in either tree is written by hand. The problem the card judges is the task's: its
shapes, every shape's input groups (the task's files, then the groups the judge's author
added), the judge (forgekernel.judge), the tolerance calibrated from the reference, and the
reference itself, sealed (forgekernel.sealref) so it rides the student's jobs without being
readable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from pathlib import Path

from forgekernel import leaderboard as lbmod
from forgekernel import sealref
from forgekernel.paths import template_dir
from forgekernel.task import MODE_BLACKBOX, MODE_GENERATE, toml_value

SUITE = "forgekernel"
PROBLEM = lbmod.PROBLEM_FILE                     # eval/leaderboard/problem.json
LB_DIR = lbmod.LEADERBOARD_DIR
KERNEL_FILE = {"cutedsl": "kernel.py", "cuda": "kernel.cu"}
CUTE_ARCH = {"sm90": "sm_90a", "sm100": "sm_100a", "sm103": "sm_103a", "sm120": "sm_120a"}
EVAL_FILES = ("leaderboard.py", "timing.py", "precision.py", "closed_ref.py", "ncu_probe.py")
# The template files every tree starts from, `<op>` substituted (card_template/).
TEMPLATE_FILES = ("README.md", "submit.local.toml.example", ".gitignore", ".gitattributes")
NCU_PROBE_MODE = "cmd:ncu --csv --metrics gpu__time_active.sum python3 eval/ncu_probe.py"
JOB_TIMEOUT_S = 1800
REF_JOB_TIMEOUT_S = 2700            # the reference tree's jobs (recon probes run longer)
FIRST_SEED = 1000
BRANCH = "main"


def copy_template(op_dir: Path, op: str) -> list[Path]:
    """The template files into a new tree ``op_dir`` (``<op>`` -> ``op``)."""
    op_dir = Path(op_dir)
    created = []
    for name in TEMPLATE_FILES:
        text = (template_dir() / name).read_text(encoding="utf-8").replace("<op>", op)
        dst = op_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        created.append(dst)
    return created


def ref_tree(op: str) -> str:
    return f"_{op}ref"


def kernel_file(task) -> str:
    return KERNEL_FILE[task.language]


def card_problem(task, check, *, reference_impl: dict | None = None,
                 tolerance: dict | None = None) -> dict:
    """problem.json for ``task`` judged by ``check`` (forgekernel.judge.Check)."""
    axes = list(task.shapes[0].axes)
    definition = {
        "name": task.name,
        "axes": {a: {"type": "var"} for a in axes},
        "inputs": check.definition["inputs"],
        "outputs": check.definition["outputs"],
        "reference": check.judge_src,
        "custom_inputs_entrypoint": "make_inputs",
    }
    workloads = []
    for s in task.shapes:
        groups = [{"name": g.name, "kind": "file", "path": _input_rel(s.name, g)}
                  for g in s.inputs]
        for k, g in enumerate(check.groups.get(s.name, [])):
            groups.append({"name": g["name"], "kind": "generated",
                           "seed": int(g.get("seed", FIRST_SEED + k)),
                           "intent": str(g.get("intent", ""))})
        workloads.append({"uuid": s.name, "axes": dict(s.axes), "input_groups": groups})
    out = {
        "schema": 1,
        "suite": SUITE,
        "definition": definition,
        "workload": workloads[0],
        "dps": False,
        "kernel_file": kernel_file(task),
        "entry": "run",
        "reference_impl": reference_impl or {"kind": "definition",
                                             "name": "the judge (not timed)"},
        "time_reference": reference_impl is not None,
        "timing": task.target.timing,
        "tolerance": {"outputs": dict(tolerance or {}),
                      "rule": "max|y - judge| / max|judge| per output, never below two units "
                              "of the output dtype's rounding"},
        "source": {"task": task.name, "upstream": "forgekernel task",
                   "dataset": "the task's own shapes and input groups"},
    }
    if len(workloads) > 1:
        out["group"] = [{"index": i, "name": w["uuid"], "workload": w}
                        for i, w in enumerate(workloads)]
        # the loop's meta tools (lever pricing, stall / SASS pricing, the timeline probe)
        # read the shapes named here one at a time: every shape, not a sample of them
        out["meta_workloads"] = [w["uuid"] for w in workloads]
    return out


def ncu_launch_count(problem: dict) -> int:
    """NCU's launch cap for a card whose profile window spreads its calls over every input
    group of every shape (eval/leaderboard.py ncu_window)."""
    pairs = sum(len(w["workload"]["input_groups"]) for w in problem.get("group") or []) \
        or len(problem["workload"]["input_groups"])
    return min(256, max(24, 8 * pairs))


def _input_rel(shape: str, group) -> str:
    return f"inputs/{shape}/{group.name}{group.file.suffix or '.safetensors'}"


def update_problem(op_dir: Path, **fields) -> dict:
    p = Path(op_dir) / PROBLEM
    problem = json.loads(p.read_text(encoding="utf-8"))
    problem.update(fields)
    p.write_text(json.dumps(problem, indent=2) + "\n", encoding="utf-8")
    return problem


# -- the card -------------------------------------------------------------------------------


def write_card(run, task, check, machine) -> Path:
    """``<run>/operators/<op>/``, committed. The reference is sealed in afterwards
    (``seal_reference``) when the task has one."""
    ops = run.operators_dir
    ops.mkdir(parents=True, exist_ok=True)
    kf = kernel_file(task)
    op_dir = ops / task.name
    if op_dir.exists():
        raise FileExistsError(f"operator tree already exists: {op_dir}")
    copy_template(op_dir, task.name)
    lb_dir = op_dir / LB_DIR
    lb_dir.mkdir(parents=True, exist_ok=True)
    for name in EVAL_FILES:
        shutil.copy2(template_dir() / "eval" / name, op_dir / "eval" / name)
    for s in task.shapes:
        for g in s.inputs:
            dst = lb_dir / _input_rel(s.name, g)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(g.file, dst)
    problem = card_problem(task, check)
    (op_dir / PROBLEM).write_text(json.dumps(problem, indent=2) + "\n", encoding="utf-8")
    fmt = {"name": task.name, "suite": SUITE, "upstream": "the task's own judge and clock",
           "workload": ", ".join(s.name for s in task.shapes),
           "group_note": "Every input group of every shape is judged and timed; the gate is "
                         "the geomean over the groups, then over the shapes.\n"}
    (op_dir / "eval" / "bench.py").write_text(lbmod.BENCH_TEMPLATE.format(**fmt), encoding="utf-8")
    (op_dir / "eval" / "ncu_run.py").write_text(lbmod.NCU_TEMPLATE.format(**fmt), encoding="utf-8")
    entry = op_dir / "gpu_entry.sh"
    text = lbmod.GPU_ENTRY_TEMPLATE.format(
        name=task.name, suite=SUITE, arch=CUTE_ARCH.get(task.arch, task.arch + "a"),
        clock="", mem_clock="", require_lock="0",
        clock_note="A task runs at the GPU's own clocks unless LB_SM_CLOCK_MHZ / "
                   "LB_MEM_CLOCK_MHZ pin one (that needs permission to lock); the eval JSON "
                   "records which (leaderboard.clock).")
    entry.write_text(text.replace("--launch-count 24", f"--launch-count {ncu_launch_count(problem)}"),
                     encoding="utf-8")
    entry.chmod(0o755)
    (op_dir / kf).write_text(render_kernel_stub(task, check), encoding="utf-8")
    (op_dir / "window.toml").write_text(render_window_toml(task, check), encoding="utf-8")
    write_submit(op_dir, task, machine)
    _ignore(op_dir, "loop.toml")
    _git_init(op_dir, f"Chore: generate the {task.name} card from its task")
    return op_dir


def seal_reference(run, task, op_dir: Path) -> dict:
    """Seal the task's reference into the card (timed beside the kernel, never readable)
    and write its recon tree; returns problem.json's reference_impl."""
    ref = task.reference
    root = ref.run_file.parent
    sources = []
    for p in sorted(x for x in root.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        try:
            sources.append({"path": str(p.relative_to(root)), "content": p.read_text(encoding="utf-8")})
        except UnicodeDecodeError:
            continue            # a binary next to the reference: not part of the payload
    sol = {"name": f"{task.name} reference", "definition": task.name,
           "spec": {"languages": ["pytorch"], "destination_passing_style": False,
                    "entry_point": f"{ref.run_file.relative_to(root)}::{ref.run_fn}"},
           "sources": sources}
    errs = sealref.solution_problems(sol, task.name, template_dir() / "eval")
    if errs:
        raise ValueError("the reference cannot be sealed: " + "; ".join(errs))
    module = sealref.module_name(sol)
    impl = sealref.reference_impl(sol, module, f"the task's reference ({ref.run_file.name}:"
                                               f"{ref.run_fn})")
    if ref.deny_modules:
        impl["deny_modules"] = list(ref.deny_modules)
    update_problem(op_dir, reference_impl=impl, time_reference=True)
    sealref.write_trees(op_dir, run.operators_dir / ref_tree(task.name), sol, module,
                        template_dir() / "eval", kernel_file=kernel_file(task),
                        deny_modules=ref.deny_modules)
    for tree in (op_dir, run.operators_dir / ref_tree(task.name)):
        if (tree / ".git").exists():
            _commit(tree, f"Chore: seal the {task.name} reference")
        else:
            _git_init(tree, f"Chore: seal the {task.name} reference")
    return impl


def set_tolerance(op_dir: Path, tolerance: dict, how: str) -> None:
    problem = json.loads((Path(op_dir) / PROBLEM).read_text(encoding="utf-8"))
    tol = dict(problem.get("tolerance") or {})
    tol["outputs"] = dict(tolerance)
    tol["calibrated"] = how
    update_problem(op_dir, tolerance=tol)
    _commit(op_dir, "Chore: freeze the judge's tolerance")
    ref = Path(op_dir).parent / ref_tree(Path(op_dir).name)
    if (ref / PROBLEM).exists():
        shutil.copy2(Path(op_dir) / PROBLEM, ref / PROBLEM)
        _commit(ref, "Chore: freeze the judge's tolerance")


BOARD_DIR = "leaderboard"           # tasks/<op>/leaderboard/{card,ref}


def install_board_trees(run, task, machine) -> Path:
    """A leaderboard task's trees: its frozen card (and recon tree) from the run's snapshot,
    with this machine's submit.toml, committed."""
    lb = task.leaderboard
    op_dir = run.operators_dir / task.name
    run.operators_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(lb.card, op_dir)
    trees = [(op_dir, JOB_TIMEOUT_S)]
    if lb.ref is not None:
        ref = run.operators_dir / ref_tree(task.name)
        shutil.copytree(lb.ref, ref)
        trees.append((ref, REF_JOB_TIMEOUT_S))
    for tree, timeout in trees:
        write_submit(tree, task, machine, timeout=timeout)
        _ignore(tree, "loop.toml")
        _git_init(tree, f"Chore: the {task.name} board problem, from its task")
    return op_dir


def write_board_loop_toml(run, task, op_dir: Path, *, deny: list[str], ncu: bool = True) -> Path:
    """loop.toml for a leaderboard task: the card's own suggestion (loop.toml.example,
    written by `forge init --leaderboard`) with this run's hardware, deny list and NCU."""
    example = Path(op_dir) / "loop.toml.example"
    cfg = tomllib.loads(example.read_text(encoding="utf-8")) if example.is_file() else {}
    lc = dict(cfg.get("loop") or {})
    lc["kb_arch"] = task.arch
    lc["gate_window"] = task.target.timing
    lc["ref_source_paths"] = sorted(dict.fromkeys(
        [str(p) for p in lc.get("ref_source_paths", [])] + list(deny)))
    lc.setdefault("ref_source_urls", [])
    lc.setdefault("ref_from_goal", "off")
    if not ncu:
        lc.update(ncu_available=False, job_env_preflight="warn", timeline_probe="off",
                  stall_pricing=False, orch_census=False)
    lines = ["# GENERATED by `forge run` from the card's loop.toml.example -- not a place to",
             "# configure anything.", "[loop]"]
    lines += [f"{k} = {toml_value(v)}" for k, v in lc.items()]
    p = Path(op_dir) / "loop.toml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_submit(tree: Path, task, machine, timeout: int = JOB_TIMEOUT_S) -> Path:
    """submit.toml: where this tree's GPU jobs run (local.env), rewritten on every launch."""
    lines = [
        "# GENERATED by `forge run` from local.env -- where this tree's GPU jobs run.",
        "[job]", f"timeout = {int(timeout)}",
        "[repo]", f'delivery = "{machine.delivery}"', f'branch = "{BRANCH}"',
        'entry_script = "gpu_entry.sh"',
        "[exec]", f'host = {json.dumps(machine.remote_host)}',
        f'workdir = {json.dumps(machine.remote_workdir)}',
        f'env_file = {json.dumps(machine.remote_env_file)}', f'gpu = "{machine.gpu}"',
    ]
    if machine.workspace:
        from forgekernel.workspace import SshWorkspace
        remote = SshWorkspace(machine.remote_host, machine.remote_workdir)
        lines.append(f'workspace = {json.dumps(remote.run_dir(task.name, tree.parents[1].name))}')
    lines += ["[env]", 'PYTHONUNBUFFERED = "1"',
        f'CUTE_DSL_ARCH = "{CUTE_ARCH.get(task.arch, task.arch + "a")}"',
    ]
    p = Path(tree) / "submit.toml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_loop_toml(run, task, op_dir: Path, *, deny: list[str], ncu: bool = True) -> Path:
    """The loop's configuration for this run: the task's language, hardware and clock,
    how the reference is reached, and what no agent may read (``deny``). Everything else is
    the loop's own default. Gitignored in the card: it names this machine's paths.

    ``ncu=False`` (the machine cannot profile): the run degrades to timing only -- no NCU
    signature is required of the reference, and the channels built on NCU (the timeline
    probe, stall pricing, the orchestration census) are off."""
    lines = [
        "# GENERATED by `forge run` for this run -- not a place to configure anything: the",
        "# task (tasks/<op>/) is the problem, and this file only says how the loop reads it.",
        "[loop]",
        f'language    = "{task.loop_language}"',
        f'kb_arch     = "{task.arch}"',
        f'gate_window = "{task.target.timing}"',
        f"ref_from_goal_hours = {task.budget.reference_hours!r}",
        f"ref_source_paths = {json.dumps(sorted(dict.fromkeys(deny)))}",
        f"ref_source_urls  = {json.dumps(list(task.reference.deny_urls) if task.reference else [])}",
    ]
    shapes = ", ".join(f"{s.name} {json.dumps(s.axes)}" for s in task.shapes)
    if task.mode == MODE_GENERATE:
        lines += [
            'oracle_mode = "source"',
            'recon_op    = ""            # no reference yet: a /goal session writes one',
            'ref_from_goal = "auto"',
            f'ref_shape   = {json.dumps(shapes)}',
            f'ref_hardware = "{task.arch}"',
        ]
    else:
        rt = ref_tree(task.name)
        name = f"the task's reference ({task.reference.run_file.name}:{task.reference.run_fn})"
        lines += [
            f'oracle_mode = "{"blackbox" if task.mode == MODE_BLACKBOX else "source"}"',
        ]
        if task.mode != MODE_BLACKBOX:
            lines.append(f"oracle      = {json.dumps(str(_snapshot_path(run, task, task.reference.source)))}")
        lines += [
            f'recon_op           = "{rt}"',
            'ref_mode           = "sealref"',
            f'ref_submit_op      = "{rt}"',
            "ref_in_job         = true",
            f"ref_name           = {json.dumps(name)}",
            f"ref_shape          = {json.dumps(shapes)}",
            f'ref_hardware       = "{task.arch}"',
            'arch_ref_mode      = "sealref"',
            f'arch_ref_submit_op = "{rt}"',
            f"arch_ref_name      = {json.dumps(name)}",
            f'ref_price_op       = "{rt}"',
            'ref_price_app      = ["python3", "sealref.py", "ncu"]',
            'ref_from_goal      = "off"',
        ]
    if not ncu:
        lines += [
            "# NCU cannot profile on this machine: timing only",
            "ncu_available     = false",
            'job_env_preflight = "warn"',
            'timeline_probe    = "off"',
            "stall_pricing     = false",
            "orch_census       = false",
        ]
    p = Path(op_dir) / "loop.toml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _snapshot_path(run, task, path: Path) -> Path:
    """``path`` (inside the task directory) as it sits in the run's snapshot."""
    return run.task_dir / Path(path).relative_to(task.dir)


# -- rendering ------------------------------------------------------------------------------


def _spec(name: str, spec: dict) -> str:
    shape = spec.get("shape")
    dims = "python scalar" if shape is None else f"[{', '.join(map(str, shape))}]"
    return f"    {name}: {spec.get('dtype')} {dims}"


def render_kernel_stub(task, check) -> str:
    shapes = "\n".join(f"    {s.name}: {json.dumps(s.axes)}" for s in task.shapes)
    ins = "\n".join(_spec(n, s) for n, s in check.definition["inputs"].items())
    outs = "\n".join(_spec(n, s) for n, s in check.definition["outputs"].items())
    if task.language == "cuda":
        params = ", ".join(lbmod._cpp_param(n, s) for n, s in check.definition["inputs"].items())
        comment = "\n".join("// " + line for line in (
            f"{task.name} -- the student's kernel, CUDA C++ (one file, its own torch binding).",
            "", "Value-returning: return the output tensor.", "", "Shapes (one kernel serves all):",
            *shapes.splitlines(), "Inputs:", *ins.splitlines(), "Outputs:", *outs.splitlines(),
            "", "Judged against the task's judge (eval/leaderboard/problem.json, \"reference\")."))
        return f"""{comment}

#include <torch/extension.h>

torch::Tensor run({params}) {{
    TORCH_CHECK(false, "write the kernel");
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &run);
}}
"""
    return f'''"""{task.name} — the student's kernel.

Value-returning: run(<inputs, in order>) returns the outputs (a tensor, or a tuple in the
order below), in the declared dtypes.

Shapes (one kernel serves all of them):
{shapes}
Inputs:
{ins}
Outputs:
{outs}

Judged against the task's judge (eval/leaderboard/problem.json, "reference") on every input
group of every shape, and timed on all of them (eval/leaderboard.py).
"""

import torch  # noqa: F401


def run({", ".join(check.inputs)}):
    raise NotImplementedError("write the kernel")
'''


def render_window_toml(task, check) -> str:
    ins = json.dumps(check.inputs)
    outs = json.dumps(check.outputs)
    return f"""# {task.name} — timed-window contract (GENERATED by `forge run`).
#
# Both sides take the task's inputs and return its outputs: the reference and the
# student's run() have the same boundary by construction, so there is nothing to declare.
# Both are value-returning and allocate their outputs inside the call (torch's caching
# allocator, no kernel).

schema_version = 1

[reference]
window  = "the task's reference, one call per timed sample"
inputs  = {ins}
outputs = {outs}

[student]
window  = "run(), one call per timed sample"
inputs  = {ins}
outputs = {outs}
"""


# -- git ------------------------------------------------------------------------------------


def _git(tree: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(tree), *args], check=True, capture_output=True,
                   env=_git_env())


def _git_env() -> dict:
    import os
    env = dict(os.environ)
    env.setdefault("GIT_AUTHOR_NAME", "forgekernel")
    env.setdefault("GIT_AUTHOR_EMAIL", "forgekernel@localhost")
    env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
    env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
    return env


def _git_init(tree: Path, message: str) -> None:
    _git(tree, "init", "-q", "-b", BRANCH)
    # the student's workspace pushes its branches back into the card
    _git(tree, "config", "receive.denyCurrentBranch", "updateInstead")
    _commit(tree, message)


def _commit(tree: Path, message: str) -> None:
    _git(tree, "add", "-A")
    status = subprocess.run(["git", "-C", str(tree), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    if status.strip():
        _git(tree, "commit", "-q", "-m", message)


def _ignore(tree: Path, *names: str) -> None:
    gi = Path(tree) / ".gitignore"
    text = gi.read_text(encoding="utf-8") if gi.exists() else ""
    have = text.split()
    add = [n for n in names if n not in have]
    if add:
        gi.write_text(text + ("" if not text or text.endswith("\n") else "\n")
                      + "\n".join(add) + "\n", encoding="utf-8")


def clone_workspace(op_dir: Path, ws: Path) -> Path:
    """The student's workspace: a single-branch clone of the card."""
    if not (ws / ".git").exists():
        subprocess.run(["git", "clone", "-q", "--single-branch", "-b", BRANCH, str(op_dir),
                        str(ws)], check=True, capture_output=True, env=_git_env())
    return ws
