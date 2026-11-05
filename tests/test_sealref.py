"""forge init --sealed-ref: a Python solution (one file or many) as a leaderboard card's
sealed in-job SOURCE reference (forgekernel.sealref) — the card, the recon tree, the key,
the fence, the guards."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from forgekernel import leaderboard as lb
from forgekernel import sealref
from forgekernel.scaffold import main as scaffold_main

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_EVAL = REPO_ROOT / "src" / "forgekernel" / "card_template" / "eval"
DEF = {
    "name": "001_rms", "description": "rmsnorm",
    "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 64}},
    "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
               "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"}},
    "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"}},
    "reference": "import torch\n\ndef run(hidden_states, weight):\n    return hidden_states * weight\n",
}
KERNEL = ("from launch_helpers import launch_the_scaled_product_kernel\n\n\n"
          "def run(hidden_states, weight, output):\n"
          "    launch_the_scaled_product_kernel(hidden_states, weight, output)\n")
HELPERS = ("def launch_the_scaled_product_kernel(hidden_states, weight, output):\n"
           "    output.append(('secret_tiling_choice_for_the_product', hidden_states, weight))\n")


def solution(**over) -> dict:
    sol = {"name": "rms_triton", "definition": "001_rms", "author": "t",
           "spec": {"languages": ["triton"], "target_hardware": ["B200"], "entry_point": "kernel.py::run",
                    "dependencies": ["triton"], "destination_passing_style": True},
           "sources": [{"path": "kernel.py", "content": KERNEL},
                       {"path": "launch_helpers.py", "content": HELPERS}]}
    sol.update(over)
    return sol


@pytest.fixture(autouse=True)
def _no_network(monkeypatch) -> None:
    monkeypatch.setattr(lb, "_http_json", lambda url, timeout=30.0: (_ for _ in ()).throw(OSError("offline")))


@pytest.fixture
def sol_root(tmp_path: Path) -> Path:
    root = tmp_path / "sol" / "benchmark"
    pdir = root / "L1" / "001_rms"
    pdir.mkdir(parents=True)
    (pdir / "definition.json").write_text(json.dumps(DEF))
    (pdir / "workload.jsonl").write_text(json.dumps(
        {"uuid": "u-0", "axes": {"batch_size": 4},
         "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}}}) + "\n")
    return root


@pytest.fixture
def base(tmp_path: Path) -> Path:
    shutil.copytree(REPO_ROOT / "meta", tmp_path / "repo" / "meta")
    shutil.copytree(REPO_ROOT / "src" / "forgekernel" / "card_template", tmp_path / "repo" / "src" / "forgekernel" / "card_template")
    return tmp_path / "repo"


def new_card(base: Path, sol_root: Path, tmp_path: Path, sol: dict | None = None,
             git: bool = False) -> tuple[Path, Path, str]:
    sp = tmp_path / "solution" / "solution.json"
    sp.parent.mkdir(exist_ok=True)
    sp.write_text(json.dumps(sol or solution()))
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                   "sol-execbench", "--dataset", str(sol_root), "--definition", "L1/001_rms",
                   "--workload", "0", "--no-sol-board", "--sealed-ref", str(sp),
                   "--ref-source-path", "/elsewhere/plain"])
    card, ref = base / "operators" / "rms", base / "operators" / "_rmsref"
    if git:
        # what `forge run` does when it installs the frozen trees into a run
        from forgekernel import taskcard
        for tree in (card, ref):
            taskcard._git_init(tree, "install")
    return card, ref, sealref.module_name(sol or solution())


def _closed_ref():
    spec = importlib.util.spec_from_file_location("_test_sealref_closed_ref", TEMPLATE_EVAL / "closed_ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(tree: Path) -> str:
    import tomllib
    return tomllib.loads((tree / "submit.local.toml").read_text())["env"]["FK_REF_KEY"]


def test_card_times_a_sealed_source_reference(base: Path, sol_root: Path, tmp_path: Path) -> None:
    card, ref, module = new_card(base, sol_root, tmp_path)
    problem = json.loads((card / "eval" / "leaderboard" / "problem.json").read_text())
    assert problem["reference_impl"] == {
        "kind": "sealed", "name": f"sealed source reference {module}",
        "path": "../../ref_bin/ref_payload.bin", "module": module, "entry": "run", "dps": True,
        "key_env": "FK_REF_KEY", "languages": ["triton"]}
    assert problem["time_reference"] is True            # SOL card, but a real kernel: timed in-job
    assert (card / "eval" / "closed_ref.py").read_text() == (TEMPLATE_EVAL / "closed_ref.py").read_text()
    loop = (card / "loop.toml.example").read_text()
    assert 'oracle_mode        = "blackbox"' in loop and 'recon_op           = "_rmsref"' in loop
    assert 'ref_mode           = "sealref"' in loop and '["python3", "sealref.py", "ncu"]' in loop
    assert json.dumps([str(tmp_path / "solution"), "/elsewhere/plain"]) in loop
    assert "sealed source reference" in (card / "window.toml").read_text()
    assert "loop.toml" in (card / ".gitignore").read_text().split()


def test_the_payload_opens_with_the_key_of_either_tree_and_no_plaintext_is_written(
        base: Path, sol_root: Path, tmp_path: Path) -> None:
    card, ref, _ = new_card(base, sol_root, tmp_path)
    cr = _closed_ref()
    payload = (card / "ref_bin" / "ref_payload.bin").read_bytes()
    assert (ref / "ref_bin" / "ref_payload.bin").read_bytes() == payload
    key = _key(card)
    assert _key(ref) == key and len(key) == 64
    assert cr.unseal_bundle(payload, key)["files"] == {"kernel.py": KERNEL, "launch_helpers.py": HELPERS}
    with pytest.raises(PermissionError):
        cr.unseal_bundle(payload)
    for tree in (card, ref):
        assert cr.check_tree(str(tree), payload, key) == []
        text = "\n".join(p.read_text(errors="replace") for p in tree.rglob("*") if p.is_file())
        assert "secret_tiling_choice" not in text and "launch_the_scaled_product_kernel" not in text
    meta = json.loads((ref / "sealref.json").read_text())
    assert meta["module"] == sealref.module_name(solution()) and meta["files"] == 2
    assert meta["payload_sha256"] == cr.payload_digest(payload)
    assert (ref / "sealref.py").read_text() == sealref.TOOL.read_text()
    assert (ref / "eval" / "leaderboard.py").read_text() == (card / "eval" / "leaderboard.py").read_text()
    assert 'ncu_window(which="ref")' in (ref / "eval" / "ncu_run.py").read_text()


def test_the_key_is_gitignored_in_both_trees_and_the_payload_committed(
        base: Path, sol_root: Path, tmp_path: Path, monkeypatch) -> None:
    for k, v in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t"),
                 ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t")):
        monkeypatch.setenv(k, v)
    card, ref, _ = new_card(base, sol_root, tmp_path, git=True)
    for tree in (card, ref):
        files = subprocess.run(["git", "-C", str(tree), "ls-files"], capture_output=True, text=True,
                               check=True).stdout.split()
        assert "ref_bin/ref_payload.bin" in files and "submit.local.toml" not in files
        assert (tree / "submit.local.toml").is_file()
    from forgekernel.submit.config import load_config
    for name in ("rms", "_rmsref"):
        (base / "operators" / name / "submit.toml").write_text(
            (base / "operators" / name / "submit.toml").read_text().replace(
                'delivery = "local"', 'delivery = "local"'))
        assert load_config(name, base_dir=base).env["FK_REF_KEY"] == _key(card)


def test_the_agent_policy_denies_both_keys(base: Path, sol_root: Path, tmp_path: Path) -> None:
    from forgekernel.loop import agent_policy
    card, ref, _ = new_card(base, sol_root, tmp_path)
    secrets = agent_policy.secret_files_for(base, "rms", [ref])
    assert card / "submit.local.toml" in secrets and ref / "submit.local.toml" in secrets


def test_recon_tree_gpu_entry_and_fresh_start(base: Path, sol_root: Path, tmp_path: Path) -> None:
    card, ref, _ = new_card(base, sol_root, tmp_path)
    for tree in (card, ref):
        assert subprocess.run(["bash", "-n", str(tree / "gpu_entry.sh")]).returncode == 0
        assert (tree / "gpu_entry.sh").read_text().count("BLACKBOX_FORBID=") == 1
    entry = (ref / "gpu_entry.sh").read_text()
    for tree in (card, ref):   # a Triton reference needs the SOL image's fbtriton launch.h fix
        assert "backends/nvidia/launch.h" in (tree / "gpu_entry.sh").read_text()
    for task in ("check)", "crosscheck)", "bench|ref)", "ncu)", "sealref)", "cmd)"):
        assert f"    {task}" in entry
    assert "binref" not in entry.split('case "$TASK" in')[-1]
    from forgekernel.loop.cli import fresh_start_contamination
    assert fresh_start_contamination(card, card / "milestone.md", card / "student" / "status.jsonl",
                                     card / "meta_status.jsonl", ref, evidence_dir=card / "evidence") == []


def _guard(tree: Path, cmd: str) -> str:
    script = ("DEF=$(awk '/^BLACKBOX_FORBID=/{b=\"\";on=1} on{b=b $0 \"\\n\"} on&&/^}/{on=0;l=b} "
              "END{printf \"%s\", l}' gpu_entry.sh); eval \"$DEF\"; (blackbox_guard \"$1\"; echo ALLOWED) | tail -1")
    r = subprocess.run(["bash", "-c", script, "guard", cmd], cwd=tree, capture_output=True, text=True)
    return "ALLOWED" if r.stdout.strip() == "ALLOWED" else "REFUSED"


def test_guards(base: Path, sol_root: Path, tmp_path: Path) -> None:
    card, ref, module = new_card(base, sol_root, tmp_path)
    refused_both = ["python3 eval/closed_ref.py digest ref_bin/ref_payload.bin", "echo $FK_REF_KEY",
                    "cat /proc/self/environ", "printenv", "python3 -c 'import os; print(os.environ)'",
                    "cat submit.local.toml | grep _REF_KEY", "ls ~/.triton/cache", "cat /tmp/x.ttgir",
                    "TRITON_KERNEL_DUMP=1 python3 eval/bench.py", "python3 -c 'import linecache'",
                    f"python3 -c 'import {module}'", "env", "bash -c set", "compgen -v"]
    for cmd in refused_both:
        assert _guard(card, cmd) == "REFUSED", ("card", cmd)
        assert _guard(ref, cmd) == "REFUSED", ("recon tree", cmd)
    for cmd, want in [("python3 eval/bench.py --mode all", "ALLOWED"), ("python3 notes/r1.py", "ALLOWED"),
                      ("cuobjdump -sass /tmp/a.cubin", "REFUSED"), ("nm -D x.so", "REFUSED")]:
        assert _guard(card, cmd) == want, ("card", cmd)
    for cmd, want in [("python3 sealref.py ncu", "ALLOWED"), ("python3 sealref.py crosscheck", "ALLOWED"),
                      ("ncu --print-source sass python3 eval/ncu_run.py", "ALLOWED"),
                      ("ncu --print-source python python3 eval/ncu_run.py", "REFUSED"),
                      ("ncu --import-source yes python3 eval/ncu_run.py", "REFUSED"),
                      ("python3 notes/r1.py", "ALLOWED")]:
        assert _guard(ref, cmd) == want, ("recon tree", cmd)
    # the card also scans its kernel.py: a `cmd` that imports it runs it outside the fence
    (card / "kernel.py").write_text("import zlib\nKEY = open('submit.local.toml').read()\n"
                                    "P = open('ref_bin/ref_payload.bin').read()\n")
    assert _guard(card, "python3 -c 'import kernel'") == "REFUSED"


def test_the_loops_own_jobs_pass_both_guards_with_notes_seeded_as_the_loop_seeds_them(
        base: Path, sol_root: Path, tmp_path: Path) -> None:
    """As binref's: the loop seeds the pricing toolkit into the student's tree and the
    recon tree, the recon toolkit and blackbox-guard.sh into the recon tree; every job
    the loop itself sends must pass, a probe reaching for the reference must not."""
    from types import SimpleNamespace
    from forgekernel import binref
    from forgekernel.loop import driver
    card, ref, module = new_card(base, sol_root, tmp_path)
    stub = SimpleNamespace(commit=lambda *a, **k: None)
    driver._seed_pricing_toolkits(card, stub)
    driver._seed_pricing_toolkits(ref, stub)
    driver._seed_sass_tools(SimpleNamespace(recon_dir=ref, oracle_mode="blackbox"), stub)
    assert (ref / "notes" / binref.SEEDED_GUARD).is_file()
    for cmd in (driver._ORCH_CENSUS_CMD, driver._LEVER_PRICING_CMD, driver._TIMELINE_CMD,
                driver._JOB_ENV_CHECK_CMD, driver._STALL_PRICE_CMD,
                # a group card's other meta workloads: the same jobs, one workload each
                driver._LEVER_PRICING_CMD + " --workload wl1",
                driver._TIMELINE_CMD + " --workload wl1", driver._STALL_PRICE_CMD + " --workload wl1"):
        assert _guard(card, cmd.removeprefix("cmd:")) == "ALLOWED", ("card", cmd)
    for cmd in (driver._STALL_PRICE_CMD.removeprefix("cmd:") + " --app python3 sealref.py ncu",
                driver._STALL_PRICE_CMD.removeprefix("cmd:") + " --workload wl1 --app python3 sealref.py ncu",
                "python3 notes/r1.py"):
        assert _guard(ref, cmd) == "ALLOWED", ("recon tree", cmd)
    (card / "notes" / "r1.py").write_text("print(open('ref_bin/ref_payload.bin').read(64))\n")
    assert _guard(card, "python3 notes/r1.py") == "REFUSED"
    (ref / "notes" / "r2.py").write_text("import linecache\nprint(linecache.cache)\n")
    assert _guard(ref, "python3 notes/r2.py") == "REFUSED"


def test_the_generic_routes_to_the_key_are_words_of_the_cmd_guard(base: Path, sol_root: Path,
                                                                   tmp_path: Path) -> None:
    card, ref, _ = new_card(base, sol_root, tmp_path)
    for cmd in ("python3 -c \"import os; print(os.getenv('FK_' + 'REF' + '_KEY'))\"",
                "cat /proc/$PPID/cmdline", "cat /proc/self/e?viron", "cat ref_bin/ref_payload.key"):
        assert _guard(card, cmd) == "REFUSED", ("card", cmd)
        assert _guard(ref, cmd) == "REFUSED", ("recon tree", cmd)


FAKE_TIMEOUT = '#!/bin/bash\n# timeout --signal=KILL <s> CMD...\nshift 2\nexec "$@"\n'
DUMP_ENV = ("import json, os, sys\n"
            "json.dump({'env': dict(os.environ), 'argv': sys.argv}, open(sys.argv[1], 'w'))\n")


@pytest.mark.parametrize("tree_name", ["card", "recon"])
def test_the_key_leaves_the_job_environment_before_any_task_runs(base: Path, sol_root: Path,
                                                                 tmp_path: Path, tree_name) -> None:
    """The key arrives in the job environment; gpu_entry.sh moves it into the key file beside
    the payload and re-executes itself without it, so a `cmd` job (or anything it starts)
    inherits none of it — while the loader still finds it."""
    card, ref, _ = new_card(base, sol_root, tmp_path)
    tree = card if tree_name == "card" else ref
    key = _key(tree)
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    for name, text in (("timeout", FAKE_TIMEOUT), ("nvidia-smi", "#!/bin/bash\necho FAKE\n")):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    (tree / "notes").mkdir(exist_ok=True)
    (tree / "notes" / "dump.py").write_text(DUMP_ENV)
    out = tmp_path / "dump.json"
    env = {k: v for k, v in os.environ.items()
           if k not in ("LB_SM_CLOCK_MHZ", "LB_MEM_CLOCK_MHZ", "LB_REQUIRE_CLOCK_LOCK")}
    env.update({"PATH": f"{bin_dir}:{os.environ['PATH']}", "LB_SM_CLOCK_MHZ": "", "LB_CLOCK_SETTLE_S": "0",
                "LB_INSTALL_CUPTI": "0", "TASK_TIMEOUT": "60", "FK_REF_KEY": key})
    proc = subprocess.run(["bash", str(tree / "gpu_entry.sh"), "cmd", f"python3 notes/dump.py {out}"],
                          cwd=tree, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    seen = json.loads(out.read_text())
    assert "FK_REF_KEY" not in seen["env"] and not any(key in v for v in seen["env"].values())
    assert key not in " ".join(seen["argv"]) and key not in proc.stdout + proc.stderr
    kf = tree / sealref.KEY_FILE
    assert kf.read_text().strip() == key and (kf.stat().st_mode & 0o777) == 0o600
    cr = _closed_ref()
    assert cr.key_file(str(tree / sealref.PAYLOAD)) == str(kf)
    assert cr.key_from_file(str(tree / sealref.PAYLOAD)) == key
    assert sealref.KEY_FILE in (tree / ".gitignore").read_text().split()


@pytest.mark.parametrize("over, why", [
    ({"spec": {"languages": ["cuda_cpp"], "entry_point": "k.cu::run"}}, "--binary-ref"),
    ({"definition": "002_other"}, "the card for"),
    ({"spec": {"languages": ["triton"], "entry_point": "kernel.py"}}, "<file>.py::<function>"),
    ({"sources": [{"path": "../kernel.py", "content": "x"}]}, "inside the bundle"),
])
def test_unusable_solutions_are_refused(base: Path, sol_root: Path, tmp_path: Path, over, why) -> None:
    with pytest.raises(SystemExit, match=re.escape(why)):
        new_card(base, sol_root, tmp_path, solution(**over))
    assert not (base / "operators" / "rms").exists() and not (base / "operators" / "_rmsref").exists()


def test_binary_and_sealed_are_exclusive(base: Path, sol_root: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        scaffold_main(["rms", "--base-dir", str(base), "--no-git", "--leaderboard", "sol-execbench",
                       "--dataset", str(sol_root), "--definition", "L1/001_rms", "--workload", "0",
                       "--binary-ref", "a.json", "--sealed-ref", "b.json"])


# --- the job side: the card's judge loads the reference, the fence ------------------

class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()
    def __call__(self, *a, **k): return _Any()
    def __iter__(self): return iter(())
    def __mro_entries__(self, bases): return (object,)
    def __hash__(self): return id(self)
    def __or__(self, o): return _Any()


@pytest.fixture
def card_lb(base: Path, sol_root: Path, tmp_path: Path, monkeypatch):
    """The card's eval/leaderboard.py, imported with a stand-in torch (the fence and the
    loader need none)."""
    card, ref, module = new_card(base, sol_root, tmp_path)
    if importlib.util.find_spec("torch") is None:
        fake = types.ModuleType("torch")
        fake.__getattr__ = lambda n: _Any()
        # a stable version string: the CUDA loader keys its build on it, and a fresh
        # _Any() per read would key two loads of the same source differently
        fake.__version__ = "0.0+fake"
        monkeypatch.setitem(sys.modules, "torch", fake)
    spec = importlib.util.spec_from_file_location("_test_card_lb", card / "eval" / "leaderboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    problem = json.loads((card / "eval" / "leaderboard" / "problem.json").read_text())
    problem["_dir"] = str(card / "eval" / "leaderboard")
    yield mod, problem, card, module
    for name in [m for m in sys.modules if m.startswith(module)]:
        sys.modules.pop(name, None)
    sys.meta_path[:] = [f for f in sys.meta_path if getattr(f, "pkg", None) != module]


def test_the_judge_loads_the_sealed_reference_with_the_job_key_and_drops_it(card_lb, monkeypatch) -> None:
    lbm, problem, card, module = card_lb
    monkeypatch.delenv("FK_REF_KEY", raising=False)
    with pytest.raises(PermissionError, match="FK_REF_KEY"):
        lbm._impl(problem, "ref")
    monkeypatch.setenv("FK_REF_KEY", _key(card))
    fn, dps = lbm._impl(problem, "ref")
    import os
    assert dps is True and "FK_REF_KEY" not in os.environ   # decoded: the key is gone
    out = []
    fn("x", "w", out)
    assert out == [("secret_tiling_choice_for_the_product", "x", "w")]
    assert f"{module}.launch_helpers" in sys.modules and "launch_helpers" not in sys.modules


@pytest.mark.parametrize("src, fenced", [
    ("import torch\n\ndef run(hidden_states, weight, output):\n    output.copy_(hidden_states * weight)\n", False),
    ("import closed_ref\n", True),
    ("P = open('../ref_bin/ref_payload.bin').read()\n", True),
    ("import os\nK = os.environ.get('FK_' + 'REF_KEY')\n", True),
    ("import linecache\n", True),
    ("import sys\nsys.settrace(lambda *a: None)\n", True),
    ("def run(*a):\n    return run.__globals__\n", True),
    ("import glob\nglob.glob('/root/.triton/cache/*/*.ttgir')\n", True),
    ("open('/proc/self/environ').read()\n", True),
    ("# closed_ref in a comment is not a reach\ndef run(*a):\n    pass\n", False),
])
def test_the_fence(card_lb, src, fenced) -> None:
    lbm, problem, card, module = card_lb
    (card / "kernel.py").write_text(src)
    assert bool(lbm.kernel_fence(problem)) is fenced
    (card / "kernel.py").write_text(f"import {module}\n")
    assert lbm.kernel_fence(problem)


def test_the_timeline_probes_preamble_passes_the_sealed_fence(card_lb) -> None:
    lbm, problem, card, module = card_lb
    from forgekernel.skills import timeline_probe
    pre = timeline_probe.stamped_preamble()
    (card / "kernel.py").write_text(pre + "\n\ndef run(*a):\n    pass\n")
    assert lbm.kernel_fence(problem) is None
