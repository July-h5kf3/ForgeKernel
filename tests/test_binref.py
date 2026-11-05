"""forge init --binary-ref: a compiled solution as a leaderboard card's sealed in-job
reference (forgekernel.binref) — the card, the recon tree, the guards, install."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from forgekernel import binref
from forgekernel import leaderboard as lb
from forgekernel.scaffold import main as scaffold_main

REPO_ROOT = Path(__file__).resolve().parents[1]
DEF = {
    "name": "001_rms", "description": "rmsnorm",
    "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 64}},
    "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
               "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"}},
    "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"}},
    "reference": "import torch\n\ndef run(hidden_states, weight):\n    return hidden_states * weight\n",
}
CU = """#include <torch/extension.h>
void run(torch::Tensor x, torch::Tensor w, torch::Tensor out) { out.copy_(x * w); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""


def solution(**over) -> dict:
    sol = {"name": "rms_cu", "definition": "001_rms", "author": "t",
           "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"], "entry_point": "kernel.cu::run",
                    "dependencies": [], "destination_passing_style": True},
           "sources": [{"path": "kernel.cu", "content": CU}]}
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


def new_card(base: Path, sol_root: Path, tmp_path: Path, sol: dict | None = None) -> tuple[Path, Path, str]:
    sp = tmp_path / "solution" / "solution.json"
    sp.parent.mkdir(exist_ok=True)
    sp.write_text(json.dumps(sol or solution()))
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard", "sol-execbench",
                   "--dataset", str(sol_root), "--definition", "L1/001_rms", "--workload", "0",
                   "--no-sol-board", "--binary-ref", str(sp), "--ref-source-path", "/elsewhere/plain"])
    return base / "operators" / "rms", base / "operators" / "_rmsref", binref.module_name(sol or solution())


def test_card_times_a_sealed_binary_reference(base: Path, sol_root: Path, tmp_path: Path) -> None:
    card, ref, module = new_card(base, sol_root, tmp_path)
    problem = json.loads((card / "eval" / "leaderboard" / "problem.json").read_text())
    assert problem["reference_impl"] == {
        "kind": "binary", "name": f"sealed binary reference {module}",
        "path": f"../../ref_bin/{module}.so", "module": module, "entry": "run", "dps": True}
    assert problem["time_reference"] is True           # SOL card, but a real kernel: timed in-job
    assert not [p for p in card.rglob("*") if p.suffix in (".cu", ".cpp", ".h")]
    assert (card / "ref_bin").is_dir()
    assert "loop.toml" in (card / ".gitignore").read_text().split()
    loop = (card / "loop.toml.example").read_text()
    assert 'oracle_mode        = "blackbox"' in loop and 'recon_op           = "_rmsref"' in loop
    assert json.dumps([str(ref / "src"), str(tmp_path / "solution"), "/elsewhere/plain"]) in loop
    assert "sealed binary reference" in (card / "window.toml").read_text()


def test_recon_tree_holds_the_sources_and_a_copy_of_the_judge(base: Path, sol_root: Path,
                                                              tmp_path: Path) -> None:
    card, ref, module = new_card(base, sol_root, tmp_path)
    assert (ref / "src" / "kernel.cu").read_text() == CU
    meta = json.loads((ref / "binref.json").read_text())
    assert meta["module"] == module and meta["sources"] == ["kernel.cu"] and meta["dps"] is True
    assert (ref / "binref.py").read_text() == binref.TOOL.read_text()
    assert (ref / "eval" / "leaderboard.py").read_text() == (card / "eval" / "leaderboard.py").read_text()
    assert json.loads((ref / "eval" / "leaderboard" / "problem.json").read_text())["reference_impl"]["path"] \
        == f"../../ref_bin/{module}.so"
    assert 'ncu_window(which="ref")' in (ref / "eval" / "ncu_run.py").read_text()
    assert "ref_bin/" in (ref / ".gitignore").read_text()
    for tree in (card, ref):
        assert subprocess.run(["bash", "-n", str(tree / "gpu_entry.sh")]).returncode == 0
        assert (tree / "gpu_entry.sh").read_text().count("BLACKBOX_FORBID=") == 1
    from forgekernel.loop.cli import fresh_start_contamination
    assert fresh_start_contamination(card, card / "milestone.md", card / "student" / "status.jsonl",
                                     card / "meta_status.jsonl", ref, evidence_dir=card / "evidence") == []


def test_recon_tree_refuses_every_task_where_the_clock_lock_does_not_hold(
        base: Path, sol_root: Path, tmp_path: Path) -> None:
    """The recon tree inherits the card's clock lock and its refusal: its build / binref /
    ncu / cmd jobs on a node at SM 1155 MHz print the line ForgeKernel withholds."""
    from forgekernel.submit.jobs import clock_lock_refusal
    card, ref, _ = new_card(base, sol_root, tmp_path)
    smi = tmp_path / "fake-smi"
    smi.write_text('#!/bin/bash\ncase "$*" in *clocks.current.graphics*) echo "1155, 3996";; '
                   '*) echo FAKE;; esac\n')
    smi.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LB_")}
    env.update(LB_NVIDIA_SMI=str(smi), LB_CLOCK_SETTLE_S="0", LB_INSTALL_CUPTI="0")
    for task in ("build", "binref", "ncu", "cmd"):
        proc = subprocess.run(["bash", str(ref / "gpu_entry.sh"), task, "true"], cwd=ref,
                              env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 3 and clock_lock_refusal(proc.stdout), (task, proc.stdout)


FAKE_TORCH = """import json, os, sys, types
from pathlib import Path
torch = types.ModuleType("torch"); torch.__version__ = "0"; torch.version = types.SimpleNamespace(cuda="0")
ext = types.ModuleType("torch.utils.cpp_extension")
def load(**kw):
    root = Path(kw["build_directory"])
    kw["_staged"] = {str(f.relative_to(root)): f.read_text() for f in sorted(root.rglob("*")) if f.is_file()}
    Path(os.environ["LOAD_LOG"]).write_text(json.dumps(kw))
    (Path(kw["build_directory"]) / (kw["name"] + ".so")).write_bytes(b"\\x7fELF")
ext.load = load
torch.utils = types.ModuleType("torch.utils"); torch.utils.cpp_extension = ext
sys.modules.update({"torch": torch, "torch.utils": torch.utils, "torch.utils.cpp_extension": ext})
"""


@pytest.mark.parametrize("opts, cuda, ld", [
    (None, [], []),                                                          # no compile_options: no flags
    ({}, ["-O3", "--use_fast_math"], ["-lcuda"]),                            # present: SOL defaults
    ({"cuda_cflags": ["-O2"]}, ["-O2"], ["-lcuda"]),                         # TMA GEMMs: driver API needs -lcuda
    ({"ld_flags": ["-lcublas"]}, ["-O3", "--use_fast_math"], ["-lcublas"]),  # explicit ld_flags replace it
    ({"ld_flags": []}, ["-O3", "--use_fast_math"], []),
])
def test_build_resolves_flags_as_sol_execbench_does(base: Path, sol_root: Path, tmp_path: Path,
                                                    opts, cuda, ld) -> None:
    spec = dict(solution()["spec"])
    if opts is not None:
        spec["compile_options"] = opts
    _, ref, module = new_card(base, sol_root, tmp_path, solution(spec=spec))
    assert json.loads((ref / "binref.json").read_text())["compile_options"] == opts
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "sitecustomize.py").write_text(FAKE_TORCH)
    log = tmp_path / "load.json"
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "fake"), "LOAD_LOG": str(log),
           "BINREF_BUILD_DIR": str(tmp_path / "build")}
    r = subprocess.run([sys.executable, "binref.py", "build"], cwd=ref, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    kw = json.loads(log.read_text())
    assert kw["extra_cuda_cflags"] == ["-gencode=arch=compute_100a,code=sm_100a"] + cuda
    assert kw["extra_ldflags"] == ld and kw["extra_cflags"] == []
    assert f"SO_LDFLAGS {json.dumps(ld)}" in r.stdout and (ref / "ref_bin" / f"{module}.so").exists()



def _fake_build(ref: Path, tmp_path: Path, **env_over) -> tuple[subprocess.CompletedProcess, dict]:
    """`binref.py build` against FAKE_TORCH: (the run, the kwargs cpp_extension.load got)."""
    (tmp_path / "fake").mkdir(exist_ok=True)
    (tmp_path / "fake" / "sitecustomize.py").write_text(FAKE_TORCH)
    log = tmp_path / "load.json"
    if log.exists():
        log.unlink()
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "fake"), "LOAD_LOG": str(log),
           "BINREF_BUILD_DIR": str(tmp_path / "build")}
    for k, v in env_over.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    r = subprocess.run([sys.executable, "binref.py", "build"], cwd=ref, env=env, capture_output=True, text=True)
    return r, (json.loads(log.read_text()) if log.exists() else {})


# A spliced solution: each component's files under cN/, a top-level cN__x.cu that
# wraps its #include in a namespace, and a dispatcher with the binding. Two components both ship a
# kernel.cu: compiled as sources they were two ninja rules for one kernel.cuda.o.
SPLICE_SOURCES = [
    ("c0/kernel.cu", "__global__ void k0() {}\n"),
    ("c0__kernel.cu", "namespace sl_c0 {\n#include \"c0/kernel.cu\"\n}\n"),
    ("c1/kernel.cu", "__global__ void k1() {}\n"),
    ("c1/util.cuh", "#pragma once\n"),
    ("c1/extra.cpp", "int c1_helper() { return 1; }\n"),
    ("c1__kernel.cu", "namespace sl_c1 {\n#include \"c1/kernel.cu\"\n}\n"),
    ("splice_dispatch.cu", CU),
]


def test_build_compiles_only_the_top_level_sources_as_sol_execbench_does(base: Path, sol_root: Path,
                                                                          tmp_path: Path) -> None:
    sol = solution(sources=[{"path": p, "content": c} for p, c in SPLICE_SOURCES])
    sol["spec"]["entry_point"] = "splice_dispatch.cu::run"
    _, ref, module = new_card(base, sol_root, tmp_path, sol)
    r, kw = _fake_build(ref, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    build = tmp_path / "build" / module
    assert kw["sources"] == [str(build / "c0__kernel.cu"), str(build / "c1__kernel.cu"),
                             str(build / "splice_dispatch.cu")]
    # every file is staged for the #includes, and the build root is on the include path
    for rel, content in SPLICE_SOURCES:
        assert kw["_staged"][rel] == content
    assert kw["extra_include_paths"][0] == str(build)
    # ... and gone once the .so is out: the staging dir sits outside the tree's clone dir on a
    # node other trees' jobs may borrow, and holds the sealed sources in the clear
    assert not build.exists() and (ref / "ref_bin" / f"{module}.so").exists()
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "binref.json").write_text(json.dumps({"module": "m", "sources": []}))
    ns: dict = {"__name__": "binref_tool_t", "__file__": str(tmp_path / "meta" / "binref.py")}
    exec(compile(binref.TOOL.read_text(), str(binref.TOOL), "exec"), ns)
    assert ns["compiled_sources"](["a/k.cu", "k.cu", "k.cuh", "b\\x.cpp", "x.c", "d/e/f.cc"]) == ["k.cu", "x.c"]


def test_a_solution_with_no_top_level_source_is_refused() -> None:
    sol = solution(sources=[{"path": "c0/kernel.cu", "content": CU}])
    sol["spec"]["entry_point"] = "c0/kernel.cu::run"
    assert any("top of the solution" in e for e in binref.solution_problems(sol, "001_rms"))
    assert not binref.solution_problems(solution(), "001_rms")


@pytest.mark.parametrize("deps", [[], ["CUTLASS_4_4"]])
def test_build_puts_cutlass_on_the_include_path_whatever_the_dependencies(base: Path, sol_root: Path,
                                                                           tmp_path: Path, deps) -> None:
    """SOL-ExecBench's build_ext: $CUTLASS_DIR (else /usr/local/cutlass) include/ and
    tools/util/include/ on every cuda_cpp build -- spec.dependencies (CUTLASS_4_4 or none) play
    no part. Headers missing there: a warning that names the directory, then the build as SOL's."""
    sol = solution()
    sol["spec"]["dependencies"] = deps
    _, ref, _ = new_card(base, sol_root, tmp_path, sol)
    cut = tmp_path / "cutlass"
    (cut / "include" / "cutlass").mkdir(parents=True, exist_ok=True)
    (cut / "include" / "cutlass" / "cutlass.h").write_text("#pragma once\n")
    r, kw = _fake_build(ref, tmp_path, CUTLASS_DIR=str(cut))
    assert r.returncode == 0, r.stdout + r.stderr
    assert kw["extra_include_paths"][1:] == [f"{cut}/include", f"{cut}/tools/util/include"]
    assert "CUTLASS headers not found" not in r.stdout
    # CUTLASS_DIR unset: SOL's image default; this test host has no headers there
    r, kw = _fake_build(ref, tmp_path, CUTLASS_DIR=None)
    assert r.returncode == 0, r.stdout + r.stderr
    assert kw["extra_include_paths"][1:] == ["/usr/local/cutlass/include",
                                             "/usr/local/cutlass/tools/util/include"]
    if not Path("/usr/local/cutlass/include/cutlass/cutlass.h").is_file():
        assert "CUTLASS headers not found under /usr/local/cutlass/include (CUTLASS_DIR unset" in r.stdout
    r, kw = _fake_build(ref, tmp_path, CUTLASS_DIR=str(tmp_path / "nowhere"))
    assert r.returncode == 0 and f"CUTLASS headers not found under {tmp_path / 'nowhere'}/include" in r.stdout


def _guard(tree: Path, cmd: str) -> str:
    # the guard's block: BLACKBOX_FORBID= through the `}` closing blackbox_guard() (a recon
    # tree's block defines blackbox_soft() first)
    script = ("DEF=$(awk '/^BLACKBOX_FORBID=/{b=\"\";on=1;g=0} on{b=b $0 \"\\n\"} /^blackbox_guard\\(\\)/{g=1} "
              "on&&g&&/^}/{on=0;l=b} END{printf \"%s\", l}' gpu_entry.sh); eval \"$DEF\"; "
              "(blackbox_guard \"$1\"; echo ALLOWED) | tail -1")
    r = subprocess.run(["bash", "-c", script, "guard", cmd], cwd=tree, capture_output=True, text=True)
    return "ALLOWED" if r.stdout.strip() == "ALLOWED" else "REFUSED"


def test_guards(base: Path, sol_root: Path, tmp_path: Path) -> None:
    card, ref, module = new_card(base, sol_root, tmp_path)
    so = f"ref_bin/{module}.so"
    for cmd, want in [("cat src/kernel.cu", "REFUSED"), ("grep run binref.json", "REFUSED"),
                      ("ncu --print-source cuda python3 eval/ncu_run.py", "REFUSED"),
                      (f"cuobjdump -ptx {so}", "REFUSED"),
                      (f"cuobjdump -sass {so}", "ALLOWED"),
                      ("ncu --print-source sass python3 eval/ncu_run.py", "ALLOWED"),
                      ("python3 notes/r1.py", "ALLOWED")]:
        assert _guard(ref, cmd) == want, ("recon tree", cmd)
    for cmd, want in [(f"cuobjdump -sass {so}", "REFUSED"), ("nm -D x.so", "REFUSED"),
                      (f"python3 -c 'import {module}'", "REFUSED"),
                      ("python3 eval/bench.py --mode all", "ALLOWED")]:
        assert _guard(card, cmd) == want, ("card", cmd)


PROBE_PY = '''import os, subprocess
SRC = """#include <cute/tensor.hpp>
#include <cuda_runtime.h>
__global__ void k(unsigned *a) { asm volatile("add.cc.u32 %0, %0, 1;" : "+r"(a[0])); }
"""
os.makedirs("/tmp/r3", exist_ok=True)
open("/tmp/r3/probe.cu", "w").write(SRC)
subprocess.run(["nvcc", "-x", "cu", "-cubin", "-o", "/tmp/r3/probe.cubin", "/tmp/r3/probe.cu"])
subprocess.run(["nvcc", "notes/p_expr.cu", "-o", "/tmp/r3/p"])
'''


@pytest.mark.parametrize("cmd, want", [
    ("nvcc -O3 notes/p_expr.cu -o /tmp/p/a.out", "ALLOWED"),
    ("nvcc -x cu -cubin -o /tmp/p.cubin /tmp/r4/probe.cu", "ALLOWED"),
    ("grep -c cta_group /usr/local/cutlass/include/cute/arch/copy_sm100.hpp", "ALLOWED"),
    ("grep -c x /usr/local/cuda-13.1/include/cuda_runtime.h", "ALLOWED"),
    ("cat /workspace/repo/k.cu", "REFUSED"),                                  # the tree
    ("cat /usr/../workspace/k.cu", "REFUSED"),
    ("ls *.cu", "REFUSED"),
    ("cat ../x.cuh", "REFUSED"),
    ("cat src/x", "REFUSED"),
    ("nvcc notes/kernel.cu", "REFUSED"),                                      # the sealed source's name
])
def test_the_recon_tree_lets_a_cpp_probe_compile_and_still_refuses_a_reach(
        base: Path, sol_root: Path, tmp_path: Path, cmd: str, want: str) -> None:
    _, ref, _ = new_card(base, sol_root, tmp_path)
    assert _guard(ref, cmd) == want, cmd


@pytest.mark.parametrize("name, text, want", [
    ("r3.py", PROBE_PY, "ALLOWED"),
    ("p_expr.cu", "#include <cutlass/cutlass.h>\n#include \"probe_util.cuh\"\n__global__ void k() {}\n", "ALLOWED"),
    ("r3.py", "import glob\nprint(glob.glob('**/*.cu', recursive=True))\n", "REFUSED"),
    ("r3.py", "import os\nprint([f for f in os.listdir('.') if f.endswith('.cuh')])\n", "REFUSED"),
    ("r3.py", "print(open('../kernel.cu').read())\n", "REFUSED"),
    ("r3.py", "print(open('notes/kernel.cu').read())\n", "REFUSED"),
    ("p_expr.cu", "#include \"../src/kernel.cu\"\n", "REFUSED"),
    ("p_expr.cu", "#include \"kernel.cu\"\n", "REFUSED"),
])
def test_the_recon_tree_scans_probe_files_after_taking_their_own_spellings_out(
        base: Path, sol_root: Path, tmp_path: Path, name: str, text: str, want: str) -> None:
    _, ref, _ = new_card(base, sol_root, tmp_path)
    (ref / "notes").mkdir(exist_ok=True)
    (ref / "notes" / name).write_text(text)
    assert _guard(ref, "python3 notes/r1.py") == want, (name, text)


def test_the_loops_own_jobs_pass_both_guards_with_notes_seeded_as_the_loop_seeds_them(
        base: Path, sol_root: Path, tmp_path: Path) -> None:
    """Production shape: the guards scan notes/, and the loop seeds it — the pricing toolkit
    into the student's tree (a clone of the card) and into the recon tree (also the ref-pricing
    tree), the recon toolkit and the shared blackbox-guard.sh into the recon tree. Every job the
    loop itself sends must pass; a probe that reaches for the reference must not."""
    from types import SimpleNamespace
    from forgekernel.loop import driver
    card, ref, module = new_card(base, sol_root, tmp_path)
    stub = SimpleNamespace(commit=lambda *a, **k: None)
    driver._seed_pricing_toolkits(card, stub)
    driver._seed_pricing_toolkits(ref, stub)
    driver._seed_sass_tools(SimpleNamespace(recon_dir=ref, oracle_mode="blackbox"), stub)
    assert driver.BLACKBOX_GUARD_NAME == binref.SEEDED_GUARD and (ref / "notes" / binref.SEEDED_GUARD).is_file()
    loop_cmds = [c.removeprefix("cmd:") for c in (
        driver._ORCH_CENSUS_CMD, driver._LEVER_PRICING_CMD, driver._TIMELINE_CMD,
        driver._JOB_ENV_CHECK_CMD, driver._STALL_PRICE_CMD)]
    # a group card's other meta workloads: the same jobs, one workload each
    loop_cmds += [c.removeprefix("cmd:") + " --workload wl1" for c in (
        driver._LEVER_PRICING_CMD, driver._TIMELINE_CMD, driver._STALL_PRICE_CMD)]
    for cmd in loop_cmds:
        assert _guard(card, cmd) == "ALLOWED", ("card", cmd)
    for cmd in (driver._STALL_PRICE_CMD.removeprefix("cmd:") + " --app python3 binref.py ncu",
                driver._STALL_PRICE_CMD.removeprefix("cmd:") + " --workload wl1 --app python3 binref.py ncu",
                "python3 notes/r1.py"):
        assert _guard(ref, cmd) == "ALLOWED", ("recon tree", cmd)
    (card / "notes" / "r1.py").write_text("import glob\nprint(open(glob.glob('ref_bin/*.so')[0], 'rb').read(64))\n")
    assert _guard(card, "python3 notes/r1.py") == "REFUSED"
    (card / "notes" / "r1.py").write_text(f"import {module}\n")
    assert _guard(card, "python3 notes/r1.py") == "REFUSED"
    (ref / "notes" / "r2.sh").write_text("cat src/kernel.cu\n")
    assert _guard(ref, "bash notes/r2.sh") == "REFUSED"


@pytest.mark.parametrize("over, why", [
    ({"spec": {"languages": ["triton"], "entry_point": "k.py::run"}}, "closed_ref"),
    ({"definition": "002_other"}, "the card for"),
    ({"sources": [{"path": "kernel.cu", "content": "PYBIND11_MODULE(my_mod, m) {}"}]}, "TORCH_EXTENSION_NAME"),
])
def test_unusable_solutions_are_refused(base: Path, sol_root: Path, tmp_path: Path, over, why) -> None:
    with pytest.raises(SystemExit, match=why):
        new_card(base, sol_root, tmp_path, solution(**over))
    assert not (base / "operators" / "rms").exists() and not (base / "operators" / "_rmsref").exists()


def test_install_commits_nothing_without_git_and_records_provenance(base: Path, sol_root: Path,
                                                                    tmp_path: Path) -> None:
    card, _, module = new_card(base, sol_root, tmp_path)
    so = tmp_path / "built.so"
    so.write_bytes(b"\x7fELF fake")
    dst = binref.install(card, so, {"image": "img:1"}, commit=False)
    assert dst == card / "ref_bin" / f"{module}.so" and dst.read_bytes() == so.read_bytes()
    prov = (card / "ref_bin" / "PROVENANCE.txt").read_text()
    assert hashlib.sha256(so.read_bytes()).hexdigest() in prov and "img:1" in prov


def test_install_commits_the_same_binary_into_card_and_recon_tree(base: Path, sol_root: Path,
                                                                  tmp_path: Path, monkeypatch) -> None:
    card, ref, module = new_card(base, sol_root, tmp_path)
    for tree in (card, ref):
        subprocess.run(["git", "init", "-q", str(tree)], check=True)
        subprocess.run(["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", "commit",
                        "-q", "--allow-empty", "-m", "init"], check=True)
    for k, v in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@t"),
                 ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@t")):
        monkeypatch.setenv(k, v)
    so = tmp_path / "built.so"
    so.write_bytes(b"\x7fELF fake")
    monkeypatch.chdir(base)  # relative paths, as the CLI is usually called
    binref.install(Path("operators/rms"), so, commit=True, ref_tree=Path("operators/_rmsref"))
    for tree in (card, ref):  # ref_bin/ and *.so are gitignored in the recon tree: forced in
        files = subprocess.run(["git", "-C", str(tree), "ls-files", "ref_bin"], capture_output=True,
                               text=True, check=True).stdout.split()
        assert files == ["ref_bin/PROVENANCE.txt", f"ref_bin/{module}.so"]
        assert (tree / "ref_bin" / f"{module}.so").read_bytes() == so.read_bytes()


def test_a_failed_build_leaves_no_staged_source_behind(base: Path, sol_root: Path, tmp_path: Path) -> None:
    sol = solution(sources=[{"path": p, "content": c} for p, c in SPLICE_SOURCES])
    sol["spec"]["entry_point"] = "splice_dispatch.cu::run"
    _, ref, module = new_card(base, sol_root, tmp_path, sol)
    (tmp_path / "fake").mkdir(exist_ok=True)
    (tmp_path / "fake" / "sitecustomize.py").write_text(FAKE_TORCH.replace(
        'def load(**kw):', 'def load(**kw):\n    raise RuntimeError("nvcc: error")'))
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "fake"), "LOAD_LOG": str(tmp_path / "l.json"),
           "BINREF_BUILD_DIR": str(tmp_path / "build")}
    r = subprocess.run([sys.executable, "binref.py", "build"], cwd=ref, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "nvcc: error" in r.stderr
    assert not (tmp_path / "build" / module).exists()
