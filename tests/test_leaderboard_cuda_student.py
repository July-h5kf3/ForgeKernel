"""A CUDA C++ student on a leaderboard card (`forge new ... --kernel-file kernel.cu`, loop
language cuda-sol): the stub, the signature check, the export as the board's cuda_cpp
solution, the card's build flags and fence for a .cu kernel, the `cmd` guard, the loop's
kernel-file default."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from forgekernel import leaderboard as lb
from forgekernel.loop import cli as loop_cli
from forgekernel.scaffold import main as scaffold_main
from test_binref import _guard, solution  # noqa: F401  (fixtures base / sol_root below)
from test_binref import base, sol_root, _no_network  # noqa: F401

STUDENT = """// fk-cuda-cflags: -Xptxas=-v -lineinfo
#include <torch/extension.h>
// a comment naming dlopen and ref_bin is not a reach
__global__ void mul(const float* x, const float* w, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = x[i] * w[i % 64];
}
void run(torch::Tensor hidden_states, torch::Tensor weight, torch::Tensor output) {
    int n = hidden_states.numel();
    mul<<<(n + 255) / 256, 256>>>(hidden_states.data_ptr<float>(), weight.data_ptr<float>(),
                                  output.data_ptr<float>(), n);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""


def cuda_card(base: Path, sol_root: Path, tmp_path: Path) -> tuple[Path, str]:
    from forgekernel import binref
    sp = tmp_path / "solution" / "solution.json"
    sp.parent.mkdir(exist_ok=True)
    sp.write_text(json.dumps(solution()))
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard", "sol-execbench",
                   "--dataset", str(sol_root), "--definition", "L1/001_rms", "--workload", "0",
                   "--no-sol-board", "--binary-ref", str(sp), "--kernel-file", "kernel.cu"])
    return base / "operators" / "rms", binref.module_name(solution())


class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()
    def __call__(self, *a, **k): return _Any()
    def __iter__(self): return iter(())
    def __mro_entries__(self, bases): return (object,)
    def __hash__(self): return id(self)
    def __or__(self, o): return _Any()


@pytest.fixture
def card_lb(base, sol_root, tmp_path, monkeypatch):
    """The card's eval/leaderboard.py, imported with a stand-in torch when there is none."""
    card, module = cuda_card(base, sol_root, tmp_path)
    if importlib.util.find_spec("torch") is None:
        fake = types.ModuleType("torch")
        fake.__getattr__ = lambda n: _Any()
        # a stable version string: the CUDA loader keys its build on it, and a fresh
        # _Any() per read would key two loads of the same source differently
        fake.__version__ = "0.0+fake"
        monkeypatch.setitem(sys.modules, "torch", fake)
    spec = importlib.util.spec_from_file_location("_test_cuda_card_lb", card / "eval" / "leaderboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    problem = json.loads((card / "eval" / "leaderboard" / "problem.json").read_text())
    problem["_dir"] = str(card / "eval" / "leaderboard")
    return mod, problem, card, module


def test_the_card_names_a_cuda_kernel_file_with_a_board_format_stub(base, sol_root, tmp_path) -> None:
    card, _ = cuda_card(base, sol_root, tmp_path)
    problem = json.loads((card / "eval" / "leaderboard" / "problem.json").read_text())
    assert problem["kernel_file"] == "kernel.cu" and not (card / "kernel.py").exists()
    stub = (card / "kernel.cu").read_text()
    assert "void run(torch::Tensor hidden_states, torch::Tensor weight, torch::Tensor output)" in stub
    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)" in stub and 'm.def("run", &run)' in stub
    assert lb.signature_findings(stub, problem) == []
    readme = (card / "README.md").read_text()
    assert "The `run` signature of `kernel.cu`" in readme and "fk-cuda-cflags" in readme
    assert loop_cli._card_kernel_file(card) == "kernel.cu"
    assert loop_cli._card_kernel_file(tmp_path) == "kernel.py"


@pytest.mark.parametrize("edit, why", [
    (lambda s: s.replace("TORCH_EXTENSION_NAME", "my_mod"), "PYBIND11_MODULE"),
    (lambda s: s.replace('m.def("run"', 'm.def("go"'), 'binds no "run"'),
    (lambda s: s.replace("torch::Tensor weight,", "torch::Tensor w,"), "the definition wants"),
    (lambda s: s.replace("void run(", "void go("), "defines no"),
])
def test_the_signature_check_reads_the_cuda_file(base, sol_root, tmp_path, edit, why) -> None:
    card, _ = cuda_card(base, sol_root, tmp_path)
    problem = json.loads((card / "eval" / "leaderboard" / "problem.json").read_text())
    assert lb.signature_findings(STUDENT, problem) == []
    assert any(why in f for f in lb.signature_findings(edit(STUDENT), problem))


def test_export_is_the_boards_cuda_cpp_solution_with_the_cards_flags(base, sol_root, tmp_path) -> None:
    card, _ = cuda_card(base, sol_root, tmp_path)
    (card / "kernel.cu").write_text(STUDENT)
    sol = lb.export_solution(card, name="s", author="a")
    assert sol["spec"]["languages"] == ["cuda_cpp"] and sol["spec"]["entry_point"] == "kernel.cu::run"
    assert sol["spec"]["compile_options"] == {"cuda_cflags": [
        "-O3", "--use_fast_math", "-Xptxas=-v", "-lineinfo", "-gencode=arch=compute_100a,code=sm_100a"]}
    assert sol["sources"] == [{"path": "kernel.cu", "content": STUDENT}]


def test_the_card_builds_with_the_flags_export_writes(card_lb) -> None:
    lbm, problem, card, module = card_lb
    for src in (STUDENT, "int x;\n", "// fk-cuda-cflags: -arch=sm_100a\n"):
        assert lbm.cuda_kernel_flags(src) == lb.cuda_kernel_flags(src)
    assert lbm.CUDA_DEFAULT_CFLAGS == lb.CUDA_DEFAULT_CFLAGS and lbm.CUDA_GENCODE == lb.CUDA_GENCODE
    with pytest.raises(ValueError, match="not allowed"):
        lbm.cuda_kernel_flags("// fk-cuda-cflags: -Xlinker ../ref_bin/x.so\n")


def test_the_card_loads_a_cuda_kernel_through_the_board_build(card_lb, monkeypatch) -> None:
    lbm, problem, card, module = card_lb
    (card / "kernel.cu").write_text(STUDENT)
    seen = {}
    fake_ext = types.ModuleType("torch.utils.cpp_extension")

    def load(**kw):
        seen.update(kw)
        return types.SimpleNamespace(run=lambda *a: "ran")
    fake_ext.load = load
    fake_utils = types.ModuleType("torch.utils")
    fake_utils.cpp_extension = fake_ext
    monkeypatch.setitem(sys.modules, "torch.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", fake_ext)
    monkeypatch.setattr(sys.modules["torch"], "utils", fake_utils, raising=False)
    monkeypatch.setenv("FK_KERNEL_BUILD_DIR", str(card.parent / "build"))
    fn, dps = lbm._impl(problem, "kernel")
    assert fn() == "ran" and dps is True
    assert seen["extra_cuda_cflags"] == lb.cuda_kernel_flags(STUDENT) and seen["extra_ldflags"] == ["-lcuda"]
    assert seen["name"].startswith("fk_kernel_") and seen["is_python_module"] is True
    built = Path(seen["sources"][0])
    assert built.read_text() == STUDENT and built.parent.parent == card.parent / "build"


@pytest.mark.parametrize("src, fenced", [
    (STUDENT, False),
    (STUDENT.replace("int n = ", 'void* h = dlopen("x", 2);\n    int n = '), True),
    (STUDENT.replace("int n = ", "auto s = py::module_::import(\"sys\");\n    int n = "), True),
    (STUDENT.replace("int n = ", 'PyObject* o = PyImport_ImportModule("x");\n    int n = '), True),
    (STUDENT.replace("int n = ", 'FILE* f = fopen("../ref_bin/a", "rb");\n    int n = '), True),
    (STUDENT.replace("int n = ", 'const char* e = getenv("FK_REF_KEY");\n    int n = '), True),
    (STUDENT.replace("int n = ", "CUmodule mm; cuModuleLoadData(&mm, nullptr);\n    int n = "), True),
    (STUDENT.replace("int n = ", "#define CAT(a, b) a##b\n    int n = "), True),
    (STUDENT.replace("-lineinfo", "-L../ref_bin"), True),
])
def test_the_cuda_fence(card_lb, src, fenced) -> None:
    lbm, problem, card, module = card_lb
    (card / "kernel.cu").write_text(src)
    assert bool(lbm.kernel_fence(problem)) is fenced, lbm.kernel_fence(problem)
    (card / "kernel.cu").write_text(STUDENT.replace("int n = ", f"// {module}\n    int n = "))
    assert lbm.kernel_fence(problem) is None                       # a comment is not a reach
    (card / "kernel.cu").write_text(STUDENT.replace("int n = ", f'const char* m = "{module}";\n    int n = '))
    assert lbm.kernel_fence(problem)                               # a string is


def test_the_cmd_guard_scans_the_cuda_kernel_and_cpp_probes(base, sol_root, tmp_path) -> None:
    card, module = cuda_card(base, sol_root, tmp_path)
    plain = STUDENT.replace("// a comment naming dlopen and ref_bin is not a reach\n", "")
    (card / "kernel.cu").write_text(plain)
    (card / "notes").mkdir(exist_ok=True)
    (card / "notes" / "probe.cu").write_text("__global__ void k() {}\nint main() { return 0; }\n")
    assert _guard(card, "nvcc -o /tmp/p notes/probe.cu && /tmp/p") == "ALLOWED"
    (card / "notes" / "probe.cu").write_text('#include <dlfcn.h>\nint main() { dlopen("x", 2); }\n')
    assert _guard(card, "nvcc -o /tmp/p notes/probe.cu && /tmp/p") == "REFUSED"
    (card / "notes" / "probe.cu").unlink()
    assert _guard(card, "nvcc -shared kernel.cu") == "ALLOWED"
    (card / "kernel.cu").write_text(plain + f'\nstatic const char* m = "../ref_bin/{module}.so";\n')
    assert _guard(card, "nvcc -shared kernel.cu") == "REFUSED"


def test_a_sealed_card_guards_cpp_probes_and_the_cuda_kernel(base, sol_root, tmp_path) -> None:
    """A sealed source reference's key file sits in both trees for the whole job: a C/C++
    probe (or a CUDA kernel file a `cmd` compiles) that opens it is refused like a Python one."""
    import test_sealref
    sp = tmp_path / "solution" / "solution.json"
    sp.parent.mkdir(exist_ok=True)
    sp.write_text(json.dumps(test_sealref.solution()))
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard", "sol-execbench",
                   "--dataset", str(sol_root), "--definition", "L1/001_rms", "--workload", "0",
                   "--no-sol-board", "--sealed-ref", str(sp), "--kernel-file", "kernel.cu"])
    card, ref = base / "operators" / "rms", base / "operators" / "_rmsref"
    plain = STUDENT.replace("// a comment naming dlopen and ref_bin is not a reach\n", "")
    (card / "kernel.cu").write_text(plain)
    assert _guard(card, "nvcc -shared kernel.cu") == "ALLOWED"
    (card / "kernel.cu").write_text(plain + '\nstatic const char* k = "ref_bin/ref_payload.key";\n')
    assert _guard(card, "nvcc -shared kernel.cu") == "REFUSED"
    (card / "kernel.cu").write_text(plain)
    for tree in (card, ref):
        (tree / "notes").mkdir(exist_ok=True)
        (tree / "notes" / "probe.cu").write_text("__global__ void k() {}\nint main() { return 0; }\n")
        assert _guard(tree, "nvcc -o /tmp/p notes/probe.cu && /tmp/p") == "ALLOWED", tree
        (tree / "notes" / "probe.cu").write_text(
            '#include <cstdio>\nint main() { FILE* f = fopen("../ref_bin/ref_payload.key", "r"); }\n')
        assert _guard(tree, "nvcc -o /tmp/p notes/probe.cu && /tmp/p") == "REFUSED", tree
