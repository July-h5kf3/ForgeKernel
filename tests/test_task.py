"""forgekernel.task: reading and validating tasks/<op>/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgekernel import task as task_mod
from forgekernel.task import TaskError, load_task

SHAPES = [{"name": "small", "axes": {"m": 128, "n": 256}},
          {"name": "large", "axes": {"m": 4096, "n": 256}}]


def _kb(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for kb in ("hw_kb/cutedsl/sm100", "hw_kb/cutedsl/sm90", "hw_kb/cuda/sm100"):
        (root / kb).mkdir(parents=True, exist_ok=True)
    return root


def _task(tmp_path: Path, toml: str, *, shapes=None, name: str = "rmsnorm",
          description: str = "out = x / rms(x) * w\n", files: dict | None = None) -> Path:
    d = tmp_path / "tasks" / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text(toml)
    if description is not None:
        (d / "description.md").write_text(description)
    if shapes is not False:
        (d / "shapes.json").write_text(json.dumps(SHAPES if shapes is None else shapes))
    for rel, text in (files or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return d


MINIMAL = '[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
WITH_REF = MINIMAL + '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n'
REF_FILES = {"reference/run.py": "def run(x, w):\n    return x\n"}


def test_minimal_task_takes_every_default(tmp_path):
    t = load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path))
    assert t.name == "rmsnorm"
    assert t.language == "cutedsl" and t.loop_language == "cutedsl"
    assert t.arch == "sm100"
    assert t.reference is None and t.mode == task_mod.MODE_GENERATE
    assert t.target.ratio == 1.2 and t.target.ms is None and t.target.timing == "device"
    assert t.teacher.backend == "claude" and t.teacher.model == "claude-opus-5-5"
    assert t.student.effort is None
    assert (t.budget.hours, t.budget.max_meta, t.budget.reference_hours) == (4.0, 3, 8.0)
    assert [s.name for s in t.shapes] == ["small", "large"]
    assert all(s.groups_to_add == 3 for s in t.shapes)
    assert any("no [reference]" in n for n in t.notes)


def test_the_task_overrides_global_config_key_by_key(tmp_path):
    toml = MINIMAL + ('[target]\nms = 0.5\n[agent]\nmodel = "claude-fable-5-1"\ntimeout_s = 600\n'
                      '[agent.student]\neffort = "low"\n[budget]\nhours = 1\n')
    t = load_task(_task(tmp_path, toml), kb_root=_kb(tmp_path))
    assert t.target.ms == 0.5 and t.target.ratio is None and t.target.timing == "device"
    # the task's [agent] beats the global [agent.teacher] key by key, its own role table beats both
    assert (t.teacher.model, t.teacher.effort, t.teacher.timeout_s) == ("claude-fable-5-1",
                                                                         "high", 600)
    assert (t.student.model, t.student.effort, t.student.timeout_s) == ("claude-fable-5-1",
                                                                         "low", 600)
    assert t.teacher.provider == t.student.provider == "native"
    assert (t.budget.hours, t.budget.max_meta) == (1.0, 3)


def test_a_provider_must_be_declared_and_its_key_set(tmp_path, monkeypatch):
    from forgekernel import agents, provider
    gc = tmp_path / "gc"
    gc.mkdir()
    (gc / "backends.toml").write_text('[gw]\nbase_url = "https://gw.example/anthropic"\n'
                                      'wire = "anthropic"\nkey_env = "GW_KEY"\n')
    monkeypatch.setattr(provider, "global_config_dir", lambda: gc)
    with pytest.raises(TaskError, match="provider must be one of"):
        load_task(_task(tmp_path, MINIMAL + '[agent]\nprovider = "nope"\n'), kb_root=_kb(tmp_path))
    t = load_task(_task(tmp_path, MINIMAL + '[agent]\nprovider = "gw"\n', name="gwtask"),
                  kb_root=_kb(tmp_path))
    with pytest.raises(provider.ProviderError, match="GW_KEY"):
        agents.agent_command(t.student, "student", env={})
    cmd = agents.agent_command(t.student, "student", env={"GW_KEY": "secret"})
    assert cmd.startswith("env ANTHROPIC_BASE_URL=https://gw.example/anthropic "
                          "FK_PROVIDER_KEY_ENV=GW_KEY bash ")
    assert "secret" not in cmd and cmd.endswith(" 7200")
    native = load_task(_task(tmp_path, MINIMAL, name="nat"), kb_root=_kb(tmp_path))
    assert agents.agent_command(native.teacher, "teacher").startswith("bash ")


def test_reference_with_source_is_source_mode(tmp_path):
    t = load_task(_task(tmp_path, WITH_REF, files=REF_FILES), kb_root=_kb(tmp_path))
    assert t.mode == task_mod.MODE_SOURCE
    assert t.reference.run_fn == "run"
    assert t.reference.run_file.name == "run.py"
    assert t.reference.source == (tmp_path / "tasks" / "rmsnorm" / "reference").resolve()


def test_reference_without_source_is_blackbox(tmp_path):
    toml = MINIMAL + '[reference]\nrun = "reference/run.py:run"\n'
    t = load_task(_task(tmp_path, toml, files=REF_FILES), kb_root=_kb(tmp_path))
    assert t.mode == task_mod.MODE_BLACKBOX and t.reference.source is None


def test_blackbox_flag_withholds_a_source_on_disk(tmp_path):
    toml = WITH_REF + "blackbox = true\n"
    t = load_task(_task(tmp_path, toml, files=REF_FILES), kb_root=_kb(tmp_path))
    assert t.mode == task_mod.MODE_BLACKBOX


def test_cuda_language_maps_to_the_loop_language(tmp_path):
    toml = '[task]\nlanguage = "cuda"\narch = "sm100"\n'
    t = load_task(_task(tmp_path, toml), kb_root=_kb(tmp_path))
    assert t.loop_language == "cuda-sol"


def test_an_arch_without_a_kb_for_the_language_is_refused(tmp_path):
    toml = '[task]\nlanguage = "cuda"\narch = "sm90"\n'
    with pytest.raises(TaskError, match="no hardware knowledge base"):
        load_task(_task(tmp_path, toml), kb_root=_kb(tmp_path))


def test_every_problem_is_reported_at_once(tmp_path):
    toml = ('[task]\nlanguage = "triton"\narch = "h100"\n'
            '[target]\nratio = 1.5\nms = 0.2\ntiming = "wall"\n'
            '[budget]\nhours = 0\nmax_meta = -1\n'
            '[agent.student]\nbackend = "unknown-agent"\n'
            '[extra]\nx = 1\n')
    with pytest.raises(TaskError) as e:
        load_task(_task(tmp_path, toml, shapes=False, description=None), kb_root=_kb(tmp_path))
    text = "\n".join(e.value.problems)
    for needle in ("language must be one of", "arch must name", "both ratio and ms",
                   "timing must be one of", "hours must be a positive", "max_meta must be",
                   "backend must be one of", "unknown table [extra]",
                   "description.md is missing", "shapes.json is missing"):
        assert needle in text, needle


def test_unknown_keys_are_mistakes(tmp_path):
    toml = MINIMAL + '[target]\nratoi = 1.1\n[agent.teacher]\nmodle = "x"\n'
    with pytest.raises(TaskError) as e:
        load_task(_task(tmp_path, toml), kb_root=_kb(tmp_path))
    assert any("'ratoi'" in p for p in e.value.problems)
    assert any("'modle'" in p for p in e.value.problems)


def test_reference_run_must_name_a_callable_inside_the_task(tmp_path):
    for run, needle in (("python reference/run.py", "must be \"<file>.py:<function>\""),
                        ("../elsewhere.py:run", "leaves the task directory"),
                        ("/abs/run.py:run", "must be relative"),
                        ("reference/missing.py:run", "no such file")):
        d = tmp_path / run.replace("/", "_").replace(" ", "_").replace(":", "_")
        toml = MINIMAL + f'[reference]\nrun = "{run}"\n'
        with pytest.raises(TaskError) as e:
            load_task(_task(d, toml, files=REF_FILES), kb_root=_kb(tmp_path))
        assert any(needle in p for p in e.value.problems), (run, e.value.problems)


def test_shapes_must_share_axes_and_have_unique_names(tmp_path):
    shapes = [{"name": "a", "axes": {"m": 1}}, {"name": "a", "axes": {"m": 2, "k": 3}},
              {"name": "b", "axes": {"m": 0}}]
    with pytest.raises(TaskError) as e:
        load_task(_task(tmp_path, MINIMAL, shapes=shapes), kb_root=_kb(tmp_path))
    text = "\n".join(e.value.problems)
    assert "duplicate shape name 'a'" in text
    assert "differ from the first shape's" in text
    assert "is not a positive integer" in text


def test_input_groups_are_counted_against_the_minimum(tmp_path):
    shapes = [{"name": "a", "axes": {"m": 1},
               "inputs": [{"name": "trace", "file": "inputs/a.safetensors"},
                          {"name": "zeros", "file": "inputs/z.safetensors"}]},
              {"name": "b", "axes": {"m": 2}}]
    files = {"inputs/a.safetensors": "x", "inputs/z.safetensors": "y"}
    t = load_task(_task(tmp_path, MINIMAL, shapes=shapes, files=files), kb_root=_kb(tmp_path))
    a, b = t.shapes
    assert [g.name for g in a.inputs] == ["trace", "zeros"]
    assert a.groups_to_add == 1 and b.groups_to_add == 3
    assert any("shape a: 2 input group(s) given, the teacher adds 1" in n for n in t.notes)


def test_input_group_files_must_exist_inside_the_task(tmp_path):
    shapes = [{"name": "a", "axes": {"m": 1},
               "inputs": [{"name": "gone", "file": "inputs/gone.pt"},
                          {"name": "out", "file": "../out.pt"}]}]
    with pytest.raises(TaskError) as e:
        load_task(_task(tmp_path, MINIMAL, shapes=shapes), kb_root=_kb(tmp_path))
    text = "\n".join(e.value.problems)
    assert "no such file" in text and "leaves the task directory" in text


def test_task_name_comes_from_the_directory(tmp_path):
    with pytest.raises(TaskError, match="not a task name"):
        load_task(_task(tmp_path, MINIMAL, name="_hidden"), kb_root=_kb(tmp_path))


def test_deny_paths_resolve_against_the_task(tmp_path):
    toml = WITH_REF + 'deny = ["../other", "/opt/lib"]\n'
    t = load_task(_task(tmp_path, toml, files=REF_FILES), kb_root=_kb(tmp_path))
    assert (tmp_path / "tasks" / "other").resolve() in t.reference.deny
    assert Path("/opt/lib").resolve() in t.reference.deny


def test_deny_urls_and_modules_are_read(tmp_path):
    toml = (MINIMAL + '[reference]\nrun = "reference/run.py:run"\n'
            'deny_urls = ["github.com/org/RefLib ", "pypi.org/project/ref-lib"]\n'
            'deny_modules = ["reflib", "ref-lib"]\n')
    t = load_task(_task(tmp_path, toml, files=REF_FILES), kb_root=_kb(tmp_path))
    assert t.mode == task_mod.MODE_BLACKBOX
    assert t.reference.deny_urls == ("github.com/org/RefLib", "pypi.org/project/ref-lib")
    assert t.reference.deny_modules == ("reflib", "ref-lib")


@pytest.mark.parametrize("line, msg", [
    ('deny_urls = "github.com/org/RefLib"', "deny_urls must be a list"),
    ('deny_urls = [""]', "deny_urls must be a list"),
    ('deny_modules = ["ref lib"]', "deny_modules must be a list of package names"),
    ('deny_modules = ["reflib|.*"]', "deny_modules must be a list of package names"),
])
def test_bad_deny_urls_and_modules_are_refused(tmp_path, line, msg):
    toml = WITH_REF + line + "\n"
    with pytest.raises(TaskError, match=msg):
        load_task(_task(tmp_path, toml, files=REF_FILES), kb_root=_kb(tmp_path))
