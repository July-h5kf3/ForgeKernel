"""An empty ``[loop] dsl_source_dirs`` on a cutedsl run defaults to the DSL
distribution installed here (loop/dsl_source.py) — checked against the declared
reference source — and the student's policy carries the result. A declared list is
used as written."""

import os
import sys
from pathlib import Path

from forgekernel.loop import agent_policy as ap
from forgekernel.loop import dsl_source as ds

PY = sys.executable


def _fake_install(site: Path, version: str = "4.6.0", *, with_meta: bool = True):
    """A site directory laid out like the nvidia-cutlass-dsl wheel family: one
    ``nvidia_cutlass_dsl/dsl_packages`` root holding ``cutlass`` (and a sibling),
    the shell dist-info carrying the version, libs-core's RECORD naming the files."""
    pkgs = site / "nvidia_cutlass_dsl" / "dsl_packages"
    (pkgs / "cutlass" / "cute").mkdir(parents=True)
    (pkgs / "iket").mkdir()
    (site / "nvidia_cutlass_dsl" / "__init__.py").write_text("")
    (pkgs / "cutlass" / "__init__.py").write_text("")
    (pkgs / "cutlass" / "cute" / "__init__.py").write_text("")
    (pkgs / "iket" / "__init__.py").write_text("")
    if not with_meta:
        return pkgs
    shell = site / f"nvidia_cutlass_dsl-{version}.dist-info"
    shell.mkdir()
    (shell / "METADATA").write_text(f"Metadata-Version: 2.1\nName: nvidia-cutlass-dsl\n"
                                    f"Version: {version}\n")
    (shell / "RECORD").write_text(f"{shell.name}/METADATA,,\n")
    core = site / f"nvidia_cutlass_dsl_libs_core-{version}.dist-info"
    core.mkdir()
    (core / "METADATA").write_text(f"Metadata-Version: 2.1\n"
                                   f"Name: nvidia-cutlass-dsl-libs-core\nVersion: {version}\n")
    rows = ["nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py,,",
            "nvidia_cutlass_dsl/dsl_packages/cutlass/cute/__init__.py,,",
            "nvidia_cutlass_dsl/dsl_packages/iket/__init__.py,,",
            f"{core.name}/METADATA,,"]
    (core / "RECORD").write_text("\n".join(rows) + "\n")
    return pkgs


def _env(site: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(site)
    return env


def _resolve(cfg, site, *, language="cutedsl"):
    return ds.resolve(cfg, language=language, python=PY, probe_env=_env(site))


# -- resolution --------------------------------------------------------------------------

def test_a_declared_list_wins_as_written(tmp_path):
    site = tmp_path / "site"
    _fake_install(site)
    mine = tmp_path / "my-dsl"
    mine.mkdir()
    r = _resolve({"loop": {"dsl_source_dirs": [str(mine)]}}, site)
    assert r.origin == ds.ORIGIN_EXPLICIT and r.dirs == (mine.resolve(),)
    assert not r.defaulted and r.describe().startswith("explicit ")


def test_an_empty_key_defaults_to_the_installed_distribution(tmp_path):
    site = tmp_path / "site"
    pkgs = _fake_install(site, "4.6.0")
    r = _resolve({"loop": {"dsl_source_dirs": []}}, site)
    assert r.defaulted and r.dirs == (pkgs.resolve(),) and r.version == "4.6.0"
    assert r.notes and r.notes[0].startswith("dsl_source_dirs: defaulted to ")
    assert "nvidia-cutlass-dsl 4.6.0" in r.notes[0]
    assert r.describe().startswith("default ")


def test_an_install_overlapping_the_reference_source_is_refused(tmp_path):
    site = tmp_path / "site"
    pkgs = _fake_install(site, "4.6.0")
    r = _resolve({"loop": {"ref_source_paths": [str(pkgs / "cutlass")]}}, site)
    assert r.origin == ds.ORIGIN_NONE and r.dirs == ()
    assert "overlaps the declared reference source" in r.notes[-1]
    # ... in either direction: a reference tree above the install is the same door.
    r2 = _resolve({"loop": {"ref_source_paths": [str(site)]}}, site)
    assert r2.origin == ds.ORIGIN_NONE


def test_nothing_installed_is_the_old_warning(tmp_path):
    site = tmp_path / "empty-site"
    site.mkdir()
    r = _resolve({"loop": {}}, site)
    assert r.origin == ds.ORIGIN_NONE and r.dirs == () and r.version is None
    assert "is not installed" in r.notes[-1] and "dsl_source_dirs" in r.notes[-1]


def test_an_interpreter_that_cannot_run_is_reported_not_raised(tmp_path):
    r = ds.resolve({"loop": {}}, language="cutedsl", python=str(tmp_path / "no-such-python"))
    assert r.origin == ds.ORIGIN_NONE and "could not be located" in r.notes[-1]


def test_another_language_gets_no_dsl_default(tmp_path):
    site = tmp_path / "site"
    _fake_install(site)
    r = _resolve({"loop": {}}, site, language="cuda-cpp")
    assert r.origin == ds.ORIGIN_NONE and r.dirs == ()
    assert "language='cuda-cpp'" in r.notes[-1]


def test_an_install_without_dist_info_is_found_by_import(tmp_path):
    site = tmp_path / "site"
    pkgs = _fake_install(site, with_meta=False)
    # No metadata → no version; the import fallback still finds the package root.
    r = ds.resolve({"loop": {}}, language="cutedsl", python=PY,
                   probe_env={**os.environ, "PYTHONPATH": str(pkgs)})
    assert r.defaulted and r.dirs == (pkgs.resolve(),) and r.version is None


# -- the policy carries it ---------------------------------------------------------------

def test_the_policy_and_notice_carry_the_default(tmp_path):
    site = tmp_path / "site"
    pkgs = _fake_install(site, "4.6.0")
    cfg = {"loop": {"dsl_source_dirs": [], "ref_source_paths": [], "ref_source_urls": []}}
    r = _resolve(cfg, site)
    pol = ap.from_config(cfg, oracle_mode="blackbox", repo_root=tmp_path / "run",
                         teacher_trees=[], dsl_source_dirs_default=r.dirs)
    assert pol.dsl_source_dirs == (pkgs.resolve(),)
    assert str(pkgs.resolve()) in pol.notice("student")
    # Readable: the default is not in any deny list, and only the reference is denied.
    assert pkgs.resolve() not in pol.deny_read("student")
    s = pol.settings("student", tmp_path / "run" / "operators" / "op-ws")
    assert str(pkgs.resolve()) not in s["sandbox"]["filesystem"]["denyRead"]


def test_a_declared_list_is_not_replaced_by_the_default(tmp_path):
    mine = tmp_path / "my-dsl"
    mine.mkdir()
    cfg = {"loop": {"dsl_source_dirs": [str(mine)]}}
    pol = ap.from_config(cfg, oracle_mode="source", repo_root=tmp_path / "run",
                         teacher_trees=[], dsl_source_dirs_default=(tmp_path / "other",))
    assert pol.dsl_source_dirs == (mine.resolve(),)


def test_declaration_warnings_defer_to_the_resolution(tmp_path):
    site = tmp_path / "site"
    pkgs = _fake_install(site, "4.6.2")
    cfg = {"loop": {"dsl_source_dirs": []}}
    # Defaulted: no warning (the loop prints the default as news instead).
    ok = _resolve(cfg, site)
    assert ok.defaulted
    assert not [w for w in ap.declaration_warnings(cfg, ok) if "dsl_source_dirs" in w]
    # Refused: the refusal's own reason is the warning, not the bare "is empty".
    bad = _resolve({"loop": {"dsl_source_dirs": [],
                             "ref_source_paths": [str(pkgs / "cutlass")]}}, site)
    ws = [w for w in ap.declaration_warnings(cfg, bad) if "dsl_source_dirs" in w]
    assert len(ws) == 1 and "overlaps the declared reference source" in ws[0]
    # Without a resolution the old warning stands.
    assert any("dsl_source_dirs is empty" in w for w in ap.declaration_warnings(cfg))
