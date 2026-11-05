"""Where an offline student may read the DSL's own Python source from.

The student runs without network (loop/agent_policy.py), so the CuTe-DSL API it writes
against has to be readable from disk. ``[loop] dsl_source_dirs`` names that checkout
explicitly; when a run leaves it empty, the student is left to discover ``cutlass.cute``
by trial and error — in one GDN run a large share of the probe failures were spellings
of the 4.6.0 API, not ideas being tested.

This module derives a default for a ``cutedsl`` run from the ``nvidia-cutlass-dsl``
distribution installed in the interpreter running the loop: the wheel family puts the
whole Python DSL under one directory (``nvidia_cutlass_dsl/dsl_packages/``, holding
``cutlass`` and its siblings; a ``.pth`` file adds it to ``sys.path``), and that
directory is what the student needs. (README: install ``nvidia-cutlass-dsl`` next to
ForgeKernel so the student can read the API it writes against.)

Two things keep the default honest:

* **Reference source.** A default that lies inside (or contains) a declared
  ``ref_source_paths`` entry would hand the student the reference through the DSL
  door. Overlap → no default, a note that says so.
* **Explicit wins.** A declared ``dsl_source_dirs`` is used as written, whatever is
  installed here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LANGUAGE_CUTEDSL = "cutedsl"
DIST_SHELL = "nvidia-cutlass-dsl"
DIST_CORE = "nvidia-cutlass-dsl-libs-core"   # the pure-Python DSL body
PROBE_TIMEOUT_S = 60

ORIGIN_EXPLICIT = "explicit"
ORIGIN_DEFAULT = "default"
ORIGIN_NONE = "none"

# Runs in the target interpreter: report the distribution's version and the package
# roots the DSL imports from. Metadata first (no import side effects, works for an
# interpreter that cannot import the DSL, e.g. a CPU-only venv holding the wheel);
# ``import cutlass`` as the fallback for an install without dist-info (a source
# checkout on sys.path).
_PROBE = r"""
import importlib.metadata as m, json, os, sys
out = {"version": None, "roots": [], "how": None}
def _dist(name):
    try:
        return m.distribution(name)
    except m.PackageNotFoundError:
        return None
for name in (%(shell)r, %(core)r):
    d = _dist(name)
    if d is not None and out["version"] is None:
        out["version"] = d.version
core = _dist(%(core)r)
roots = []
if core is not None and core.files:
    for f in core.files:
        parts = f.parts
        if len(parts) >= 3 and parts[-1].endswith(".py") and "dsl_packages" in parts:
            i = parts.index("dsl_packages")
            roots.append(str((core.locate_file(f).resolve().parents[len(parts) - i - 2])))
    out["how"] = "metadata"
if not roots:
    try:
        import cutlass
        roots = [str(os.path.dirname(os.path.abspath(p))) for p in cutlass.__path__]
        out["how"] = "import"
    except Exception:
        pass
seen = []
for r in roots:
    if r not in seen:
        seen.append(r)
out["roots"] = seen
print(json.dumps(out))
"""


@dataclass(frozen=True)
class DslSource:
    """The resolved ``dsl_source_dirs`` and where they came from.

    ``origin``: ``explicit`` (declared in loop.toml), ``default`` (derived from an
    installed distribution), ``none`` (nothing usable). ``version`` is the installed
    distribution's version when one was found, whether or not it was accepted.
    ``notes`` are what the loop prints — one line each, ``[loop]``-prefixed by the
    caller."""

    dirs: tuple[Path, ...] = ()
    origin: str = ORIGIN_NONE
    version: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def defaulted(self) -> bool:
        return self.origin == ORIGIN_DEFAULT

    def describe(self) -> str:
        """One line: the dirs and where they came from."""
        if self.origin == ORIGIN_EXPLICIT:
            return f"explicit {[str(p) for p in self.dirs]}"
        if self.origin == ORIGIN_DEFAULT:
            return (f"default {[str(p) for p in self.dirs]}  "
                    f"({DIST_SHELL} {self.version}, from the installed distribution)")
        why = self.notes[-1] if self.notes else "not declared"
        return f"(none — {why})"


def _abs(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def probe_interpreter(python: str, *, env: dict | None = None,
                      timeout_s: float = PROBE_TIMEOUT_S) -> dict | None:
    """``{"version", "roots", "how"}`` from ``python``, or None when the probe could
    not run (missing interpreter, timeout, unparsable output)."""
    code = _PROBE % {"shell": DIST_SHELL, "core": DIST_CORE}
    try:
        proc = subprocess.run([python, "-c", code], capture_output=True, text=True,
                              timeout=timeout_s, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("roots"), list):
        return None
    return data


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def resolve(cfg: dict, *, language: str, python: str | None = None,
            probe_env: dict | None = None) -> DslSource:
    """The run's ``dsl_source_dirs``: as declared, else derived, else none.

    ``cfg`` is the parsed loop.toml; ``language`` the resolved kernel language;
    ``python`` overrides the interpreter probed (``sys.executable`` by default)."""
    lc = cfg.get("loop", {}) if isinstance(cfg, dict) else {}
    declared = lc.get("dsl_source_dirs") or []
    if isinstance(declared, list) and declared:
        return DslSource(dirs=tuple(_abs(p) for p in declared), origin=ORIGIN_EXPLICIT)
    if language != LANGUAGE_CUTEDSL:
        return DslSource(notes=(f"language={language!r}: no DSL source to default",))

    interp = python or sys.executable
    interp_note = "the loop's interpreter"
    data = probe_interpreter(str(interp), env=probe_env)
    if data is None:
        return DslSource(notes=(
            f"dsl_source_dirs is empty and {DIST_SHELL} could not be located in "
            f"{interp_note} ({interp}) — the student is offline, so it can read the "
            f"DSL's API only from a local checkout named in [loop] dsl_source_dirs",))
    version = data.get("version")
    roots = [Path(r).resolve() for r in data.get("roots", []) if r]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        return DslSource(version=version, notes=(
            f"dsl_source_dirs is empty and {DIST_SHELL} is not installed in "
            f"{interp_note} ({interp}) — the student is offline, so it can read the "
            f"DSL's API only from a local checkout named in [loop] dsl_source_dirs",))

    refs = [_abs(p) for p in (lc.get("ref_source_paths") or [])
            if isinstance(p, str)]
    clash = [(r, ref) for r in roots for ref in refs if _overlaps(r, ref)]
    if clash:
        r, ref = clash[0]
        return DslSource(version=version, notes=(
            f"dsl_source_dirs is empty; the installed {DIST_SHELL} at {r} overlaps the "
            f"declared reference source {ref} — not defaulting (the DSL door must not "
            f"open onto the reference). Declare [loop] dsl_source_dirs explicitly",))

    return DslSource(
        dirs=tuple(roots), origin=ORIGIN_DEFAULT, version=version,
        notes=(f"dsl_source_dirs: defaulted to {[str(r) for r in roots]} "
               f"({DIST_SHELL} {version or 'unversioned'}, {interp_note})",))
