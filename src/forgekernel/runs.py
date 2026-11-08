"""Runs: one directory per run of a task (``.artifacts/<task>_<id>/``, as in ForgeTrain).

::

    .artifacts/<task>_<id>/
      run.json      what was launched: the task, its snapshot digest, the harness commit,
                    the machine, the run's status
      task/         the task as it was at launch (every later step reads this copy)
      operators/    the trees the loop works on: the student's, the reference's
      ws/           the student's workspace
      run.log       the run's output

A new run never inherits anything from an earlier one; resuming a run reads its own
directory only. The directory is not under version control.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from forgekernel.paths import repo_root

RUNS_ENV = "FORGEKERNEL_RUNS_DIR"
RUNS_DIR = ".artifacts"
RUN_FILE = "run.json"
SNAPSHOT_DIR = "task"
_SNAPSHOT_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".DS_Store")

STATUS_CREATED = "created"
STATUS_RUNNING = "running"
STATUS_STOPPED = "stopped"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


class RunError(ValueError):
    pass


def runs_root() -> Path:
    env = os.environ.get(RUNS_ENV, "").strip()
    return Path(env).expanduser().resolve() if env else repo_root() / RUNS_DIR


@dataclass(frozen=True)
class Run:
    dir: Path
    task_name: str
    id: str

    @property
    def name(self) -> str:
        return self.dir.name

    @property
    def task_dir(self) -> Path:
        return self.dir / SNAPSHOT_DIR

    @property
    def operators_dir(self) -> Path:
        return self.dir / "operators"

    @property
    def ws(self) -> Path:
        return self.dir / "ws"

    @property
    def log(self) -> Path:
        return self.dir / "run.log"

    def record(self) -> dict:
        return json.loads((self.dir / RUN_FILE).read_text(encoding="utf-8"))

    def update(self, **fields) -> dict:
        rec = self.record()
        rec.update(fields)
        rec["updated"] = _now()
        _write_json(self.dir / RUN_FILE, rec)
        return rec


def snapshot_digest(d: Path) -> str:
    """sha256 over the snapshot's relative paths and bytes, in path order."""
    h = hashlib.sha256()
    for p in sorted(x for x in Path(d).rglob("*") if x.is_file()):
        h.update(str(p.relative_to(d)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def new_run(task, *, root: Path | None = None, machine=None, now=time.time) -> Run:
    """Create ``<root>/<task>_<id>/`` with the task snapshotted into it."""
    if root is None:
        root = runs_root()
        if machine is not None and machine.workspace:
            root = root / task.name
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%y%m%d-%H%M%S", time.localtime(now()))
    for _ in range(8):
        rid = f"{stamp}-{secrets.token_hex(2)}"
        d = root / f"{task.name}_{rid}"
        try:
            d.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RunError(f"could not allocate a run directory under {root}")
    shutil.copytree(task.dir, d / SNAPSHOT_DIR, ignore=_SNAPSHOT_IGNORE)
    rec = {
        "task": task.name,
        "task_source": str(task.dir),
        "id": rid,
        "created": _now(),
        "updated": _now(),
        "snapshot_sha256": snapshot_digest(d / SNAPSHOT_DIR),
        "forgekernel_commit": _harness_commit(),
        "machine": ({"delivery": machine.delivery, "gpu": machine.gpu,
                     "host": machine.remote_host or None} if machine is not None else None),
        "status": STATUS_CREATED,
    }
    if machine is not None and machine.workspace:
        rec["remote_workspace"] = {
            "host": machine.remote_host,
            "directory": machine.workspace.run_dir(task.name, d.name),
        }
    _write_json(d / RUN_FILE, rec)
    return Run(dir=d, task_name=task.name, id=rid)


def open_run(path: str | Path) -> Run:
    """An existing run directory; RunError when ``path`` is not one, or its snapshot no
    longer matches what was launched (a resumed run must solve the same problem)."""
    d = Path(path).expanduser().resolve()
    f = d / RUN_FILE
    if not f.is_file():
        raise RunError(f"{d} is not a run directory (no {RUN_FILE})")
    try:
        rec = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RunError(f"{f} does not parse: {e}") from None
    for key in ("task", "id", "snapshot_sha256"):
        if not rec.get(key):
            raise RunError(f"{f} has no {key!r}")
    if not (d / SNAPSHOT_DIR).is_dir():
        raise RunError(f"{d} has lost its task snapshot ({SNAPSHOT_DIR}/)")
    if snapshot_digest(d / SNAPSHOT_DIR) != rec["snapshot_sha256"]:
        raise RunError(f"{d / SNAPSHOT_DIR} was edited after launch; a run cannot change its "
                       f"task -- start a new run instead")
    return Run(dir=d, task_name=rec["task"], id=rec["id"])


def list_runs(root: Path | None = None) -> list[Run]:
    root = Path(root) if root is not None else runs_root()
    out = []
    if root.is_dir():
        records = set(root.glob("*/run.json")) | set(root.glob("*/*/run.json"))
        for d in sorted(p.parent for p in records):
            if (d / RUN_FILE).is_file():
                try:
                    rec = json.loads((d / RUN_FILE).read_text(encoding="utf-8"))
                    out.append(Run(dir=d, task_name=rec["task"], id=rec["id"]))
                except (OSError, ValueError, KeyError):
                    continue
    return out


def _harness_commit() -> str | None:
    try:
        p = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        if p.returncode != 0:
            return None
        return p.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
