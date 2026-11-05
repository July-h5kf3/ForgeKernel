"""Accounting for the main-repo artifacts nothing commits.

The teacher writes two things into the *main* repo working tree: a certified
`hw_kb/` recipe (G3) and the per-op milestone (distill/patch). Every git call
the driver makes is scoped to the student workspace (``RealBackend`` runs
``git -C <workspace>``), so neither is ever committed — a recipe survives only
as a dirty file, and an unattended run drops it the moment someone checks out
another branch in that clone.

Two guards, both cheap:

- refuse to start on an unharvested ``hw_kb`` — otherwise a second run stacks
  its recipes on top of the first one's and the pile gets harder to review;
- report what is left behind when the run ends, with the absolute repo root,
  since ``paths.repo_root()`` follows the installed package rather than the cwd and
  the tree that received the recipes is not always the one being watched.

The trajectory files (``status.jsonl`` / ``meta_status.jsonl``) are deliberately
out of scope: the loop appends to them every iteration, so they are always dirty
and would drown the signal.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Change:
    """One porcelain entry: the two-letter status code and the path it applies to."""
    status: str
    path: str

    @property
    def is_new(self) -> bool:
        """Untracked or newly added — i.e. a recipe that exists nowhere else."""
        return self.status in {"??", "A ", "AM"}


def parse_porcelain(text: str) -> list[Change]:
    """Parse ``git status --porcelain`` output into changes.

    Renames (``R  old -> new``) are recorded under their destination path; the
    caller only ever needs to know which file to look at now.
    """
    changes: list[Change] = []
    for line in text.splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:]
        if " -> " in path:  # rename/copy: keep the destination
            path = path.split(" -> ", 1)[1]
        changes.append(Change(status, path.strip('"')))
    return changes


def git_status(root: Path, paths: list[str]) -> list[Change] | None:
    """Porcelain status for ``paths`` under ``root``. None if git cannot answer.

    A missing git, a non-repo root or a bad pathspec must not take the loop
    down: harvesting is bookkeeping, not part of the optimization.
    """
    try:
        proc = subprocess.run(
            ["git", "-c", "core.quotepath=false", "-C", str(root),
             "status", "--porcelain", "--", *paths],
            capture_output=True, text=True, timeout=30.0, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return parse_porcelain(proc.stdout)


def format_report(changes: list[Change], root: Path) -> str:
    """A human-readable block naming the absolute paths still uncommitted."""
    if not changes:
        return "[harvest] nothing left behind in the main repo"
    new = [c for c in changes if c.is_new]
    modified = [c for c in changes if not c.is_new]
    header = (f"[harvest] {len(changes)} uncommitted file(s) in {root} — nothing commits "
              f"these, review and commit them or they are lost:")
    lines = [header]
    for label, group in (("new", new), ("modified", modified)):
        for c in group:
            lines.append(f"[harvest]   {label:8} {c.path}")
    return "\n".join(lines)


def kb_blockers(root: Path, kb_dir: Path) -> list[Change]:
    """Uncommitted entries under ``kb_dir`` — a non-empty list should block a start."""
    try:
        rel = kb_dir.resolve().relative_to(root.resolve())
    except ValueError:  # kb_dir outside the repo (custom layout): nothing to guard
        return []
    return git_status(root, [str(rel)]) or []
