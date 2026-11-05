"""The template files every generated operator tree starts from (``taskcard.copy_template``,
``src/forgekernel/card_template/``), and what `git archive` ships of such a tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from forgekernel import taskcard

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_copy_template_writes_the_template_files_with_the_op_name(tmp_path: Path) -> None:
    op = tmp_path / "operators" / "demo_op"
    created = taskcard.copy_template(op, "demo_op")
    assert {p.name for p in created} == set(taskcard.TEMPLATE_FILES) == {
        "README.md", "submit.local.toml.example", ".gitignore", ".gitattributes"}
    assert all(p.is_file() for p in created)
    readme = (op / "README.md").read_text(encoding="utf-8")
    assert "demo_op" in readme and "<op>" not in readme
    # the template holds no file the harness later generates (they would be overwritten)
    for generated in ("window.toml", "submit.toml", "loop.toml", "gpu_entry.sh", "kernel.py"):
        assert not (REPO_ROOT / "src" / "forgekernel" / "card_template" / generated).exists()


def test_no_milestone_is_in_the_template() -> None:
    """The milestone is the distill step's output, not a file to be filled in.

    A placeholder is a liability: if it ever satisfies the §9 gate, _needs_authoring
    goes False and distillation silently skips, leaving the student to optimize against
    literal placeholders. Absence cannot drift into looking authored.
    """
    assert "milestone.md" not in taskcard.TEMPLATE_FILES
    assert not (REPO_ROOT / "src" / "forgekernel" / "card_template" / "milestone.md").exists()


def test_no_operator_specific_leak_in_meta_prompts() -> None:
    """The committed meta/ prompts must be fully operator-agnostic."""
    student = (REPO_ROOT / "meta" / "student_prompt.md").read_text(encoding="utf-8")
    for leak in ["flash_bwd.py", "flash_attn_dsl", "flash-attn-4", "FA4", "FA3", "cuDNN"]:
        assert leak not in student, f"operator-specific leak in meta prompt: {leak}"


def test_the_dispatcher_refuses_unknown_commands(capsys) -> None:
    """The pre-V1 routes (`forge init <op> --kernel-file`, `forge <op> baseline|window|monitor|
    leaderboard|submit`) are gone: `forge init` takes only the op (argparse refuses the rest),
    and an unknown `forge <op> <verb>` prints the usage instead of reaching the submit CLI."""
    from forgekernel.__main__ import main

    with pytest.raises(SystemExit):
        main(["init", "demo_op", "--kernel-file", "foo/bar.py"])
    for verb in ("baseline", "window", "monitor", "leaderboard", "submit"):
        assert main(["demo_op", verb]) == 2
    err = capsys.readouterr().err
    assert "unknown command" in err and "forge new tasks/<op>" in err


# --- job payload: the archive must not carry local-only artifacts -------------
#
# Every job extracts `git archive <branch>` fresh. The student and meta agents read the
# workspace directory directly and the driver reads the archives, so none of that needs
# to reach the GPU machine.

def _members(tar: bytes) -> set:
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        return {m.name for m in tf.getmembers() if m.isfile()}


def _git(op: Path, *a: str) -> bytes:
    import subprocess
    return subprocess.run(["git", "-C", str(op), *a], check=True, capture_output=True).stdout


def test_a_tree_keeps_local_only_artifacts_out_of_the_archive(tmp_path: Path) -> None:
    op = tmp_path / "operators" / "demo_op"
    taskcard.copy_template(op, "demo_op")
    # the artifacts a live run accumulates, none of which the node opens
    (op / "milestone.md").write_text("x" * 4096, encoding="utf-8")
    (op / "meta_status.jsonl").write_text("{}\n", encoding="utf-8")
    (op / "arch_baseline.json").write_text("{}", encoding="utf-8")
    (op / "student").mkdir(exist_ok=True)
    (op / "student" / "status.jsonl").write_text("{}\n", encoding="utf-8")
    # and the things it does: the kernel, the harness, a probe
    (op / "kernel.py").write_text("k = 1\n", encoding="utf-8")
    (op / "submit.toml").write_text("[job]\ntimeout = 1\n", encoding="utf-8")
    (op / "notes").mkdir(exist_ok=True)
    (op / "notes" / "probe_x.py").write_text("p = 1\n", encoding="utf-8")

    _git(op, "init", "-q")
    _git(op, "config", "user.email", "t@t")
    _git(op, "config", "user.name", "t")
    _git(op, "add", "-A")
    _git(op, "commit", "-q", "-m", "t")
    shipped = _members(_git(op, "archive", "--format=tar", "HEAD"))
    for local_only in ("milestone.md", "README.md", "meta_status.jsonl",
                       "arch_baseline.json", "student/status.jsonl"):
        assert local_only not in shipped, f"{local_only} must not ride to the node"
    # probes really do run on the node, so notes/ stays
    assert "notes/probe_x.py" in shipped
    assert "kernel.py" in shipped and "submit.toml" in shipped


def test_an_uncommitted_gitattributes_does_not_take_effect(tmp_path: Path) -> None:
    """`git archive <tree-ish>` reads attributes from the ARCHIVED TREE: an export-ignore
    list added to the working directory is inert until it is committed (which is why the
    harness commits the template before the first job)."""
    op = tmp_path / "operators" / "demo_op"
    taskcard.copy_template(op, "demo_op")
    (op / "milestone.md").write_text("x" * 4096, encoding="utf-8")
    (op / "kernel.py").write_text("k = 1\n", encoding="utf-8")
    attrs = (op / ".gitattributes").read_text(encoding="utf-8")
    (op / ".gitattributes").unlink()

    _git(op, "init", "-q")
    _git(op, "config", "user.email", "t@t")
    _git(op, "config", "user.name", "t")
    _git(op, "add", "-A")
    _git(op, "commit", "-q", "-m", "no attributes yet")

    def shipped():
        return _members(_git(op, "archive", "--format=tar", "HEAD"))

    assert "milestone.md" in shipped()          # nothing excluded yet
    (op / ".gitattributes").write_text(attrs, encoding="utf-8")
    assert "milestone.md" in shipped()          # written but NOT committed: inert
    _git(op, "add", "-A")
    _git(op, "commit", "-q", "-m", "commit the attributes")
    assert "milestone.md" not in shipped()      # committed: now it bites


def test_copy_template_refuses_nothing_but_write_card_refuses_an_existing_tree(tmp_path) -> None:
    """write_card raises on an existing tree (a run never regenerates its card in place)."""
    op = tmp_path / "operators" / "demo_op"
    op.mkdir(parents=True)
    run = type("Run", (), {"operators_dir": tmp_path / "operators", "dir": tmp_path})()
    task = type("Task", (), {"name": "demo_op", "language": "cutedsl"})()
    with pytest.raises(FileExistsError):
        taskcard.write_card(run, task, None, None)
