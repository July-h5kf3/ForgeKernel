"""Deterministic leak lint for ``hw_kb/`` entries.

``hw_kb/`` is the one artifact every later operator's student reads in full, so
a recipe that cites the reference implementation — its file names, internal
function names, line numbers, narrated structure or measured footprints — hands
the reference solution to same-operator students through a side door. An exact
public-repo symbol is worse than a paraphrase: it is a retrieval key into the
student model's pretraining memory of that codebase.

The discipline (see ``docs/kb-provenance/README.md``): entries may cite official
docs, public-infra API surface and this repo's GPU probes; narrated provenance
goes to ``docs/kb-provenance/<entry>.md``, which no agent read set includes.
Prompts drift, so the rule is enforced here mechanically, at three points: loop
start (whole KB tree, catches manual edits), recipe certification (the file the
teacher just wrote, before the student can read it), and the harvest (what a run
writes into the KB when it stops). ``tests/test_kb_lint.py`` keeps the shipped tree clean.

Three checks per file, findings are ``path:line: reason`` strings:

1. banned words — ``oracle`` / ``teacher`` / ``reference implementation``.
   Recipes have no business mentioning the reference at all; rule text that
   must (README, TEMPLATE) is skipped by name. Boundaries are ASCII-only:
   Python's ``\\b`` treats CJK as word characters, so ``oracle`` followed
   directly by a CJK character would slip through a ``\\b``-delimited pattern
   — and an agent may still write CJK prose into an entry.
2. reference tokens — known reference-implementation file names and internal
   symbols (case-insensitive substring). The static list covers references seen
   so far; ``derive_reference_tokens`` extends it mechanically from the resolved
   oracle tree, so a source-mode operator is covered without anyone remembering
   to edit this file. Closed-binary runs still need their recon symbol names
   added to ``REFERENCE_TOKENS`` by hand.
3. line-number citations (``L123`` / ``foo.py:123`` / ``line N``)
   whose own line — or the head line of their citation block — names no
   whitelisted source. A bare line ref is almost always a
   reference-implementation cite with the attribution stripped.
4. experiment narrative — run dates (MMDD), calendar dates, ``iter N``,
   commit / cubin shas, ``<date> <op> run``. A recipe is part of the exam paper;
   what happened on which run belongs in ``docs/kb-provenance/<entry>.md``, the
   commit message and the run records. Inside ``hw_kb/`` a bare job id
   is still allowed: it is the certification pointer the harvest prompt requires.

The same narrative rule covers the other exam surfaces — prompts (``meta/``),
the operator template, the milestone template and the prompt string constants
inside the loop's Python (docstrings and comments are exempt; agents never see
them). ``lint_exam_surfaces`` walks those; ``tests/test_kb_lint.py`` keeps them
clean, and there job ids are banned too (nothing in a prompt should point at
one run).
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Callable
from pathlib import Path

# Files under hw_kb/ that state the rules rather than teach a mechanism. They
# legitimately mention the banned words while forbidding them.
SKIP_NAMES = {"README.md", "TEMPLATE.md"}


def _word(term: str) -> re.Pattern[str]:
    """Case-insensitive match with ASCII-only boundaries: flags ``oracle`` + a CJK character,
    ``oracle_kernel`` and ``ORACLE:`` alike, but not ``coracle``."""
    return re.compile(rf"(?<![A-Za-z0-9]){term}(?![A-Za-z0-9])", re.IGNORECASE)


BANNED_WORDS = [
    (_word("oracle"), "mentions the reference implementation ('oracle')"),
    (_word("teacher"), "mentions the teacher agent"),
    (re.compile(r"(?<![A-Za-z0-9])reference[- ]implementations?(?![A-Za-z0-9])", re.I),
     "mentions the reference implementation"),
]

# File names and internal symbols of known reference implementations.
# Case-insensitive substring match. derive_reference_tokens() extends this from
# the resolved oracle tree for source runs; for closed-binary runs, add the
# recon-report symbol names here when onboarding the operator.
REFERENCE_TOKENS = [
    # flash-attention CuTe DSL ports (attn fwd/bwd references)
    "flash-attention",   # the repo name itself (also blocks github.com/.../flash-attention cites)
    "flash_fwd_sm90",
    "flash_bwd",         # flash_bwd.py, flash_bwd_sm90.py, flash_bwd_postprocess...
    "mainloop_bwd_sm90",
    "flash_attn_dsl",
    "flash_attn_cute",
    "flash_attn/cute",
    "mma_one_n_block",
    "mma_one_m_block",
    "intrawg_overlap",
    "RescaleOBeforeGemm",
    "first_half_block_overlap",
    "last_half_block_overlap",
    # persistent-GEMM / attn-fwd CuTe DSL reference internals. NOT
    # `_make_tma_atoms_and_tensors`: that name is the CUTLASS official
    # dense_gemm example's helper, which entries legitimately cite as public
    # infra — only the reference's own variants are banned.
    "_sm90_compute_tile_shape_or_override",
    "_make_smem_layouts",
    "_make_tma_store_atoms_and_tensors",
    "_get_tiled_mma",
    "_get_smem_layout_atom",
    # CUDA GEMM reference workload
    "workload/src",
    "acc_to_smem_epi_tile",
    # DSA-chain closed references (recon symbol names, added per the
    # closed-binary duty above)
    "flash_mla",
    "flashmla",
    "sparse_attn_fwd_kernel",
    "deep_gemm",
]

# Symbols too generic to be retrieval keys — never derived as tokens.
_DERIVE_STOPLIST = {
    "forward", "backward", "kernel", "kernels", "common", "config", "configs",
    "helpers", "launch", "params", "static", "template", "test", "tests",
    "utils", "wrapper", "interface", "benchmark", "setup", "schedule", "run",
    # CuTe DSL idioms every warp-specialized kernel spells the same way — a KB
    # entry naming them cites the DSL, not the reference.
    "sharedstorage", "namedbarrier",
}
_SOURCE_EXTS = {".py", ".cu", ".cuh", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".h"}
_PY_DEF = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_]\w{5,})", re.MULTILINE)


def derive_reference_tokens(
        oracle: Path | str | None,
        report: Callable[..., None] | None = None) -> list[str]:
    """Reference tokens read mechanically off the resolved oracle tree: source
    file basenames plus (for Python) def/class names. Complements the static
    list so protection does not depend on a human editing it per operator.
    Returns [] when there is no readable source oracle (closed-binary runs).

    ``report`` surfaces the two things lint's startup line should say: every
    oracle path that is *missing* (so a mispointed or outdated oracle cannot
    silently degrade the check to just the static table) and, at the end, one
    summary line with the derived-token count and the number of real paths
    that contributed.  ``report("missing", path)`` is called once per missing
    listed path; ``report("summary", "{n} tokens from {k}/{m} oracle paths")``
    once at the end, only when ``oracle`` resolved to at least one real path."""
    def _emit(kind: str, msg: str) -> None:
        if report is not None:
            report(kind, msg)

    if not oracle:
        return []
    # A source run may list several oracle paths; cli joins them with ", " into
    # one string. Every listed path contributes; a missing one is skipped (and a
    # joined string long enough to trip PATH_MAX must not raise ENAMETOOLONG).
    # A missing path is reported, not swallowed: degrade loudly, not silently.
    parts = [oracle] if isinstance(oracle, Path) else str(oracle).split(", ")
    roots: list[Path] = []
    n_missing = 0
    for part in parts:
        cand = Path(str(part).strip())
        try:
            if cand.exists():
                roots.append(cand)
            else:
                n_missing += 1
                _emit("missing", str(cand))
        except OSError:
            n_missing += 1
            _emit("missing", str(cand))
    if not roots:
        return []
    files: list[Path] = []
    for root in roots:
        files += [root] if root.is_file() else [
            f for f in sorted(root.rglob("*")) if f.suffix in _SOURCE_EXTS]
    tokens: list[str] = []
    seen = set()

    def add(name: str) -> None:
        low = name.lower()
        stem = low.rsplit(".", 1)[0] if "." in low else low
        if low.startswith("__"):          # dunders (__init__, __post_init__, __call__)
            return
        if len(name) >= 6 and stem not in _DERIVE_STOPLIST and low not in seen:
            seen.add(low)
            tokens.append(name)

    for f in files[:500]:
        if f.suffix not in _SOURCE_EXTS:
            continue
        add(f.name)       # flash_bwd.py
        add(f.stem)       # flash_bwd (catches the name cited without extension)
        if f.suffix == ".py":
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for sym in _PY_DEF.findall(text):
                add(sym)
    _emit("summary", "%d tokens from %d/%d oracle paths"
          % (len(tokens), len(roots), len(parts)))
    return tokens


# A citation line containing any of these (or whose block-head line does) is
# talking about official docs, public infra or this repo's own probes — line
# numbers are then fine. Matched per line, not per block: one stray "probe" in
# a block must not bless every unattributed line ref around it.
SOURCE_WHITELIST = [
    "docs.nvidia.com",
    "nvidia.github.io",
    "forums.developer.nvidia.com",
    "PTX ISA",
    "cutlass",       # covers cutlass/, nvidia-cutlass-dsl, cutlass_dsl
    "CUTLASS",
    "cute/",         # include/cute/... headers
    "CuTeDSL",
    "cpasync",       # cutlass.cute.nvgpu.cpasync submodule, often cited bare
    "quack",
    "third_party",
    "examples/python",
    "probe_",        # WORKSPACE/notes/probe_*.py — the path form, deliberately not
    "notes/",        # bare prose "probe": that word must not bless a line ref
    "scripts/",
    "hw_kb/",
    ".md)",          # markdown cross-link to another hw_kb entry on the same line
]

# L123 (single digit deliberately excluded: L1/L2 are cache levels), path:lineno,
# "line N" / "lines N–M". "pipeline 2" is safe: \b needs a boundary.
_LINE_REF = re.compile(
    r"\bL\d{2,}\b"
    r"|\.(?:py|cu|cuh|cc|cpp|cxx|hpp|hh|h):\d+"
    r"|\blines?\s*~?\s*\d+")
_BLOCK_START = re.compile(r"^(?:[-*] |\d+\. |#)")


def _blocks(lines: list[str]):
    """Group lines into citation blocks: a top-level bullet / heading starts a
    block, its continuation lines belong to it. Yields (start_lineno, lines)."""
    start, buf = 1, []
    for i, line in enumerate(lines, start=1):
        if _BLOCK_START.match(line) or not line.strip():
            if buf:
                yield start, buf
            start, buf = i, [line] if line.strip() else []
        else:
            buf.append(line)
    if buf:
        yield start, buf


def _has_source(line: str, head: str) -> bool:
    return any(w in line or w in head for w in SOURCE_WHITELIST)


# Experiment narrative: the tokens by which "what happened on run X" leaks into
# an exam surface. Each pattern is deliberately narrow — the tree carries many
# legitimate 4-digit numbers, so a run date is only MMDD in the 07–09 range
# (later months collide with ordinary numbers like 1024 and are caught through
# the ``<date> … run`` form instead, whose middle token must contain ``_`` or
# ``-`` so "1024 threads run" stays prose while "1015 dsa_sparse run" is flagged),
# a sha needs both a digit and a hex letter, and ``iter N`` needs a digit after it
# (``every eval iter`` as a noun is prose).
NARRATIVE_RULES = [
    (re.compile(r"(?<![0-9A-Za-z.\-_/])0[789][0-3][0-9](?![0-9A-Za-z.])"),
     "run date (MMDD) — narrative belongs in docs/kb-provenance/ or the commit message"),
    (re.compile(r"(?<![0-9])20[0-9]{2}-[01][0-9]-[0-3][0-9](?![0-9])"),
     "calendar date"),
    (re.compile(r"(?<![0-9A-Za-z_])(?:0[789]|1[012])[0-3][0-9]\s*"
                r"(?:[A-Za-z0-9]*[_\-][A-Za-z0-9_\-]*\s+)?(?:arm|run)\b"),
     "run reference (<date> … run)"),
    (re.compile(r"(?<![A-Za-z0-9_])iter\s*[0-9]+(?![0-9])"),
     "iteration number of a past run"),
    (re.compile(r"(?<![A-Za-z0-9_/.\-])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,12}(?![A-Za-z0-9_\-])"),
     "commit / cubin sha"),
]
# 5-12 digits; same width as
# driver._KB_CERT_MARK so what certifies an entry is what a template may not carry.
_JOB_ID = re.compile(r"(?<![A-Za-z0-9_])job\s+\*{0,2}[0-9]{5,12}(?![0-9])", re.IGNORECASE)


def lint_narrative_text(text: str, path: str = "", *,
                        allow_job_ids: bool = False) -> list[str]:
    """Experiment-narrative findings for one surface's content."""
    findings: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        for pat, why in NARRATIVE_RULES:
            if pat.search(line):
                findings.append(f"{path}:{i}: experiment narrative — {why}")
        if not allow_job_ids and _JOB_ID.search(line):
            findings.append(f"{path}:{i}: experiment narrative — job id in a prompt / template")
    return findings


def lint_text(text: str, path: str = "",
              extra_tokens: list[str] | None = None) -> list[str]:
    """All findings for one entry's content."""
    findings: list[str] = []
    lines = text.splitlines()
    tokens = [(t, t.lower()) for t in REFERENCE_TOKENS + (extra_tokens or [])]
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        for pat, why in BANNED_WORDS:
            if pat.search(line):
                findings.append(f"{path}:{i}: {why}")
        for tok, tok_low in tokens:
            if tok_low in low:
                findings.append(f"{path}:{i}: reference-implementation token '{tok}'")
    for start, block in _blocks(lines):
        head = block[0]
        for j, line in enumerate(block):
            if _LINE_REF.search(line) and not _has_source(line, head):
                findings.append(
                    f"{path}:{start + j}: line-number citation without a whitelisted "
                    f"source (official docs / public infra / probe) on the line or "
                    f"its block head")
    findings.extend(lint_narrative_text(text, path, allow_job_ids=True))
    return findings


def lint_file(file: Path, rel_to: Path | None = None,
              extra_tokens: list[str] | None = None) -> list[str]:
    label = str(file.relative_to(rel_to)) if rel_to else str(file)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as e:
        return [f"{label}:0: unreadable ({e})"]
    return lint_text(text, label, extra_tokens=extra_tokens)


def lint_kb_dir(kb_dir: Path,
                extra_tokens: list[str] | None = None) -> list[str]:
    """Findings across every recipe in the KB tree (rule files skipped).

    A missing tree is a finding, not a clean pass: a mispointed kb_dir (e.g. a
    shared venv resolving to the wrong worktree) must not certify as leak-free
    a tree that was never scanned."""
    if not kb_dir.is_dir():
        return [f"{kb_dir}:0: kb_dir does not exist — nothing was scanned (fail closed)"]
    findings: list[str] = []
    for file in sorted(kb_dir.rglob("*.md")):
        if file.name in SKIP_NAMES:
            continue
        findings.extend(lint_file(file, rel_to=kb_dir.parent, extra_tokens=extra_tokens))
    return findings


# The other exam surfaces (paths relative to the repo root). Markdown / TOML
# are linted whole; Python only through its non-docstring string constants.
EXAM_SURFACE_DIRS = ("meta", "src/forgekernel/card_template", "skills")
EXAM_SURFACE_FILES = ("docs/milestones-template.md",)
EXAM_SURFACE_PY = ("src/forgekernel", "skills")
# The card template's eval/ code is copied into cards, not rendered into prompts; its
# strings pin upstream versions (commit + date) on purpose. Its Markdown / TOML is linted.
EXAM_SURFACE_PY_SKIP = ("src/forgekernel/card_template",)
_SURFACE_EXTS = {".md", ".toml"}


def _py_prompt_strings(text: str):
    """(lineno, value) for every string constant that is not a docstring."""
    tree = ast.parse(text)
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                doc_ids.add(id(body[0].value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in doc_ids):
            yield node.lineno, node.value


def lint_exam_surfaces(repo_root: Path) -> list[str]:
    """Narrative findings across prompts, templates and the loop's prompt
    strings. Job ids are banned here: a prompt must not point at one run."""
    root = Path(repo_root)
    findings: list[str] = []
    files: list[Path] = []
    for d in EXAM_SURFACE_DIRS:
        base = root / d
        if base.is_dir():
            files.extend(f for f in sorted(base.rglob("*")) if f.suffix in _SURFACE_EXTS)
    files.extend(root / f for f in EXAM_SURFACE_FILES if (root / f).is_file())
    for file in files:
        label = str(file.relative_to(root))
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as e:
            findings.append(f"{label}:0: unreadable ({e})")
            continue
        findings.extend(lint_narrative_text(text, label))
    for d in EXAM_SURFACE_PY:
        base = root / d
        if not base.is_dir():
            continue
        for file in sorted(base.rglob("*.py")):
            label = str(file.relative_to(root))
            if any(label.startswith(s + "/") for s in EXAM_SURFACE_PY_SKIP):
                continue
            try:
                text = file.read_text(encoding="utf-8")
                strings = list(_py_prompt_strings(text))
            except (OSError, SyntaxError) as e:
                findings.append(f"{label}:0: unreadable ({e})")
                continue
            for lineno, value in strings:
                for f in lint_narrative_text(value, label):
                    # re-anchor the finding to the constant's own line span
                    rel = int(f.split(":", 2)[1]) - 1
                    findings.append(f"{label}:{lineno + rel}: {f.split(': ', 1)[1]}")
    return findings


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    surfaces = "--surfaces" in args
    args = [a for a in args if a != "--surfaces"]
    # --oracle <joined-paths>: a source-mode oracle so the CLI can extend the
    # reference-token list the same way the loop does, and say so on the line it
    # prints.  Without it the oracle list is trusted but never exercised.
    oracle = None
    if "--oracle" in args:
        oi = args.index("--oracle")
        if oi + 1 >= len(args):
            print("[kb-lint] error: --oracle needs a value (or the joined oracle paths)")
            return 2
        oracle = args[oi + 1]
        args = args[:oi] + args[oi + 2:]
    kb_dir = Path(args[0]) if args else Path("hw_kb")
    missing: list[str] = []
    summary: list[str] = []

    def _report(kind: str, msg: str) -> None:
        (missing if kind == "missing" else summary).append(msg)

    tokens = derive_reference_tokens(oracle, report=_report) if oracle else []
    findings = lint_kb_dir(kb_dir, extra_tokens=tokens)
    if surfaces:
        findings += lint_exam_surfaces(kb_dir.resolve().parent)
    for m in missing:
        print(f"[kb-lint] warning: oracle path not readable — {m}")
    for s in summary:
        print(f"[kb-lint] derived {s}")
    for f in findings:
        print(f"[kb-lint] {f}")
    if findings:
        print(f"[kb-lint] {len(findings)} finding(s) — reference-implementation "
              f"provenance belongs in docs/kb-provenance/, not in hw_kb/ "
              f"(see docs/kb-provenance/README.md)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
