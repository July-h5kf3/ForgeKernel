"""Tests for the KB leak lint: reference-implementation provenance must not
appear inside hw_kb entries (it belongs in docs/kb-provenance/)."""

from __future__ import annotations

from pathlib import Path


from forgekernel.loop.kb_lint import derive_reference_tokens, lint_kb_dir, lint_text

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- the three checks ------------------------------------------------------

def test_banned_words_flagged():
    for word in ("the oracle does X", "the Oracle's spelling", "teacher-only path",
                 "footprint of the reference implementation",
                 "same as the Reference-Implementations do"):
        assert lint_text(word, "e.md"), word


def test_banned_word_is_word_bounded():
    # 'oracular'/'Teachers' should not trip the ASCII-bounded patterns
    assert not lint_text("an oracular statement by teachers\n", "e.md")


def test_banned_word_cjk_and_underscore_adjacency_flagged():
    # Python's \b treats CJK as \w, so a \b-bounded pattern misses a banned
    # word glued to CJK prose. Boundaries must be ASCII-only.
    for s in ("oracle\u7684 per-element segment structure", "teacher\u5728G3 certification",
              "oracle_kernel spelling", "ORACLE\uff1a closed binary"):
        assert lint_text(s, "e.md"), s


def test_reference_tokens_flagged():
    for tok in ("in flash_bwd_sm90.py", "see mma_one_n_block_intrawg_overlap",
                "the RescaleOBeforeGemm variant", "workload/src/gemm/gemm.cu",
                "csrc of github.com/Dao-AILab/flash-attention"):
        assert lint_text(tok, "e.md"), tok


def test_reference_tokens_case_insensitive():
    assert lint_text("the mainloop of Flash_bwd_sm90.py", "e.md")


def test_derive_reference_tokens_from_oracle_tree(tmp_path):
    o = tmp_path / "oracle"
    o.mkdir()
    (o / "ref_persistent_gemm.py").write_text(
        "def _compute_epi_tile_override(x):\n    pass\n\ndef main():\n    pass\n")
    (o / "sched.cuh").write_text("// scheduler\n")
    (o / "run.py").write_text("def run():\n    pass\n")
    toks = derive_reference_tokens(o)
    assert "ref_persistent_gemm.py" in toks and "ref_persistent_gemm" in toks
    assert "_compute_epi_tile_override" in toks
    assert "sched.cuh" in toks
    assert "main" not in toks  # generic/short names are not retrieval keys
    assert "run.py" not in toks
    assert lint_text("- the epilogue extracted from ref_persistent_gemm.py", "e.md",
                     extra_tokens=toks)
    assert not lint_text("- the epilogue extracted from ref_persistent_gemm.py", "e.md")
    assert derive_reference_tokens(None) == []
    assert derive_reference_tokens(tmp_path / "missing") == []


def test_line_ref_without_source_flagged():
    findings = lint_text("- order of the steady-state body: `some_fn` L1410-1477\n", "e.md")
    assert findings and "line-number citation" in findings[0]


def test_line_ref_extra_citation_forms_flagged():
    # forms present verbatim in the migrated provenance content
    for s in ("- the swap face is at lines 469–474\n", "- see line ~236 for the swap\n",
              "- the barrier count at ref.cuh:123\n"):
        findings = lint_text(s, "e.md")
        assert findings and "line-number citation" in findings[0], s


def test_cache_level_names_are_not_line_refs():
    assert not lint_text("- the sector relation between the L2 cache and L1TEX\n", "e.md")


def test_line_ref_with_whitelisted_source_passes():
    for line in (
        "- quack `sm90_utils.gemm` L163 fences unconditionally on every call\n",
        "- `cutlass/cute/core.py` `composition` L120-140\n",
        "- official examples/python/CuTeDSL/hopper/dense_gemm.py L88\n",
        "- GPU-certified: `WORKSPACE/notes/probe_kb_x.py`, contract at L40\n",
        "- see the consumer side in [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md) L30-31\n",
    ):
        assert not lint_text(line, "e.md"), line


def test_whitelist_scoped_to_line_or_block_head():
    # the block head names the source -> its continuation line refs are blessed
    text = ("- public infra module (quack `sm90_utils.py`):\n"
            "  - `make_smem_layout` L14-38 (not staged)\n"
            "  - `gemm` L149-173 (fence order)\n")
    assert not lint_text(text, "e.md")
    # but a whitelist word on a SIBLING line must not bless another line's ref
    bad = ("- mechanism notes:\n"
           "  this entry is already GPU-certified by probe_kb_x.py.\n"
           "  see also the segment structure at new_ref_kernel.py:1518-1552.\n")
    assert lint_text(bad, "e.md")


def test_bare_probe_prose_does_not_bless_a_line_ref():
    # "probe" as prose was a wildcard: nearly every Sources block names a probe, so
    # per-block substring matching made check 3 toothless. Only the path form
    # (probe_ / notes/) whitelists.
    assert lint_text("- probe verified; segment structure at new_ref_kernel.py:1518-1552\n", "e.md")
    assert not lint_text("- GPU-certified probe: `x-ws/notes/probe_kb_y.py` from L42\n", "e.md")


def test_whitelist_does_not_bless_banned_tokens():
    # A reference token stays flagged even next to a whitelisted source.
    assert lint_text("- same as quack: flash_bwd_sm90.py L1497\n", "e.md")


# --- tree walk -------------------------------------------------------------

def test_rule_files_are_skipped(tmp_path):
    kb = tmp_path / "hw_kb"
    (kb / "sm90").mkdir(parents=True)
    (kb / "README.md").write_text("Rule: never cite the oracle.\n")
    (kb / "TEMPLATE.md").write_text("The word teacher must not appear.\n")
    (kb / "sm90" / "ok.md").write_text("# clean recipe\n")
    assert lint_kb_dir(kb) == []


def test_tree_walk_names_the_offending_file(tmp_path):
    kb = tmp_path / "hw_kb"
    (kb / "sm90").mkdir(parents=True)
    (kb / "sm90" / "bad.md").write_text("extracted from the teacher oracle\n")
    findings = lint_kb_dir(kb)
    assert findings and all("sm90/bad.md" in f for f in findings)


def test_shipped_hw_kb_tree_is_clean():
    """Regression: the repo's own KB must stay leak-free."""
    assert lint_kb_dir(REPO_ROOT / "hw_kb") == []


def test_missing_kb_dir_is_a_finding(tmp_path):
    # a mispointed kb_dir must not certify as leak-free a tree it never scanned
    findings = lint_kb_dir(tmp_path / "nope")
    assert findings and "fail closed" in findings[0]


# --- experiment narrative (check 4 + the other exam surfaces) ---------------
from forgekernel.loop.kb_lint import lint_exam_surfaces, lint_narrative_text  # noqa: E402


def test_narrative_arm_date_flagged():
    assert lint_narrative_text("measured (0902 dsa_sparse best tree)", "x.md")


def test_narrative_iteration_and_sha_flagged():
    found = lint_narrative_text("this is the shape from iter 12, on best tree 06b8493", "x.md")
    assert any("iteration" in f for f in found)
    assert any("sha" in f for f in found)


def test_narrative_arm_word_after_date_flagged():
    assert lint_narrative_text("the 1015 dsa_sparse run guessed wrong here", "x.md")
    assert lint_narrative_text("the 1015 run guessed wrong here", "x.md")


def test_narrative_generic_prose_passes():
    ok = ("An observed failure mode: several blackbox runs in a row guessed wrong at the same "
          "spot. Every eval iter is censused once in the NCU job; L2::256B, 0xffffffff, "
          "1024 bytes, version 4.6.1, 0.0902 ms; 1024 threads run the epilogue.")
    assert lint_narrative_text(ok, "x.md") == []


def test_narrative_job_id_policy():
    for line in ("`[PRICED: -6.7%, job 870023]`",        # Hopper pools: 5-7 digits
                 "`[PRICED: -6.7%, job 1789171541]`"):   # Blackwell pool: 10 digits
        assert lint_narrative_text(line, "x.md", allow_job_ids=True) == []
        assert lint_narrative_text(line, "x.md") != [], line


def test_kb_lint_text_includes_narrative_check():
    # hw_kb entries get check 4 through lint_text, with job ids allowed
    assert any("narrative" in f for f in lint_text("on the 0902 blackbox run's best tree", "sm90/x.md"))
    assert lint_text("job 870023, three rounds PASS", "sm90/x.md") == []


def test_exam_surfaces_python_docstrings_exempt(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "p.md").write_text("Generic wording, no dates.\n", encoding="utf-8")
    py = tmp_path / "src" / "forgekernel"
    py.mkdir(parents=True)
    (py / "m.py").write_text(
        '"""The 0902 run docstring may carry a date."""\n'
        "# 0813 bwd run: comments are exempt too\n"
        "def f():\n"
        '    """The iter 12 docstring is exempt too."""\n'
        '    return "prompt body\\nthe second line mentions 0902 dsa_sparse"\n',
        encoding="utf-8")
    found = lint_exam_surfaces(tmp_path)
    assert len(found) == 1 and found[0].startswith("src/forgekernel/m.py:6:")  # 2nd line of the constant


def test_shipped_exam_surfaces_have_no_experiment_narrative():
    """Regression: prompts, templates and the loop's prompt strings carry no
    run dates / iteration numbers / shas / job ids — lessons go in generic
    wording, the evidence chain lives in docs/kb-provenance/."""
    assert lint_exam_surfaces(REPO_ROOT) == []


def test_derive_tokens_from_a_multi_path_oracle_string(tmp_path):
    """cli joins a list-valued [loop].oracle with ", "; every path contributes and
    a joined string longer than PATH_MAX must not raise (regression: ENAMETOOLONG crash)."""
    a = tmp_path / "fused_rule_sm90.py"
    a.write_text("def fused_rule_kernel():\n    pass\n")
    b = tmp_path / "helpers_extra.py"
    b.write_text("class ExtraHelperThing:\n    pass\n")
    long_missing = tmp_path / ("x" * 300 + ".py")
    joined = ", ".join([str(a), str(long_missing), str(b)])
    toks = derive_reference_tokens(joined)
    assert "fused_rule_kernel" in toks and "ExtraHelperThing" in toks
    assert "fused_rule_sm90.py" in toks and "helpers_extra" in toks


def test_derive_tokens_skips_dunders_and_stoplisted_file_names(tmp_path):
    """Generic names read off the oracle must not turn every KB entry into a leak
    (observed: helpers.py / __init__ / SharedStorage / NamedBarrier tripped 99 findings)."""
    (tmp_path / "helpers.py").write_text(
        "class SharedStorage:\n    pass\nclass NamedBarrier:\n    pass\n"
        "def __post_init__(self):\n    pass\ndef specific_thing_here():\n    pass\n")
    (tmp_path / "schedule.py").write_text("def __init__(self):\n    pass\n")
    toks = derive_reference_tokens(tmp_path)
    assert "specific_thing_here" in toks
    for bad in ("helpers.py", "helpers", "schedule.py", "schedule", "__init__",
                "__post_init__", "SharedStorage", "NamedBarrier"):
        assert bad not in toks


def test_derive_tokens_reports_missing_paths_and_a_summary(tmp_path):
    """A source-mode oracle list is all absolute machine paths; when one goes
    stale the lint must say so and say how many tokens it still derived,
    instead of silently shrinking to the static table."""
    a = tmp_path / "fused_rule_sm90.py"
    a.write_text("def fused_rule_kernel():\n    pass\n")
    gone = tmp_path / "moved_elsewhere.py"
    events: list[tuple[str, str]] = []
    toks = derive_reference_tokens(", ".join([str(a), str(gone)]),
                                   report=lambda k, m: events.append((k, m)))
    assert "fused_rule_kernel" in toks
    kinds = [k for k, _ in events]
    assert kinds.count("missing") == 1 and kinds.count("summary") == 1
    assert str(gone) in dict(events)["missing"]
    assert dict(events)["summary"].endswith("from 1/2 oracle paths")
    # no report callable: same tokens, no side effects (the old contract)
    assert derive_reference_tokens(", ".join([str(a), str(gone)])) == toks
    # a null oracle (closed-binary run) reports nothing at all
    silent: list = []
    assert derive_reference_tokens(None, report=lambda k, m: silent.append(k)) == []
    assert silent == []


def test_kb_lint_cli_prints_derived_tokens_and_missing_oracle_paths(tmp_path, capsys):
    from forgekernel.loop.kb_lint import main
    kb = tmp_path / "hw_kb"
    kb.mkdir()
    (kb / "clean.md").write_text("# clean\n\n- cites only the official docs.\n")
    a = tmp_path / "fused_rule_sm90.py"
    a.write_text("def fused_rule_kernel():\n    pass\n")
    gone = tmp_path / "moved_elsewhere.py"
    rc = main([str(kb), "--oracle", ", ".join([str(a), str(gone)])])
    out = capsys.readouterr().out
    assert rc == 0
    assert "warning: oracle path not readable" in out and str(gone) in out
    assert "derived" in out and "from 1/2 oracle paths" in out
    # the derived tokens are actually applied to the lint
    (kb / "leak.md").write_text("# x\n\n- see how fused_rule_kernel spells it.\n")
    rc = main([str(kb), "--oracle", str(a)])
    out = capsys.readouterr().out
    assert rc == 1 and "fused_rule_kernel" in out
