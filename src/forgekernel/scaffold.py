"""``forge init <op> --leaderboard <suite> ...``: the card of one leaderboard problem.

A leaderboard task is one FlashInfer-Bench / SOL-ExecBench problem, judged and timed the
board's way (``forgekernel.leaderboard``, docs/leaderboards.md). This module builds its card
(and, with ``--sealed-ref`` / ``--binary-ref``, the recon tree of its sealed reference) under
``<base>/operators/``; ``forge init`` freezes the result under ``tasks/<op>/leaderboard/`` and
``forge run`` installs it into a run (``forgekernel.taskcard.install_board_trees``). The problem
is resolved before anything is written, so a bad choice leaves no half card.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from forgekernel import taskcard
from forgekernel.paths import repo_root, template_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="forge init <op> --leaderboard",
        description="The card of one leaderboard problem, judged and timed the way the board does.")
    p.add_argument("op", help="operator name (becomes operators/<op>/)")
    p.add_argument("--kernel-file", default="kernel.py",
                   help="the student's kernel file: kernel.py (default) or kernel.cu for a "
                        "CUDA C++ student")
    p.add_argument("--base-dir", default=None,
                   help="where operators/ is written (default: this repository)")
    p.add_argument("--leaderboard", choices=["flashinfer-bench", "sol-execbench"],
                   required=True, help="the board")
    p.add_argument("--dataset", default=None,
                   help="flashinfer-trace clone (flashinfer-bench) / the tree "
                        "download_solexecbench.py writes (sol-execbench)")
    p.add_argument("--definition", default=None,
                   help="definition name (flashinfer-bench) / <subset>/<problem> (sol-execbench)")
    p.add_argument("--workload", default=None,
                   help="workload uuid (or unique prefix) or 0-based index; or a GROUP of "
                        "workloads, comma-separated items each an index, an inclusive lo-hi "
                        "range or a uuid prefix (e.g. 3,5,9 or 0-6): one card judges all of "
                        "them and gates on the geomean of their readings")
    p.add_argument("--check-workload", default=None,
                   help="correctness-only workloads, --workload's syntax (e.g. 8-15): judged on "
                        "every run of the card (all must pass), never timed, never in the gate "
                        "-- workloads the kernel is scored on elsewhere (the rest of a size group "
                        "wider than the card times); ones --workload already names are dropped")
    p.add_argument("--list-workloads", action="store_true",
                   help="print the definition's workloads and exit")
    p.add_argument("--baseline", default=None,
                   help="flashinfer-bench: which dataset baseline solution is the in-job "
                        "reference (default: the first)")
    p.add_argument("--blob-root", action="append", default=[],
                   help="sol-execbench: extra root for safetensors blobs (a "
                        "flashinfer-trace clone for its FlashInfer-Bench subset)")
    p.add_argument("--value-returning", action="store_true",
                   help="kernel returns its outputs instead of writing pre-allocated "
                        "ones (default: DPS, what both boards default to)")
    board = p.add_mutually_exclusive_group()
    board.add_argument("--sol-board", default=None, metavar="FILE",
                       help="sol-execbench: the board's record for the problem (a saved "
                            "kernels/<id> response) instead of fetching it — its T_b / T_SOL "
                            "are the card's SOL score and target")
    board.add_argument("--no-sol-board", action="store_true",
                       help="sol-execbench: do not fetch the board's T_b / T_SOL (the card "
                            "then has no SOL score and the task needs [target] ms)")
    p.add_argument("--binary-ref", default=None, metavar="SOLUTION_JSON",
                   help="a compiled (cuda_cpp) solution of this problem becomes the card's "
                        "sealed in-job reference: ref_bin/<module>.so, built by the recon "
                        "tree operators/_<op>ref/ (forgekernel.binref) — for blackbox runs")
    p.add_argument("--binary-ref-name", default=None,
                   help="the binary reference's display name")
    p.add_argument("--sealed-ref", default=None, metavar="SOLUTION_JSON",
                   help="a Python solution (PyTorch / Triton / CuTe DSL, one file or many) of "
                        "this problem becomes the card's sealed in-job reference: every source "
                        "sealed into ref_bin/ref_payload.bin with a key kept in the gitignored "
                        "submit.local.toml, decoded in memory inside the job; recon tree "
                        "operators/_<op>ref/ (forgekernel.sealref) — for blackbox runs")
    p.add_argument("--sealed-ref-name", default=None,
                   help="the sealed source reference's display name")
    p.add_argument("--ref-source-path", action="append", default=[],
                   help="with --binary-ref / --sealed-ref: another place the reference's "
                        "source lives (loop.toml ref_source_paths; the solution's directory, "
                        "and a binary reference's recon src/, are added)")
    args = p.parse_args(argv)
    if args.binary_ref and args.sealed_ref:
        p.error("--binary-ref and --sealed-ref are two kinds of one reference: choose one")

    from forgekernel import leaderboard as lbmod
    template_eval = template_dir() / "eval"
    problem = _leaderboard_problem(args)
    if problem is None:
        return 0            # --list-workloads
    sol = module = None
    if args.binary_ref:
        from forgekernel import binref
        sol = binref.load_solution(args.binary_ref)
        if errs := binref.solution_problems(sol, problem.definition["name"]):
            raise SystemExit("forge init: --binary-ref: " + "; ".join(errs))
        module = binref.module_name(sol)
        problem.reference_impl = binref.reference_impl(sol, module, args.binary_ref_name)
    elif args.sealed_ref:
        from forgekernel import sealref
        sol = sealref.load_solution(args.sealed_ref)
        if errs := sealref.solution_problems(sol, problem.definition["name"], template_eval):
            raise SystemExit("forge init: --sealed-ref: " + "; ".join(errs))
        module = sealref.module_name(sol)
        problem.reference_impl = sealref.reference_impl(sol, module, args.sealed_ref_name)

    base = Path(args.base_dir) if args.base_dir else repo_root()
    op_dir = base / "operators" / args.op
    if op_dir.exists():
        raise FileExistsError(f"operator tree already exists: {op_dir}")
    created = taskcard.copy_template(op_dir, args.op)
    created += lbmod.write_card(op_dir, problem, template_eval,
                                dps=not args.value_returning, kernel_file=args.kernel_file)
    for note in problem.notes:
        print(f"[init] NOTE {note}")
    if (nbytes := lbmod.blob_bytes(problem)) > lbmod.BLOB_WARN_BYTES:
        print(f"[init] WARNING the card's safetensors blobs are {nbytes / 2**20:.0f} "
              f"MiB — every job sends them to the GPU machine, which slows each job")
    if sol is not None:
        from forgekernel import binref
        ref_dir = base / "operators" / taskcard.ref_tree(args.op)
        if args.sealed_ref:
            from forgekernel import sealref
            made, _ = sealref.write_trees(op_dir, ref_dir, sol, module, template_eval,
                                           kernel_file=args.kernel_file)
            created += made
            sources = [str(Path(args.sealed_ref).resolve().parent)]
            if extra := sealref.non_python_sources(sol):
                print(f"[init] WARNING the solution's non-Python files {extra} travel in the "
                      f"payload but are not materialised on the node: a reference that reads "
                      f"them fails")
            print(f"[init] sealed source reference {module}: payload + key in {op_dir} and the "
                  f"recon tree {ref_dir} (key in each tree's gitignored submit.local.toml) — on "
                  f"a GPU run `bash gpu_entry.sh crosscheck` there and `cardref` here")
        else:
            created += binref.write_ref_tree(ref_dir, op_dir, sol, module)
            entry = op_dir / "gpu_entry.sh"
            entry.write_text(binref.guard_card_gpu_entry(entry.read_text(), module,
                                                         args.kernel_file))
            (op_dir / "ref_bin").mkdir(exist_ok=True)
            sources = [str(ref_dir / "src"), str(Path(args.binary_ref).resolve().parent)]
            print(f"[init] binary reference {module}: recon tree {ref_dir} — on a GPU run its "
                  f"`bash gpu_entry.sh build`, then `python -m forgekernel.binref install "
                  f"{op_dir} <the .so> --ref-tree {ref_dir}`; then `crosscheck` there and "
                  f"`cardref` here")
        binref.fill_ref_source_paths(op_dir, list(dict.fromkeys(sources + args.ref_source_path)))
        taskcard._ignore(op_dir, "loop.toml")

    print(f"[init] leaderboard card: {op_dir}")
    for path in dict.fromkeys(created):
        print(f"  + {path.relative_to(op_dir.parent)}")
    print(f"[init] {problem.suite} {problem.name} "
          f"{'workload group' if problem.group else 'workload'} "
          f"{lbmod.describe_workloads(problem)} — review loop.toml.example, then "
          f"`bash gpu_entry.sh cardref` on a GPU before the first run")
    if problem.group:
        print(f"[init] meta workloads (pricing / stall / timeline read each): "
              f"{', '.join(lbmod.meta_workload_names(problem))}")
    if problem.check_only:
        print(f"[init] correctness-only workloads (judged on every run, never timed): "
              f"{', '.join(g['name'] for g in problem.check_only)}")
    return 0


def _leaderboard_problem(args):
    """The board problem named on the command line, or None after --list-workloads."""
    from forgekernel import leaderboard as lbmod
    if not args.dataset or not args.definition:
        raise SystemExit("forge init: --leaderboard needs --dataset and --definition")
    try:
        if args.list_workloads:
            rows = _leaderboard_rows(lbmod, args)
            print("\n".join(lbmod.list_workloads(rows)))
            return None
        if args.leaderboard == lbmod.SUITE_FIB:
            if args.sol_board or args.no_sol_board:
                raise ValueError("--sol-board / --no-sol-board are sol-execbench options")
            return lbmod.fib_problem(args.dataset, args.definition, args.workload,
                                     baseline=args.baseline, check_workloads=args.check_workload)
        if args.sol_board:
            board, source = lbmod.load_sol_board(args.sol_board), str(args.sol_board)
        elif args.no_sol_board:
            board, source = None, ""
        else:
            # the board's T_b / T_SOL decide when a SOL run is done: fetched by default,
            # and a board that is unreachable leaves a NOTE rather than a failure
            board, source = lbmod.fetch_sol_board, ""
        return lbmod.sol_problem(args.dataset, args.definition, args.workload,
                                 blob_roots=args.blob_root, board=board,
                                 board_source=source, check_workloads=args.check_workload)
    except (OSError, ValueError) as e:
        raise SystemExit(f"forge init: {e}") from None


def _leaderboard_rows(lbmod, args) -> list:
    root = Path(args.dataset).expanduser()
    if args.leaderboard == lbmod.SUITE_FIB:
        found = sorted(root.glob(f"workloads/*/{args.definition}.jsonl"))
    else:
        found = sorted(root.glob(f"{args.definition}/workload.jsonl")) or sorted(
            root.glob(f"*/{args.definition}/workload.jsonl"))
    if len(found) != 1:
        raise FileNotFoundError(f"{args.definition}: {len(found)} workload files under {root}")
    return lbmod.workload_rows(found[0])


if __name__ == "__main__":
    raise SystemExit(main())
