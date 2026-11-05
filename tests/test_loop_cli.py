"""Tests for forge loop CLI oracle resolution + bounded GPU poll."""

import pytest

from forgekernel.baseline import RefBaseline
from forgekernel.loop.cli import (
    ORACLE_BLACKBOX,
    ORACLE_SOURCE,
    _resolve_oracle,
    _resolve_oracle_mode,
    _resolve_target_ms,
    _will_distill,
    poll_until_terminal,
)
from forgekernel.loop.driver import BackendError


class _Clock:
    """Deterministic fake clock: now() advances by `step` on every sleep(_)."""
    def __init__(self, step=20.0):
        self.t = 0.0
        self.step = step
        self.sleeps = 0
    def now(self):
        return self.t
    def sleep(self, _):
        self.sleeps += 1
        self.t += self.step


CFG = {"loop": {"oracle": "/ref/gemm.cu"}}


def test_oracle_default_on_from_config():
    assert _resolve_oracle(ORACLE_SOURCE, None, CFG) == "/ref/gemm.cu"


def test_explicit_oracle_overrides_config():
    assert _resolve_oracle(ORACLE_SOURCE, "/other.cu", CFG) == "/other.cu"


def test_oracle_accepts_several_files():
    """One file was not the whole reference: the kernel it pointed at takes tile shapes
    and stage counts as constructor arguments, and the values for a given head_dim live
    in a sibling that was never in the reading set. The teacher inferred them from a
    resource budget and produced a self-consistent architecture that was not the
    reference's — invisible downstream, since the student implements what it is told."""
    cfg = {"loop": {"oracle": ["/ref/kernel.py", "/ref/interface.py"]}}
    assert _resolve_oracle(ORACLE_SOURCE, None, cfg) == "/ref/kernel.py, /ref/interface.py"
    # a lone string still resolves as before
    assert _resolve_oracle(
        ORACLE_SOURCE, None, {"loop": {"oracle": "/ref/kernel.py"}}) == "/ref/kernel.py"


def test_no_config_no_flag_is_none():
    assert _resolve_oracle(ORACLE_SOURCE, None, {}) is None


def test_blackbox_withholds_a_configured_oracle():
    """The mode exists to deny source the operator does happen to have on disk.

    Resolving the path anyway and relying on the prompt not to mention it would leave
    the run's central claim — that the teacher never read the reference — resting on
    the prompt rather than on the plumbing.
    """
    assert _resolve_oracle(ORACLE_BLACKBOX, None, CFG) is None
    assert _resolve_oracle(ORACLE_BLACKBOX, "/explicit.cu", CFG) is None


def test_blackbox_still_distills_and_therefore_still_needs_a_baseline(tmp_path):
    """Blackbox mode withholds the oracle PATH but still has an input to distil from
    (reconnaissance of the reference), so it still measures the baseline distillation
    anchors to; a source-mode run with nothing to read has no input."""
    ms = tmp_path / "milestone.md"
    ms.write_text("# empty skeleton, fails §9\n", encoding="utf-8")
    assert _will_distill(ms, None, ORACLE_BLACKBOX) is True
    assert _will_distill(ms, "/ref/gemm.cu", ORACLE_SOURCE) is True
    # source mode with nothing to read still has no input
    assert _will_distill(ms, None, ORACLE_SOURCE) is False


def test_oracle_mode_resolution_and_conflicts():
    assert _resolve_oracle_mode({}) == ORACLE_SOURCE
    assert _resolve_oracle_mode({"loop": {"oracle_mode": "blackbox"}}) == ORACLE_BLACKBOX
    with pytest.raises(SystemExit):
        _resolve_oracle_mode({"loop": {"oracle_mode": "typo"}})
    with pytest.raises(SystemExit):
        _resolve_oracle_mode({"loop": {"oracle_mode": "none"}})


def test_language_defaults_to_cutedsl():
    from forgekernel.loop.cli import _resolve_language
    assert _resolve_language(None, {}) == "cutedsl"


def test_language_from_loop_toml():
    from forgekernel.loop.cli import _resolve_language
    assert _resolve_language(None, {"loop": {"language": "inline-ptx"}}) == "inline-ptx"


def test_language_cli_flag_overrides_config():
    from forgekernel.loop.cli import _resolve_language
    assert _resolve_language("triton", {"loop": {"language": "cutedsl"}}) == "triton"


def test_gate_window_defaults_config_flag_and_validation():
    from forgekernel.loop.cli import _resolve_gate_window
    assert _resolve_gate_window(None, {}) == "device"
    assert _resolve_gate_window(None, {"loop": {"gate_window": "e2e"}}) == "e2e"
    assert _resolve_gate_window("device", {"loop": {"gate_window": "e2e"}}) == "device"
    with pytest.raises(SystemExit):
        _resolve_gate_window(None, {"loop": {"gate_window": "wallclock"}})


def test_target_derivation_follows_the_gate_window():
    ref = RefBaseline(op="demo", ms=0.2796, ms_device=0.2491)
    assert _resolve_target_ms(None, ref, 0.95) == pytest.approx(0.2491 / 0.95)
    assert _resolve_target_ms(None, ref, 0.95, gate_window="e2e") == \
        pytest.approx(0.2796 / 0.95)
    # a device gate over an archive with no device reading must refuse, not
    # silently derive the stop threshold from the e2e number
    with pytest.raises(ValueError, match="ms_device"):
        _resolve_target_ms(None, RefBaseline(op="demo", ms=0.2796), 0.95,
                           gate_window="device")


def test_poll_returns_on_first_terminal():
    clk = _Clock()
    status = poll_until_terminal(
        lambda: "Succeeded", "tasks/1",
        poll_s=20, timeout_s=1000, now=clk.now, sleep=clk.sleep)
    assert status == "Succeeded" and clk.sleeps == 0  # no wait when already terminal


def test_poll_returns_after_a_few_non_terminal():
    seq = iter(["", "Running", "Running", "Succeeded"])
    clk = _Clock()
    status = poll_until_terminal(
        lambda: next(seq), "1",
        poll_s=20, timeout_s=1000, now=clk.now, sleep=clk.sleep)
    assert status == "Succeeded" and clk.sleeps == 3


def test_poll_raises_backenderror_on_timeout():
    # a job stuck non-terminal forever (the iter4 hang) must degrade cleanly, not loop
    clk = _Clock(step=20.0)
    with pytest.raises(BackendError) as e:
        poll_until_terminal(
            lambda: "Running", "tasks/42",
            poll_s=20, timeout_s=100, now=clk.now, sleep=clk.sleep)
    assert "tasks/42" in str(e.value) and "Running" in str(e.value)


def test_killed_counts_as_terminal():
    """Anything the poller does not recognise as still-moving ends the poll: an
    allowlist of terminal states that missed one would poll the full hour and then
    raise."""
    clk = _Clock()
    for status in ("Killed", "OOMKilled", "SomeStatusNobodyEnumerated"):
        assert poll_until_terminal(
            lambda s=status: s, "1",
            poll_s=20, timeout_s=1000, now=clk.now, sleep=clk.sleep) == status


def test_running_states_keep_polling():
    seq = iter(["Running", "", "Running", "Running", "Lost"])
    clk = _Clock()
    assert poll_until_terminal(
        lambda: next(seq), "1",
        poll_s=20, timeout_s=1000, now=clk.now, sleep=clk.sleep) == "Lost"
    assert clk.sleeps == 4


def test_poll_raises_on_empty_status_forever():
    # the actual root cause: _status_of parse yields "" (never terminal) -> was an
    # infinite loop; now bounded -> BackendError once the deadline passes.
    clk = _Clock(step=20.0)
    with pytest.raises(BackendError):
        poll_until_terminal(
            lambda: "", "tasks/7",
            poll_s=20, timeout_s=60, now=clk.now, sleep=clk.sleep)



# --- target_ms resolution ---------------------------------------------------

_BASE = RefBaseline(op="demo", ms=2.0, ms_device=2.0, job="1")


def test_target_derived_from_the_measured_baseline():
    assert _resolve_target_ms(None, _BASE, 0.95) == pytest.approx(2.10526, rel=1e-4)


def test_explicit_target_overrides_the_baseline():
    """An operator whose bar is not its own ref (e.g. beating a published number)."""
    assert _resolve_target_ms(1.9, _BASE, 0.95) == 1.9


def test_target_prefers_the_architecture_reference_over_the_library_baseline():
    """The two references answer different questions and are ~20% apart here. Deriving
    the stop threshold from the library one gave target_ms=2.4933 against an oracle that
    runs at 1.978 — the loop would have reported success at ~three quarters of the
    performance it was asked to reproduce."""
    from forgekernel.baseline import ROLE_ARCHITECTURE

    arch = RefBaseline(op="demo", ms=1.978, ms_device=1.978, role=ROLE_ARCHITECTURE)
    library = RefBaseline(op="demo", ms=2.3686, ms_device=2.3686, job="655416")
    assert _resolve_target_ms(None, library, 0.95, arch=arch) == pytest.approx(2.0821, rel=1e-3)
    # without one declared it still falls back rather than refusing to run
    assert _resolve_target_ms(None, library, 0.95, arch=None) == pytest.approx(2.4933, rel=1e-3)
    # an explicit flag still wins over both
    assert _resolve_target_ms(1.9, library, 0.95, arch=arch) == 1.9


def test_no_target_and_no_baseline_is_unresolved():
    """Caller must refuse rather than invent a stop threshold."""
    assert _resolve_target_ms(None, None, 0.95) is None


# --- job-id provenance ------------------------------------------------------

def _submitting_backend(monkeypatch, tmp_path, statuses, log="log body", **kw):
    """A RealBackend whose submit prints job=<n> (counting up from 655000), whose status
    reads walk ``statuses``, and whose log is ``log``."""
    from types import SimpleNamespace

    from forgekernel.loop import cli
    n = {"submits": 0}

    def fake_forge(op, verb, *a, **k):
        assert verb == "submit", verb
        n["submits"] += 1
        return SimpleNamespace(returncode=0, stderr="",
                               stdout=f"[submit] job={654999 + n['submits']}\n")

    seq = iter(statuses)
    monkeypatch.setattr(cli, "_forge", fake_forge)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    be = cli.RealBackend("demo", tmp_path, None, "n", "e", poll_s=0.0, **kw)
    monkeypatch.setattr(be, "_job_status", lambda job: next(seq))
    monkeypatch.setattr(be, "_fetch_log", lambda job, status, **k: log)
    return be, n


def test_submit_records_the_job_id_for_provenance(monkeypatch, tmp_path):
    """A baseline cites its job in every gate it anchors, and the fetched logs do
    not carry the id — so submit_gpu has to hand it back."""
    be, _ = _submitting_backend(monkeypatch, tmp_path, ["Running", "Succeeded"])
    assert be.last_job_id == ""
    assert be.submit_gpu("br", "cmd:true") == "log body"
    assert be.last_job_id == "655000"


def test_a_job_that_lost_its_runner_is_resubmitted_not_failed(monkeypatch, tmp_path):
    """A runner that went away without a verdict (a reboot, an OOM kill) never produced
    output, so failing the eval would charge the agent for the machine."""
    be, n = _submitting_backend(monkeypatch, tmp_path, ["Lost", "Lost", "Succeeded"],
                                lost_retries=3)
    assert be.submit_gpu("br", "all") == "log body"
    assert n["submits"] == 3 and be.last_job_id == "655002"


def test_repeatedly_lost_jobs_eventually_give_up_without_a_verdict(monkeypatch, tmp_path):
    from forgekernel.loop import cli
    from forgekernel.loop.driver import NoRunError
    be, _ = _submitting_backend(monkeypatch, tmp_path, ["Lost"] * 3, lost_retries=2)
    with pytest.raises(cli.RunnerLostError, match="lost its runner 3 times") as e:
        be.submit_gpu("br", "all")
    assert isinstance(e.value, NoRunError)


def test_a_job_that_never_reached_the_gpu_machine_is_a_transport_error(monkeypatch, tmp_path):
    from forgekernel.loop import cli
    from forgekernel.submit.jobs import TRANSPORT_ERROR_MARKER
    be, _ = _submitting_backend(monkeypatch, tmp_path, ["Failed"],
                                log=f"{TRANSPORT_ERROR_MARKER} 3 transport failure(s) on h\n")
    with pytest.raises(cli.JobTransportError, match="could not be delivered"):
        be.submit_gpu("br", "all")


def _job_tree(tmp_path, monkeypatch, *, log: str | None, status: str):
    """operators/demo with one finished job 77, the way its runner leaves it."""
    import json as _json

    monkeypatch.setenv("FORGEKERNEL_OPERATORS_DIR", str(tmp_path / "operators"))
    op = tmp_path / "operators" / "demo"
    jd = op / ".jobs" / "77"
    jd.mkdir(parents=True)
    (op / "submit.toml").write_text('[repo]\ndelivery = "local"\nbranch = "main"\n')
    (jd / "result.json").write_text(_json.dumps({"id": 77, "status": status}))
    if log is not None:
        (jd / "log.txt").write_text(log)


def test_a_jobs_status_and_log_are_read_from_its_files(monkeypatch, tmp_path):
    from forgekernel.loop import cli
    _job_tree(tmp_path, monkeypatch, log="PRICE-OK\n=== gpu_entry.sh done (task=cmd) ===\n",
              status="Succeeded")
    monkeypatch.setattr(cli, "_forge", lambda *a, **k: pytest.fail("no subprocess needed"))
    be = cli.RealBackend("demo", tmp_path, None, "n", "e", poll_s=0.0)
    assert be._job_status("77") == "Succeeded"
    assert "PRICE-OK" in be._fetch_log("77", "Succeeded", task="cmd")
    be.last_job_id, be.last_mode = "77", "cmd:python3 x.py"
    assert "PRICE-OK" in be.reread_log("77")


def test_succeeded_job_with_no_log_raises_instead_of_blaming_the_agent(monkeypatch, tmp_path):
    from forgekernel.loop import cli
    _job_tree(tmp_path, monkeypatch, log=None, status="Succeeded")
    be = cli.RealBackend("demo", tmp_path, None, "n", "e", poll_s=0.0)
    with pytest.raises(cli.BackendError, match="left no log"):
        be._fetch_log("77", "Succeeded")
    # a job that failed legitimately has nothing to read — that is the agent's signal
    assert be._fetch_log("77", "Failed") == ""


# --- the loop measures its own baseline -------------------------------------

def _conformant_milestone() -> str:
    """The schema-clean sample from the milestone tests — one source, not a copy."""
    from test_milestone import VALID  # pytest puts tests/ on sys.path
    return VALID


_REF_SPEC = {"loop": {"ref_cmd": "python3 eval/measure_target.py",
                      "ref_name": "cuDNN", "ref_hardware": "H100"}}


def test_ref_spec_read_from_loop_toml():
    from forgekernel.loop.cli import _resolve_ref_spec
    assert _resolve_ref_spec(_REF_SPEC)["ref_cmd"] == "python3 eval/measure_target.py"
    assert _resolve_ref_spec({}) == {}


def test_an_archived_baseline_is_reused_not_remeasured(tmp_path, monkeypatch):
    """One measurement per operator, reused across runs — that is the whole
    reason the baseline is an archive rather than a step of the loop."""
    from forgekernel.baseline import RefBaseline, save_baseline
    from forgekernel.loop.cli import _ensure_ref_baseline

    save_baseline(tmp_path, RefBaseline(op="demo", ms=2.0, job="1"))
    called = []
    monkeypatch.setattr("forgekernel.baseline_cli.measure_and_archive",
                        lambda *a, **k: called.append(1))
    got = _ensure_ref_baseline("demo", tmp_path, tmp_path / "m.md", "oracle.cu",
                               _REF_SPEC, 3)
    assert got.ms == 2.0
    assert not called, "an existing archive must not trigger a GPU job"


def test_missing_baseline_is_measured_when_the_operator_declares_its_ref(tmp_path, monkeypatch):
    from forgekernel.baseline import RefBaseline
    from forgekernel.loop.cli import _ensure_ref_baseline

    seen = {}
    def _fake(op, op_dir, spec, *, repeat):
        seen.update(spec=spec, repeat=repeat)
        return RefBaseline(op=op, ms=2.36, job="9")

    monkeypatch.setattr("forgekernel.baseline_cli.measure_and_archive", _fake)
    got = _ensure_ref_baseline("demo", tmp_path, tmp_path / "absent.md", "oracle.cu",
                               _REF_SPEC, 3)
    assert got.ms == 2.36
    assert seen["spec"]["ref_cmd"] == "python3 eval/measure_target.py"
    assert seen["repeat"] == 3


def test_a_task_reference_measured_as_the_architecture_one_is_not_measured_twice(
        tmp_path, monkeypatch):
    from forgekernel.baseline import ROLE_ARCHITECTURE, ROLE_LIBRARY, RefBaseline, load_baseline
    from forgekernel.loop.cli import _ensure_ref_baseline

    monkeypatch.setattr("forgekernel.baseline_cli.measure_and_archive",
                        lambda *a, **k: pytest.fail("the same reference measured twice"))
    cfg = {"loop": {"ref_mode": "sealref", "ref_submit_op": "_demoref", "ref_name": "ref",
                    "arch_ref_mode": "sealref", "arch_ref_submit_op": "_demoref"}}
    arch = RefBaseline(op="demo", ms=0.157, role=ROLE_ARCHITECTURE, job="7",
                       samples_ms=[0.18, 0.157, 0.16])
    got = _ensure_ref_baseline("demo", tmp_path, tmp_path / "absent.md", "oracle.cu",
                               cfg, 3, arch=arch)
    assert got.ms == 0.157 and got.role == ROLE_LIBRARY and got.job == "7"
    assert load_baseline(tmp_path).ms == 0.157

    # a different reference (another tree) is still measured
    seen = []
    monkeypatch.setattr("forgekernel.baseline_cli.measure_and_archive",
                        lambda op, op_dir, spec, *, repeat: seen.append(spec) or
                        RefBaseline(op=op, ms=0.2, job="8"))
    other = {"loop": dict(cfg["loop"], ref_submit_op="_libref")}
    assert _ensure_ref_baseline("demo", tmp_path / "b", tmp_path / "absent.md", "oracle.cu",
                                other, 3, arch=arch).ms == 0.2
    assert seen and seen[0]["submit_op"] == "_libref"


def test_no_ref_declared_leaves_the_driver_to_refuse(tmp_path):
    """Without ref_cmd the loop cannot measure; the driver reports it rather than
    the CLI guessing a command."""
    from forgekernel.loop.cli import _ensure_ref_baseline
    assert _ensure_ref_baseline("demo", tmp_path, tmp_path / "absent.md",
                                "oracle.cu", {}, 3) is None


def test_a_run_with_nothing_to_distill_does_not_measure(tmp_path, monkeypatch):
    """Resuming against an existing milestone must not spend GPU on a re-measure."""
    from forgekernel.loop.cli import _ensure_ref_baseline
    monkeypatch.setattr("forgekernel.baseline_cli.measure_and_archive",
                        lambda *a, **k: pytest.fail("must not measure"))
    ms = tmp_path / "m.md"
    ms.write_text(_conformant_milestone(), encoding="utf-8")
    assert _ensure_ref_baseline("demo", tmp_path, ms, "oracle.cu", _REF_SPEC, 3) is None


def test_no_oracle_means_no_distillation_hence_no_measurement(tmp_path, monkeypatch):
    from forgekernel.loop.cli import _ensure_ref_baseline
    monkeypatch.setattr("forgekernel.baseline_cli.measure_and_archive",
                        lambda *a, **k: pytest.fail("must not measure"))
    assert _ensure_ref_baseline("demo", tmp_path, tmp_path / "absent.md", None,
                                _REF_SPEC, 3) is None


def test_measure_false_reports_the_measurement_without_making_it(tmp_path, monkeypatch):
    """--harvest-only builds its context without buying a job."""
    from forgekernel.loop.cli import _ensure_ref_baseline
    monkeypatch.setattr("forgekernel.baseline_cli.measure_and_archive",
                        lambda *a, **k: pytest.fail("dry-run must not submit a job"))
    assert _ensure_ref_baseline("demo", tmp_path, tmp_path / "absent.md", "oracle.cu",
                                _REF_SPEC, 3, measure=False) is None


def test_workspace_without_a_remote_is_detected(tmp_path):
    """A workspace that cannot push fails at iteration 0, after distillation has
    already spent an agent call — worth catching before the run starts."""
    import subprocess

    from forgekernel.loop.cli import _workspace_can_push
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    assert not _workspace_can_push(ws)
    subprocess.run(["git", "remote", "add", "origin", str(tmp_path)], cwd=ws, check=True)
    assert _workspace_can_push(ws)


def test_missing_workspace_is_not_pushable(tmp_path):
    from forgekernel.loop.cli import _workspace_can_push
    assert not _workspace_can_push(tmp_path / "does-not-exist")


# --- the architecture reference ----------------------------------------------
# This is the anchor target_ms is derived from, and it had no coverage at all while
# it was silently derived from the library baseline instead — a 26% error that ran
# for a full run. The tests below pin the two properties that failure violated:
# an architecture reference is preferred over the library one, and a declared
# reference keeps its provenance rather than being presented as measured.

_ARCH_CFG = {
    "loop": {
        "arch_ref_ms": 1.978,
        "arch_ref_device_ms": 1.978,
        "arch_ref_name": "flash-attn cute (sm90)",
        "arch_ref_job": "660168",
        "ref_shape": "B=10 N=4096 H=16/2 D=64 causal",
        "ref_hardware": "H100-SXM",
        "arch_ref_provenance": "hand-run on a GPU node with the oracle toolchain",
    }
}


def test_arch_baseline_is_built_from_the_declared_reference(tmp_path):
    from forgekernel.baseline import ROLE_ARCHITECTURE
    from forgekernel.loop.cli import _ensure_arch_baseline

    b = _ensure_arch_baseline("flash_attn_bwd", tmp_path, _ARCH_CFG)
    assert b.role == ROLE_ARCHITECTURE
    assert b.ms == 1.978
    assert b.raw_stdout                      # provenance survives into the archive
    assert (tmp_path / "arch_baseline.json").exists()


def test_a_stale_declared_arch_archive_refreshes_from_the_toml(tmp_path):
    """The declared fallback saves its values into arch_baseline.json, and the
    archive used to win unconditionally on the next start — so an operator hit by
    the device default's ms_device refusal was told to fix loop.toml, did, and was
    refused again until they knew to hand-delete the archive. A declared-sourced
    archive is just the toml's saved image: when the toml moves, it follows."""
    from forgekernel.baseline import load_baseline, ROLE_ARCHITECTURE
    from forgekernel.loop.cli import _ensure_arch_baseline

    old = {"loop": {k: v for k, v in _ARCH_CFG["loop"].items()
                    if k != "arch_ref_device_ms"}}
    assert _ensure_arch_baseline("flash_attn_bwd", tmp_path, old).ms_device is None

    b = _ensure_arch_baseline("flash_attn_bwd", tmp_path, _ARCH_CFG)
    assert b.ms_device == 1.978                # the toml's new reading, not the image
    assert load_baseline(tmp_path, ROLE_ARCHITECTURE).ms_device == 1.978  # persisted


def test_a_measured_arch_archive_is_never_clobbered_by_declared_values(tmp_path):
    """samples_ms is the discriminator: measure_and_archive always records its runs,
    the declared path never does. A real measurement outranks a hand-typed constant."""
    from forgekernel.baseline import RefBaseline, ROLE_ARCHITECTURE, save_baseline
    from forgekernel.loop.cli import _ensure_arch_baseline

    measured = RefBaseline(op="flash_attn_bwd", ms=2.05, role=ROLE_ARCHITECTURE,
                           job="660899", samples_ms=[2.05, 2.07, 2.06])
    save_baseline(tmp_path, measured)
    b = _ensure_arch_baseline("flash_attn_bwd", tmp_path, _ARCH_CFG)
    assert b.ms == 2.05 and b.samples_ms      # the measurement, not the toml


def test_arch_baseline_measures_best_of_three(tmp_path, monkeypatch):
    """The architecture measurement anchors target_ms, so it gets the same repeat=3
    the library baseline does. A repeat=1 sample once landed 5% into the node-
    contention band and the loop stopped a run at 0.897x while calling it 0.95x."""
    from forgekernel import baseline_cli
    from forgekernel.loop.cli import _ensure_arch_baseline

    seen = {}

    def fake_measure(op, op_dir, spec, *, repeat, role):
        seen["repeat"] = repeat
        raise baseline_cli.RefMeasurementError("stub: measurement not exercised here")

    monkeypatch.setattr(baseline_cli, "measure_and_archive", fake_measure)
    cfg = {"loop": {**_ARCH_CFG["loop"], "arch_ref_mode": "gemmref",
                    "arch_ref_submit_op": "_recon"}}
    _ensure_arch_baseline("flash_attn_bwd", tmp_path, cfg)
    assert seen["repeat"] == 3


def test_arch_baseline_absent_when_nothing_is_declared(tmp_path):
    """Operators with no architecture reference keep the library-only behaviour."""
    from forgekernel.loop.cli import _ensure_arch_baseline
    assert _ensure_arch_baseline("demo", tmp_path, {"loop": {}}) is None


def test_arch_baseline_carries_the_ncu_signature(tmp_path):
    """The mechanism gate lines anchor to the reference's NCU numbers. Without this
    they could only anchor to the library baseline — a different algorithm, so a
    number that says nothing about whether the taught machine was reproduced."""
    from forgekernel.loop.cli import _ensure_arch_baseline

    cfg = {"loop": {**_ARCH_CFG["loop"],
                    "arch_ref_ncu": {"registers_per_thread": 168, "smem_per_block": 114688}}}
    b = _ensure_arch_baseline("flash_attn_bwd", tmp_path, cfg)
    assert b.has_ncu and b.ncu["registers_per_thread"] == 168
    assert "NCU signature" in b.render()


def test_target_ms_prefers_the_architecture_reference(tmp_path):
    """The regression that motivated the split: with both present, the target must
    come from the machine being reproduced, not from the library it competes with."""
    from forgekernel.baseline import ROLE_ARCHITECTURE
    from forgekernel.loop.cli import _ensure_arch_baseline, _resolve_target_ms

    library = RefBaseline(op="flash_attn_bwd", ms=2.3686, ms_device=2.3686,
                          ref_name="cuDNN")
    arch = _ensure_arch_baseline("flash_attn_bwd", tmp_path, _ARCH_CFG)
    assert arch.role == ROLE_ARCHITECTURE

    assert _resolve_target_ms(None, library, 0.95, arch=arch) == pytest.approx(1.978 / 0.95)
    # ...and without one, the library baseline is still better than nothing
    assert _resolve_target_ms(None, library, 0.95, arch=None) == pytest.approx(2.3686 / 0.95)
    # an explicit --target-ms outranks both
    assert _resolve_target_ms(1.90, library, 0.95, arch=arch) == 1.90


# --- a failing call keeps the short budget ------------------------------------

class _FakeProc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _backend(monkeypatch, **kw):
    from pathlib import Path

    from forgekernel.loop.cli import RealBackend
    monkeypatch.setattr("forgekernel.loop.cli.time.sleep", lambda *_: None)
    return RealBackend("op", Path("/ws"), None, "n", "e", **kw)


def test_an_ordinary_failure_keeps_the_short_budget(monkeypatch):
    from forgekernel.loop.driver import BackendError
    b = _backend(monkeypatch, git_retries=3)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return _FakeProc(1, stderr="fatal: could not read from remote")

    with pytest.raises(BackendError):
        b._retry_call(fn, what="git push")
    assert calls["n"] == 3


# --- a server-side API outage is not a broken agent ---------------------------
# Both arrive as rc=1 from the agent CLI, and spending the 90-second hiccup budget
# on a 529 incident ended a run overnight (three tries inside the
# outage window, agent_failed, loop gone until morning). Server trouble gets its
# own patience budget; auth/config failures keep the short one.

def _agent_backend(monkeypatch, replies, **kw):
    """RealBackend whose agent turn yields each _FakeProc in turn.

    The seam is `_agent_turn` (the heartbeat wrapper around Popen), not
    `subprocess.run`: the turn runs on a pump thread now so the loop can print
    progress while the agent thinks."""
    b = _backend(monkeypatch, **kw)
    b.agent_cmd = "fake-agent"
    it = iter(replies)

    def _turn(cmd, prompt, role):
        p = next(it)
        return p.returncode, p.stdout, p.stderr

    monkeypatch.setattr(b, "_agent_turn", _turn)
    return b


def test_a_529_overload_waits_instead_of_spending_the_hiccup_budget(monkeypatch):
    overloaded = _FakeProc(1, stdout="API Error: 529 Overloaded. This is a "
                                     "server-side issue, usually temporary")
    b = _agent_backend(monkeypatch,
                       [overloaded] * 5 + [_FakeProc(0, stdout="done")],
                       agent_retries=3, api_outage_retries=8)
    assert b.run_agent("go") == "done"   # five outage replies never burned an attempt


def test_a_persistent_outage_still_gives_up(monkeypatch):
    from forgekernel.loop.driver import AgentError
    overloaded = _FakeProc(1, stdout="API Error: 529 Overloaded")
    b = _agent_backend(monkeypatch, [overloaded] * 20,
                       agent_retries=3, api_outage_retries=4)
    with pytest.raises(AgentError, match="after 3 attempts"):
        b.run_agent("go")


def test_a_stream_cut_waits_instead_of_spending_the_hiccup_budget(monkeypatch):
    # An upstream SSE stream that dies mid-generation (the "upstream stream ended
    # without a terminal frame" cut that hit closed-source runs) must go
    # to the outage patience, not the 3-attempt budget — it is a transient
    # disconnect, and burning a hard attempt on it would kill the loop after three.
    cut = _FakeProc(1, stdout="API Error: upstream stream ended without a terminal "
                              "frame (no message_delta, no message_stop); the "
                              "response is truncated and incomplete; received 609 "
                              "frames; reason=eof")
    b = _agent_backend(monkeypatch,
                       [cut] * 5 + [_FakeProc(0, stdout="done")],
                       agent_retries=3, api_outage_retries=8)
    assert b.run_agent("go") == "done"   # five stream cuts never burned an attempt


def test_a_persistent_stream_cut_still_gives_up(monkeypatch):
    from forgekernel.loop.driver import AgentError
    cut = _FakeProc(1, stdout="API Error: upstream stream ended without a terminal "
                              "frame; received 1220 frames; reason=eof")
    b = _agent_backend(monkeypatch, [cut] * 20,
                       agent_retries=3, api_outage_retries=4)
    with pytest.raises(AgentError, match="after 3 attempts"):
        b.run_agent("go")



def test_an_auth_failure_keeps_the_short_agent_budget(monkeypatch):
    from forgekernel.loop.driver import AgentError
    b = _backend(monkeypatch, agent_retries=3, api_outage_retries=8)
    b.agent_cmd = "fake-agent"
    calls = {"n": 0}

    def fake_turn(cmd, prompt, role):
        calls["n"] += 1
        return 1, "Invalid API key · Please run /login", ""

    monkeypatch.setattr(b, "_agent_turn", fake_turn)
    with pytest.raises(AgentError, match="after 3 attempts"):
        b.run_agent("go")
    assert calls["n"] == 3      # no outage patience spent on a misconfig


# --- a job outlives the loop that submitted it ---------------------------------
# A job's runner keeps running when the loop dies; resubmitting on restart pays for
# the same eval twice and discards a Succeeded result nobody read.

def _adoption_ws(tmp_path):
    (tmp_path / ".git").mkdir()
    return tmp_path


def _fake_forge(calls, *, submit_id="5", status="Succeeded", log="log body"):
    def fake_forge(op, verb, *a, **k):
        calls.append(verb)
        if verb == "submit":
            return _FakeProc(0, stdout=f"[submit] job={submit_id}\n")
        if verb == "status":
            return _FakeProc(0, stdout=f'{{"id": 9, "status": "{status}"}}\n')
        return _FakeProc(0, stdout=log)
    return fake_forge


def test_an_interrupted_submit_is_adopted_not_resubmitted(monkeypatch, tmp_path):
    import json as _json

    from forgekernel.loop import cli

    ws = _adoption_ws(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "_forge", _fake_forge(calls))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0)
    monkeypatch.setattr(cli.RealBackend, "_branch_sha", lambda self, b: "abc")
    (ws / ".git" / "fk-inflight.json").write_text(_json.dumps(
        {"job": "9", "branch": "br", "mode": "all", "sha": "abc"}))

    assert be.submit_gpu("br", "all") == "log body"
    assert "submit" not in calls          # the recorded job was adopted
    assert be.last_job_id == "9"          # provenance still flows
    assert not (ws / ".git" / "fk-inflight.json").exists()  # consumed on success


def test_a_stale_inflight_record_is_ignored_and_deleted(monkeypatch, tmp_path):
    """A student commit after the record was written means the job answers an older
    question — adopting it would score stale code."""
    import json as _json

    from forgekernel.loop import cli

    ws = _adoption_ws(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "_forge", _fake_forge(calls))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0)
    monkeypatch.setattr(cli.RealBackend, "_branch_sha", lambda self, b: "new-sha")
    (ws / ".git" / "fk-inflight.json").write_text(_json.dumps(
        {"job": "9", "branch": "br", "mode": "all", "sha": "old-sha"}))

    assert be.submit_gpu("br", "all") == "log body"
    assert "submit" in calls              # fresh submit, not adoption
    assert be.last_job_id == "5"


def test_an_aborted_poll_leaves_the_job_adoptable(monkeypatch, tmp_path):
    import json as _json

    from forgekernel.loop import cli

    ws = _adoption_ws(tmp_path)
    monkeypatch.setattr(cli, "_forge", _fake_forge([], status="Running"))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0, poll_timeout_s=0.0)
    with pytest.raises(cli.BackendError, match="job 5"):
        be.submit_gpu("br", "all")
    rec = _json.loads((ws / ".git" / "fk-inflight.json").read_text())
    assert rec["job"] == "5"              # the next run can pick it back up


# --- the architecture reference is a precondition, not an input ----------------
# Every mechanism gate line in a distilled milestone anchors to this reference's NCU
# signature, and v1 is the one write with no rollback behind it. Measuring it after
# distillation means authoring those gates against numbers that do not exist yet --
# which is how they ended up anchored to the library baseline, a different algorithm.

_ARCH_MEASURED_CFG = {"loop": {**_ARCH_CFG["loop"],
                               "arch_ref_submit_op": "_arch_ref",
                               "arch_ref_mode": "fa4ref"}}


def test_a_declared_measurement_is_run_rather_than_transcribed(tmp_path, monkeypatch):
    from forgekernel.baseline import ROLE_ARCHITECTURE, RefBaseline
    from forgekernel.loop.cli import _ensure_arch_baseline

    seen = {}

    def fake_measure(op, op_dir, spec, *, repeat, role):
        seen.update(spec=spec, role=role)
        return RefBaseline(op=op, ms=2.0, role=role, ncu={"time_ns": 1})

    monkeypatch.setattr("forgekernel.baseline_cli.measure_and_archive", fake_measure)
    b = _ensure_arch_baseline("flash_attn_bwd", tmp_path, _ARCH_MEASURED_CFG)
    assert b.ms == 2.0                                  # measured, not the declared 1.978
    assert seen["role"] == ROLE_ARCHITECTURE
    assert seen["spec"]["ref_mode"] == "fa4ref"
    assert seen["spec"]["submit_op"] == "_arch_ref"     # not the operator's own tree


def test_the_declared_values_are_the_fallback_when_measuring_fails(tmp_path, monkeypatch):
    """A hand-transcribed constant is second choice, not the plan: it cannot tell that
    the shape or the image moved under it. But it beats having no reference at all."""
    from forgekernel.loop.cli import _ensure_arch_baseline

    def boom(*a, **k):
        raise RuntimeError("cluster unreachable")

    monkeypatch.setattr("forgekernel.baseline_cli.measure_and_archive", boom)
    b = _ensure_arch_baseline("flash_attn_bwd", tmp_path, _ARCH_MEASURED_CFG)
    assert b.ms == 1.978 and b.raw_stdout             # fell back, provenance intact


def test_an_archived_reference_is_never_remeasured(tmp_path, monkeypatch):
    """Two runs of one operator must compare against identical numbers."""
    from forgekernel.loop.cli import _ensure_arch_baseline

    first = _ensure_arch_baseline("flash_attn_bwd", tmp_path, _ARCH_CFG)
    monkeypatch.setattr("forgekernel.baseline_cli.measure_and_archive",
                        lambda *a, **k: pytest.fail("must reuse the archive"))
    assert _ensure_arch_baseline("flash_attn_bwd", tmp_path, _ARCH_MEASURED_CFG).ms == first.ms


def test_distillation_is_refused_without_an_ncu_signature():
    from forgekernel.baseline import ROLE_ARCHITECTURE, RefBaseline
    from forgekernel.loop.cli import check_arch_ref_ready

    no_ncu = RefBaseline(op="o", ms=2.1, role=ROLE_ARCHITECTURE, ref_name="FA4")
    why = check_arch_ref_ready(_ARCH_MEASURED_CFG, no_ncu, will_distill=True)
    assert why and "NCU signature" in why

    assert check_arch_ref_ready(_ARCH_MEASURED_CFG, None, will_distill=True)


def test_a_reference_with_a_signature_lets_the_run_start():
    from forgekernel.baseline import ROLE_ARCHITECTURE, RefBaseline
    from forgekernel.loop.cli import check_arch_ref_ready

    ok = RefBaseline(op="o", ms=2.1, role=ROLE_ARCHITECTURE, ncu={"time_ns": 2077216})
    assert check_arch_ref_ready(_ARCH_MEASURED_CFG, ok, will_distill=True) is None


def test_an_already_distilled_milestone_does_not_need_the_reference_reachable():
    """Re-running a run must not require the reference to be measurable again --
    its numbers are already baked into the milestone."""
    from forgekernel.loop.cli import check_arch_ref_ready
    assert check_arch_ref_ready(_ARCH_MEASURED_CFG, None, will_distill=False) is None


def test_operators_without_an_architecture_reference_are_unaffected():
    """An operator whose reference *is* the library has nothing to reproduce."""
    from forgekernel.loop.cli import check_arch_ref_ready
    assert check_arch_ref_ready({"loop": {}}, None, will_distill=True) is None


def test_ref_rides_job_judgment():
    # Misjudging True is the sample-burn bug: the confirmation stage then expects a
    # same-job ratio the bench cannot produce. Only a plain ref_cmd (measured
    # through the operator's own tree, importable in student jobs) rides the job.
    from forgekernel.loop.cli import _ref_rides_job
    assert _ref_rides_job({"ref_cmd": "python eval/bench_ref.py"})
    assert not _ref_rides_job({})  # target-ms-only run: no ref declared
    assert not _ref_rides_job({"ref_mode": "faref", "submit_op": "_recon"})
    assert not _ref_rides_job({"ref_cmd": "python ref.py", "submit_op": "_recon"})


def test_unrecognized_model_flake_is_transient():
    """The CLI's model recognition tracks a dynamic config fetch and fails
    intermittently — including on auxiliary queries (a run died on the
    generate_session_title path). It must route into the outage budget, not burn
    the 3 hard attempts."""
    from forgekernel.loop.cli import _TRANSIENT_AGENT_ERR
    observed = ('[claude-code:unrecognized_model] '
                '{"model":"CLAUDE_ew682d","query_source":"generate_session_title"}')
    assert _TRANSIENT_AGENT_ERR.search(observed)
    assert not _TRANSIENT_AGENT_ERR.search("Execution error: assertion failed in kernel.py")


# --- fresh-start contamination gate ----------------------------------------

def _fresh_layout(tmp_path):
    """A minimal operators/ layout the gate walks: op dir, status file, recon tree."""
    op_dir = tmp_path / "operators" / "demo"
    (op_dir / "student").mkdir(parents=True)
    status = op_dir / "student" / "status.jsonl"
    status.touch()
    meta_status = op_dir / "meta_status.jsonl"
    meta_status.touch()
    recon = tmp_path / "operators" / "_recon"
    recon.mkdir()
    return op_dir, op_dir / "milestone.md", status, meta_status, recon


def test_fresh_clean_tree_has_no_findings(tmp_path):
    from forgekernel.loop.cli import fresh_start_contamination
    assert fresh_start_contamination(*_fresh_layout(tmp_path)) == []


def test_every_phase_product_is_named(tmp_path):
    """One finding per artifact, each naming the phase that would be skipped —
    the launch-script preflights this gate replaces printed exactly which file
    blocked the start, and losing that would make refusals slower to act on."""
    from forgekernel.loop.cli import fresh_start_contamination
    op_dir, milestone, status, meta_status, recon = _fresh_layout(tmp_path)
    milestone.write_text("# v1\n", encoding="utf-8")
    (status.parent / "recon_complete.json").write_text("{}", encoding="utf-8")
    meta_status.write_text('{"meta_step": 0}\n', encoding="utf-8")
    (op_dir / "arch_baseline.json").write_text("{}", encoding="utf-8")
    (op_dir / "ref_baseline.json").write_text("{}", encoding="utf-8")
    for name in ("report.md", "candidates.md", "derivation.md"):
        (recon / name).write_text("x\n", encoding="utf-8")
    (recon / "notes").mkdir()
    (recon / "notes" / "probe_1.py").write_text("pass\n", encoding="utf-8")
    findings = fresh_start_contamination(op_dir, milestone, status, meta_status, recon)
    assert len(findings) == 9
    joined = "\n".join(findings)
    assert "distillation would be skipped" in joined
    assert "reconnaissance would be skipped" in joined
    assert "meta history" in joined
    assert "arch_baseline.json" in joined and "ref_baseline.json" in joined
    assert "notes/ is not empty" in joined


def test_records_in_status_admit_everything(tmp_path):
    """Resume is the case the products exist FOR: any recorded iteration means the
    run already owns what is on disk, and re-litigating it would turn every guard
    restart into a fresh-start refusal."""
    from forgekernel.loop.cli import fresh_start_contamination
    op_dir, milestone, status, meta_status, recon = _fresh_layout(tmp_path)
    milestone.write_text("# v1\n", encoding="utf-8")
    (op_dir / "arch_baseline.json").write_text("{}", encoding="utf-8")
    status.write_text('{"iter": 0}\n', encoding="utf-8")
    assert fresh_start_contamination(op_dir, milestone, status, meta_status, recon) == []


def test_blank_lines_are_not_records(tmp_path):
    """touch + stray newlines must still read as a fresh start, or a crashed
    launch that got as far as `touch status.jsonl` would slip past the gate."""
    from forgekernel.loop.cli import _status_has_records
    p = tmp_path / "status.jsonl"
    p.write_text("\n  \n", encoding="utf-8")
    assert not _status_has_records(p)
    assert not _status_has_records(tmp_path / "missing.jsonl")


def test_no_recon_dir_skips_recon_checks(tmp_path):
    """Source-mode runs have no recon tree; a report.md elsewhere is not theirs."""
    from forgekernel.loop.cli import fresh_start_contamination
    op_dir, milestone, status, meta_status, recon = _fresh_layout(tmp_path)
    (recon / "report.md").write_text("x\n", encoding="utf-8")
    assert fresh_start_contamination(op_dir, milestone, status, meta_status, None) == []


def test_empty_notes_dir_is_clean(tmp_path):
    """The factory ships the recon tree with an empty notes/; its existence alone
    must not read as contamination."""
    from forgekernel.loop.cli import fresh_start_contamination
    op_dir, milestone, status, meta_status, recon = _fresh_layout(tmp_path)
    (recon / "notes").mkdir()
    assert fresh_start_contamination(op_dir, milestone, status, meta_status, recon) == []


def _pre_iter0_crash_residue(op_dir, milestone, status, meta_status, recon):
    """What a blackbox run's tree looks like after a crash before iteration 0:
    baselines measured, recon seeded/committed and mid-write, no record yet."""
    (op_dir / "arch_baseline.json").write_text("{}", encoding="utf-8")
    (op_dir / "ref_baseline.json").write_text("{}", encoding="utf-8")
    (recon / "report.md").write_text("findings so far\n", encoding="utf-8")
    (recon / "notes").mkdir()
    (recon / "notes" / "sass_tools.py").write_text("pass\n", encoding="utf-8")


def test_admitted_launch_resumes_pre_iter0_products(tmp_path):
    """The window the sentinel exists for: baselines + recon can run for hours
    before the first status record, and a blackbox run's recon tree is seeded
    and committed from round one — the guard's archive list does not cover it,
    so without admission every restart in that window would refuse (exit 9),
    be read as transient, restart, and refuse again until max_restarts."""
    from forgekernel.loop.cli import _admit_launch, fresh_start_contamination
    op_dir, milestone, status, meta_status, recon = _fresh_layout(tmp_path)
    _admit_launch(op_dir, status)  # what the first launch's gate pass wrote
    _pre_iter0_crash_residue(op_dir, milestone, status, meta_status, recon)
    assert fresh_start_contamination(op_dir, milestone, status, meta_status, recon) == []


def test_foreign_sentinel_is_not_an_admission(tmp_path):
    """A hand-copied op dir carries its sentinel along; one naming a different
    path is a previous run's admission, not this one's."""
    import json
    from forgekernel.loop.cli import fresh_start_contamination
    op_dir, milestone, status, meta_status, recon = _fresh_layout(tmp_path)
    (status.parent / "launch_admitted.json").write_text(json.dumps(
        {"op_dir": str(tmp_path / "operators" / "other"), "admitted_epoch": 1.0}),
        encoding="utf-8")
    _pre_iter0_crash_residue(op_dir, milestone, status, meta_status, recon)
    findings = fresh_start_contamination(op_dir, milestone, status, meta_status, recon)
    assert len(findings) == 4


def test_corrupt_sentinel_is_ignored(tmp_path):
    from forgekernel.loop.cli import fresh_start_contamination
    op_dir, milestone, status, meta_status, recon = _fresh_layout(tmp_path)
    (status.parent / "launch_admitted.json").write_text("not json", encoding="utf-8")
    (op_dir / "ref_baseline.json").write_text("{}", encoding="utf-8")
    findings = fresh_start_contamination(op_dir, milestone, status, meta_status, recon)
    assert len(findings) == 1


def test_banked_pricing_evidence_is_a_phase_product(tmp_path):
    """A source-mode run banks ref/student pricing evidence under operators/<op>/
    evidence/ (there is no recon tree to put it beside); a fresh run must not
    read a previous run's role table as its own."""
    from forgekernel.loop.cli import fresh_start_contamination
    op_dir, milestone, status, meta_status, recon = _fresh_layout(tmp_path)
    ev = op_dir / "evidence"
    ev.mkdir()
    assert fresh_start_contamination(op_dir, milestone, status, meta_status, None,
                                     evidence_dir=ev) == []
    (ev / "ref_role_exposure.txt").write_text("ROLE-JSON: {}\n", encoding="utf-8")
    (recon / "student_orch_dossier.txt").write_text("x\n", encoding="utf-8")
    findings = fresh_start_contamination(op_dir, milestone, status, meta_status, recon,
                                         evidence_dir=ev)
    assert len(findings) == 2 and all("pricing evidence" in f for f in findings)



def test_ref_pricing_defaults_to_the_recon_tree_on_a_blackbox_run(tmp_path, capsys):
    """The ref's overlap account is priced on the recon tree unless loop.toml
    opts out explicitly — a loop.toml that predated the knob once left a whole
    dsa_sparse run with a null ref exposure column."""
    from forgekernel.loop.cli import _resolve_ref_pricing
    recon = tmp_path / "operators" / "_dsaref"
    (recon / ".git").mkdir(parents=True)
    made = []
    mk = lambda op, d: made.append((op, d)) or object()
    recon_be = object()

    # blackbox, tree has the student-style launch target -> same repo, same backend
    (recon / "eval").mkdir()
    (recon / "eval" / "ncu_run.py").write_text("pass\n")
    d, be, app = _resolve_ref_pricing({"loop": {"recon_op": "_dsaref"}}, tmp_path,
                                      recon, recon_be, mk)
    assert (d, be, app) == (recon, recon_be, None) and made == []

    # no launch target and no ref_price_app -> loud preflight WARNING, but the
    # tree stays wired: the driver re-checks at every pricing call, so a teacher
    # that adds eval/ncu_run.py during reconnaissance still gets the ref priced
    (recon / "eval" / "ncu_run.py").unlink()
    d, be, app = _resolve_ref_pricing({"loop": {"recon_op": "_dsaref"}}, tmp_path,
                                      recon, recon_be, mk)
    assert d == recon and be is recon_be
    assert "OVERLAP ACCOUNT" in capsys.readouterr().err
    # the normalized recon name wins over the raw key (a blackbox card with
    # recon_op = "" still reconnoitres in _recon)
    d, be, app = _resolve_ref_pricing({"loop": {"recon_op": ""}}, tmp_path,
                                      recon, recon_be, mk, recon_op="_dsaref")
    assert d == recon

    # ref_price_app names the tree's own entry -> priced with --app
    d, be, app = _resolve_ref_pricing(
        {"loop": {"recon_op": "_dsaref",
                  "ref_price_app": ["python3", "dsa_ref.py", "ncu"]}},
        tmp_path, recon, recon_be, mk)
    assert d == recon and be is recon_be and app == ["python3", "dsa_ref.py", "ncu"]

    # explicit opt-out
    d, be, app = _resolve_ref_pricing({"loop": {"recon_op": "_dsaref", "ref_price_op": ""}},
                                      tmp_path, recon, recon_be, mk)
    assert d is None and be is None

    # a source-mode run names its own tree and gets its own backend
    src = tmp_path / "operators" / "_myref"
    (src / ".git").mkdir(parents=True)
    (src / "eval").mkdir()
    (src / "eval" / "ncu_run.py").write_text("pass\n")
    d, be, app = _resolve_ref_pricing({"loop": {"ref_price_op": "_myref"}}, tmp_path,
                                      None, None, mk)
    assert d == src and made == [("_myref", src)]
    # source-mode run with no knob: nothing to price, and no default either
    assert _resolve_ref_pricing({"loop": {}}, tmp_path, None, None, mk) == (None, None, None)


# --- source-mode runs may reconnoitre (opt-in via [loop] recon_op) -------------

def test_recon_op_defaults_on_for_every_distilling_run():
    from forgekernel.loop.cli import _recon_op
    assert _recon_op({}, ORACLE_BLACKBOX) == "_recon"
    assert _recon_op({"loop": {"recon_op": "_ref"}}, ORACLE_BLACKBOX) == "_ref"
    # a source-mode run reconnoitres by default, from the same default tree
    assert _recon_op({}, ORACLE_SOURCE) == "_recon"
    assert _recon_op({"loop": {"oracle": "x.cu"}}, ORACLE_SOURCE) == "_recon"
    assert _recon_op({"loop": {"recon_op": "_ref"}}, ORACLE_SOURCE) == "_ref"
    # a source-only control run is declared, not implied
    assert _recon_op({"loop": {"recon_op": ""}}, ORACLE_SOURCE) is None
    assert _recon_op({"loop": {"recon_op": False}}, ORACLE_SOURCE) is None
    # blackbox cannot opt out: it has nothing else to distil from
    assert _recon_op({"loop": {"recon_op": ""}}, ORACLE_BLACKBOX) == "_recon"


def test_blackbox_recon_opt_out_is_refused_out_loud(capsys):
    """A blackbox card that wrote recon_op = "" still reconnoitres from the
    default tree — but the config said one thing and the run does another, so
    the rewrite is logged rather than silent."""
    from forgekernel.loop.cli import _recon_op
    assert _recon_op({"loop": {"recon_op": ""}}, ORACLE_BLACKBOX) == "_recon"
    err = capsys.readouterr().err
    assert "cannot opt out" in err and "_recon" in err
    # a real tree name is taken as written, without noise
    assert _recon_op({"loop": {"recon_op": "_ref"}}, ORACLE_BLACKBOX) == "_ref"
    assert capsys.readouterr().err == ""


# --- [loop] ref_in_job -------------------------------------------------------------

def test_ref_in_job_is_meaningful_only_with_a_separately_measured_ref(capsys):
    from forgekernel.loop.cli import _resolve_ref_in_job
    sep = {"ref_mode": "fp8gref", "submit_op": "_fp8gref"}
    assert _resolve_ref_in_job({"loop": {"ref_in_job": True}}, sep) is True
    assert _resolve_ref_in_job({"loop": {}}, sep) is False
    assert capsys.readouterr().err == ""
    # the ref already rides the job: redundant, said out loud
    assert _resolve_ref_in_job({"loop": {"ref_in_job": True}}, {"cmd": "python3 ref.py"}) is False
    assert "redundant" in capsys.readouterr().err
    # no reference at all: nothing to be in the job
    assert _resolve_ref_in_job({"loop": {"ref_in_job": True}}, {}) is False
    assert "no reference is declared" in capsys.readouterr().err

# --- teacher / student agent split ---------------------------------------------
# A run may put its teacher turns on one model and the student on another
# (--teacher-agent-cmd). The role is chosen per call; a backend with no teacher
# command serves both roles from --agent-cmd, exactly as single-model runs do.

def test_run_agent_routes_the_teacher_role_to_the_teacher_command(monkeypatch):
    b = _backend(monkeypatch, agent_retries=1)
    b.agent_cmd = "student-agent"
    b.teacher_agent_cmd = "teacher-agent"
    seen = []

    def fake_turn(cmd, prompt, role):
        seen.append(cmd)
        return 0, "ok", ""

    monkeypatch.setattr(b, "_agent_turn", fake_turn)
    assert b.run_agent("p") == "ok"
    assert b.run_agent("p", role="teacher") == "ok"
    assert b.run_agent("p", role="student") == "ok"
    assert seen == ["student-agent", "teacher-agent", "student-agent"]


def test_run_agent_without_a_teacher_command_serves_both_roles(monkeypatch):
    b = _backend(monkeypatch, agent_retries=1)
    b.agent_cmd = "only-agent"
    seen = []

    def fake_turn(cmd, prompt, role):
        seen.append(cmd)
        return 0, "ok", ""

    monkeypatch.setattr(b, "_agent_turn", fake_turn)
    b.run_agent("p", role="teacher")
    assert seen == ["only-agent"]
    assert b.agent_cmd_for("teacher") == "only-agent"



def test_an_adopted_job_whose_log_never_arrives_is_not_adopted_again(monkeypatch, tmp_path):
    """A terminal job with no log raises from _fetch_log — and used to leave its record
    behind, so every restart adopted the same dead job, failed the same way, and never
    resubmitted. The record counts its misses and is given up once retrying is clearly
    futile."""
    import json as _json

    from forgekernel.loop import cli

    ws = _adoption_ws(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "_forge", _fake_forge(calls, log=""))   # nothing ever logged
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0)
    monkeypatch.setattr(cli.RealBackend, "_branch_sha", lambda self, b: "abc")
    (ws / ".git" / "fk-inflight.json").write_text(_json.dumps(
        {"job": "9", "branch": "br", "mode": "all", "sha": "abc"}))

    rec_path = ws / ".git" / "fk-inflight.json"
    for attempt in range(1, cli._INFLIGHT_LOG_ATTEMPTS + 1):
        with pytest.raises(cli.BackendError, match="no log"):
            be.submit_gpu("br", "all")
        assert "submit" not in calls                            # always adopted, never resubmitted
        if attempt < cli._INFLIGHT_LOG_ATTEMPTS:
            assert rec_path.exists(), "given up on the first miss"
            assert _json.loads(rec_path.read_text())["log_failures"] == attempt
    assert not rec_path.exists()                                # …and eventually let go


# --- a turn in progress says so, and a dead one says that too -----------------
# Observed: a teacher turn ran long, the run log stayed empty for five hours
# because the agent ran under a blocking `subprocess.run` with the output
# captured, and a watchdog that judges a wedged loop by log silence reclaimed
# a turn that was still alive. The turn's own failures were likewise
# invisible: they went to the agent wrapper's log, never to the run log.

def test_a_long_turn_prints_a_heartbeat(monkeypatch, tmp_path, capsys):
    b = _backend(monkeypatch)
    b.workspace = tmp_path      # the turn really runs; it needs a real cwd
    b.agent_cmd = "fake-agent"
    b.agent_heartbeat_s = 0.05
    # A command that outlives several heartbeats, then answers.
    out = b._agent_turn("sleep 0.35; echo done", "prompt", "teacher")
    assert out[0] == 0 and out[1].strip() == "done"
    beats = [ln for ln in capsys.readouterr().out.splitlines()
             if "agent turn (teacher) in progress" in ln]
    assert beats, "a turn longer than the heartbeat printed nothing"


def test_the_heartbeat_can_be_switched_off(monkeypatch, tmp_path, capsys):
    b = _backend(monkeypatch)
    b.workspace = tmp_path
    b.agent_cmd = "fake-agent"
    b.agent_heartbeat_s = 0
    assert b._agent_turn("echo done", "prompt", "student")[1].strip() == "done"
    assert "in progress" not in capsys.readouterr().out


def test_a_failed_attempt_reaches_the_run_log(monkeypatch, capsys):
    from forgekernel.loop.driver import AgentError
    b = _agent_backend(monkeypatch, [_FakeProc(124, stderr="killed at 9000s")] * 3,
                       agent_retries=3)
    with pytest.raises(AgentError):
        b.run_agent("go")
    printed = capsys.readouterr().out
    assert "agent turn (student) failed, attempt 1/3" in printed
    assert "killed at 9000s" in printed


def test_an_interrupted_turn_kills_the_agent_process(monkeypatch, tmp_path):
    """Ctrl-C lands on the waiting thread while the pump owns the pipes.
    `subprocess.run` killed the child on its way out; the Popen driven from a
    thread must do the same — a claude process that outlives the loop keeps
    spending API usage."""
    import threading as _threading
    import types
    from forgekernel.loop import cli as cli_mod
    b = _backend(monkeypatch)
    b.workspace = tmp_path
    b.agent_cmd = "fake-agent"
    b.agent_heartbeat_s = 0.05
    spawned = []
    real_popen = cli_mod.subprocess.Popen

    def spy(*a, **k):
        p = real_popen(*a, **k)
        spawned.append(p)
        return p

    monkeypatch.setattr(cli_mod.subprocess, "Popen", spy)

    class Interrupting(_threading.Thread):
        hits = 0

        def join(self, timeout=None):
            Interrupting.hits += 1
            if Interrupting.hits == 1:
                raise KeyboardInterrupt        # the operator hit Ctrl-C mid-turn
            return super().join(timeout)

    monkeypatch.setattr(cli_mod, "threading", types.SimpleNamespace(Thread=Interrupting))
    with pytest.raises(KeyboardInterrupt):
        b._agent_turn("sleep 30", "prompt", "teacher")
    assert spawned, "the turn never started a process"
    assert spawned[0].wait(timeout=5) is not None      # killed, not orphaned


def test_the_heartbeat_interval_is_a_launch_knob(monkeypatch):
    # --agent-heartbeat-s reaches the backend; 0 switches the heartbeat off
    assert _backend(monkeypatch, agent_heartbeat_s=0).agent_heartbeat_s == 0
    assert _backend(monkeypatch).agent_heartbeat_s == 600.0


def test_harvest_only_runs_the_write_up_and_nothing_else(tmp_path, monkeypatch, capsys):
    """A run stopped before its finish path left its axis bill unpaid: --harvest-only
    runs the one write-up turn — no loop, no confirmation job, no publish — and
    refuses without a teacher or without a trajectory to harvest."""
    import types
    from forgekernel.loop import cli
    status = tmp_path / "status.jsonl"
    status.write_text('{"iter": 0}\n', encoding="utf-8")
    ctx = types.SimpleNamespace(status_path=status, kb_dir=tmp_path / "hw_kb",
                                kb_lint_tokens=None)
    called = []
    monkeypatch.setattr(cli, "run_kb_harvest",
                        lambda c: called.append(c) or ["hw_kb/cutedsl/sm100/x.md"])
    monkeypatch.setattr(cli, "git_status", lambda root, paths: [])
    monkeypatch.setattr(cli, "run_two_layer", lambda *a, **k: pytest.fail("no loop"))
    monkeypatch.setattr(cli, "run_best_confirmation", lambda *a, **k: pytest.fail("no job"))
    args = types.SimpleNamespace(teacher_agent_cmd=None, agent_cmd="bash scripts/agent-legs.sh")
    assert cli._harvest_only(ctx, args, tmp_path, tmp_path / "m.md", None) == 0
    assert called == [ctx] and ctx.kb_lint_tokens == []
    assert '[harvest-only] done: {"harvest_only": true, "kb_harvest": ["hw_kb/cutedsl/sm100/x.md"]' \
        in capsys.readouterr().out
    args.agent_cmd = None
    assert cli._harvest_only(ctx, args, tmp_path, tmp_path / "m.md", None) == 2
    args.agent_cmd = "bash scripts/agent-legs.sh"
    status.write_text("", encoding="utf-8")
    assert cli._harvest_only(ctx, args, tmp_path, tmp_path / "m.md", None) == 2
    assert len(called) == 1
