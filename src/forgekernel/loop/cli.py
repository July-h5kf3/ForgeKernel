"""``forge <op> loop`` — run the two-layer loop unattended on a generated card.

Wires the deterministic driver core (``forgekernel.loop.driver``) to real
adapters: an agent CLI (for student/meta turns), the GPU harness
(``forgekernel.submit``), and git (workspace commit/push/reset). The core loop is
unit-tested with a fake backend; this module is the side-effecting boundary.

``forge run`` (``forgekernel.launch``) is the only caller: it generates the card and
its loop.toml from the task and passes the budget, the target and the two agent
commands. Everything else the loop needs is read from the card (loop.toml,
problem.json); the knobs that used to be arguments are the constants below.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import tempfile
import sys
import threading
import time
import tomllib
from dataclasses import asdict, replace as dc_replace
from pathlib import Path

from forgekernel.baseline import (
    ARCH_BASELINE_FILENAME,
    BASELINE_FILENAME,
    ROLE_ARCHITECTURE,
    ROLE_LIBRARY,
    RefBaseline,
    baseline_path,
    load_baseline,
    save_baseline,
    workloads_mismatch,
)
from forgekernel.inner.trajectory import GATE_WINDOW_DEFAULT, GATE_WINDOWS
from forgekernel.loop import agent_policy as agent_policy_mod
from forgekernel.loop import dsl_source as dsl_source_mod
from forgekernel.loop import goal_ref as goal_ref_mod
from forgekernel.loop import harness_guard, precision_gate
from forgekernel.loop.agent_policy import AgentPolicy
from forgekernel.loop.driver import (
    EVIDENCE_FILE_NAMES,
    JOB_ENV_PREFLIGHT_POLICIES,
    _RECON_STATE_NAME,
    AgentError,
    BackendError,
    LoopContext,
    NoRunError,
    apply_window_credit,
    job_env_preflight,
    kb_read_dir,
    run_best_confirmation,
    run_kb_harvest,
    run_two_layer,
    archive_is_whole,
)
from forgekernel.loop.harvest import format_report, git_status, kb_blockers
from forgekernel.loop.parse import ENTRY_DONE_RE
from forgekernel import monitor as monitor_mod
from forgekernel.window_contract import (
    EXIT_WINDOW_UNDECLARED,
    WINDOW_FILENAME,
    WindowContract,
    bench_prep_findings,
    bench_times_workloads,
    bench_workloads_preflight,
    load_window_contract,
    window_findings,
    window_warnings,
)
from forgekernel.loop.kb_lint import derive_reference_tokens, lint_kb_dir
from forgekernel.loop import lang_mode
from forgekernel.paths import operators_dir, repo_root
from forgekernel.submit.jobs import (STATUS_LOST, STATUS_RUNNING,
                                     STATUS_SUCCEEDED, TRANSPORT_ERROR_MARKER,
                                     clock_lock_refusal, is_job, job_log, job_status)

# The run's outcome as the loop ends it (``stop``: target, meta_budget, infeasible, ...),
# beside the operator tree; `forge run` reads it for its exit code.
SUMMARY_FILE = "loop_summary.json"

# -- the loop's fixed settings ---------------------------------------------------------------
# What used to be `forge <op> loop` arguments and loop.toml knobs (docs/design/v1-launch.md
# §12, Appendix A): set to the values the runs that worked well used, not configurable.
GIT_NAME = "forge-loop"
GIT_EMAIL = "forge-loop@local"
REF_REPEAT = 3                   # reference measurements; the fastest is kept
AGENT_HEARTBEAT_S = 600.0        # a progress line while a turn runs (0 = none)
KB_TRIPWIRE_ROUNDS = 6           # a student probe this many GPU rounds deep earns a KB write-up
CONFIRM_JOBS = 3                 # the finish re-measures the best sha this many times
MAX_PROBE_ROUNDS = 4             # GPU layout-probe rounds per student iteration
MAX_VERDICT_PROBE_ROUNDS = 4     # GPU verdict-experiment rounds per diagnose step
INFEASIBLE_MIN_DIAGNOSES = 1     # plateau diagnoses before an INFEASIBLE stop can be accepted
LEVER_PRICING_ROUNDS = 3         # interleaved timing rounds per variant in a pricing job
LEVER_PRICING_MAX_LEVERS = 5
MAX_CONSECUTIVE_REVERTS = 3      # before the inner loop yields to the teacher (stuck)
REOPEN_MAX_FRUITLESS = 2         # fruitless reopen transactions before reopening closes
REPRICE_MIN_GAIN_PCT = 3.0       # a new best this much faster re-prices the cited levers
WATERMARK_MARGIN = 0.03          # a new best beyond this fraction is re-measured first
REANCHOR_BAND = 0.06             # a gap residual inside this band re-anchors the ref first

# Statuses a job can still move on from. Everything else is terminal.
_NON_TERMINAL = {STATUS_RUNNING}

# Terminal *and* the job actually ran, so its log must exist (see _fetch_log).
_SUCCEEDED = {STATUS_SUCCEEDED}
# The operator entry script's closing line (gpu_entry.sh prints it after the
# task, whatever the task was): once it is the log's LAST line, the log is
# whole (driver.archive_is_whole — anywhere else it is a nested entry's).
_ENTRY_DONE_RE = ENTRY_DONE_RE

# The runner went away without a verdict (a reboot, an OOM kill): nothing says the job
# ran, so failing the eval would charge the kernel for the machine — resubmit instead.
_LOST = {STATUS_LOST}

# How many times a job's log may fail to arrive before the in-flight record is given
# up: a restart re-adopts a recorded job, and one whose log never comes must not be
# re-adopted forever.
_INFLIGHT_LOG_ATTEMPTS = 3


class RunnerLostError(NoRunError):
    """Every attempt of the resubmit budget lost its runner without a verdict. Nothing
    ran to completion, so no caller may charge it to the tree."""


class ClockLockRefusedError(NoRunError):
    """The card's gpu_entry kept refusing to time because the board's clock lock did
    not hold on the GPU (jobs.CLOCK_LOCK_REFUSED_MARKER). Nothing was judged or timed,
    so it is a NoRunError — no caller may charge it to the tree, the iteration budget,
    or a meta tool's attempts."""


class JobTransportError(BackendError):
    """The job could not be delivered to the GPU machine (TRANSPORT_ERROR_MARKER in
    its log: the copy or the ssh session failed every attempt). Not the kernel's
    failure; the fix is on the machine side (the ssh alias, the host, its disk)."""

def _end_process_group(proc: subprocess.Popen, grace_s: float = 8.0) -> None:
    """SIGTERM the process group, then SIGKILL whatever is left after ``grace_s``.

    The group is the one Popen(start_new_session=True) created, so this reaches the leg
    scripts and the model process without touching anything else on the machine."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if proc.poll() is None:
        proc.kill()


# Server-side API trouble the agent CLI surfaces as a non-zero exit: overload (529),
# rate limits (429), 5xx, and connection-level drops. These clear on their own — the
# only wrong response is giving up inside the outage window, which is exactly what
# the three-attempt hiccup budget did (90 s of patience against an hours-long 529
# incident ended a run overnight). Auth/config failures deliberately do
# NOT match: retrying a bad model name or an expired login only hides the misconfig.
_TRANSIENT_AGENT_ERR = re.compile(
    r"API Error[^\n]{0,60}\b(?:408|409|429|5\d\d)\b"
    r"|\boverloaded\b|rate.?limit|too many requests|service unavailable"
    r"|internal server error|connection (?:reset|refused|closed|error)"
    r"|ECONNRESET|ETIMEDOUT|EAI_AGAIN|fetch failed|request timed? ?out"
    # An upstream SSE stream cut mid-generation ("upstream stream ended without a
    # terminal frame", reason=eof) is a transient API-side disconnect, not a
    # misconfig: it repeatedly killed closed-source runs, and the
    # string carries no HTTP status code so none of the branches above matched.
    r"|stream ended without a terminal frame"
    # claude-code's model recognition tracks a dynamic config fetch and fails
    # intermittently. It can fire on
    # auxiliary queries too — a run died 4h49m in when it hit the
    # generate_session_title path — so it belongs to the outage budget, not the
    # 3-attempt one.
    r"|\[claude-code:unrecognized_model\]",
    re.I)



def poll_until_terminal(status_fn, job, *, poll_s, timeout_s, now, sleep):
    """Poll status_fn() until it reports something outside _NON_TERMINAL, or raise
    BackendError once timeout_s elapses.

    An empty status stays non-terminal on purpose: that is a read failure, not a
    verdict, and treating it as done would hand the caller a job whose real state is
    unknown. Clock/sleep are injected so the deadline is unit-testable without
    wall-clock waits.
    """
    deadline = now() + timeout_s
    while True:
        status = status_fn()
        if status and status not in _NON_TERMINAL:
            return status
        if now() >= deadline:
            raise BackendError(
                f"job {job} did not reach a terminal state within {timeout_s:.0f}s "
                f"(last status={status!r}); giving up to avoid an unbounded poll")
        sleep(poll_s)


def _forge(op: str, *args: str, timeout: float = 600.0,
           env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "forgekernel.submit", op, *args],
        capture_output=True, text=True, cwd=repo_root(), timeout=timeout,
        env=None if env is None else {**os.environ, **env},
    )


def _entry_task(mode: str) -> str:
    """The gpu_entry task a submit ``--mode`` runs: the mode itself, or ``cmd``
    for ``cmd:<command>`` — what the entry script's closing line names."""
    return (mode or "").split(":", 1)[0]



class RealBackend:
    """Agent CLI + GPU harness + git. cwd for git ops is the student workspace."""

    inflight_file = "fk-inflight.json"

    # A job the card's gpu_entry refused to time because the board's clock lock did not
    # hold on the GPU (jobs.CLOCK_LOCK_REFUSED_MARKER) is never returned to a caller:
    # submit_gpu resubmits it up to this many times, then stops with
    # ClockLockRefusedError (nothing is charged). The state rarely clears by waiting:
    # a GPU that would not lock its clocks kept refusing — a few retries rule out a
    # one-off; the machine needs a look.
    clock_lock_retries = 3
    clock_lock_retry_sleep_s = 30.0

    def __init__(self, op: str, workspace: Path, agent_cmd: str | None,
                 git_name: str, git_email: str, poll_s: float = 3.0,
                 agent_retries: int = 3, agent_retry_sleep: float = 15.0,
                 agent_heartbeat_s: float = 600.0,
                 git_retries: int = 3, git_retry_sleep: float = 10.0,
                 poll_timeout_s: float = 14400.0, lost_retries: int = 3,
                 api_outage_retries: int = 30, api_outage_sleep_cap: float = 600.0,
                 teacher_agent_cmd: str | None = None,
                 agent_policy: AgentPolicy | None = None):
        self.op = op
        self.workspace = workspace
        self.agent_cmd = agent_cmd
        # What the agent may reach from this machine (network, reference source); None
        # = unrestricted, which only tests and GPU-only backends construct.
        self.agent_policy = agent_policy
        # Teacher turns (recon / distill / diagnose / patch / KB) may run on a
        # different model than the student; None = the student's command serves
        # both roles.
        self.teacher_agent_cmd = teacher_agent_cmd
        self.git_name = git_name
        self.git_email = git_email
        # A job's status and log are files its runner writes (forgekernel.submit.jobs),
        # so a poll costs a file read.
        self.poll_s = poll_s
        self.agent_retries = agent_retries
        self.agent_retry_sleep = agent_retry_sleep
        # How often a turn in progress says so: a teacher turn legitimately runs for
        # hours, and a run log that goes quiet that long reads as a wedged loop.
        # 0 disables the heartbeat.
        self.agent_heartbeat_s = agent_heartbeat_s
        self.git_retries = git_retries
        self.git_retry_sleep = git_retry_sleep
        self.lost_retries = lost_retries  # extra submits after a job lost its runner
        # The machine the most recent GPU submission ran on — the driver stamps every
        # record with it. "" until the first submit, or when the submit config cannot
        # be read.
        self.last_cluster = ""
        # Agent API-outage patience: a separate budget so server-side trouble does not
        # consume the hiccup one (~4 h at the default cap).
        self.api_outage_retries = api_outage_retries
        self.api_outage_sleep_cap = api_outage_sleep_cap
        # [loop] job_env_files: tree path -> the card's copy, overlaid on every
        # sha this backend pushes for a job (see _with_job_env). Empty = off.
        self.job_env_files: dict[str, Path] = {}
        self._job_env_announced: dict[str, str] = {}
        # [loop] harness_guard (loop/harness_guard.py): the card whose tracked harness
        # files (gpu_entry.sh, eval/, ref_bin/, .gitattributes) every job tree carries
        # instead of the lineage's; None = no harness overlay. kernel_file is never
        # harness (it is the student's).
        self.harness_card: Path | None = None
        self.kernel_file = ""
        self.last_mode = ""  # the gpu_entry mode of last_job_id (its closing line names the task)
        self.poll_timeout_s = poll_timeout_s  # per-job poll bound
        self.last_job_id = ""  # set by submit_gpu; provenance for whoever records results

    @property
    def machine(self) -> str:
        """Where this op's jobs run ("local" or the ssh host); "" when unreadable."""
        try:
            from forgekernel.submit.config import load_config
            return load_config(self.op).cluster
        except Exception:
            return ""

    def agent_cmd_for(self, role: str = "student") -> str | None:
        if role == "teacher" and self.teacher_agent_cmd:
            return self.teacher_agent_cmd
        return self.agent_cmd

    def _agent_turn(self, cmd: str, prompt: str, role: str) -> tuple[int, str, str]:
        """One agent invocation, announcing itself while it thinks.

        `subprocess.run` would block with the output captured, so a 40-minute
        teacher turn and a wedged one look identical in the run log — a watchdog
        that reclaims a loop whose log has gone quiet cannot tell them apart and
        can reclaim a live turn. The work is the same; a thread
        drains the pipes so this side can print a line every
        ``agent_heartbeat_s`` seconds.
        """
        proc = subprocess.Popen(
            cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=self.workspace, start_new_session=True,
        )
        box: dict = {}

        def _pump() -> None:
            try:
                box["out"], box["err"] = proc.communicate(prompt)
            except BaseException as e:      # noqa: BLE001 — re-raised on the caller's thread
                box["exc"] = e

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()
        beat = self.agent_heartbeat_s
        started = time.monotonic()
        try:
            while True:
                pump.join(timeout=beat if beat and beat > 0 else None)
                if not pump.is_alive():
                    break
                print(f"[loop] agent turn ({role}) in progress — "
                      f"{(time.monotonic() - started) / 60:.0f} min so far", flush=True)
        except BaseException:
            # Ctrl-C / SystemExit lands on THIS thread while the pump owns the
            # pipes. `subprocess.run` killed the child on its way out; a Popen
            # driven from a thread must do that by hand, or the agent (a claude
            # process) outlives the loop and keeps spending API usage.
            # Give a transport adapter time to stop its Orca-owned terminal as well.
            _end_process_group(proc, grace_s=35.0)
            pump.join(timeout=5)
            raise
        if "exc" in box:
            raise box["exc"]
        return proc.returncode, box.get("out", ""), box.get("err", "")

    def run_agent(self, prompt: str, *, role: str = "student") -> str:
        cmd = self.agent_cmd_for(role)
        if not cmd:
            raise AgentError("no --agent-cmd configured; cannot spawn an agent")
        if self.agent_policy is not None:
            # Rewritten before every turn, not once: teacher and student turns share the
            # working directory, and nothing a turn does to the file may outlive it.
            self.agent_policy.install(role, self.workspace)
            prompt = self.agent_policy.notice(role) + prompt
        last = ""
        attempt = 0
        outage_waits = 0
        while attempt < self.agent_retries:
            rc, out, err = self._agent_turn(cmd, prompt, role)
            if "FK_AGENT_UNCERTAIN" in err or "FK_AGENT_UNSUPPORTED" in err:
                raise AgentError(err.strip())
            if rc == 0 and out.strip():
                from forgekernel.workspace import sync_agent_workspace
                sync_agent_workspace(self.workspace)
                return out
            last = (f"rc={rc} "
                    f"stderr={err[-300:].strip()!r} stdout={out[-200:].strip()!r}")
            # Say it in the run log, not only in whatever the leg script keeps:
            # a turn that died is the other half of the heartbeat, and without it
            # a reader cannot tell a retrying loop from a stalled one.
            print(f"[loop] agent turn ({role}) failed, attempt "
                  f"{attempt + 1}/{self.agent_retries}: {last}", flush=True)
            if (_TRANSIENT_AGENT_ERR.search(out[-2000:] + err[-2000:])
                    and outage_waits < self.api_outage_retries):
                outage_waits += 1
                sleep_s = min(self.api_outage_sleep_cap,
                              self.agent_retry_sleep * (2 ** outage_waits))
                print(f"[loop] agent API outage ({outage_waits}/{self.api_outage_retries}) "
                      f"— waiting {sleep_s:.0f}s before retrying: {last}", flush=True)
                time.sleep(sleep_s)
                continue  # deliberately does not consume a hiccup attempt
            attempt += 1
            time.sleep(self.agent_retry_sleep * attempt)  # transient backoff
        raise AgentError(f"agent command failed after {self.agent_retries} attempts: {last}")


    def submit_gpu(self, branch: str, mode: str) -> str:
        """Submit, wait, and return the job's log (_submit_gpu_once) — never a log in
        which the card's gpu_entry refused to time because the board's clock lock did not
        hold on the GPU.

        Such a job judged and timed nothing: handed back, parse_eval reads "no eval
        JSON" as a correctness FAIL, the driver records a REVERT and spends an
        iteration, and a meta tool spends an attempt — on the machine, not the payload.
        So it is resubmitted here, up to ``clock_lock_retries`` times in a row; past
        that the submit stops with ClockLockRefusedError — a NoRunError, which no caller
        charges.
        """
        refusals = 0
        while True:
            log = self._submit_gpu_once(branch, mode)
            why = clock_lock_refusal(log)
            if why is None:
                return log
            refusals += 1
            head = (f"[loop] job {self.last_job_id or '?'} ({mode}): the board's clock lock "
                    f"did not hold on {self.last_cluster or 'the GPU'} ({why[:160]}) — a "
                    f"machine fault, not the kernel: no verdict is recorded and nothing is "
                    f"charged")
            if refusals >= max(1, int(self.clock_lock_retries)):
                raise ClockLockRefusedError(f"{head} ({refusals} refusal(s) in a row)")
            print(f"{head}; resubmitting ({refusals}/{self.clock_lock_retries})", flush=True)
            time.sleep(self.clock_lock_retry_sleep_s)

    def _submit_gpu_once(self, branch: str, mode: str) -> str:
        """Submit (or adopt), poll, and return the job's log; a job that lost its runner
        without a verdict never produced output, so it is resubmitted rather than
        charged to the kernel.

        Before submitting, an in-flight job recorded by an interrupted run is adopted if
        it is still for this exact work (same branch/mode AND the branch has not moved):
        a job's runner outlives the loop that submitted it, and a restart would
        otherwise pay for the same eval twice.
        """
        self.last_cluster = self.machine
        adopted = self._adopt_inflight(branch, mode)
        lost = 0
        while True:
            if adopted:
                job, adopted = adopted, None
                print(f"[loop] adopting in-flight job {job} from an interrupted run "
                      f"instead of resubmitting", flush=True)
            else:
                sub = self._retry_call(
                    lambda: _forge(self.op, "submit", "--branch", branch, "--mode", mode),
                    what=f"submit {mode}")
                m = re.search(r"job=(\d+)", sub.stdout) or re.search(r'"id":\s*(\d+)', sub.stdout)
                if not m:
                    raise BackendError(
                        f"could not parse job id from submit output:\n{sub.stdout[-800:]}")
                job = m.group(1)
                self._record_inflight(job, branch, mode)
            # the job id is provenance the caller may need to record (a baseline cites it
            # in every gate it anchors); it is not recoverable from the logs this returns
            self.last_job_id = job
            self.last_mode = mode
            status = poll_until_terminal(
                lambda: self._job_status(job), job, poll_s=self.poll_s,
                timeout_s=self.poll_timeout_s, now=time.time, sleep=time.sleep)
            if status in _LOST:
                self._clear_inflight()  # a lost job is gone; never re-adopt it
                lost += 1
                print(f"[loop] job {job} lost its runner without a verdict; resubmitting "
                      f"({lost}/{self.lost_retries})", flush=True)
                if lost > self.lost_retries:
                    raise RunnerLostError(
                        f"job lost its runner {lost} times in a row (last: {job}); nothing "
                        f"ran to completion")
                continue
            try:
                log = self._fetch_log(job, status, task=_entry_task(mode))
            except BackendError:
                # A record whose log never comes must not be adopted forever: it counts
                # its misses and is given up once retrying is clearly futile.
                if self._note_inflight_log_failure(job) >= _INFLIGHT_LOG_ATTEMPTS:
                    print(f"[loop] job {job}: no log after {_INFLIGHT_LOG_ATTEMPTS} "
                          f"attempts — giving the job up rather than re-adopting a dead "
                          f"one", flush=True)
                    self._clear_inflight()
                raise
            self._clear_inflight()
            if TRANSPORT_ERROR_MARKER in log:
                detail = next((ln.strip() for ln in log.splitlines()
                               if TRANSPORT_ERROR_MARKER in ln), TRANSPORT_ERROR_MARKER)
                raise JobTransportError(
                    f"the job could not be delivered to {self.last_cluster or 'the GPU machine'}"
                    f" — check the ssh alias and the host: {detail[:600]}")
            return log

    def _op_dir(self) -> Path | None:
        try:
            from forgekernel.submit.config import load_config
            return Path(load_config(self.op).op_dir)
        except Exception:
            return None

    def _job_status(self, job: str) -> str:
        """The job's status where its runner writes it, else through `submit status`
        ("" when unreadable: the poll treats that as not done yet)."""
        op_dir = self._op_dir()
        try:
            if op_dir is not None and is_job(op_dir, job):
                return str(job_status(op_dir, job).get("status") or "")
            out = _forge(self.op, "status", job).stdout
            return str(json.loads(out.strip().splitlines()[-1]).get("status") or "")
        except Exception:
            return ""

    # In-flight job provenance lives in the workspace's git dir: survives the loop
    # process and every `git reset/add -A` the driver does, and can never be committed
    # into the operator branch or ride into a GPU job's tree. A workspace that is not a
    # git checkout (unit tests) simply has no state — the feature degrades to the old
    # always-resubmit behavior.
    def _inflight_path(self) -> Path | None:
        gd = self.workspace / ".git"
        if gd.is_dir():
            return gd / self.inflight_file
        try:
            text = gd.read_text()
        except OSError:
            return None
        if text.startswith("gitdir:"):  # linked worktree: .git is a pointer file
            p = Path(text.split(":", 1)[1].strip())
            if not p.is_absolute():
                p = (self.workspace / p).resolve()
            if p.is_dir():
                return p / self.inflight_file
        return None

    def _branch_sha(self, branch: str) -> str:
        proc = subprocess.run(["git", "-C", str(self.workspace), "rev-parse", branch],
                              capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else ""

    def _note_inflight_log_failure(self, job: str) -> int:
        """Count one failed attempt to read ``job``'s log; return the new count.

        Lives in the record itself, so the count survives the process — the misses
        that matter are spread across guard restarts, which is exactly when an
        unreadable job would otherwise be re-adopted forever.
        """
        path = self._inflight_path()
        if path is None:
            return _INFLIGHT_LOG_ATTEMPTS      # no record to count in: old behaviour
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return _INFLIGHT_LOG_ATTEMPTS
        if rec.get("job") != job:
            return 0
        rec["log_failures"] = int(rec.get("log_failures", 0)) + 1
        try:
            path.write_text(json.dumps(rec))
        except OSError:
            pass
        return rec["log_failures"]

    def _record_inflight(self, job: str, branch: str, mode: str) -> None:
        path = self._inflight_path()
        if path is None:
            return
        try:
            path.write_text(json.dumps(
                {"job": job, "branch": branch, "mode": mode,
                 "sha": self._branch_sha(branch),
                 # the machine the job went to: an adopted job's record must
                 # be stamped with where it RAN, not where the next submit
                 # would go
                 "cluster": self.last_cluster}))
        except OSError:
            pass  # provenance is best-effort; losing it costs one resubmit, not the run

    def _clear_inflight(self) -> None:
        path = self._inflight_path()
        if path is not None:
            path.unlink(missing_ok=True)

    def _adopt_inflight(self, branch: str, mode: str) -> str | None:
        """The recorded job id, iff it is still an answer to this exact question.

        Same branch and mode, and the branch tip still at the sha that was submitted —
        a student commit in between means the record answers an older question, and
        adopting it would score stale code. A stale or unreadable record is deleted so
        it cannot be re-examined forever; a valid one is kept until the log is fetched,
        so an outage that outlives this process still leaves the job adoptable.
        """
        path = self._inflight_path()
        if path is None or not path.exists():
            return None
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            rec = {}
        job = rec.get("job")
        if (not job or rec.get("branch") != branch or rec.get("mode") != mode
                or (rec.get("sha") and rec["sha"] != self._branch_sha(branch))):
            path.unlink(missing_ok=True)
            return None
        if rec.get("cluster"):
            self.last_cluster = rec["cluster"]
        return job

    def reread_log(self, job_id: str) -> str:
        """A finished job's log, read again by a caller that found the first read short
        of its verdict lines. Raises BackendError when there is none."""
        return self._fetch_log(job_id, STATUS_SUCCEEDED,
                               task=_entry_task(self.last_mode) if job_id == self.last_job_id else "")

    def _fetch_log(self, job: str, status: str, *, task: str = "") -> str:
        """A terminal job's log. The runner closes the log before it writes the verdict
        (jobs.runner_main), so what is there is all there will ever be.

        A succeeded job with no log is an infrastructure symptom, not an agent one, so
        it raises rather than returning "" to be misread.
        """
        op_dir = self._op_dir()
        if op_dir is not None and is_job(op_dir, job):
            log = job_log(op_dir, job)
        else:
            try:
                res = _forge(self.op, "logs", job)
                log = res.stdout if res.returncode == 0 else ""
            except subprocess.TimeoutExpired:
                log = ""
        if log.strip() and not archive_is_whole(log, task):
            print(f"[loop] job {job}: no closing entry line for task={task or '?'}; a "
                  f"finished job's log is final — taking it as complete ({len(log)} chars)",
                  flush=True)
        if not log.strip() and status in _SUCCEEDED:
            raise BackendError(
                f"job {job} reported {status} but left no log — refusing to report this "
                f"as an agent-side failure")
        return log

    def _retry_call(self, fn, what: str):
        """Retry a subprocess.CompletedProcess-returning call on non-zero rc / OS error,
        with a short backoff: a git or local hiccup clears in seconds."""
        last = ""
        for attempt in range(1, self.git_retries + 1):
            try:
                proc = fn()
            except Exception as e:  # OS hiccup
                last = repr(e)
            else:
                if getattr(proc, "returncode", 0) == 0:
                    return proc
                last = f"rc={proc.returncode} {getattr(proc, 'stderr', '')[-200:].strip()!r}"
            time.sleep(self.git_retry_sleep * attempt)  # transient backoff
        raise BackendError(f"{what} failed after {self.git_retries} attempts: {last}")

    def _git(self, *args: str, what: str) -> None:
        ws = str(self.workspace)
        self._retry_call(
            lambda: subprocess.run(["git", "-C", ws, *args], capture_output=True, text=True),
            what=what)

    def commit(self, message: str) -> str:
        self._git("add", "-A", what="git add")
        self._git("-c", f"user.name={self.git_name}", "-c", f"user.email={self.git_email}",
                  "commit", "-q", "--allow-empty", "-m", message, what="git commit")
        sha = subprocess.run(["git", "-C", str(self.workspace), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True)
        return sha.stdout.strip()

    def commit_paths(self, base: str, head: str) -> list[str] | None:
        """Paths ``head`` changes relative to ``base`` (committed trees, so exactly
        what a job of either would ship); None when git cannot say. The driver
        archives a commit that changes nothing outside notes/ on the evaluated
        reading of ``base`` instead of buying a second eval of the same bytes."""
        out = subprocess.run(["git", "-C", str(self.workspace), "diff", "--name-only",
                              "--no-renames", base, head], capture_output=True, text=True)
        if out.returncode != 0:
            return None
        return [p for p in out.stdout.splitlines() if p.strip()]

    # -- the card-owned evaluation harness (loop/harness_guard.py) -------------------

    def harness_changes(self, base: str, head: str) -> list[str] | None:
        """The harness paths ``head`` changes relative to ``base`` (added, removed or
        edited; committed trees). None when git cannot say."""
        out = subprocess.run(["git", "-C", str(self.workspace), "diff", "--name-only",
                              "--no-renames", base, head, "--", *harness_guard.HARNESS_PATHSPECS],
                             capture_output=True, text=True)
        if out.returncode != 0:
            return None
        return harness_guard.harness_only(out.stdout.splitlines(), self.kernel_file)

    def harness_drift(self, anchor: str) -> list[str] | None:
        """Harness paths where the WORKING TREE differs from ``anchor``: committed or not,
        and untracked files under the harness directories (a reset keeps those, the next
        ``git add -A`` would commit them). None when git cannot say."""
        ws = str(self.workspace)
        diff = subprocess.run(["git", "-C", ws, "diff", "--name-only", "--no-renames", anchor,
                               "--", *harness_guard.HARNESS_PATHSPECS],
                              capture_output=True, text=True)
        new = subprocess.run(["git", "-C", ws, "ls-files", "--others", "--exclude-standard",
                              "--", *harness_guard.HARNESS_PATHSPECS],
                             capture_output=True, text=True)
        if diff.returncode != 0 or new.returncode != 0:
            return None
        return harness_guard.harness_only(diff.stdout.splitlines() + new.stdout.splitlines(),
                                          self.kernel_file)

    def restore_harness(self, anchor: str, paths: list[str], message: str) -> str:
        """Put ``paths`` back to their ``anchor`` content (removed when ``anchor`` has no
        such file) and commit exactly those paths; returns the new HEAD. Nothing else in
        the working tree is staged: a kept probe or an uncommitted note stays as it was."""
        ws = str(self.workspace)
        for rel in paths:
            there = subprocess.run(["git", "-C", ws, "cat-file", "-e", f"{anchor}:{rel}"],
                                   capture_output=True, text=True).returncode == 0
            if there:
                self._git("checkout", anchor, "--", rel, what="git checkout (harness restore)")
            else:
                subprocess.run(["git", "-C", ws, "rm", "-q", "-f", "--cached",
                                "--ignore-unmatch", "--", rel], capture_output=True, text=True)
                try:
                    (self.workspace / rel).unlink()
                except OSError:
                    pass
        # `checkout <anchor> -- path` updates the index too, and the removals were taken
        # out of it above: the index holds exactly the restore, nothing else is staged
        staged = subprocess.run(["git", "-C", ws, "diff", "--cached", "--quiet"],
                                capture_output=True, text=True)
        if staged.returncode != 0:
            self._git("-c", f"user.name={self.git_name}", "-c", f"user.email={self.git_email}",
                      "commit", "-q", "-m", message, what="git commit (harness restore)")
        return self.head()

    def dirty(self) -> bool:
        """Does this tree have changes git would commit? Used to skip an empty
        bookkeeping commit; a git failure reads as "dirty" so the caller banks
        rather than silently skipping evidence."""
        out = subprocess.run(["git", "-C", str(self.workspace), "status", "--porcelain"],
                             capture_output=True, text=True)
        return out.returncode != 0 or bool(out.stdout.strip())

    def current_branch(self) -> str:
        """The checked-out branch name (what `submit --branch` of this tree's
        own repo must name); "main" when detached."""
        out = subprocess.run(["git", "-C", str(self.workspace), "rev-parse",
                              "--abbrev-ref", "HEAD"], capture_output=True, text=True)
        name = out.stdout.strip()
        return name if out.returncode == 0 and name and name != "HEAD" else "main"

    def _overlay_on(self) -> bool:
        return bool(self.job_env_files) or self.harness_card is not None

    def push_branch(self, branch: str) -> None:
        ref = self._with_job_env(self.head()) if self._overlay_on() else "HEAD"
        self._git("push", "origin", f"{ref}:refs/heads/{branch}", "--force", what="git push")

    def push_ref(self, ref: str, branch: str) -> None:
        """Push an arbitrary commit (not HEAD) — stall pricing runs against the
        best-so-far sha without touching the working tree."""
        if self._overlay_on():
            ref = self._with_job_env(ref)
        self._git("push", "origin", f"{ref}:refs/heads/{branch}", "--force",
                  what="git push ref")

    def _overlay_wanted(self, ref: str) -> tuple[dict, list]:
        """({tree path: bytes} the job tree must hold, [tree paths it must not hold]) for
        ``ref``: the card's [loop] job_env_files, then -- under the harness guard -- every
        harness file the card tracks, and the removal of harness files the card does not
        have (a student-added eval/sitecustomize.py is imported by every bench)."""
        want: dict = {}
        for path, src in sorted(self.job_env_files.items()):
            try:
                want[path] = src.read_bytes()
            except OSError as e:
                print(f"[loop] job-env: cannot read the card's {src} ({e}) — {path} rides "
                      f"the tree's own copy", flush=True)
        drop: list = []
        if self.harness_card is not None:
            card = harness_guard.card_harness_files(self.harness_card, self.kernel_file)
            tree = harness_guard.tree_harness_files(self.workspace, ref, self.kernel_file)
            if card is None or tree is None:
                print(f"[loop] harness overlay: cannot list the harness of the card "
                      f"{self.harness_card} or of {ref[:10]} — this job runs the tree's own "
                      f"copies", flush=True)
            else:
                for path, src in card.items():
                    try:
                        want.setdefault(path, src.read_bytes())
                    except OSError as e:
                        print(f"[loop] harness overlay: cannot read the card's {src} ({e})",
                              flush=True)
                drop = [p for p in sorted(tree) if p not in card and p not in want]
        return want, drop

    def _with_job_env(self, ref: str) -> str:
        """``ref`` with the card's job-environment files ([loop] job_env_files) and, under
        the harness guard, the card's evaluation harness overlaid — the sha a job actually
        runs. ``ref`` itself when every file already matches the card, so a run whose tree
        never diverged pushes exactly what it did before.

        This is the channel a job-environment fix never had: gpu_entry.sh and eval/
        ride the student's lineage, which only the student commits to and every
        rollback resets, so an operator's or a teacher's fix to them could not reach
        a job without entering that lineage — one fix was observed lost three
        times that way (a pricing reset, a verdict-probe reset, a card-dir
        edit no job ever read). Overlaid here, the fix lives in the card and reaches
        every job, the lineage stays the student's, and records keep the tree's sha.
        The same channel keeps the judge the card's: a student edit to eval/ or
        gpu_entry.sh never reaches a job (loop/harness_guard.py). One child commit built
        from a temporary index, bytes for bytes (binary blobs and sealed references
        included): working tree, index and HEAD are never touched."""
        want, drop = self._overlay_wanted(ref)
        ws = self.workspace
        tree = harness_guard.tree_harness_files(ws, ref, "") or {}
        changes: list = []
        try:
            for path, data in sorted(want.items()):
                have = tree.get(path)
                if have is None and not harness_guard.is_harness_path(path):
                    shown = subprocess.run(["git", "-C", str(ws), "ls-tree", ref, "--", path],
                                           capture_output=True, text=True).stdout.split()
                    have = (shown[0], shown[2]) if len(shown) >= 3 else None
                blob = harness_guard.blob_sha(ws, data, write=True)
                src = self.job_env_files.get(path) or (self.harness_card / path
                                                       if self.harness_card else None)
                mode = "100755" if (path.endswith(".sh") or (src and os.access(src, os.X_OK))) \
                    else "100644"
                if have is None or have[1] != blob:
                    changes.append((path, mode, blob))
            changes += [(path, None, None) for path in drop]
            sha = ref
            if changes:
                sha = self._commit_overlay(ref, changes)
        except (OSError, BackendError) as e:
            print(f"[loop] job-env: could not overlay the card's files ({e}) — this job "
                  f"runs the tree's own copies", flush=True)
            sha, changes = ref, []
        changed = [p for p, _, _ in changes]
        key = ",".join(changed)
        if self._job_env_announced.get("last") != key:
            self._job_env_announced["last"] = key
            if changed:
                print(f"[loop] job-env: jobs now run the card's {harness_guard.describe(changed)} "
                      f"over the tree's copies ([loop] job_env_files / harness_guard; the "
                      f"lineage is unchanged)", flush=True)
            else:
                print("[loop] job-env: the tree's copies match the card — no overlay",
                      flush=True)
        return sha

    def _commit_overlay(self, ref: str, changes: list) -> str:
        """A child commit of ``ref`` carrying ``changes`` [(path, mode, blob) | (path,
        None, None) = remove], built from a temporary index."""
        ws = str(self.workspace)

        def step(what: str, *args: str, env=None) -> str:
            p = subprocess.run(["git", "-C", ws, *args], capture_output=True, text=True,
                               env=env)
            if p.returncode != 0:
                raise BackendError(f"{what} failed on {ref[:10]}: "
                                   f"{(p.stderr or p.stdout).strip()[:300]}")
            return p.stdout.strip()

        fd, index = tempfile.mkstemp(prefix="fk-overlay-index-")
        os.close(fd)
        os.unlink(index)
        env = dict(os.environ, GIT_INDEX_FILE=index)
        try:
            step("git read-tree", "read-tree", ref, env=env)
            for path, mode, blob in changes:
                if blob is None:
                    step("git update-index", "update-index", "--force-remove", "--", path, env=env)
                else:
                    step("git update-index", "update-index", "--add", "--cacheinfo",
                         f"{mode},{blob},{path}", env=env)
            tree = step("git write-tree", "write-tree", env=env)
            names = harness_guard.describe([p for p, _, _ in changes])
            return step("git commit-tree", "-c", f"user.name={self.git_name}",
                        "-c", f"user.email={self.git_email}", "commit-tree", tree, "-p", ref,
                        "-m", f"job-env: the card's {names} (overlay for this job; not part "
                              f"of the lineage)")
        finally:
            try:
                os.unlink(index)
            except OSError:
                pass

    def changed_paths(self, base: str) -> list[str]:
        """Tracked paths whose content differs between ``base`` and the working
        tree, committed on top of ``base`` or not: what ``reset --hard base`` throws
        away (untracked files survive a hard reset and are not listed)."""
        out = subprocess.run(["git", "-C", str(self.workspace), "diff", "--name-only",
                              base, "--"], capture_output=True, text=True)
        if out.returncode != 0:
            return []
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]

    def with_file(self, ref: str, path: str, text: str, message: str) -> str:
        """The sha of ``ref`` with ``path`` holding ``text``: ``ref`` itself when
        it already does, else a new commit on top of ``ref`` carrying that one
        change. Built from a temporary index (read-tree / update-index /
        write-tree / commit-tree), so the working tree, the index and HEAD are
        untouched — a probe can run a harness-maintained file against an older
        tree (the best sha) while the tree under judgment stays what it was."""
        ws = str(self.workspace)

        def step(what: str, *args: str, env=None, inp=None) -> str:
            p = subprocess.run(["git", "-C", ws, *args], capture_output=True, text=True,
                               env=env, input=inp)
            if p.returncode != 0:
                raise BackendError(f"{what} failed on {ref[:10]}: "
                                   f"{(p.stderr or p.stdout).strip()[:300]}")
            return p.stdout.strip()

        have = subprocess.run(["git", "-C", ws, "show", f"{ref}:{path}"],
                              capture_output=True, text=True)
        if have.returncode == 0 and have.stdout == text:
            return ref
        fd, index = tempfile.mkstemp(prefix="fk-overlay-index-")
        os.close(fd)
        os.unlink(index)            # read-tree writes a fresh index at that path
        env = dict(os.environ, GIT_INDEX_FILE=index)
        try:
            blob = step("git hash-object", "hash-object", "-w", "--stdin", inp=text)
            step("git read-tree", "read-tree", ref, env=env)
            step("git update-index", "update-index", "--add", "--cacheinfo",
                 f"100644,{blob},{path}", env=env)
            tree = step("git write-tree", "write-tree", env=env)
            return step("git commit-tree", "-c", f"user.name={self.git_name}",
                        "-c", f"user.email={self.git_email}",
                        "commit-tree", tree, "-p", ref, "-m", message)
        finally:
            try:
                os.unlink(index)
            except OSError:
                pass

    def reset(self, ref: str) -> None:
        self._git("reset", "--hard", ref, what="git reset")

    def head(self) -> str:
        out = subprocess.run(["git", "-C", str(self.workspace), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True)
        return out.stdout.strip()


def _workspace_can_push(workspace: Path) -> bool:
    """Whether the student workspace has a remote to push branches to."""
    try:
        out = subprocess.run(["git", "-C", str(workspace), "remote"],
                             capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return "origin" in out.stdout.split()


def _resolve_ref_spec(cfg: dict) -> dict:
    """How to measure this operator's reference, from ``[loop]`` in loop.toml.

    Declaring it there is what lets the loop measure the baseline itself instead of
    telling the operator author to go run a second command: what the ref *is* stays
    an operator fact (torch SDPA backward here, cuBLAS there), but nobody should
    have to restate it on the command line every time.
    """
    loop = cfg.get("loop", {})
    spec = {k: loop[k] for k in ("ref_cmd", "ref_ncu_cmd", "ref_name", "ref_shape",
                                 "ref_hardware") if loop.get(k)}
    # A library baseline may live in a different submission tree, the same way the
    # architecture reference does (measure_and_archive already understands ref_mode +
    # submit_op). The run that needs this is the one whose library reference IS the
    # reconnoitred implementation: measuring it through the student harness would put
    # the reference package inside a student job, which is a source leak.
    if loop.get("ref_mode"):
        spec["ref_mode"] = loop["ref_mode"]
    if loop.get("ref_submit_op"):
        spec["submit_op"] = loop["ref_submit_op"]
    return spec


def _ref_rides_job(spec: dict) -> bool:
    """Whether the declared ref can be timed inside the student eval job.

    Out-of-job when measured through a separate submission tree — a ref_mode task
    OR a ref_cmd routed via submit_op — and trivially when no ref is declared at
    all (target-ms-only runs): nothing rides. Misjudging True is not harmless: the
    confirmation stage would then expect a same-job ratio the bench cannot produce
    (the driver degrades to the absolute median, but the misjudgment also skips
    the fresh ref re-anchor that runs without an in-job ref rely on).
    """
    return bool(spec) and not spec.get("ref_mode") and not spec.get("submit_op")


def _precision_preflight(op_dir: Path, contract: str, defaulted: bool, *, resuming: bool,
                         dry_run: bool) -> int:
    """Say which precision contract the run is held to, and refuse a fresh ``strict``
    start whose bench cannot report the precision block (EXIT_PRECISION_UNREPORTED); a
    resume, a dry run or an ``official`` card only warns."""
    kind = "leaderboard card" if precision_gate.is_leaderboard_card(op_dir) else "own bench"
    print(f"[loop] precision_contract = {contract}"
          + (f" (default for a {kind})" if defaulted else "")
          + {"strict": " — a failed precision check REVERTs like a correctness failure",
             "official": " — precision is recorded, never gated",
             "off": (" — precision is neither measured nor read (the board's own verdict "
                     "only)" if kind == "leaderboard card" else " — precision is not read")
             }[contract], flush=True)
    if contract == precision_gate.OFF:
        return 0
    findings = precision_gate.bench_findings(op_dir)
    if not findings:
        return 0
    for f in findings:
        sys.stderr.write(f"[loop]   precision: {f}\n")
    if contract == precision_gate.STRICT and not resuming and not dry_run:
        sys.stderr.write("[loop] error: refusing a fresh start under precision_contract = "
                         "\"strict\" with a bench that reports no precision block — wire "
                         "eval/precision.py into the bench, or set [loop] precision_contract = "
                         "\"official\" (record only) / \"off\" to run without it on purpose.\n")
        return EXIT_PRECISION_UNREPORTED
    sys.stderr.write("[loop] WARNING: the bench reports no precision block — rows record "
                     "`unreported` and the contract is not applied.\n")
    return 0


def _harness_guard_preflight(op_dir: Path, workspace: Path, kernel_file: str, backend,
                             *, resuming: bool, dry_run: bool) -> int:
    """Turn the harness overlay on for ``backend`` and compare the workspace's harness
    with the card's. 0 = go on; EXIT_HARNESS_DRIFT = a fresh start over a workspace whose
    gpu_entry.sh / eval/ / ref_bin/ differ from the card's."""
    if not harness_guard.is_git_repo(op_dir):
        sys.stderr.write(f"[loop] WARNING: harness_guard: the card {op_dir} is not a git "
                         f"repository — its tracked harness files cannot be listed, so jobs "
                         f"run the lineage's own gpu_entry.sh / eval/ (the driver still "
                         f"reverts deliveries that edit them).\n")
        return 0
    backend.harness_card = op_dir
    if not (Path(workspace) / ".git").exists():
        return 0
    drift = harness_guard.harness_drift(op_dir, workspace, "HEAD", kernel_file)
    if drift is None:
        sys.stderr.write("[loop] WARNING: harness_guard: cannot compare the workspace's "
                         "harness with the card's (git failed)\n")
        return 0
    if not drift:
        print(f"[loop] harness_guard: jobs run the card's gpu_entry.sh / eval/ / ref_bin/ "
              f"(identical to the workspace's); a delivery that edits them is reverted",
              flush=True)
        return 0
    sys.stderr.write(f"[loop] harness_guard: the workspace's harness differs from the "
                     f"card's ({op_dir}): {harness_guard.describe(drift, 12)} "
                     f"(~ content, + only in the card, - only in the workspace)\n")
    if resuming or dry_run:
        sys.stderr.write("[loop] WARNING: resuming over a lineage whose evaluation harness is "
                         "not the card's — every job from now on runs the card's copies, so "
                         "readings before and after this launch may not compare; check the "
                         "lineage history of these files.\n")
        return 0
    sys.stderr.write("[loop] error: refusing a fresh start whose workspace carries another "
                     "judge than the card — re-clone the workspace from the card, or commit "
                     "the intended harness to the card itself; [loop] harness_guard = \"off\" "
                     "turns the guard off on purpose.\n")
    return EXIT_HARNESS_DRIFT


def resolve_job_env_files(cfg: dict, op_dir: Path, *, kernel_file: str = "") -> dict:
    """[loop] job_env_files -> {tree path: the card's copy of it}.

    Each entry names a file of the operator card (operators/<op>/ — the workspace's
    origin) that every job must run from the card instead of from the student's
    lineage: the job environment (``gpu_entry.sh``, ``eval/ncu_run.py``, ...). See
    RealBackend._with_job_env. Relative, inside the card and present — a typo would
    overlay nothing and read as "on" — and never the kernel file or notes/: those
    are the student's tree, and overlaying them would replace its work."""
    raw = (cfg.get("loop", {}) or {}).get("job_env_files") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
        raise ValueError(f"[loop] job_env_files must be a list of paths, got {raw!r}")
    out: dict = {}
    for p in raw:
        rel = Path(p.strip())
        key = rel.as_posix()
        if not key or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"[loop] job_env_files entry {p!r} must be a path inside "
                             f"the card ({op_dir})")
        if key == str(kernel_file) or key.split("/")[0] == "notes":
            raise ValueError(f"[loop] job_env_files entry {p!r} is the student's tree "
                             f"(the kernel file / notes/), not the job environment")
        src = op_dir / rel
        if not src.is_file():
            raise ValueError(f"[loop] job_env_files entry {p!r}: {src} does not exist")
        out[key] = src
    return out


def _resolve_ref_in_job(cfg: dict, spec: dict) -> bool:
    """[loop] ref_in_job — the bench's in-job comparator is the reference itself.

    Meaningful only on runs whose reference is measured through a separate tree
    (ref_mode / submit_op): those are the runs the single-ref contract strips the
    in-job reading from. When the ref already rides the student job the reading
    is kept anyway, and a target-ms-only run has no reference to be "in" the
    job — both are told the flag is inert rather than silently accepted.
    """
    raw = cfg.get("loop", {}).get("ref_in_job", False)
    if not raw:
        return False
    if not spec:
        sys.stderr.write("[loop] warning: [loop] ref_in_job = true but no reference is "
                         "declared (target-ms-only run) — ignored\n")
        return False
    if _ref_rides_job(spec):
        sys.stderr.write("[loop] note: [loop] ref_in_job = true is redundant here — the "
                         "reference already rides the student job (no separate tree)\n")
        return False
    return True


ORACLE_SOURCE = "source"
ORACLE_BLACKBOX = "blackbox"
ORACLE_MODES = (ORACLE_SOURCE, ORACLE_BLACKBOX)


def _will_distill(milestone: Path, oracle: str | None, mode: str = ORACLE_SOURCE) -> bool:
    """True when this run is going to author a milestone, so it needs a baseline.

    A resumed run against a milestone that already exists must not re-measure: the
    archive is reused precisely so two runs of one operator compare against the
    same numbers.
    """
    if not _has_distillation_input(oracle, mode):
        return False
    from forgekernel.milestone import validate_milestone
    try:
        return bool(validate_milestone(milestone.read_text(encoding="utf-8")))
    except OSError:
        return True  # missing milestone == not distilled yet


def _status_has_records(status_path: Path) -> bool:
    """True once any iteration is on record — the loop is resuming, not starting."""
    try:
        text = status_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return any(line.strip() for line in text.splitlines())


def _oracle_lint_tokens(cfg: dict, oracle, report=None) -> list[str]:
    """The oracle-derived KB lint tokens, or [] when the run turns them off.

    ``[loop] kb_lint_oracle_tokens = false`` is for a source-mode run whose student STARTS
    from the oracle (its kernel file is the oracle, verbatim): the lint has nothing to
    protect there, and an oracle that took its helper names from the KB (an earlier
    run's kernel) matches the KB's own generic names everywhere -- which would refuse
    the launch and, mid-run, quarantine certified entries. The static token table
    still applies."""
    if cfg.get("loop", {}).get("kb_lint_oracle_tokens", True) is False:
        sys.stderr.write("[loop] kb-lint reference tokens: off ([loop] kb_lint_oracle_tokens = false)\n")
        return []
    return derive_reference_tokens(oracle, report=report)


def _harvest_only(ctx, args, root: Path, milestone: Path, oracle, cfg: dict | None = None) -> int:
    """`loop --harvest-only`: the end-of-run KB write-up, and nothing else.

    A run stopped before its finish path (a killed session, `stop.sh`) never
    pays its kb-tripwire / axis bill: the KEEPs that moved a SASS contract axis
    stay "queued for the end-of-run KB harvest write-up" forever. This builds the
    same context a launch would (agent policy, teacher channel, KB tree, lint
    tokens) without measuring or refusing anything, then runs only run_kb_harvest —
    one teacher turn, zero GPU jobs, no confirmation: what the teacher
    wrote is left uncommitted for a human to review, as at the end of every run."""
    if not (args.teacher_agent_cmd or args.agent_cmd):
        sys.stderr.write("[loop] error: --harvest-only needs a teacher: --teacher-agent-cmd / "
                         "--agent-cmd\n")
        return 2
    if not _status_has_records(ctx.status_path):
        sys.stderr.write(f"[loop] error: --harvest-only: {ctx.status_path} has no records "
                         f"— there is no run to harvest\n")
        return 2
    ctx.kb_lint_tokens = _oracle_lint_tokens(cfg or {}, oracle)
    recorded = run_kb_harvest(ctx)
    summary: dict = {"harvest_only": True, "kb_harvest": recorded}
    provenance_dir = lang_mode.provenance_dir(root, getattr(ctx, "language", None))
    left = git_status(root, [str(ctx.kb_dir), str(milestone), str(provenance_dir)])
    if left is None:
        print(f"[harvest] could not read git status under {root} — check it by hand")
    else:
        print(format_report(left, root))
        summary["unharvested"] = [c.path for c in left]
    print(f"[harvest-only] done: {json.dumps(summary)}")
    return 0


_LAUNCH_SENTINEL_NAME = "launch_admitted.json"


def _launch_sentinel_path(status_path: Path) -> Path:
    """Where "this launch passed the fresh-start gate" is remembered.

    Lives next to status.jsonl for the same reasons recon_complete.json does:
    student/ is gitignored in the operator repo, so the marker can never ride a
    factory clone or a GPU tarball into another run.
    """
    return status_path.parent / _LAUNCH_SENTINEL_NAME


def _launch_admitted(op_dir: Path, status_path: Path) -> bool:
    """True when an earlier launch of THIS operator dir already passed the gate.

    Iterations only start recording after the whole pre-iteration pipeline —
    baseline jobs, reconnaissance, distillation — which can run for hours with
    status.jsonl still empty. A crash in that window (429 storms, quota, a
    machine reboot) leaves products the gate would flag but that this run
    authored, and blackbox runs cannot be repaired by archiving alone: the
    reconnaissance tree is seeded and committed from round one, and the guard's
    archive list has never covered it. So admission is remembered: from the
    moment the gate passes, everything on disk belongs to this run, and every
    later restart resumes it as-is. Trust is conditional on the sentinel naming
    this exact operator dir — a hand-copied op dir carries its sentinel along,
    and one that names a different path is foreign, not an admission.
    """
    try:
        state = json.loads(
            _launch_sentinel_path(status_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(state, dict) and state.get("op_dir") == str(op_dir.resolve())


def _admit_launch(op_dir: Path, status_path: Path) -> None:
    """Persist that this launch passed the fresh-start gate (see _launch_admitted)."""
    _launch_sentinel_path(status_path).write_text(json.dumps({
        "op_dir": str(op_dir.resolve()),
        "admitted_epoch": time.time(),
    }, indent=2), encoding="utf-8")


def fresh_start_contamination(
        op_dir: Path, milestone: Path, status_path: Path,
        meta_status_path: Path, recon_dir: Path | None,
        evidence_dir: Path | None = None) -> list[str]:
    """Phase products a fresh start would silently inherit; empty means clean.

    Every phase of the loop resumes from its on-disk products: an existing
    milestone skips distillation, recon_complete.json skips reconnaissance, an
    archived baseline skips its measurement, and findings left in the recon tree
    are read as this run's own. That is exactly right for a crash resume
    (status.jsonl has records) and exactly wrong for a new run: products left by
    a previous run are inherited with no record that they were, and the run's
    from-scratch claim silently becomes false. Until now the refusal lived in
    each run's launch script, copied forward by convention — a loop started
    without the script had no protection at all. Deliberate reuse (a fix-rerun
    holding reconnaissance constant, a smoke run over a seeded baseline) is
    declared with --allow-preexisting-artifacts.

    Two things count as resume, and both admit everything: a recorded iteration,
    and an admission sentinel from an earlier launch of this same operator dir —
    the pre-iteration pipeline runs for hours before the first record exists,
    and a crash there must restart into a resume, not a refusal.
    """
    if _status_has_records(status_path):
        return []
    if _launch_admitted(op_dir, status_path):
        return []
    findings: list[str] = []
    if milestone.is_file():
        findings.append(f"{milestone} — distillation would be skipped")
    marker = status_path.parent / _RECON_STATE_NAME
    if marker.is_file():
        findings.append(f"{marker} — reconnaissance would be skipped")
    if _status_has_records(meta_status_path):
        findings.append(f"{meta_status_path} — a previous run's meta history would "
                        f"be read as this run's")
    for name in (ARCH_BASELINE_FILENAME, BASELINE_FILENAME):
        if (op_dir / name).is_file():
            findings.append(f"{op_dir / name} — its measurement would be skipped")
    if recon_dir is not None:
        for name in ("report.md", "candidates.md", "derivation.md"):
            if (recon_dir / name).is_file():
                findings.append(f"{recon_dir / name} — a previous run's "
                                f"reconnaissance would be read as this run's")
        notes = recon_dir / "notes"
        if notes.is_dir() and any(notes.iterdir()):
            findings.append(f"{notes}/ is not empty — previous probe scripts would "
                            f"ride into this run's reconnaissance")
    goal_state = goal_ref_mod.Paths.of(op_dir).state
    if goal_state.is_file():
        findings.append(f"{goal_state} — a previous run's generated reference "
                        f"(ref_from_goal) would be reused as this run's oracle")
    for d in {recon_dir, evidence_dir} - {None}:
        for name in EVIDENCE_FILE_NAMES:
            if (d / name).is_file():
                findings.append(f"{d / name} — a previous run's pricing evidence "
                                f"would be read as this run's")
    return findings


class ArchRefUnavailable(RuntimeError):
    """The architecture reference could not be established. Fatal before distillation."""


def _arch_ref_spec(cfg: dict) -> dict:
    """How to measure the architecture reference, or {} if this operator has none.

    Not every operator has one: an operator whose reference *is* the library has a
    library baseline and nothing else to reproduce. Absence is therefore legitimate,
    and the mandatory-first-step rule below applies only once an operator declares it.
    """
    loop = cfg.get("loop", {})
    if not loop.get("arch_ref_mode"):
        return {}
    return {
        "ref_mode": loop["arch_ref_mode"],
        "submit_op": loop.get("arch_ref_submit_op"),
        "ref_name": loop.get("arch_ref_name", ""),
        "ref_shape": loop.get("ref_shape", ""),
        "ref_hardware": loop.get("ref_hardware", ""),
    }


def _declared_arch_baseline(op: str, cfg: dict) -> RefBaseline | None:
    """The architecture reference as declared in loop.toml, or None if undeclared.

    Built (not saved) here so _ensure_arch_baseline can both fall back to it and
    compare an existing archive against it: a declared-sourced archive is just the
    saved image of these values, and must follow the toml rather than shadow it.
    """
    ms = cfg.get("loop", {}).get("arch_ref_ms")
    if ms is None:
        return None
    dev = cfg["loop"].get("arch_ref_device_ms")  # hand-declared device-window reading
    # A hand-declared device number must say which window produced it, or the archive
    # inherits the ambiguity this whole field exists to remove. Undeclared means
    # unknown, which the audit treats as plain batch and leaves unchecked.
    basis = str(cfg["loop"].get("arch_ref_device_basis", "") or "").strip()
    graph = cfg["loop"].get("arch_ref_graph_ms")
    enqueue = cfg["loop"].get("arch_ref_host_enqueue_ms")
    # A declared number is one input set's unless the toml names the workloads
    # it is the geometric mean over (a multi-workload card's gate reading is).
    workloads = [str(w) for w in cfg["loop"].get("arch_ref_workloads") or []]
    return RefBaseline(
        op=op, ms=float(ms), role=ROLE_ARCHITECTURE,
        ms_device=float(dev) if dev is not None else None,
        device_window_basis=basis,
        ms_device_graph=float(graph) if graph is not None else None,
        host_enqueue_ms=float(enqueue) if enqueue is not None else None,
        n_workloads=max(1, len(workloads)), workloads=workloads,
        job=cfg["loop"].get("arch_ref_job", ""),
        ref_name=cfg["loop"].get("arch_ref_name", ""),
        shape=cfg["loop"].get("ref_shape", ""),
        hardware=cfg["loop"].get("ref_hardware", ""),
        ncu=dict(cfg["loop"].get("arch_ref_ncu", {})),
        raw_stdout=cfg["loop"].get("arch_ref_provenance", ""),
    )


def _cluster_guarded(op, baseline):
    """``baseline``, unless it was measured on a different machine than this op's
    jobs run on now — then None, so the caller re-measures.

    Absolute latency for one unchanged kernel moves more between machines than most
    single-round gains, so a target anchored to another machine's numbers is wrong
    in an unbounded direction. An unstamped side ("" — an unreadable submit config)
    passes: guarding is only possible where provenance exists.
    """
    if baseline is None:
        return None
    stamped = getattr(baseline, "cluster", "")
    if not stamped:
        return baseline
    try:
        from forgekernel.submit.config import load_config
        cfg = load_config(op)
        home = cfg.cluster
    except Exception:
        return baseline
    if home and stamped != home:
        print(f"[loop] archived {baseline.role} baseline was measured on cluster "
              f"{stamped!r} but this operator's primary cluster is {home!r} — "
              f"discarding it; the reference must be re-measured on the primary "
              f"before any gate can anchor to it", flush=True)
        return None
    return baseline


def _ensure_arch_baseline(op: str, op_dir: Path, cfg: dict, *, measure: bool = True):
    """The architecture reference: the implementation the milestone teaches.

    Distinct from the library baseline, and the distinction is load-bearing. The
    library one (cuDNN here) answers "is this competitive with what you would
    otherwise call". The architecture one answers "did you reproduce the machine the
    decision graph describes" — and for this operator they are ~20% apart, so a
    target derived from the library stops the loop while it is still far short of the
    thing it was asked to reproduce. That is what happened: target_ms came out at
    2.4933 against an oracle that runs at ~1.98, and the operator README had said so
    in prose the whole time without anything reading it.

    Measured here, before anything is distilled, because every mechanism gate line in
    the milestone anchors to this reference's NCU signature. Distilling first and
    measuring later means authoring gates against numbers that do not exist yet, and
    the fallback in that situation is to anchor them to the library baseline — a
    different algorithm, so a comparison that says nothing either way.

    ``arch_ref_ms``/``arch_ref_device_ms``/``arch_ref_ncu`` in loop.toml remain as a
    declared fallback for when the reference cannot be run at all (a device-gated
    run — the default — needs ``arch_ref_device_ms``, since target derivation
    refuses to substitute the e2e number). They are second choice on purpose: a
    hand-transcribed constant cannot tell that the shape or the image moved under it,
    and an earlier run of this operator was anchored to one that had.
    """
    existing = _cluster_guarded(op, load_baseline(op_dir, ROLE_ARCHITECTURE))
    declared = _declared_arch_baseline(op, cfg)
    if existing:
        # A declared-sourced archive (the only writer that leaves samples_ms empty;
        # measure_and_archive always records its runs) is the saved image of
        # loop.toml. When the toml moves — arch_ref_device_ms added for the device
        # default, a corrected reading — the archive must follow it, not shadow it:
        # returning the stale image here sent operators into a dead end where the
        # prescribed fix (declare the device reading) changed nothing until they
        # knew to hand-delete arch_baseline.json. A MEASURED archive is never
        # overwritten by declared values.
        if declared is not None and not existing.samples_ms \
                and asdict(existing) != asdict(declared):
            print(f"[loop] the archived architecture baseline is a saved image of "
                  f"loop.toml's declared values and the toml changed — refreshing "
                  f"the archive from loop.toml")
            save_baseline(op_dir, declared)
            return declared
        return existing

    spec = _arch_ref_spec(cfg)
    if spec and measure:
        from forgekernel import baseline_cli
        print(f"[loop] measuring the architecture reference "
              f"({spec.get('ref_name') or spec['ref_mode']}) before distillation…")
        try:
            # repeat=3, same as the library baseline: this number anchors target_ms,
            # and a single sample sits anywhere in the node-contention band (±5% on
            # this cluster). One repeat=1 sample landed 5% slow and the loop stopped
            # a run at 0.897x while calling it 0.95x — the target is the one number
            # a noisy measurement is most expensive on.
            return baseline_cli.measure_and_archive(
                op, op_dir, spec, repeat=3, role=ROLE_ARCHITECTURE)
        except Exception as e:
            # Not fatal here: the declared fallback may still carry a usable signature,
            # and the caller decides whether what it got is enough to start.
            print(f"[loop] architecture reference measurement failed ({e}); "
                  f"falling back to the declared values in loop.toml")

    if declared is None:
        return None
    save_baseline(op_dir, declared)
    return declared


def ncu_available(cfg: dict) -> bool:
    """[loop] ncu_available (default true): false on a machine where NCU cannot profile --
    the run degrades to timing only instead of refusing to start."""
    return bool((cfg.get("loop", {}) or {}).get("ncu_available", True))


def check_arch_ref_ready(cfg: dict, arch_baseline, *, will_distill: bool) -> str | None:
    """Why the run may not start, or None if it may.

    The architecture reference is a precondition of distillation, not an input the
    teacher can do without: the template requires every mechanism gate line to anchor
    to a measured signature, and the only other signature available is the library
    baseline's — a different algorithm, where agreeing and disagreeing are equally
    uninformative. A milestone authored without it is wrong in a way no later step
    detects, and v1 has no rollback behind it.

    Only enforced when the operator declares an architecture reference, and only while
    a milestone still has to be written: an already-distilled milestone has its numbers
    and re-running it must not require the reference to be reachable again.
    """
    if not will_distill or not _arch_ref_spec(cfg):
        return None
    if arch_baseline is None:
        return ("this operator declares an architecture reference (arch_ref_mode) but "
                "none could be measured or loaded")
    if not arch_baseline.has_ncu and not ncu_available(cfg):
        # A machine where NCU cannot read the counters (`forge run` probed it): the run
        # goes on timing only, and the milestone's mechanism lines have nothing to anchor
        # to -- said once here, the distillation prompt carries the reference without one.
        print("[loop] NCU is unavailable on this machine ([loop] ncu_available = false): the "
              "architecture reference has no NCU signature, so the milestone is distilled "
              "from timings alone", flush=True)
        return None
    if not arch_baseline.has_ncu:
        return (f"the architecture reference ({arch_baseline.ref_name or 'unnamed'}) has "
                "no NCU signature, so every mechanism gate line would have to anchor to "
                "the library baseline instead — a different algorithm")
    return None


def _ensure_ref_baseline(op: str, op_dir: Path, milestone: Path, oracle: str | None,
                         cfg: dict, repeat: int, *, mode: str = ORACLE_SOURCE,
                         measure: bool = True, arch: RefBaseline | None = None):
    """The archived baseline, measuring it first if this run needs one and can.

    Returns None when the loop does not need a baseline (nothing to distill) or
    cannot produce one (no ref_cmd declared) — the driver reports the latter.

    ``measure=False`` reports what a real run would do without doing it (the
    --harvest-only path builds its context without buying a job).

    When the declared reference is the architecture reference itself (a task's one
    reference is both) and ``arch`` was measured, its reading is archived again
    under this role instead of buying the same jobs twice.
    """
    existing = _cluster_guarded(op, load_baseline(op_dir))
    if existing or not _will_distill(milestone, oracle, mode):
        return existing
    if not measure:
        return None
    spec = _resolve_ref_spec(cfg)
    if not (spec.get("ref_cmd") or spec.get("ref_mode")):
        return None
    aspec = _arch_ref_spec(cfg)
    if arch is not None and arch.samples_ms and not spec.get("ref_cmd") \
            and spec.get("ref_mode") == aspec.get("ref_mode") \
            and spec.get("submit_op") == aspec.get("submit_op"):
        print(f"[loop] the ref baseline is the architecture reference — reusing its "
              f"reading ({arch.ms} ms, job {arch.job or '?'})")
        same = dc_replace(arch, role=ROLE_LIBRARY, ref_name=spec.get("ref_name") or arch.ref_name)
        save_baseline(op_dir, same)
        return same
    print(f"[loop] no ref baseline for {op} — measuring it first "
          f"({repeat}x '{spec.get('ref_cmd') or spec['ref_mode']}')")
    from forgekernel.baseline_cli import measure_and_archive
    return measure_and_archive(op, op_dir, spec, repeat=repeat)


def _resolve_gate_window(flag: str | None, cfg: dict) -> str:
    """Which timing window the loop gates on: --gate-window, else loop.toml, else device.

    Mirrors _resolve_language. "device" (pipelined batch ≈ device time — the
    kernel-quality claim, and the default) or "e2e" (per-call+sync,
    host tax inside; per-card opt-in for kernels whose deployment pays the host tax
    on every call). Per-card because the right window depends on how the kernel is
    deployed, and validated here because a typo'd window silently gating on the
    wrong number is the failure mode this whole knob exists to prevent.
    """
    chosen = flag or cfg.get("loop", {}).get("gate_window") or GATE_WINDOW_DEFAULT
    if chosen not in GATE_WINDOWS:
        raise SystemExit(
            f"gate_window must be one of {GATE_WINDOWS}, got {chosen!r} "
            f"(--gate-window / loop.toml [loop].gate_window)")
    return chosen


def _home_cluster(op: str) -> str:
    """The card's primary cluster — the one unstamped records and the archived
    baselines belong to. "" when the submit config is unreadable."""
    try:
        from forgekernel.submit.config import load_config
        cfg = load_config(op)
        return cfg.cluster
    except Exception:
        return ""


def _resolve_target_ms(flag: float | None, baseline, ratio: float, arch=None,
                       gate_window: str = GATE_WINDOW_DEFAULT,
                       allow_host_bound: bool = False) -> float | None:
    """The stop threshold: an explicit flag, else derived from a reference.

    Deriving it is the point of measuring first — a target estimated from oracle
    source is the one number a wrong guess is most expensive on, since it decides
    when the run stops.

    Which reference it derives from decides what "done" means. The architecture one
    wins when present: reproducing the implementation the milestone teaches is the
    experiment, and the library baseline is only the competitiveness check alongside
    it. Deriving from the library instead set this operator's target 26% slower than
    its oracle, so the loop would have declared success having reproduced about
    three quarters of the performance it was asked for.
    """
    if flag is not None:
        return flag
    ref = arch or baseline
    # target_ms(gate_window=…) raises when the archive never measured the gate
    # window; let it propagate — main() turns it into a clean refusal to start.
    return ref.target_ms(ratio, gate_window,
                         allow_host_bound=allow_host_bound) if ref else None


def _target_workloads_problems(op: str, op_dir: Path, refs) -> list[str]:
    """Reference archives read over other workloads than the bench's gate reading.

    Static: whether eval/bench.py reports a multi-workload geomean
    (``measure_windows_workloads``) against each archive's ``n_workloads``. The
    workload NAMES are only known once an eval ran; the driver holds them
    against every KEEP (``_target_workloads_mismatch``).
    """
    multi = bench_times_workloads(op_dir)
    if multi is None:
        return []
    out = []
    for ref in refs:
        if ref is None:
            continue
        what = (f"{baseline_path(op_dir, ref.role).name} ({ref.role} reference, "
                f"{ref.ms} ms)")
        if multi and ref.n_workloads <= 1:
            out.append(
                f"{what} is a reading of ONE input set, but eval/bench.py reports the "
                f"geomean over several workloads — re-archive it from a ref script that "
                f"prints the same workloads' geomean block (timing.workloads_performance), "
                f"e.g. `forge {op} baseline --force --ref-cmd <ref_cmd>`; a declared "
                f"arch_ref_ms names the workloads it is the geomean over in [loop] "
                f"arch_ref_workloads")
        elif not multi and ref.n_workloads > 1:
            out.append(
                f"{what} is the geomean over {ref.n_workloads} workloads, but "
                f"eval/bench.py times one input set — time the same workloads through "
                f"timing.measure_windows_workloads, or re-archive the reference on the "
                f"bench's one input set")
    return out


def _fresh_ref_workloads_mismatch(runs: list[dict], confirm: dict) -> str | None:
    """Why the finish's fresh ref runs cannot divide the confirmed student median.

    On a multi-workload card the student median is a geomean over the bench's
    workloads (``confirm["workloads"]`` names them); a ref run parsed over other
    workloads (``parse_ref_stdout``'s ``n_workloads`` / ``workloads``) is a
    different quantity, and the median-vs-median ratio would be the cost mix.
    """
    names = [str(w.get("name")) for w in confirm.get("workloads") or []]
    return next((why for r in runs if (why := workloads_mismatch(
        r.get("n_workloads", 1), r.get("workloads"), len(names) or 1, names))), None)


def _arm_has_reference(cfg: dict) -> bool:
    """Is there a second timed window to hold the student's boundary against?

    A reference measured by the loop (``ref_cmd`` / ``ref_mode``), an
    architecture reference, or one declared as archived values all define a
    reference window; a target-ms-only run defines none.
    """
    loop = cfg.get("loop", {})
    return bool(_resolve_ref_spec(cfg) or _arch_ref_spec(cfg)
                or loop.get("arch_ref_ms") is not None
                or loop.get("arch_ref_device_ms") is not None)


def _window_preflight(op_dir: Path, cfg: dict, *, resuming: bool
                      ) -> tuple[str, WindowContract | None, list[str]]:
    """Check the timed-window contract. Returns (state, contract, problems):

    * ``"ok"`` — declared and consistent (problems = non-refusing notes);
    * ``"n/a"`` — no reference declared, nothing to hold the boundary against;
    * ``"warn"`` — undeclared / inconsistent, but the run is resuming (it
      predates the rule, or is mid-run) — say so, do not refuse;
    * ``"refuse"`` — undeclared / inconsistent on a fresh start.

    A file that exists but cannot be parsed is a refusal on any start: a
    half-read contract is worse than none. A contract that symmetrizes inputs
    through a prep step is also held to the bench: ``eval/bench.py`` must pass
    ``prep=`` to ``measure_windows`` (``window_contract.bench_prep_findings``),
    or the declaration is on paper only.
    """
    try:
        contract = load_window_contract(op_dir)
    except ValueError as e:
        return "refuse", None, [str(e)]
    if not _arm_has_reference(cfg):
        return "n/a", contract, []
    findings = window_findings(contract)
    if not findings:
        findings = bench_prep_findings(op_dir, contract)
    if findings:
        return ("warn" if resuming else "refuse"), contract, findings
    return "ok", contract, window_warnings(contract)


def _teacher_only_trees(cfg: dict, root: Path, op: str, oracle_mode: str,
                        oracle_paths: list[str] | None = None, *,
                        goal_ref: bool = True) -> list[Path]:
    """Trees the student must not read: the reconnaissance / reference-measurement /
    reference-pricing trees (teacher-side by construction: they hold the reference's
    harness and the teacher's findings) and a source-mode run's oracle. The operator's own
    tree is never one of them, even when a loop.toml routes a reference through it.

    An oracle contributes the file itself AND the directory holding it: the oracle is
    one file of a reference package, and the values the rest of the package carries are
    exactly what the student must not copy (a parameterised kernel's tile shapes live in
    the interface.py beside it). Anything that would swallow the workspace — the repository
    root, a home directory — is left out; the agent policy drops what still contains the
    agent's own working directory.

    A reference generated by ref_from_goal (loop/goal_ref.py) is teacher-side whether
    or not this run uses it: its directory under the card (the oracle, every candidate,
    the session's workspace) and its submission tree, once they exist. ``goal_ref=False`` leaves them out
    — for the generating session itself, whose working directory is inside them."""
    lc = cfg.get("loop", {})
    names = [_recon_op(cfg, oracle_mode), _resolve_ref_spec(cfg).get("submit_op"),
             lc.get("arch_ref_submit_op"), lc.get("ref_price_op")]
    trees = [operators_dir(root) / n for n in names if n and n != op]
    if goal_ref:      # only once one exists: a run that never generated is unchanged
        trees += [p for p in goal_ref_mod.teacher_only_paths(root, op) if p.exists()]
    too_broad = {Path(os.path.expanduser("~")).resolve(), Path(root).resolve(), Path("/")}
    for raw in oracle_paths or []:
        p = Path(os.path.expanduser(str(raw)))
        trees.append(p)
        parent = p.parent
        # ``.suffix`` rather than ``is_file``: the declaration is read before anything
        # is measured, and an oracle on a detached checkout may not be mounted yet.
        if (p.suffix or p.is_file()) and parent.resolve() not in too_broad:
            trees.append(parent)
    return trees


def _run_finish_monitor(ctx: LoopContext, backend, op_dir: Path, confirm: dict | None,
                        summary: dict, profiles, kernel_file: str) -> dict | None:
    """The finish's correctness monitor on the confirmed (else best green) sha.
    Never raises: a monitor that cannot run is logged and the run finishes as it
    would have."""
    from forgekernel.loop.driver import _best_good_ref_here, load_records
    try:
        sha = (confirm or {}).get("sha") or _best_good_ref_here(
            load_records(ctx.status_path), ctx)
    except Exception as e:  # noqa: BLE001
        print(f"[loop] correctness monitor: no sha ({e})", flush=True)
        return None
    if not sha:
        print("[loop] correctness monitor: no green KEEP on record — nothing to monitor",
              flush=True)
        return None
    try:
        rec = monitor_mod.run(backend, sha, branch=f"{ctx.branch_prefix}-monitor",
                              op_dir=op_dir, profiles=profiles, kernel_file=kernel_file,
                              trigger="finish", log=lambda s: print(f"[loop] {s}", flush=True))
    except Exception as e:  # noqa: BLE001 — advisory; the harvest below must still run
        print(f"[loop] correctness monitor: could not run ({e.__class__.__name__}: {e})",
              flush=True)
        return None
    brief = {k: rec[k] for k in ("status", "sha", "job")}
    brief["alarms"] = [f"{f['profile']}: {f['what']}" for f in rec["findings"]
                       if f["severity"] == "alarm"]
    summary["correctness_monitor"] = rec["status"]
    if confirm:
        confirm["correctness_monitor"] = brief
        out = op_dir / "best_confirmation.json"
        try:
            out.write_text(json.dumps(confirm, indent=1), encoding="utf-8")
        except OSError as e:
            print(f"[loop] correctness monitor: could not update {out} ({e})", flush=True)
    return rec


def _load_loop_config(op_dir: Path) -> dict:
    cfg = op_dir / "loop.toml"
    if cfg.exists():
        return tomllib.loads(cfg.read_text())
    return {}


def _has_distillation_input(oracle: str | None, mode: str) -> bool:
    """Whether the teacher has anything to distill a milestone *from*: the reference's
    source (source mode), or black-box reconnaissance of it (blackbox mode)."""
    if mode == ORACLE_BLACKBOX:
        return True
    return bool(oracle)


def _milestone_valid(milestone: Path) -> bool:
    """A milestone on disk that passes the §9 gate (what _will_distill would skip on)."""
    from forgekernel.milestone import validate_milestone
    try:
        return not validate_milestone(Path(milestone).read_text(encoding="utf-8"))
    except OSError:
        return False


def goal_ref_backend(op: str, cfg: dict, git_name: str, git_email: str):
    """The ref_from_goal session's job channel: the submission tree's own backend."""
    def make(tree_op: str, tree: Path):
        return RealBackend(tree_op, tree, None, git_name, git_email)
    return make


def _resolve_oracle_mode(cfg: dict) -> str:
    """Which channel the teacher distills through: loop.toml [loop] oracle_mode, else source."""
    configured = cfg.get("loop", {}).get("oracle_mode")
    if configured:
        if configured not in ORACLE_MODES:
            raise SystemExit(
                f"loop.toml [loop] oracle_mode must be one of {ORACLE_MODES}, got {configured!r}")
        return configured
    return ORACLE_SOURCE


def _recon_op(cfg: dict, oracle_mode: str) -> str | None:
    """The reconnaissance tree's operator name, or None when this run does not
    reconnoitre.

    Every distilling run reconnoitres by default (tree ``_recon``, or the
    ``[loop] recon_op`` the card names). Under blackbox the report is the input
    that stands in for the source; under source it is the second input beside
    it, because "the oracle answers the same questions at no cost" turned out
    false for templated references: a grouped-GEMM source-mode run inferred a
    runtime split test and a ping-pong phase that the SASS never had and lost
    30% at M0 to them, while a blackbox run on the same card measured
    both and landed M0 at the source-mode run's final ratio. A source-mode run that
    wants the old source-only behavior (a control run) sets ``recon_op = ""``;
    blackbox cannot opt out — it has nothing else to distil from.
    """
    lc = cfg.get("loop", {})
    if oracle_mode == ORACLE_BLACKBOX:
        raw = lc.get("recon_op", "_recon")
        if raw is False or raw is None or str(raw).strip() == "":
            sys.stderr.write(
                f"[loop] warning: [loop] recon_op = {raw!r} ignored — a blackbox run "
                f"cannot opt out of reconnaissance (nothing else to distil from); "
                f"using the default tree operators/_recon\n")
            return "_recon"
        return str(raw)
    if oracle_mode == ORACLE_SOURCE:
        raw = lc.get("recon_op", "_recon")
        if raw is False or raw is None or str(raw).strip() == "":
            return None
        return str(raw)
    return None


def _resolve_oracle(mode: str, oracle_flag: str | None, cfg: dict) -> str | None:
    """The oracle path, or None when this run's teacher does not read source.

    Only ORACLE_SOURCE resolves a path. ORACLE_BLACKBOX withholds it deliberately —
    the point of that mode is that the reference is a binary you can run and profile
    but not read.

    Accepts several paths, because one file is often not the whole reference. One
    operator's pointed at flash_bwd_sm90.py — a parameterised kernel whose tile
    shapes, stage counts and atom layouts are all constructor arguments. The values
    for a given head_dim live in interface.py, which was never in the teacher's
    reading set, so it inferred them from a resource budget and got a self-consistent
    architecture that is not the reference's. Handing over the numbers would have
    removed the distillation work rather than fixed the input; listing both files
    leaves the reading to the teacher.
    """
    return ", ".join(_oracle_paths(mode, oracle_flag, cfg)) or None


def _oracle_paths(mode: str, oracle_flag: str | None, cfg: dict) -> list[str]:
    """The oracle as a LIST of paths — what the agent policy needs to deny them.

    ``_resolve_oracle`` renders the same answer as one comma-joined string, which is how
    the teacher's prompt shows it. Feeding that string to anything path-shaped produces
    one path that does not exist, which is how a list-valued oracle went undenied to the
    student: keep the two apart."""
    if mode != ORACLE_SOURCE:
        return []
    raw = oracle_flag or cfg.get("loop", {}).get("oracle") or cfg.get("oracle")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(p) for p in raw]
    else:
        # A --oracle flag may carry several paths the way the prompt shows them.
        items = str(raw).split(",")
    return [s.strip() for s in items if s.strip()]


def _student_denied_oracle(mode: str, oracle_flag: str | None, cfg: dict,
                           declared: list[str]) -> list[str]:
    """The oracle paths denied to the student: this run's oracle AND the one declared
    at launch. A reused ref_from_goal reference replaces the declared oracle as the
    run's input — it drops out of ``cfg`` — but it is still a reference the student
    must not read."""
    return list(dict.fromkeys([*_oracle_paths(mode, oracle_flag, cfg), *declared]))


def _card_kernel_file(op_dir: Path) -> str:
    """The kernel file a leaderboard card names (eval/leaderboard/problem.json kernel_file —
    kernel.cu for a CUDA student), else kernel.py."""
    try:
        return json.loads((op_dir / "eval" / "leaderboard" / "problem.json").read_text(
            encoding="utf-8")).get("kernel_file") or "kernel.py"
    except (OSError, ValueError):
        return "kernel.py"


def _resolve_language(language_flag: str | None, cfg: dict,
                      kernel_file: str | None = None) -> str:
    """Kernel language: --language wins, else loop.toml [loop].language, else what
    the kernel file implies (a ``.cu`` is the CUDA mode's ``cuda-sol``), else cutedsl.

    Mirrors _resolve_oracle. Governs which language the student writes the kernel in
    (injected as {LANGUAGE} into the student/distill prompts) and, through
    loop/lang_mode.py, which mode the run uses: the CUDA mode's prompt set, KB root
    and tools, or the CuTe DSL mode's. _main refuses a language the kernel file
    contradicts (EXIT_LANGUAGE_MISMATCH)."""
    if language_flag:
        return language_flag
    return (cfg.get("loop", {}).get("language")
            or lang_mode.language_for_kernel_file(kernel_file) or "cutedsl")


KB_ARCH_DEFAULT = "sm90"
# Every code through 13 already names another refusal (4-8 preflight, 9 fresh-start
# contamination, 10 workspace without a remote, 11 and 12 retired, 13 window undeclared);
# tests/test_kb_arch.py reads the literal return sites to keep this unique.
EXIT_NO_REMOTE = 10
EXIT_KB_ARCH_MISSING = 14
# 15: the run's reference is not declared for the agent policy (loop/agent_policy.py) —
# a blackbox run with no [loop] ref_source_paths / ref_source_urls would deny nothing.
EXIT_AGENT_POLICY_UNDECLARED = 15
# 16: the card's eval/bench.py times one input set instead of the geomean over
# several workloads of its shape (window_contract.bench_workloads_findings).
EXIT_SINGLE_WORKLOAD = 16
# 17: a reference archive the target / cluster scale divide by is read over other
# workloads than the bench's gate reading (_target_workloads_problems).
EXIT_TARGET_WORKLOADS = 17
# 18: the language and the kernel file name different modes (lang_mode.language_problem):
# a CUDA-mode language over a kernel.py, or a .cu kernel under a CuTe DSL language.
EXIT_LANGUAGE_MISMATCH = 18
# 19: a fresh start whose workspace's evaluation harness (gpu_entry.sh / eval/ / ref_bin/)
# differs from the card's (loop/harness_guard.py).
EXIT_HARNESS_DRIFT = 19
# 20: a fresh start under [loop] precision_contract = "strict" whose card cannot report the
# precision block (no eval/precision.py, or a bench.py that never calls PrecisionRecorder).
EXIT_PRECISION_UNREPORTED = 20
# ref_from_goal (loop/goal_ref.py): the generation cannot start (no goal-capable agent,
# "always" on a non-source run, an earlier generation in this op dir failed) ...
EXIT_GOAL_REF_UNAVAILABLE = 21
# ... or it ran and produced no correct reference that survived acceptance.
EXIT_GOAL_REF_FAILED = 22


def _resolve_kb_arch(flag: str | None, cfg: dict) -> str:
    """Which hw_kb/<language>/<arch>/ tree the run reads and writes: --kb-arch, else
    loop.toml [loop].kb_arch, else sm90.

    Mirrors _resolve_language. hw_kb/ is one tree per architecture with no
    shared root, and a run sees exactly one of them (driver.kb_read_dir): the
    KB an sm100 run opens must not be Hopper's, and a recipe it certifies must
    not be filed under sm90/. The default is what every card written before the
    key existed was hard-coded to, so a card that does not set it renders and
    reads exactly what it always did. Validated against the tree on disk by
    _kb_arch_problem, not against a list spelled here: the trees are the list.
    """
    chosen = flag or cfg.get("loop", {}).get("kb_arch") or KB_ARCH_DEFAULT
    if not re.fullmatch(r"[a-z][a-z0-9_]*", chosen):
        raise SystemExit(
            f"kb_arch must be a plain directory name like sm90 / sm100, got {chosen!r} "
            f"(--kb-arch / loop.toml [loop].kb_arch)")
    return chosen


def _kb_arch_problem(kb_dir: Path, kb_arch: str) -> str | None:
    """Why ``hw_kb/<language>/<kb_arch>/`` cannot be a run's KB tree, or None.

    Fail closed at launch, not at the first KB: citation hours later: a tree
    with no index.md is not a KB the prompts can point at (every write-side
    prompt registers the recipe in ``{KB_DIR}/{KB_ARCH}/index.md``), and a typo'd
    arch would otherwise start a run that reads nothing and files its recipes
    into a directory nobody indexes."""
    tree = kb_dir / kb_arch
    if not tree.is_dir():
        have = sorted(d.name for d in kb_dir.iterdir()
                      if d.is_dir() and (d / "index.md").is_file()) if kb_dir.is_dir() else []
        return (f"hw_kb/{kb_dir.name} has no {kb_arch}/ tree (available: {', '.join(have) or 'none'}) "
                f"— set loop.toml [loop].kb_arch / --kb-arch to one of them, or open the "
                f"tree with an index.md per hw_kb/<language>/README.md")
    if not (tree / "index.md").is_file():
        return (f"{tree} has no index.md — the prompts register every recipe there; "
                f"open the tree per hw_kb/<language>/README.md before pointing a run at it")
    return None


def main(argv: list[str] | None = None) -> int:
    return _main(argv)


def _resolve_ref_pricing(cfg: dict, root, recon_dir, recon_backend, make_backend,
                         recon_op: "str | None" = None):
    """(ref_price_dir, ref_price_backend, ref_price_app) for the driver's frozen
    pricing job on the reference tree — the ref's overlap account (role table
    with tensor + phase timeline).

    A blackbox run has that tree already (the recon tree), so it is the
    default; a source-mode run names one with ``[loop] ref_price_op``. Opt-out is
    explicit (``ref_price_op = ""``): a loop.toml copied from before the knob
    existed once left a 17h dsa_sparse run diagnosing against a ref exposure
    column of nulls. ``[loop] ref_price_app`` names the
    profiler-window entry when the tree has no ``eval/ncu_run.py``.
    """
    loop_cfg = cfg.get("loop", {})
    ref_price_op = loop_cfg.get("ref_price_op")
    if ref_price_op is None and recon_backend is not None:
        # the normalized recon tree name (cli._recon_op), never the raw key:
        # a blackbox card with recon_op = "" still reconnoitres in _recon
        ref_price_op = recon_op if recon_op is not None else recon_dir.name
    ref_price_app = loop_cfg.get("ref_price_app") or None
    if ref_price_app is not None and not (
            isinstance(ref_price_app, list) and all(isinstance(a, str) for a in ref_price_app)):
        sys.stderr.write("[loop] warning: [loop] ref_price_app must be a list of strings "
                         f"(got {ref_price_app!r}) — ignored.\n")
        ref_price_app = None
    if not ref_price_op:
        return None, None, ref_price_app
    ref_price_dir = operators_dir(root) / ref_price_op
    if not (ref_price_dir / ".git").is_dir():
        sys.stderr.write(f"[loop] warning: [loop] ref_price_op = {ref_price_op!r} "
                         f"but {ref_price_dir} is not a git repo — the reference "
                         f"will not be priced (role/dossier tables stay one-sided).\n")
        return None, None, ref_price_app
    if ref_price_app is not None:
        script = next((a for a in ref_price_app if a.endswith(".py")), None)
        if script is None or not (ref_price_dir / script).is_file():
            sys.stderr.write(
                f"[loop] WARNING: [loop] ref_price_app = {ref_price_app!r} names no .py "
                f"entry that exists in {ref_price_dir} — the driver runs the app AS "
                f"CONFIGURED (it does not fall back to eval/ncu_run.py), so the "
                f"reference will not be priced until loop.toml is fixed.\n")
    if ref_price_app is None and not (ref_price_dir / "eval" / "ncu_run.py").is_file():
        # Warn, but keep the tree wired: the driver re-checks the launch target
        # at every pricing call, so a teacher that adds eval/ncu_run.py during
        # reconnaissance (the gate's option 1) gets the ref priced.
        sys.stderr.write(
            f"[loop] WARNING: {ref_price_dir} has no eval/ncu_run.py and loop.toml "
            f"names no [loop] ref_price_app — the pricing recipe has no launch "
            f"target there, so the reference's OVERLAP ACCOUNT (role table with "
            f"tensor, phase timeline) will not exist unless the teacher adds "
            f"eval/ncu_run.py to that tree during reconnaissance; the recon finish "
            f"gate will otherwise hold the teacher to a ROLE/PMTL waiver. Add the "
            f"file or set ref_price_app = [\"python3\", \"<entry>.py\", \"ncu\"].\n")
    if recon_backend is not None and ref_price_dir == recon_dir:
        return ref_price_dir, recon_backend, ref_price_app      # same repo, one backend
    return ref_price_dir, make_backend(ref_price_op, ref_price_dir), ref_price_app


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="forge <op> loop",
        description="Run the two-layer loop unattended on a card `forge run` generated.",
    )
    p.add_argument("op")
    p.add_argument("--workspace", required=True,
                   help="the student's workspace (a single-branch clone of the card)")
    p.add_argument("--hours", type=float, default=4.0, help="inner-loop wall-clock budget")
    p.add_argument("--target-ms", type=float, default=None,
                   help="stop threshold; default: derived from the measured reference "
                        "via --equivalence-ratio")
    p.add_argument("--equivalence-ratio", type=float, default=1.0,
                   help="reproduce the ref to within this throughput ratio (default 1.0: "
                        "match the reference outright; above 1 = beat it, e.g. 1.05 = 5%% "
                        "more throughput; the stop target is ref_ms / ratio either way)")
    p.add_argument("--agent-cmd", default=None,
                   help="headless agent: reads prompt on stdin, prints final message to stdout")
    p.add_argument("--teacher-agent-cmd", default=None,
                   help="agent command for the TEACHER turns (recon, distill, diagnose/"
                        "patch, KB certify/harvest, lever design); the student keeps "
                        "--agent-cmd. Default: --agent-cmd serves both roles")
    p.add_argument("--max-meta", type=int, default=3)
    p.add_argument("--allow-preexisting-artifacts", action="store_true",
                   help="start a fresh run (empty status.jsonl) even though phase "
                        "products — milestone.md, recon findings, recon_complete.json, "
                        "archived baselines, meta history — already exist on disk "
                        "(default: refuse, so a new run cannot silently inherit a "
                        "previous run's products; `forge run` passes this on a resume)")
    p.add_argument("--allow-dirty-kb", action="store_true",
                   help="start even if hw_kb has uncommitted changes (default: refuse, so a "
                        "certified recipe from an earlier run is not buried under a new one)")
    p.add_argument("--harvest-only", action="store_true",
                   help="run only the end-of-run KB write-up (kb-tripwire + axis bill) on "
                        "the existing trajectory — for a run stopped before its finish "
                        "path. No measurement, no GPU job, no loop; one teacher turn; "
                        "nothing is published")
    args = p.parse_args(argv)
    # --harvest-only builds the loop's context from what is archived and refuses nothing:
    # a write-up of a stopped run must not re-measure or re-gate it.
    plan_only = bool(args.harvest_only)
    agent_cmds = [c for c in (args.agent_cmd, args.teacher_agent_cmd) if c]

    root = repo_root()
    op_dir = operators_dir(root) / args.op
    kernel_file = _card_kernel_file(op_dir)
    cfg = _load_loop_config(op_dir)
    workspace = Path(args.workspace).resolve()
    milestone = op_dir / "milestone.md"
    status_path = op_dir / "student" / "status.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.touch(exist_ok=True)
    meta_status_path = op_dir / "meta_status.jsonl"
    meta_status_path.touch(exist_ok=True)
    branch_prefix = f"forge-student-{args.op}"
    oracle_mode = _resolve_oracle_mode(cfg)
    oracle = _resolve_oracle(oracle_mode, None, cfg)
    # No reference to distil from (loop/goal_ref.py): a reference this op dir already
    # generated is wired in as the oracle right here, before anything reads one; a
    # generation still to run (or to resume) runs after the fresh-start gate below.
    declared_oracle = _oracle_paths(oracle_mode, None, cfg)
    try:
        goal_mode, goal_mode_src = goal_ref_mod.resolve_mode(cfg)
        goal_settings = goal_ref_mod.settings(cfg)
        goal_plan = goal_ref_mod.plan(
            op_dir, mode=goal_mode, oracle_mode=oracle_mode, oracle_paths=declared_oracle,
            milestone_valid=_milestone_valid(milestone))
    except goal_ref_mod.GoalRefError as e:
        sys.stderr.write(f"[loop] error: ref_from_goal: {e}\n")
        if not plan_only:
            return EXIT_GOAL_REF_UNAVAILABLE
        goal_mode, goal_mode_src = "?", "error"
        goal_settings = goal_ref_mod.Settings()
        goal_plan = goal_ref_mod.Plan("off", f"a real run would refuse: {e}")
    if goal_plan.action == "reuse":
        if not plan_only:
            try:
                goal_ref_mod.ensure_reference_tree(root, args.op, op_dir, goal_plan.state)
            except goal_ref_mod.GoalRefError as e:
                sys.stderr.write(f"[loop] error: ref_from_goal: {e}\n")
                return EXIT_GOAL_REF_UNAVAILABLE
        cfg, goal_arch = goal_ref_mod.apply(cfg, goal_plan.state, target_ms=args.target_ms)
        oracle_mode = ORACLE_SOURCE
        oracle = _resolve_oracle(oracle_mode, None, cfg)
        print(f"[loop] ref_from_goal: {goal_plan.why} — oracle = {oracle}; target: "
              f"{goal_arch['why']}", flush=True)
    elif goal_plan.runs_session:
        print(f"[loop] ref_from_goal = {goal_mode} ({goal_mode_src}): {goal_plan.why} — "
              f"a /goal session generates the reference first "
              f"({goal_settings.hours:g} h budget)", flush=True)
    language = _resolve_language(None, cfg, kernel_file)
    language_problem = lang_mode.language_problem(language, kernel_file)
    if language_problem and not plan_only:
        sys.stderr.write(f"[loop] error: {language_problem}\n")
        return EXIT_LANGUAGE_MISMATCH
    gate_window = _resolve_gate_window(None, cfg)
    kb_arch = _resolve_kb_arch(None, cfg)
    kb_root = root / lang_mode.kb_root_name(language)
    kb_arch_problem = _kb_arch_problem(kb_root, kb_arch)
    if kb_arch_problem and not plan_only:
        sys.stderr.write(f"[loop] error: kb_arch = {kb_arch!r}: {kb_arch_problem}\n")
        return EXIT_KB_ARCH_MISSING

    # Agent policy declaration: static, before anything is measured. The student is
    # kept offline and away from the reference source in every run; a blackbox run's
    # teacher is kept away from it too, which needs the card to say where it lives.
    # An empty dsl_source_dirs on a cutedsl run is derived from the DSL distribution
    # installed here (loop/dsl_source.py), checked against the declared reference
    # source before the student is pointed at it.
    dsl_source = dsl_source_mod.resolve(cfg, language=language)
    if dsl_source.defaulted:
        for n in dsl_source.notes:
            sys.stderr.write(f"[loop] {n}\n")
    for w in agent_policy_mod.declaration_warnings(cfg, dsl_source, language=language):
        sys.stderr.write(f"[loop] warning: {w}\n")
    policy_problems = agent_policy_mod.declaration_problems(cfg, oracle_mode)
    # The policy is delivered as project settings in the agent's working directory; a
    # leg that skips project settings or leaves the directory would run unrestricted.
    policy_problems += agent_policy_mod.command_findings(
        agent_cmds, dict(os.environ), args.workspace)
    if policy_problems:
        for f in policy_problems:
            sys.stderr.write(f"[loop]   {f}\n")
        if not plan_only:
            sys.stderr.write(
                f"[loop] error: refusing to start: the agent policy (student offline, no "
                f"reference source) could not be guaranteed — fix the findings above "
                f"(operators/{args.op}/loop.toml / the agent command).\n")
            return EXIT_AGENT_POLICY_UNDECLARED

    # Timed-window CONTENT contract preflight (window.toml): the reference's
    # timed call and the student's must be declared to share a boundary, or
    # every reference input the student does not receive must be classed. Static,
    # costs nothing, and must come before any GPU job — a run judged on a ratio
    # with an undeclared contract tax spends its whole budget diagnosing a
    # residual no kernel lever can remove. Only runs with a reference are held
    # to it (a target-ms-only run has no second window to compare against), and
    # only fresh starts refuse: a run already on record predates the rule and
    # resumes with a warning.
    resuming = _status_has_records(status_path) or _launch_admitted(op_dir, status_path)
    window_state, window_contract, window_problems_w = _window_preflight(
        op_dir, cfg, resuming=resuming)
    if window_problems_w:
        for f in window_problems_w:
            sys.stderr.write(f"[loop]   {f}\n")
        if window_state == "refuse" and not plan_only:
            sys.stderr.write(
                f"[loop] error: refusing to start with an undeclared / inconsistent "
                f"timed-window contract (operators/{args.op}/{WINDOW_FILENAME}) — `forge "
                f"run` generates it from the judge's signature; see the findings above.\n")
            return EXIT_WINDOW_UNDECLARED  # NOT 12: that already means "chain contract"
        elif window_state == "warn":
            sys.stderr.write(
                "[loop] WARNING: resuming a run whose timed-window contract is "
                "undeclared / inconsistent — the credited ratio will not be "
                "computed; declare it before the next fresh start.\n")

    # Multi-workload perf preflight: the card's perf verdict is the geomean over
    # several workloads of its fixed shape. Static and free; fresh starts refuse,
    # resuming runs (which predate the rule or are mid-run) only warn.
    workload_problems = bench_workloads_preflight(op_dir)
    if workload_problems:
        for f in workload_problems:
            sys.stderr.write(f"[loop]   {f}\n")
        if resuming or plan_only:
            sys.stderr.write(
                "[loop] WARNING: this card's perf verdict rests on one input set — "
                "the kept kernel can be tuned to it and regress on the shape's other "
                "workloads.\n")
        else:
            sys.stderr.write(
                "[loop] error: refusing to start a card whose eval/bench.py does not "
                "time several workloads of its shape (timing.measure_windows_workloads).\n")
            return EXIT_SINGLE_WORKLOAD

    # Fresh-start contamination gate: checked before anything is measured or
    # distilled, so refusal costs nothing — after it, every phase would resume
    # from whatever products are on disk without recording whose they were.
    # Applies only to runs that author their own products (there is distillation
    # input).
    fresh_findings: list[str] = []
    if ((_has_distillation_input(oracle, oracle_mode) or goal_plan.runs_session)
            and not args.allow_preexisting_artifacts):
        recon_probe = None
        if (recon_op := _recon_op(cfg, oracle_mode)) is not None:
            recon_probe = operators_dir(root) / recon_op
        fresh_findings = fresh_start_contamination(
            op_dir, milestone, status_path, meta_status_path, recon_probe,
            evidence_dir=op_dir / "evidence")
    if fresh_findings and not plan_only:
        sys.stderr.write(
            f"[loop] error: fresh start (no records in {status_path}) but "
            f"{len(fresh_findings)} phase product(s) already on disk — a previous "
            f"run's products would be silently inherited:\n")
        for f in fresh_findings:
            sys.stderr.write(f"[loop]   {f}\n")
        sys.stderr.write(
            "[loop] archive them first (e.g. move crash residue to "
            ".forge-loop/crash-backup-<ts>/), or pass "
            "--allow-preexisting-artifacts to declare the reuse.\n")
        return 9  # NOT 8: that already means "KB leak lint found provenance"
    if not plan_only:
        # Ownership transfer: the tree just passed the gate (or the reuse was
        # declared), so everything the run writes from here on — baselines, recon
        # findings, the milestone — belongs to it, and a crash anywhere in the
        # hours-long pre-iteration pipeline must restart into a resume, not a
        # refusal (a blackbox run's seeded recon tree cannot be repaired by
        # archiving: the guard's archive list has never covered it).
        _admit_launch(op_dir, status_path)

    # ref_from_goal: generate (or finish generating) the reference now — after the gate,
    # so its products belong to this run, and before the architecture reference and the
    # library baseline, which it may stand in for (goal_ref.apply).
    if goal_plan.runs_session and not plan_only:
        if not agent_cmds:
            sys.stderr.write("[loop] error: ref_from_goal: no --agent-cmd / "
                             "--teacher-agent-cmd to run the /goal session with\n")
            return EXIT_GOAL_REF_UNAVAILABLE
        gen = goal_ref_mod.Generation(
            op=args.op, op_dir=op_dir, root=root, cfg=cfg, kernel_file=kernel_file,
            language=language, gate_window=gate_window, target_ms=args.target_ms,
            agent_cmd=args.teacher_agent_cmd or args.agent_cmd,
            # the student's deny list, with the network on (no cluster CLIs)
            policy=dc_replace(agent_policy_mod.from_config(
                cfg, oracle_mode=ORACLE_SOURCE, repo_root=root, op=args.op,
                teacher_trees=_teacher_only_trees(cfg, root, args.op, oracle_mode,
                                                  declared_oracle, goal_ref=False),
                dsl_source_dirs_default=dsl_source.dirs if dsl_source.defaulted else (),
                language=language), student_online=True,
                deny_write=tuple(goal_ref_mod.session_read_only_paths(op_dir))),
            make_backend=goal_ref_backend(args.op, cfg, GIT_NAME, GIT_EMAIL),
            settings=goal_settings,
            kb_dirs=[str(kb_root / kb_arch)] if (kb_root / kb_arch).is_dir() else [],
            heartbeat_s=AGENT_HEARTBEAT_S)
        gen.read_dirs = [str(p) for p in gen.policy.dsl_source_dirs]
        try:
            gen_state = goal_ref_mod.generate(
                gen, resume=goal_plan.state if goal_plan.action == "resume" else None)
        except goal_ref_mod.GoalRefError as e:
            sys.stderr.write(f"[loop] error: {e}\n")
            return EXIT_GOAL_REF_FAILED
        cfg, goal_arch = goal_ref_mod.apply(cfg, gen_state, target_ms=args.target_ms)
        oracle_mode = ORACLE_SOURCE
        oracle = _resolve_oracle(oracle_mode, None, cfg)
        print(f"[loop] ref_from_goal: oracle = {oracle}; target: {goal_arch['why']}",
              flush=True)

    # The architecture reference goes first, before the library baseline and before
    # anything is distilled. It is what target_ms anchors to and what every mechanism
    # gate line in the milestone anchors to, and v1 is the one write with no rollback
    # behind it — so measuring it after distillation would mean authoring those gates
    # against numbers that do not exist yet. Measuring it first also fails fast: it is
    # the step most likely to be broken, and finding that out costs one job instead of
    # a library measurement plus a distillation.
    will_distill = _will_distill(milestone, oracle, oracle_mode)
    arch_baseline = _ensure_arch_baseline(
        args.op, op_dir, cfg, measure=not plan_only)
    if not plan_only and (why := check_arch_ref_ready(
            cfg, arch_baseline, will_distill=will_distill)):
        sys.stderr.write(
            f"[loop] error: refusing to distil a milestone without the architecture "
            f"reference — {why}.\n"
            f"[loop] measure it, then re-run:\n"
            f"[loop]   python -m forgekernel.submit "
            f"{cfg.get('loop', {}).get('arch_ref_submit_op', '_arch_ref')} submit "
            f"--mode {cfg.get('loop', {}).get('arch_ref_mode', '<mode>')}\n")
        return 6

    # The archived ref baseline is what every gate in v1 is anchored to, and where
    # target_ms comes from unless overridden.
    try:
        ref_baseline = _ensure_ref_baseline(
            args.op, op_dir, milestone, oracle, cfg, REF_REPEAT,
            mode=oracle_mode, measure=not plan_only, arch=arch_baseline)
    except ValueError as e:
        sys.stderr.write(f"[loop] error: {e}\n")
        return 4
    try:
        target_ms = _resolve_target_ms(
            args.target_ms, ref_baseline, args.equivalence_ratio, arch=arch_baseline,
            gate_window=gate_window)
    except ValueError as e:
        # gate_window="device" over a baseline that never measured the device
        # window, or measured it under a broken precondition (host-bound reference):
        # refuse to start rather than derive the stop threshold from a number that
        # is the reference's host tax rather than its kernel time.
        sys.stderr.write(f"[loop] error: {e}\n")
        return 7
    if target_ms is None and not plan_only:
        declared = _resolve_ref_spec(cfg).get("ref_cmd")
        sys.stderr.write(
            f"[loop] error: no --target-ms and no ref baseline for {args.op}.\n")
        if declared:
            sys.stderr.write(
                f"[loop] the operator declares ref_cmd ({declared}) but measuring it "
                f"produced nothing — see the errors above.\n")
        else:
            sys.stderr.write(
                f"[loop] the task declares no reference to measure and no [target] ms: "
                f"one of the two is needed to derive the stop threshold\n")
        return 5

    # The absolute target, the cluster scale and the finish's median-vs-median
    # ratio divide the gate reading by these archives: on a multi-workload card
    # the gate reading is a geomean, so each archive must be the geomean over the
    # same workloads. Fresh starts refuse; resuming runs warn, and the driver
    # suspends the absolute-target currency while they disagree.
    target_wl_problems = _target_workloads_problems(args.op, op_dir,
                                                    (arch_baseline, ref_baseline))
    if target_wl_problems:
        for f in target_wl_problems:
            sys.stderr.write(f"[loop]   {f}\n")
        if resuming or plan_only:
            sys.stderr.write(
                "[loop] WARNING: the reference archive and the gate reading cover different "
                "workloads — the driver will not stop on, or re-anchor from, the absolute "
                "target while they disagree.\n")
        else:
            sys.stderr.write(
                "[loop] error: refusing to start with a target reference read over other "
                "workloads than the bench's gate reading — re-archive the reference as the "
                "geomean over the same workloads.\n")
            return EXIT_TARGET_WORKLOADS

    policy = agent_policy_mod.from_config(
        cfg, oracle_mode=oracle_mode, repo_root=root, op=args.op,
        teacher_trees=_teacher_only_trees(
            cfg, root, args.op, oracle_mode,
            _student_denied_oracle(oracle_mode, None, cfg, declared_oracle)),
        dsl_source_dirs_default=dsl_source.dirs if dsl_source.defaulted else (),
        language=language)
    backend = RealBackend(args.op, workspace, args.agent_cmd, GIT_NAME, GIT_EMAIL,
                          teacher_agent_cmd=args.teacher_agent_cmd, agent_policy=policy,
                          agent_heartbeat_s=AGENT_HEARTBEAT_S)
    try:
        backend.job_env_files = resolve_job_env_files(cfg, op_dir, kernel_file=kernel_file)
    except ValueError as e:
        sys.stderr.write(f"[loop] error: {e}\n")
        return 2
    if backend.job_env_files:
        print(f"[loop] job_env_files: every job runs the card's "
              f"{', '.join(sorted(backend.job_env_files))} (from {op_dir}); fix the job "
              f"environment there, not in the workspace", flush=True)
    # The evaluation harness is the card's (loop/harness_guard.py): every job tree
    # carries the card's tracked gpu_entry.sh / eval/ / ref_bin/ / .gitattributes, and
    # the driver reverts a delivery that edits them. A workspace whose harness already
    # differs from the card's is refused on a fresh start (the student would read one
    # judge and be judged by another) and only warned about on a resume (the overlay
    # puts the card's judge back under the run from the next job on).
    try:
        guard_mode = harness_guard.resolve_guard_mode(cfg)
    except ValueError as e:
        sys.stderr.write(f"[loop] error: {e}\n")
        return 2
    backend.kernel_file = kernel_file
    if guard_mode != harness_guard.GUARD_OFF:
        rc = _harness_guard_preflight(
            op_dir, workspace, kernel_file, backend, resuming=resuming, dry_run=plan_only)
        if rc:
            return rc
    # The precision gate (loop/precision_gate.py): what a failed precision check costs.
    try:
        precision_contract, precision_defaulted = precision_gate.resolve_contract(cfg, op_dir)
    except ValueError as e:
        sys.stderr.write(f"[loop] error: {e}\n")
        return 2
    rc = _precision_preflight(op_dir, precision_contract, precision_defaulted,
                              resuming=resuming, dry_run=plan_only)
    if rc:
        return rc
    precision_bench_reports = (precision_contract != precision_gate.OFF
                               and not precision_gate.bench_findings(op_dir))
    # Blackbox reconnaissance runs against its own submission tree (operators/_recon
    # by default): probe scripts and findings are teacher-only, and the student's GPU
    # job receives its operator tree as a tarball — anything placed there would leak.
    # The tree must be a git repo because delivery is `git archive` of committed files.
    # A source-mode run gets the same tree when its loop.toml names a recon_op: the
    # teacher then distils from source AND from what it measured, and the
    # measurements never reach the student either way.
    recon_backend, recon_dir, recon_path = None, None, None
    if (recon_op := _recon_op(cfg, oracle_mode)) is not None:
        recon_dir = operators_dir(root) / recon_op
        if (recon_dir / ".git").is_dir():
            recon_backend = RealBackend(recon_op, recon_dir, args.agent_cmd,
                                        GIT_NAME, GIT_EMAIL,
                                        teacher_agent_cmd=args.teacher_agent_cmd,
                                        agent_policy=policy,
                                        agent_heartbeat_s=AGENT_HEARTBEAT_S)
            recon_path = recon_dir / "report.md"
        elif oracle_mode == ORACLE_BLACKBOX:
            sys.stderr.write(
                f"[loop] warning: oracle_mode=blackbox but {recon_dir} is not a git "
                f"repo — reconnaissance will be skipped and the teacher distils from "
                f"the ref/arch baselines alone.\n")
        else:
            sys.stderr.write(
                f"[loop] warning: source-mode run with no reconnaissance tree — {recon_dir} "
                f"is not a git repo, so reconnaissance is skipped and the teacher "
                f"distils from the oracle alone (a weaker run: the SASS-level facts the "
                f"source states only as a template go unmeasured). Create the tree, "
                f"point [loop] recon_op at one, or set recon_op = \"\" to declare a "
                f"source-only control run on purpose.\n")
    evidence_dir = recon_dir if recon_backend is not None else op_dir / "evidence"
    ref_price_dir, ref_price_backend, ref_price_app = _resolve_ref_pricing(
        cfg, root, recon_dir, recon_backend,
        lambda op, d: RealBackend(op, d, None, GIT_NAME, GIT_EMAIL),
        recon_op=_recon_op(cfg, oracle_mode))
    ref_in_job = _resolve_ref_in_job(cfg, _resolve_ref_spec(cfg))
    # The channels that need NCU (the timeline probe, stall pricing, the orchestration
    # census) are the only loop.toml knobs left: `forge run` turns them off on a machine
    # whose counters cannot be read, and a leaderboard card turns the timeline probe on.
    lc = cfg.get("loop", {})
    source_brief_file = lc.get("source_brief_file")
    ctx = LoopContext(
        op=args.op, workspace=workspace, milestone=milestone, kb_dir=kb_root,
        status_path=status_path, meta_dir=root / "meta", kernel_file=kernel_file,
        meta_status_path=meta_status_path,
        branch_prefix=branch_prefix, target_ms=target_ms,
        language=language, gate_window=gate_window, kb_arch=kb_arch,
        ref_baseline=ref_baseline, arch_baseline=arch_baseline,
        window_contract=window_contract,
        deadline_epoch=time.time() + args.hours * 3600, backend=backend, now=time.time,
        decisions_dag=[],  # empty -> the driver parses the milestone §4 DAG per meta step
        oracle_path=oracle, oracle_mode=oracle_mode,
        source_brief=Path(source_brief_file).read_text(encoding="utf-8") if source_brief_file else "",
        recon_backend=recon_backend, recon_dir=recon_dir, recon_path=recon_path,
        max_probe_rounds=MAX_PROBE_ROUNDS,
        kb_tripwire_rounds=KB_TRIPWIRE_ROUNDS,
        max_verdict_probe_rounds=MAX_VERDICT_PROBE_ROUNDS,
        infeasible_min_diagnoses=INFEASIBLE_MIN_DIAGNOSES,
        lever_pricing_rounds=LEVER_PRICING_ROUNDS,
        lever_pricing_max_levers=LEVER_PRICING_MAX_LEVERS,
        reopen_max_fruitless=REOPEN_MAX_FRUITLESS,
        harness_guard=guard_mode,
        precision_contract=precision_contract,
        precision_bench_reports=precision_bench_reports,
        max_consecutive_reverts=MAX_CONSECUTIVE_REVERTS,
        stall_pricing=bool(lc.get("stall_pricing", True)),
        reprice_min_gain_pct=REPRICE_MIN_GAIN_PCT,
        timeline_probe=str(lc.get("timeline_probe", "off")),
        orch_census=bool(lc.get("orch_census", True)),
        # Single-ref contract: a ref measured through a separate submission tree
        # (ref_mode task or ref_cmd via submit_op — blackbox runs, source-leak)
        # never rides the student job, and a target-ms-only run has no ref to ride;
        # the driver then drops any in-job comparator readings the bench smuggles.
        ref_rides_job=_ref_rides_job(_resolve_ref_spec(cfg)),
        # [loop] ref_in_job: the comparator bench.py times inside the student job
        # IS the declared reference (same build the ref tree measures). The
        # reading is then node- and period-matched and becomes the parity
        # currency; the ref tree keeps the launch baseline / NCU / recon duties.
        ref_in_job=ref_in_job,
        equivalence_ratio=float(args.equivalence_ratio),
        # Suspicious-watermark confirmation margin: a new best beyond this fraction is
        # re-measured before it anchors meta transactions.
        watermark_margin=WATERMARK_MARGIN,
        # Records are stamped with the machine they ran on and judged against
        # same-machine rows; on another machine (a resume elsewhere) the target is
        # restated from a reference measured there (the reanchor_ref channel below).
        home_cluster=_home_cluster(args.op),
        # Phantom-gap re-anchor band: a gap-driven diagnosis whose residual sits inside
        # this fraction first re-anchors the ref same-period — the ref-side mirror of
        # the watermark margin.
        reanchor_band=REANCHOR_BAND,
        evidence_dir=evidence_dir, ref_price_dir=ref_price_dir,
        ref_price_backend=ref_price_backend, ref_price_app=ref_price_app,
    )
    # The re-anchor's measurement channel: the same fresh-ref path the finish
    # confirmation uses (measure_ref_fresh — parsed readings, never archived, so
    # the loop-start archive every past gate was judged against stays untouched),
    # tagged with the machine the samples actually ran on. Two consumers: the
    # phantom-gap re-anchor (same machine as the launch anchor, else refused) and
    # the per-machine target scale (any machine, min-vs-min against the launch
    # reading of this same channel).
    # A target-ms-only run declares no ref; the driver then skips both.
    if reanchor_spec := _resolve_ref_spec(cfg):
        def _fresh_ref_samples(_op=args.op, _spec=reanchor_spec,
                               _win=ctx.gate_window) -> tuple[list[float], str]:
            from forgekernel.baseline_cli import measure_ref_fresh
            runs = measure_ref_fresh(_op, _spec, repeat=3)
            key = "ms_device" if _win == "device" else "ms"
            # The label is where the ref jobs actually RAN (each run records
            # the cluster its submit resolved to), not where the student is
            # believed to be: if the channel failed to follow, the mismatch
            # must be visible to both consumers. Runs spread over several
            # clusters get a label no cluster matches.
            seen = sorted({r.get("cluster", "") for r in runs if r.get("cluster")})
            label = seen[0] if len(seen) == 1 else ("mixed:" + ",".join(seen) if seen else "")
            return ([r[key] for r in runs if r.get(key) is not None], label)
        ctx.reanchor_ref = _fresh_ref_samples
        # Drift can only be transported for a target PAIRED with a launch-period
        # node sample: the CLI-derived seed qualifies (it came off a measured
        # baseline); an explicit --target-ms does not (nobody sampled the nodes it
        # was set under). The anchor is the loop-start reading of the SAME channel
        # measure_ref_fresh re-measures — baseline_cli keeps the fastest, so the
        # driver pairs it min-vs-min.
        if args.target_ms is None and ref_baseline is not None:
            anchor_ms = (ref_baseline.ms_device if ctx.gate_window == "device"
                         else ref_baseline.ms)
            if anchor_ms:
                ctx.target_anchor_ms = float(anchor_ms)
                ctx.target_anchor_cluster = ref_baseline.cluster or ""

    if args.harvest_only:
        return _harvest_only(ctx, args, root, milestone, oracle, cfg)

    if not args.agent_cmd:
        sys.stderr.write("[loop] error: --agent-cmd is required\n")
        return 2

    # The workspace must be able to push: every student iteration commits there and
    # pushes the branch the GPU job is built from. Without a remote the loop only
    # finds out after distilling — one wasted agent call and a run that stops at
    # iteration 0 with a git error that says nothing about setup.
    if not _workspace_can_push(workspace):
        sys.stderr.write(
            f"[loop] error: {workspace} has no 'origin' remote. The student workspace "
            f"should be a single-branch clone of the operator repo:\n"
            f"[loop]   git clone --single-branch --branch main "
            f"{op_dir} {workspace}\n")
        return EXIT_NO_REMOTE

    # Nothing in this system commits hw_kb, so an unharvested recipe from an earlier
    # run is one `git checkout` away from gone. Starting on top of it also stacks a
    # second run's recipes onto the same review.
    # docs/kb-provenance rides along: it is where the teacher archives narrated
    # provenance, so an unharvested archive is as one-checkout-from-gone as an
    # unharvested recipe, and a reviewer commits the two together.
    # The gap-queue archive under docs/kb-provenance/gaps/ is regenerated from
    # the recon table by every run and is human-review material, not a recipe:
    # it must not refuse the next start the way an unharvested recipe does.
    # Each mode guards its own KB root and its own provenance (lang_mode): a CUDA
    # run's unharvested recipe does not hold up a CuTe DSL run, nor the reverse.
    provenance_dir = lang_mode.provenance_dir(root, getattr(ctx, "language", None))
    blockers = kb_blockers(root, ctx.kb_dir) + [
        c for c in kb_blockers(root, provenance_dir)
        if not c.path.startswith("docs/kb-provenance/gaps/")
        and lang_mode.owns_provenance(c.path, ctx.language)]
    if blockers and not args.allow_dirty_kb:
        sys.stderr.write(
            f"[loop] error: {ctx.kb_dir} / {provenance_dir} have {len(blockers)} uncommitted "
            f"change(s) — harvest them first (commit or discard), or pass --allow-dirty-kb:\n")
        for c in blockers:
            sys.stderr.write(f"[loop]   {c.status} {c.path}\n")
        return 3

    # Every student of every later operator reads hw_kb in full, so an entry that
    # cites the reference implementation leaks the answer across runs. The lint is
    # the mechanical form of the discipline in docs/kb-provenance/README.md. For a
    # source-mode run, the reference's own file and symbol names are read off the
    # resolved oracle tree so protection does not depend on a hand-kept list.
    _kb_missing: list[str] = []

    def _kb_report(kind: str, msg: str) -> None:
        if kind == "summary":
            sys.stderr.write(f"[loop] kb-lint reference tokens: {msg}\n")
        elif kind == "missing":
            _kb_missing.append(msg)

    oracle_tokens = _oracle_lint_tokens(cfg, oracle, report=_kb_report)
    for m in _kb_missing:
        sys.stderr.write(f"[loop] warning: oracle path not readable — {m}\n")
    # Scoped to the run's own tree: another architecture's entries are not in
    # this run's read set, so a leak there cannot reach this student and must
    # not stop this launch (it stops that architecture's).
    leaks = lint_kb_dir(kb_read_dir(ctx), extra_tokens=oracle_tokens)
    if leaks:
        sys.stderr.write(
            f"[loop] error: KB leak lint found {len(leaks)} finding(s) in {kb_read_dir(ctx)} — move "
            f"reference provenance to docs/kb-provenance/:\n")
        for f in leaks:
            sys.stderr.write(f"[loop]   {f}\n")
        return 8  # NOT 7: that already means "target_ms underivable for gate_window"
    ctx.kb_lint_tokens = oracle_tokens  # run_kb_certify lints with the same token set

    # Job-environment preflight: two short jobs, before anything is iterated, that
    # ask whether the harness's own channels (NCU, and the tools that re-spawn a
    # bare python3) can start on this image at all. The reference tree already had
    # such a gate (no NCU signature -> no distillation); the student tree had none,
    # and a run once went 2.5 days with every NCU / timeline / stall reading
    # dead on two imports. Checked on every start: a resumed run can
    # land on a new node, and a fix in the card is what a relaunch should prove.
    jpolicy = str(cfg.get("loop", {}).get("job_env_preflight", "refuse"))
    if jpolicy not in JOB_ENV_PREFLIGHT_POLICIES:
        sys.stderr.write(f"[loop] error: [loop] job_env_preflight = {jpolicy!r} — expected "
                         f"one of {', '.join(JOB_ENV_PREFLIGHT_POLICIES)}\n")
        return 2
    if (why := job_env_preflight(ctx, jpolicy)) and jpolicy == "refuse":
        sys.stderr.write(
            f"[loop] error: the job environment cannot run the harness's own channels — "
            f"{why}.\n"
            f"[loop] every NCU digest / timeline / stall ledger / lever price of this run "
            f"would be missing. Fix the card's gpu_entry.sh / eval/ (list them in [loop] "
            f"job_env_files so a fix in the card reaches the jobs of a run in progress), or set "
            f"[loop] job_env_preflight = \"warn\" to run without those channels on purpose.\n")
        return 13  # NOT 12: that already means "chain member's bench breaks its contract"

    try:
        summary = run_two_layer(ctx, max_meta=args.max_meta)
    except (AgentError, BackendError) as e:
        # The inner loop degrades cleanly on these, but the meta layer (recon,
        # distill, meta patches, kb certification) had no net: one starved GPU job
        # in recon ended a whole 20h run. Same clean-stop
        # semantics as run_inner_loop, so the harvest below still runs.
        kind = "backend_error" if isinstance(e, BackendError) else "agent_failed"
        print(f"[loop] meta-layer {kind} — stopping cleanly: {e}", flush=True)
        summary = {"stop": kind, "meta_steps": -1}
    # Best-sha confirmation runs after the stop so it never competes with iterations
    # for budget, and before harvest so the ratio lands in the log next to the run's
    # verdicts. Its own failure must not cost the harvest below.
    confirm = None
    # A same-job parity stop named its own sha (not necessarily the absolute
    # best) and already bought same-period samples of it: confirm THAT sha,
    # counting those samples, or the report would carry a different sha —
    # possibly reading below parity — next to the verdict that stopped the run.
    paired = getattr(ctx, "paired_stop", None) or {}
    try:
        confirm = run_best_confirmation(
            ctx, jobs=CONFIRM_JOBS, ref=paired.get("sha"),
            seed_samples=paired.get("samples"))
    except Exception as e:
        print(f"[loop] confirm: stage failed ({e}) — continuing to harvest", flush=True)
        confirm = None
    if confirm and "median_ratio" not in confirm:
        # No same-job ratio came back (out-of-job ref, target-ms-only, or a
        # bench that produced no in-job ref reading), so the ratio must be
        # median-vs-median: re-anchor the ref through its own channel NOW —
        # against the frozen loop-start archive, node drift over the run's
        # hours would read as kernel change.
        ref_spec = _resolve_ref_spec(cfg)
        if not ref_spec:
            print("[loop] confirm: no ref declared (target-ms-only run) — "
                  "the absolute median is the claim", flush=True)
        else:
            try:
                from forgekernel.baseline_cli import measure_ref_fresh
                runs = measure_ref_fresh(args.op, ref_spec, repeat=CONFIRM_JOBS)
                key = "ms_device" if ctx.gate_window == "device" else "ms"
                ref_samples = [r[key] for r in runs if r.get(key) is not None]
                if wl_why := _fresh_ref_workloads_mismatch(runs, confirm):
                    confirm["ref_workloads_mismatch"] = wl_why
                    print(f"[loop] confirm: no median-vs-median ratio — {wl_why}; "
                          f"reporting the student median only", flush=True)
                elif ref_samples:
                    confirm["ref_samples_ms"] = ref_samples
                    confirm["ref_median_ms"] = statistics.median(ref_samples)
                    confirm["median_ratio"] = confirm["ref_median_ms"] / confirm["median_ms"]
                    print(f"[loop] confirm: median-vs-median ratio "
                          f"{confirm['median_ratio']:.4f} (student {confirm['median_ms']} ms "
                          f"over {len(confirm['jobs'])} job(s), ref {confirm['ref_median_ms']} ms "
                          f"over {len(ref_samples)} job(s), {ctx.gate_window} window)", flush=True)
                else:
                    print("[loop] confirm: ref re-anchor yielded no usable reading — "
                          "reporting the student median only", flush=True)
            except Exception as e:
                print(f"[loop] confirm: ref re-anchor failed ({e}) — "
                      f"reporting the student median only", flush=True)
    if confirm:
        # The declared contract tax, priced off the confirmed sha's per-kernel
        # table against the ref median settled above (either path). Reported
        # next to the raw ratio, never in its place.
        try:
            credit = apply_window_credit(ctx, confirm)
        except Exception as e:  # noqa: BLE001 — a diagnostic must not cost the harvest
            print(f"[loop] confirm: window credit failed ({e})", flush=True)
            credit = None
        summary["confirm"] = {k: confirm[k] for k in
                              ("median_ms", "median_ratio") if k in confirm}
        if credit and credit.get("credited_ratio") is not None:
            summary["confirm"]["credited_ratio"] = credit["credited_ratio"]
        summary["confirm"]["n"] = len(confirm["jobs"])
        summary["confirm"]["sha"] = confirm["sha"][:10]
        out = op_dir / "best_confirmation.json"
        out.write_text(json.dumps(confirm, indent=1), encoding="utf-8")
        print(f"[loop] confirm: archived -> {out}", flush=True)
    # Correctness monitor on the sha the run's claim rests on: the card's own
    # correctness task under fresh seeds and reshaped inputs, plus a watch on the
    # kernel's outputs (forgekernel.monitor). Advisory: it is recorded next to the
    # confirmation and in correctness_monitor.jsonl, and costs nothing else.
    if summary.get("stop") != "backend_error":
        _run_finish_monitor(ctx, backend, op_dir, confirm, summary,
                            monitor_mod.DEFAULT_PROFILES, kernel_file)
    # kb-tripwire settlement: before scanning for harvest, give the expensive probe
    # victories their write-up turn so the recipes land in hw_kb and ride the same
    # harvest report below. Skipped when the run died on the agent
    # or backend — the write-up needs a working agent. run_kb_harvest itself never
    # raises (total internal catch): nothing may stand between the run and the
    # harvest report below.
    if summary.get("stop") not in ("agent_failed", "backend_error"):
        recorded = run_kb_harvest(ctx)
        if recorded:
            summary["kb_harvest"] = recorded
    # The out-of-run gap queue (candidates rows nobody verified) is part of the
    # run's outcome: re-banked from the table as it stands now, archived for
    # human review under docs/kb-provenance/gaps/ (never hw_kb — it is the
    # reference's structure as narrated) and counted in the summary so an
    # unattended run's reader sees how many guesses the run left standing.
    try:
        from forgekernel.loop.driver import _persist_gap_queue
        gaps = _persist_gap_queue(ctx)
        if gaps:
            summary["gap_queue"] = [f"{g.decision} ({g.reason})" for g in gaps]
            print(f"[loop] gap queue: {len(gaps)} unverified candidates row(s) — "
                  f"see evidence/gap_queue.md and docs/kb-provenance/gaps/ "
                  f"(not KB material)", flush=True)
    except Exception as e:  # noqa: BLE001 — bookkeeping, never the run's fate
        print(f"[loop] gap queue unavailable ({e.__class__.__name__}: {e})", flush=True)
    # The teacher wrote into this tree; the driver only ever commits the workspace.
    left = git_status(root, [str(ctx.kb_dir), str(milestone), str(provenance_dir)])
    if left is None:
        print(f"[harvest] could not read git status under {root} — check it by hand")
    else:
        print(format_report(left, root))
        # Recipes certified this run stay as uncommitted files: a human reviews them
        # and commits or discards them, and until then the next run's guard refuses
        # to start on top of them.
        summary["unharvested"] = [c.path for c in left]
    print(f"[loop] done: {json.dumps(summary)}")
    (op_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    records = status_path.read_text().count("\n")
    print(f"[loop] status.jsonl now has {records} records at {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
