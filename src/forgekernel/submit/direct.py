"""Run one gpu_entry task on this machine (``local``) or on one ssh host (``ssh``).

A job's bookkeeping (its id, log, verdict, the detached runner) is forgekernel.submit.jobs;
this module is what the runner does.

One run:

1. ``git archive`` the requested branch of the op's tree and extract it fresh -- locally
   under ``<op>/.exec/tree``, or on the host
   under ``<workdir>/<run>/<op>`` -- so a run never inherits the last run's build products.
2. Take the GPU lock, then ``bash gpu_entry.sh <mode>`` with the tree's ``[env]``,
   ``CUDA_VISIBLE_DEVICES`` and ``TASK_TIMEOUT``. The lock is per GPU and per machine
   (two runs on one GPU would time each other), shared by every op and every run.
3. Map the exit: rc 0 -> Succeeded; the task ran and failed -> Failed; ssh could not
   deliver it (no begin marker) -> retried, then Failed with TRANSPORT_ERROR_MARKER, which
   the loop reports as a transport problem rather than charging the kernel.

Nothing here kills processes it did not start: on a user's machine a GPU held by someone
else is theirs. The lock is what keeps forgekernel's own runs apart.
"""

from __future__ import annotations

import fcntl
import io
import lzma
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path

from forgekernel.submit.config import SubmitConfig, is_secret_env
from forgekernel.submit.jobs import (BEGIN_MARKER, RUN_HEADROOM_S, STATUS_FAILED,
                                     STATUS_SUCCEEDED, TRANSPORT_ERROR_MARKER, locked)

LOCAL_TREE = ".exec/tree"
CMD_PREFIX = "cmd:"
# On the ssh host a job runs in a session of its own, and its session id is written here
# (in the job's tree): stopping the job ends its whole session. A process group is not
# enough -- gpu_entry's `timeout` moves itself and the task into a group of their own.
SESSION_FILE = ".fk_job.sid"
# Written in the session the job runs in: its id, then gpu_entry under the GPU lock
# ("$0" is the lock, "$@" the entry command).
_SESSION_SCRIPT = 'echo $$ > ' + SESSION_FILE + ' && exec flock "$0" "$@"'
STOP_GRACE_S = 5
LOCK_DIR_ENV = "FORGEKERNEL_LOCK_DIR"
TRANSPORT_ATTEMPTS = 3
TRANSPORT_RETRY_S = 10.0
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6")


def lock_dir() -> Path:
    """Where this machine's GPU locks live (one directory for every forgekernel run)."""
    env = os.environ.get(LOCK_DIR_ENV, "").strip()
    return Path(env) if env else Path.home() / ".cache" / "forgekernel" / "locks"


@contextmanager
def gpu_lock(gpu: str, log_file=None, poll_s: float = 30.0):
    """Hold this machine's lock on GPU ``gpu`` for the duration; say so while waiting."""
    d = lock_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"gpu{gpu}.lock", "w") as fh:
        waited = 0.0
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if log_file is not None and waited == 0.0:
                    log_file.write(f"[direct] GPU {gpu} is in use by another forgekernel run "
                                   f"-- waiting\n")
                    log_file.flush()
                time.sleep(poll_s)
                waited += poll_s
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def remote_tree(cfg: SubmitConfig) -> str:
    """``<workdir>/<run>/<op>``: one tree per op and run, so concurrent runs on one host
    never share a directory."""
    op_dir = Path(cfg.op_dir)
    if cfg.exec_workspace:
        return f"{cfg.exec_workspace.rstrip('/')}/jobs/{op_dir.name}"
    run = op_dir.parents[1].name if len(op_dir.parents) > 1 else "repo"
    return f"{cfg.exec_workdir.rstrip('/')}/{run}/{op_dir.name}"


def _payload(cfg: SubmitConfig, branch: str) -> bytes:
    # The tree is this run's own: trusted even when its files belong to another uid (a
    # container's root reading a checkout it did not create), which git would refuse.
    proc = subprocess.run(["git", "-c", f"safe.directory={cfg.op_dir}", "-C", cfg.op_dir,
                           "archive", "--format=tar", branch], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git archive {branch} failed in {cfg.op_dir}: "
                           f"{proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def _entry_argv(cfg: SubmitConfig, mode: str) -> list[str]:
    if mode.startswith(CMD_PREFIX):
        return ["bash", cfg.entry_script, "cmd", *shlex.split(mode[len(CMD_PREFIX):])]
    return ["bash", cfg.entry_script, mode]


def _job_env(cfg: SubmitConfig) -> dict:
    return {**cfg.env, "TASK_TIMEOUT": str(cfg.timeout), "CUDA_VISIBLE_DEVICES": cfg.exec_gpu}


class DirectRunner:
    """Runs one mode of ``cfg``'s tree; ``run_mode`` returns the job's status."""

    def __init__(self, cfg: SubmitConfig, run=subprocess.run, popen=subprocess.Popen,
                 sleep=time.sleep):
        self.cfg = cfg
        self._run = run
        self._popen = popen
        self._sleep = sleep

    @property
    def deadline_s(self) -> float:
        return float(self.cfg.timeout) + RUN_HEADROOM_S

    def run_mode(self, branch: str, mode: str, log_file) -> str:
        with locked(Path(self.cfg.op_dir)):
            payload = _payload(self.cfg, branch)
            if self.cfg.delivery == "local":
                return self._run_local(payload, branch, mode, log_file)
            return self._run_ssh(payload, branch, mode, log_file)

    # -- local ----------------------------------------------------------------------------

    def _run_local(self, payload: bytes, branch: str, mode: str, log_file) -> str:
        tree = Path(self.cfg.op_dir) / LOCAL_TREE
        shutil.rmtree(tree, ignore_errors=True)
        tree.mkdir(parents=True)
        _extract(payload, tree)
        env = {**os.environ, **_job_env(self.cfg)}
        log_file.write(f"[direct] local GPU {self.cfg.exec_gpu} branch={branch} mode={mode}\n")
        log_file.flush()
        with gpu_lock(self.cfg.exec_gpu, log_file):
            log_file.write(f"{BEGIN_MARKER}\n")
            log_file.flush()
            proc = self._popen(_entry_argv(self.cfg, mode), cwd=tree, env=env,
                               stdout=log_file, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                rc = proc.wait(timeout=self.deadline_s)
            except subprocess.TimeoutExpired:
                _kill_session(proc)
                log_file.write(f"[direct] gpu_entry {mode} exceeded {self.deadline_s:.0f}s "
                               f"-- killed\n")
                log_file.flush()
                return STATUS_FAILED
            except BaseException:   # stopped (SIGTERM -> SystemExit) or interrupted
                _kill_session(proc)
                raise
        if rc == 0:
            return STATUS_SUCCEEDED
        log_file.write(f"[direct] gpu_entry {mode} exited rc={rc}\n")
        log_file.flush()
        return STATUS_FAILED

    # -- ssh ------------------------------------------------------------------------------

    def _ssh(self, remote_cmd: str) -> list[str]:
        return ["ssh", *SSH_OPTS, self.cfg.exec_host, remote_cmd]

    def _extract_cmd(self) -> str:
        d = shlex.quote(remote_tree(self.cfg))
        return f"rm -rf {d} && mkdir -p {d} && tar xJ -C {d} --no-same-owner"

    def _exec_cmd(self, mode: str) -> str:
        env = _job_env(self.cfg)
        secrets = sorted(k for k in env if is_secret_env(k))
        parts = [f"cd {shlex.quote(remote_tree(self.cfg))}"]
        if self.cfg.exec_env_file:
            parts.append(f". {shlex.quote(self.cfg.exec_env_file)}")
            parts.append(f"cd {shlex.quote(remote_tree(self.cfg))}")
        parts += [f"export {k}={shlex.quote(str(v))}" for k, v in sorted(env.items())
                  if k not in secrets]
        # a secret is read off stdin, never spelled on a command line `ps` shows
        parts += [f"{{ IFS= read -r {k} || true; }} && export {k}" for k in secrets]
        lock = shlex.quote(f"{self.cfg.exec_workdir.rstrip('/')}/.gpu{self.cfg.exec_gpu}.lock")
        entry = " ".join(shlex.quote(a) for a in _entry_argv(self.cfg, mode))
        parts.append(f"echo {BEGIN_MARKER}")
        parts.append(f"setsid --wait sh -c {shlex.quote(_SESSION_SCRIPT)} {lock} {entry}")
        return " && ".join(parts)

    def _stop_cmd(self) -> str:
        f = shlex.quote(f"{remote_tree(self.cfg)}/{SESSION_FILE}")
        return (f"[ -f {f} ] || exit 0; s=$(cat {f}); pkill -TERM -s \"$s\"; "
                f"sleep {STOP_GRACE_S}; pkill -KILL -s \"$s\"; rm -f {f}; exit 0")

    def _stop_remote(self, log_file) -> None:
        """End the job's session on the host (the ssh client going away does not)."""
        try:
            self._run(self._ssh(self._stop_cmd()), capture_output=True, timeout=120)
            log_file.write(f"[direct] stopped the job's session on {self.cfg.exec_host}\n")
        except Exception as e:  # noqa: BLE001 -- best effort; the verdict is written either way
            log_file.write(f"[direct] could not stop the job on {self.cfg.exec_host}: {e!r}\n")
        log_file.flush()

    def _run_ssh(self, payload: bytes, branch: str, mode: str, log_file) -> str:
        packed = lzma.compress(payload, preset=6)
        env = _job_env(self.cfg)
        secrets = "".join(f"{env[k]}\n" for k in sorted(env) if is_secret_env(k)) or None
        failures: list[str] = []
        while True:
            sync = self._run(self._ssh(self._extract_cmd()), input=packed, capture_output=True)
            if sync.returncode != 0:
                err = sync.stderr.decode(errors="replace") if isinstance(sync.stderr, bytes) \
                    else str(sync.stderr or "")
                failures.append(f"sync to {self.cfg.exec_host} (rc={sync.returncode}): "
                                f"{err.strip()[-200:]}")
            else:
                log_file.write(f"[direct] ssh {self.cfg.exec_host} GPU {self.cfg.exec_gpu} "
                               f"branch={branch} mode={mode}\n")
                log_file.flush()
                start = log_file.tell() if hasattr(log_file, "tell") else None
                proc = self._popen(self._ssh(self._exec_cmd(mode)), stdout=log_file,
                                   stderr=subprocess.STDOUT, text=True,
                                   stdin=subprocess.PIPE if secrets else subprocess.DEVNULL,
                                   start_new_session=True)
                if secrets:
                    try:
                        proc.stdin.write(secrets)
                        proc.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                try:
                    rc = proc.wait(timeout=self.deadline_s)
                except subprocess.TimeoutExpired:
                    self._stop_remote(log_file)
                    _kill_group(proc)
                    log_file.write(f"[direct] gpu_entry {mode} exceeded {self.deadline_s:.0f}s "
                                   f"-- killed\n")
                    log_file.flush()
                    return STATUS_FAILED
                except BaseException:   # stopped (SIGTERM -> SystemExit) or interrupted
                    self._stop_remote(log_file)
                    _kill_group(proc)
                    raise
                if rc == 0:
                    return STATUS_SUCCEEDED
                if _began(log_file, start, BEGIN_MARKER):
                    log_file.write(f"[direct] gpu_entry {mode} exited rc={rc}\n")
                    log_file.flush()
                    return STATUS_FAILED
                failures.append(f"gpu_entry {mode} on {self.cfg.exec_host} (rc={rc}, never "
                                f"started)")
            if len(failures) >= TRANSPORT_ATTEMPTS:
                log_file.write(f"{TRANSPORT_ERROR_MARKER} {len(failures)} transport failure(s) "
                               f"on {self.cfg.exec_host}: {' | '.join(failures)}\n")
                log_file.flush()
                return STATUS_FAILED
            log_file.write(f"[direct] {failures[-1]} -- retrying in {TRANSPORT_RETRY_S:.0f}s\n")
            log_file.flush()
            self._sleep(TRANSPORT_RETRY_S)


def _extract(payload: bytes, dest: Path) -> None:
    """Extract a ``git archive`` tar into ``dest``, refusing members that leave it."""
    with tarfile.open(fileobj=io.BytesIO(payload)) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:   # a Python without extraction filters
            root = dest.resolve()
            for m in tf.getmembers():
                target = (dest / m.name).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"payload member {m.name!r} leaves the tree")
                if m.issym() or m.islnk():
                    link = (target.parent / m.linkname).resolve()
                    if link != root and root not in link.parents:
                        raise RuntimeError(f"payload link {m.name!r} leaves the tree")
            tf.extractall(dest)


def _began(log_file, start, marker: str) -> bool:
    """Did the remote command print the begin marker after ``start`` in ``log_file``?"""
    try:
        log_file.flush()
        with open(log_file.name, encoding="utf-8", errors="replace") as fh:
            if start:
                fh.seek(start)
            return any(line.strip() == marker for line in fh)
    except (OSError, AttributeError):
        return False


def _descendants(root: int) -> list[int]:
    """Every process below ``root`` right now (ps works the same on Linux and macOS)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    kids: dict[int, list[int]] = {}
    for line in out.splitlines():
        try:
            pid, ppid = (int(x) for x in line.split())
        except ValueError:
            continue
        kids.setdefault(ppid, []).append(pid)
    found, stack = [], [root]
    while stack:
        for child in kids.get(stack.pop(), []):
            if child not in found:
                found.append(child)
                stack.append(child)
    return found


def _kill_session(proc, grace_s: float = STOP_GRACE_S) -> None:
    """End ``proc`` and everything below it. Its process group is not enough: gpu_entry's
    `timeout` moves the task into a group of its own. The tree is read before the first
    signal, while every member still hangs below ``proc``."""
    tree = [proc.pid, *_descendants(proc.pid)]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in tree:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGTERM:
            time.sleep(min(grace_s, 1.0))   # the rest of the tree, a moment to go too
    _kill_group(proc)


def _kill_group(proc) -> None:
    try:
        os.killpg(proc.pid, 9)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001 -- best effort; the verdict is written either way
        pass
