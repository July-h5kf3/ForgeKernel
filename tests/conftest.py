"""Shared test fixtures.

Truncation free-retry (loop/driver.TRUNCATION_*) is disabled by default in
tests: the fixture tests script exact agent-reply sequences against
FakeBackend, and many deliberately include short, traceless turns (a stalling
teacher, a yielded write-up) whose PRODUCTION treatment is now "retry after a
60-600s backoff without charging the round". Under test that treatment would
consume the next scripted reply and sleep for real minutes. Setting the backoff
tuple empty turns the retry loop off entirely; tests that exercise the retry
behavior itself opt back in with an explicit zero-sleep tuple.
"""
import pytest

import forgekernel.loop.driver as _driver
from forgekernel import workspace as _workspace
from forgekernel import localenv as _localenv


@pytest.fixture(autouse=True)
def _no_truncation_retries(monkeypatch):
    monkeypatch.setattr(_driver, "TRUNCATION_BACKOFF_S", ())


@pytest.fixture(autouse=True)
def _no_user_ssh_workspace(monkeypatch, tmp_path):
    """A developer's local.env must never send a unit test to a real server."""
    monkeypatch.setattr(_workspace, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(_localenv, "repo_root", lambda: tmp_path)
    for key in (*_workspace.KEYS, _workspace.CONTEXT_ENV):
        monkeypatch.delenv(key, raising=False)
