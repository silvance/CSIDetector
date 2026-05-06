"""Alert-mode config: TOML loader + build_notifier factory matrix.

Verifies the config plumbing without ever talking to Telegram or
opening a network socket. A telegram config that's well-formed
produces a TelegramNotifier wrapped in a QueuingNotifier (queue is
on by default); a config with no [notifier] table produces a bare
NullNotifier; missing creds raise; bad TOML exits cleanly.
"""

from __future__ import annotations

import os
import textwrap
import tempfile
from pathlib import Path

import pytest

from csidetector.modes.alert.config import load_config, build_notifier
from csidetector.modes.alert.notifier import NullNotifier
from csidetector.modes.alert.queue import QueuingNotifier
from csidetector.modes.alert.telegram import TelegramNotifier


@pytest.fixture
def write_toml(tmp_path):
    """Helper: dump a TOML payload to a temp file and return its path."""
    def _write(payload: str) -> str:
        p = tmp_path / "alert.toml"
        p.write_text(textwrap.dedent(payload))
        return str(p)
    return _write


# --------------------------------------------------------------------------
# load_config — file handling.
# --------------------------------------------------------------------------

def test_load_config_none_returns_empty_dict():
    assert load_config(None) == {}


def test_load_config_missing_path_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        load_config(str(tmp_path / "does-not-exist.toml"))
    assert "config file not found" in str(excinfo.value)


def test_load_config_malformed_toml_exits_cleanly(write_toml):
    path = write_toml("this is = not valid [toml")
    with pytest.raises(SystemExit) as excinfo:
        load_config(path)
    assert "failed to parse" in str(excinfo.value)


def test_load_config_valid_telegram_payload(write_toml):
    path = write_toml("""
        [notifier]
        type = "telegram"
        bot_token = "abc123"
        chat_id = "42"

        [alert]
        cooldown_s = 30
        clear_on_exit = true
        location_label = "Office"

        [queue]
        enabled = false
    """)
    cfg = load_config(path)
    assert cfg["notifier"]["type"] == "telegram"
    assert cfg["notifier"]["bot_token"] == "abc123"
    assert cfg["alert"]["cooldown_s"] == 30
    assert cfg["queue"]["enabled"] is False


# --------------------------------------------------------------------------
# build_notifier — factory matrix.
# --------------------------------------------------------------------------

def test_build_notifier_no_config_returns_null():
    n = build_notifier({})
    assert isinstance(n, NullNotifier)


def test_build_notifier_unknown_type_returns_null():
    """A [notifier] table with an unsupported type degrades to NullNotifier
    rather than crashing — keeps detect mode usable while the user fixes
    the config."""
    n = build_notifier({"notifier": {"type": "smoke-signals"}})
    assert isinstance(n, NullNotifier)


def test_build_notifier_telegram_returns_queue_wrapped(tmp_path):
    """Default behavior: telegram config gets wrapped in QueuingNotifier
    so transient outages don't lose alerts."""
    cfg = {
        "notifier": {"type": "telegram", "bot_token": "t", "chat_id": "1"},
        "queue": {"path": str(tmp_path / "q.db")},
    }
    n = build_notifier(cfg)
    try:
        assert isinstance(n, QueuingNotifier)
    finally:
        n.close()


def test_build_notifier_queue_disabled_returns_bare_telegram():
    cfg = {
        "notifier": {"type": "telegram", "bot_token": "t", "chat_id": "1"},
        "queue": {"enabled": False},
    }
    n = build_notifier(cfg)
    try:
        assert isinstance(n, TelegramNotifier)
        assert not isinstance(n, QueuingNotifier)
    finally:
        n.close()


def test_build_notifier_missing_token_raises():
    with pytest.raises(ValueError, match="bot_token"):
        build_notifier({"notifier": {"type": "telegram",
                                     "bot_token": "", "chat_id": "1"}})


def test_build_notifier_missing_chat_id_raises():
    with pytest.raises(ValueError, match="chat_id"):
        build_notifier({"notifier": {"type": "telegram",
                                     "bot_token": "t", "chat_id": ""}})


def test_queue_path_expands_tilde_and_creates_parent_dir(tmp_path, monkeypatch):
    """`~/...` in the queue path is expanded, and the parent directory
    is created on first use (so a fresh deploy doesn't crash on
    'no such file or directory')."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = {
        "notifier": {"type": "telegram", "bot_token": "t", "chat_id": "1"},
        "queue": {"path": "~/some/nested/csidetector/q.db"},
    }
    n = build_notifier(cfg)
    try:
        expected = tmp_path / "some/nested/csidetector/q.db"
        assert expected.parent.exists()
        # The DB file itself is created lazily by sqlite3; opening the
        # connection in _Store.__init__ does that.
        assert expected.exists()
    finally:
        n.close()
