"""Telegram notifier — HTTP response handling.

Mocks ``urllib.request.urlopen`` so we exercise the error-mapping logic
without ever talking to api.telegram.org. The key invariants:

  - 4xx classified as PERMANENT (queue marks the row dead, no retry)
    EXCEPT 408 / 429 which are transient (rate limiting / timeout)
  - 5xx + URLError classified as TRANSIENT (queue retries with backoff)
  - 2xx with non-JSON body classified as PERMANENT (captive portal /
    transparent proxy intercepting the request)
  - 2xx with ``ok: false`` classified as PERMANENT (Telegram-side rejection)
"""

from __future__ import annotations

import io
import json
from unittest import mock

import pytest
import urllib.error

from csidetector.modes.alert.notifier import Event, PermanentNotifierError
from csidetector.modes.alert.telegram import TelegramNotifier


def _make_event() -> Event:
    return Event.now("MOTION", "test")


def _http_error(code: int, reason: str = "test"):
    return urllib.error.HTTPError(
        url="https://api.telegram.org/botX/sendMessage",
        code=code, msg=reason, hdrs=None, fp=None,
    )


def _fake_response(body: bytes):
    """Stand-in for urlopen's context-manager return value."""
    resp = mock.MagicMock()
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda self, *a: False
    resp.read.return_value = body
    return resp


@pytest.fixture
def notifier():
    return TelegramNotifier(bot_token="t", chat_id="42")


# --------------------------------------------------------------------------
# Permanent failures — caller should NOT retry.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_permanent_4xx_codes_raise_permanent(notifier, code):
    with mock.patch("urllib.request.urlopen", side_effect=_http_error(code)):
        with pytest.raises(PermanentNotifierError):
            notifier.send(_make_event())


def test_2xx_non_json_body_is_permanent(notifier):
    """A captive portal or transparent proxy returns HTML on a 2xx —
    without bounding this, the queue retries forever."""
    html = b"<html><head><title>Captive Portal</title></head><body>...</body></html>"
    with mock.patch("urllib.request.urlopen", return_value=_fake_response(html)):
        with pytest.raises(PermanentNotifierError, match="captive portal"):
            notifier.send(_make_event())


def test_2xx_ok_false_is_permanent(notifier):
    body = json.dumps({"ok": False, "description": "chat not found"}).encode()
    with mock.patch("urllib.request.urlopen", return_value=_fake_response(body)):
        with pytest.raises(PermanentNotifierError, match="chat not found"):
            notifier.send(_make_event())


# --------------------------------------------------------------------------
# Transient failures — caller (queue) SHOULD retry with backoff.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code", [408, 429])
def test_transient_4xx_codes_reraise_for_retry(notifier, code):
    """408 (Request Timeout) and 429 (Too Many Requests) are transient
    despite being in the 4xx range. Re-raising the HTTPError (instead
    of wrapping as Permanent) lets the queue's exponential backoff
    deliver after the rate-limit window passes."""
    err = _http_error(code, reason="rate limited")
    with mock.patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(urllib.error.HTTPError):
            notifier.send(_make_event())
        # Critical: must NOT be a PermanentNotifierError, otherwise
        # the queue marks the row dead and stops trying.


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_5xx_codes_reraise_for_retry(notifier, code):
    with mock.patch("urllib.request.urlopen", side_effect=_http_error(code)):
        with pytest.raises(urllib.error.HTTPError):
            notifier.send(_make_event())


def test_url_error_reraises_for_retry(notifier):
    """Network-layer error (DNS failure, connection refused). Caller
    retries."""
    with mock.patch("urllib.request.urlopen",
                    side_effect=urllib.error.URLError("nodename nor servname")):
        with pytest.raises(urllib.error.URLError):
            notifier.send(_make_event())


# --------------------------------------------------------------------------
# Success path.
# --------------------------------------------------------------------------

def test_2xx_ok_true_returns_silently(notifier):
    body = json.dumps({"ok": True, "result": {"message_id": 1}}).encode()
    with mock.patch("urllib.request.urlopen", return_value=_fake_response(body)):
        notifier.send(_make_event())   # no exception → success
