"""The headless `alert detect` loop (what the Pi appliance's service runs)."""

from __future__ import annotations

import json
import math
import random
import types

from csidetector.modes.alert import detect


def _line(seq: int, sigma: float, rng: random.Random) -> str:
    iq = []
    for k in range(64):
        amp = 0 if (k < 6 or k > 58 or k == 32) else \
            max(1, min(120, round(20 + 6 * math.sin(k / 5) + rng.gauss(0, sigma))))
        iq += [0, amp]
    fields = ["CSI_DATA", str(seq), "aa:bb:cc:dd:ee:01", "-52", "11", "1", "0", "0", "0", "0",
              "0", "0", "0", "0", "-92", "0", "11", "0", str(seq * 10000), "0", "0", "0",
              str(len(iq)), "0", '"' + json.dumps(iq).replace(" ", "") + '"']
    return ",".join(fields)


def test_says_it_is_watching_and_alerts_on_the_first_motion_after_boot(
        tmp_path, monkeypatch, capsys):
    """The log isn't silent until something moves, and the first motion
    alerts even when the monotonic clock is near zero (just booted)."""
    rng = random.Random(0)
    capture = tmp_path / "capture.log"
    capture.write_text("\n".join([_line(i, 0.6, rng) for i in range(200)]
                                 + [_line(200 + i, 4.0, rng) for i in range(200)]) + "\n")
    monkeypatch.setattr(detect, "time", types.SimpleNamespace(monotonic=lambda: 5.0))  # just booted
    sent = []

    class Notifier:
        def send(self, event):
            sent.append(event)

        def close(self):
            pass

    assert detect.run_detect(str(capture), 0.65, settle=0.0, notifier=Notifier(),
                             location_label="Office") == 0
    out = capsys.readouterr().out.splitlines()
    assert "watching" in out[0] and str(capture) in out[0]
    assert any(" MOTION " in line for line in out)
    assert len(sent) == 1 and "Office" in sent[0].message
