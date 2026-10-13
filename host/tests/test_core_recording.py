"""Recording files: what the alert monitor's Record button writes."""

from __future__ import annotations

import gzip

import pytest

from csidetector.core import collector, recording


def _write(tmp_path, lines, meta=None, marks_after=()):
    w = recording.RecordingWriter(tmp_path / "rec" / "x.csi.gz",
                                  meta or {"condition": "walk", "note": "döor open"})
    for i, line in enumerate(lines):
        w.write(line)
        if i in marks_after:
            w.mark()
    return w


def test_round_trip_keeps_only_csi_lines(tmp_path, csi_data_line):
    lines = [csi_data_line(seq=i) + "\r\n" for i in range(3)]
    lines.insert(1, "I (1234) wifi:connected with HomeWiFi, password hunter22\n")
    lines.append("CSI_DATA,broken\n")
    w = _write(tmp_path, lines, marks_after={2})
    summary = w.close("timer")
    assert summary["samples"] == 3 and summary["bad_lines"] == 1
    assert summary["other_lines_dropped"] == 1 and summary["marks"] == 1
    assert summary["tx_macs"] == {"ac:a7:04:2c:42:54": 3} and summary["stopped_by"] == "timer"

    raw = w.path.read_bytes()
    assert raw[:2] == b"\x1f\x8b"                          # gzip
    text = gzip.decompress(raw).decode("ascii")
    assert "hunter22" not in text and "HomeWiFi" not in text   # firmware logs left out
    rec = recording.read_recording(w.path)
    assert rec.meta == {"condition": "walk", "note": "döor open"}
    assert rec.marks == [{"n": 1, "t": rec.marks[0]["t"], "samples": 2, "label": ""}]
    assert rec.end == summary
    assert [s.seq for s in rec.samples()] == [0, 1, 2]
    # Replays wherever a capture file is accepted.
    assert sum(1 for _ in collector.open_source(str(w.path))) == 3


def test_a_recording_cut_off_mid_write_still_replays(tmp_path, csi_data_line):
    w = _write(tmp_path, [csi_data_line(seq=i) for i in range(400)])
    w._f.flush()                         # as the periodic flush does
    raw = w.path.read_bytes()
    w._f.close()                         # the app died: no end line, cut short
    w.path.write_bytes(raw[: len(raw) - 10])
    rec = recording.read_recording(w.path)
    assert rec.end is None and rec.meta["condition"] == "walk"
    assert 300 < sum(1 for _ in rec.samples()) <= 400


def test_silence_at_the_start_and_end_counts_as_a_gap(tmp_path, monkeypatch, csi_data_line):
    """A transmitter that starts late or dies early mustn't read as a
    continuous recording."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(recording.time, "monotonic", lambda: clock["t"])
    w = recording.RecordingWriter(tmp_path / "late.csi.gz", {})
    clock["t"] += 30.0                              # 30 s before the first sample
    for i in range(100):
        w.write(csi_data_line(seq=i))
        clock["t"] += 0.01
    clock["t"] += 60.0                              # then it went quiet
    s = w.close("timer")
    assert s["gaps_over_1s"] == 2 and s["max_gap_s"] == pytest.approx(60.0, abs=0.1)
    assert s["first_sample_s"] == pytest.approx(30.0) and s["duration_s"] == pytest.approx(91.0)
    empty = recording.RecordingWriter(tmp_path / "empty.csi.gz", {})
    clock["t"] += 5.0
    assert empty.close("timer")["gaps_over_1s"] == 1


def test_recordings_in_the_same_second_get_their_own_files(tmp_path, csi_data_line):
    when = 1_760_000_000.0
    a = recording.new_recording(tmp_path, "walk", {"n": 1}, when)
    a.write(csi_data_line())
    a.close("timer")
    b = recording.new_recording(tmp_path, "walk", {"n": 2}, when)
    b.abandon()                                     # must not delete the first
    c = recording.new_recording(tmp_path, "walk", {"n": 3}, when)
    c.close("timer")
    assert a.path.exists() and c.path.name.endswith("_walk-2.csi.gz")
    assert recording.read_recording(a.path).meta == {"n": 1}
    with pytest.raises(FileExistsError):
        recording.RecordingWriter(a.path, {})       # never over an existing file
    assert recording.read_recording(a.path).end["samples"] == 1


def test_abandon_removes_the_file(tmp_path):
    w = _write(tmp_path, [])
    w.abandon()
    assert not w.path.exists()


def test_not_a_recording(tmp_path, csi_data_line):
    plain = tmp_path / "capture.txt"
    plain.write_text(csi_data_line() + "\n")
    with pytest.raises(ValueError):
        recording.read_recording(plain)
    assert sum(1 for _ in collector.iter_file(str(plain))) == 1   # still a capture


def test_recording_name():
    when = 1_760_000_000.0
    name = recording.recording_name("walk", when)
    assert name.endswith("_walk.csi.gz") and name[:4].isdigit()
    assert recording.recording_name("", when).endswith("_recording.csi.gz")
    assert {k for k, _ in recording.CONDITIONS} == {"empty", "still", "walk", "edge", "other"}
