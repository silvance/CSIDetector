"""Recordings of the raw CSI stream, for working on detection offline.

The alert monitor's Record button writes one file per recording: the
receiver's CSI_DATA lines exactly as it printed them, gzip-compressed,
with comment lines around them:

    # csidetector-recording/1
    # meta {"condition": "walk", "note": "...", "receiver_mac": ...}
    CSI_DATA,...                                    the stream, unchanged
    # mark {"n": 1, "t": 12.3, "samples": 1180}     the Mark button
    # end {"duration_s": 120.0, "samples": 11904, "stopped_by": "timer", ...}

parse_line skips the comments, so a recording replays anywhere a capture
file is accepted (``alert calibrate``, ``alert detect``, the heatmap),
and read_recording() gives back the metadata, marks and summary.

Only CSI_DATA lines are kept, as ``capture`` does. The firmware's other
log lines aren't needed for analysis, and leaving them out means nothing
else the board prints ends up in a file that gets sent around.
"""

from __future__ import annotations

import collections
import dataclasses
import gzip
import json
import re
import time
from pathlib import Path
from typing import Iterator, Optional

from csidetector.core.collector import CSISample, iter_file, open_text, parse_line

FORMAT = "csidetector-recording/1"
SUFFIX = ".csi.gz"

# (key, what the person recording picks). The key goes in the file name
# and the metadata.
CONDITIONS = [
    ("empty", "Empty room (nobody inside)"),
    ("still", "Someone sitting still"),
    ("walk", "Someone walking through"),
    ("edge", "Someone walking along the walls"),
    ("other", "Other (describe it in the note)"),
]

# A pause in the stream longer than this is counted in the summary.
GAP_S = 1.0
# How often buffered data is pushed to disk, so a crash or a killed
# process loses little.
_FLUSH_S = 2.0


def recording_name(condition: str, when: Optional[float] = None, n: int = 1) -> str:
    """``2026-10-13_213102_walk.csi.gz``: sorts by time, says what it is.
    ``n`` > 1 adds ``-2``, ``-3``… for recordings started in the same second."""
    slug = re.sub(r"[^a-z0-9]+", "-", condition.lower()).strip("-") or "recording"
    return (time.strftime("%Y-%m-%d_%H%M%S", time.localtime(when)) + f"_{slug}"
            + (f"-{n}" if n > 1 else "") + SUFFIX)


def new_recording(folder: Path, condition: str, meta: dict,
                  when: Optional[float] = None) -> "RecordingWriter":
    """A writer for a new file in ``folder``, never over an existing one."""
    for n in range(1, 1000):
        try:
            return RecordingWriter(Path(folder) / recording_name(condition, when, n), meta)
        except FileExistsError:
            continue
    raise FileExistsError(f"no free recording name in {folder}")


class RecordingWriter:
    """Writes one recording. Not thread-safe: one thread writes and marks.
    Refuses to replace an existing file (FileExistsError)."""

    def __init__(self, path: Path, meta: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = gzip.open(self.path, "xt", encoding="ascii", errors="replace",
                            newline="\n")
        self._t0 = time.monotonic()
        self._last_flush = self._t0
        self._first_sample: Optional[float] = None
        self._last_sample: Optional[float] = None
        self.samples = 0
        self.bad_lines = 0        # CSI_DATA lines that didn't parse (kept anyway)
        self.other_lines = 0      # firmware log lines (not kept)
        self.marks = 0
        self.gaps = 0
        self.max_gap_s = 0.0
        self.tx_macs: collections.Counter = collections.Counter()
        self.channels: collections.Counter = collections.Counter()
        self._closed = False
        try:
            self._f.write(f"# {FORMAT}\n")
            self._comment("meta", meta)
        except BaseException:
            self.abandon()            # don't leave a header-only file behind
            raise

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def write(self, line: str) -> Optional[CSISample]:
        """Keep one line from the board; returns its sample if it parsed."""
        line = line.strip()
        if not line.startswith("CSI_DATA,"):
            if line:
                self.other_lines += 1
            return None
        self._f.write(line + "\n")
        sample = parse_line(line)
        now = time.monotonic()
        if sample is None:
            self.bad_lines += 1
        else:
            self.samples += 1
            self.tx_macs[sample.mac] += 1
            self.channels[sample.channel] += 1
            if self._last_sample is None:
                self._first_sample = now
                self._gap(now - self._t0)         # silence before the first one
            else:
                self._gap(now - self._last_sample)
            self._last_sample = now
        if now - self._last_flush >= _FLUSH_S:
            self._f.flush()
            self._last_flush = now
        return sample

    def mark(self, label: str = "") -> None:
        self.marks += 1
        self._comment("mark", {"n": self.marks, "t": round(self.elapsed, 2),
                               "samples": self.samples, "label": label})

    def close(self, stopped_by: str) -> dict:
        """Write the summary and close; returns the summary. Silence at
        the start or the end (transmitter late, or gone) counts as a gap,
        so a half-empty recording doesn't read as continuous."""
        now = time.monotonic()
        duration = now - self._t0
        if self._last_sample is not None:
            self._gap(now - self._last_sample)    # silence after the last one
        else:
            self._gap(duration)
        rel = lambda t: round(t - self._t0, 2) if t is not None else None  # noqa: E731
        summary = {
            "duration_s": round(duration, 1),
            "samples": self.samples,
            "rate_hz": round(self.samples / duration, 1) if duration > 0 else 0.0,
            "first_sample_s": rel(self._first_sample),
            "last_sample_s": rel(self._last_sample),
            "bad_lines": self.bad_lines,
            "other_lines_dropped": self.other_lines,
            "marks": self.marks,
            "gaps_over_1s": self.gaps,
            "max_gap_s": round(self.max_gap_s, 2),
            "tx_macs": dict(self.tx_macs.most_common()),
            "channels": {str(k): v for k, v in self.channels.most_common()},
            "stopped_by": stopped_by,
        }
        try:
            self._comment("end", summary)
        finally:
            self._close_file()
        return summary

    def abandon(self) -> None:
        """Close and delete (nothing worth keeping was recorded)."""
        try:
            self._close_file()
        finally:
            self.path.unlink(missing_ok=True)

    def _close_file(self) -> None:
        if not self._closed:
            self._closed = True
            self._f.close()

    def _gap(self, seconds: float) -> None:
        self.max_gap_s = max(self.max_gap_s, seconds)
        if seconds > GAP_S:
            self.gaps += 1

    def _comment(self, kind: str, data: dict) -> None:
        # ensure_ascii (the default) keeps a note in any language on one
        # ASCII line.
        self._f.write(f"# {kind} {json.dumps(data, sort_keys=True)}\n")


@dataclasses.dataclass
class Recording:
    path: Path
    meta: dict
    marks: list
    end: Optional[dict]          # None: cut off before it finished

    def samples(self) -> Iterator[CSISample]:
        return iter_file(str(self.path))


def read_recording(path) -> Recording:
    """The metadata, marks and summary of a recording (samples via
    ``.samples()``). Raises ValueError if it isn't one."""
    meta: Optional[dict] = None
    marks: list = []
    end: Optional[dict] = None
    with open_text(str(path)) as f:
        try:
            first = f.readline().strip()
            if first != f"# {FORMAT}":
                raise ValueError(f"{path} isn't a {FORMAT} file")
            for line in f:
                if not line.startswith("# "):
                    continue
                kind, _, data = line[2:].partition(" ")
                try:
                    value = json.loads(data)
                except ValueError:
                    continue
                if kind == "meta":
                    meta = value
                elif kind == "mark":
                    marks.append(value)
                elif kind == "end":
                    end = value
        except EOFError:
            pass                  # cut off: what was written before is kept
    if meta is None:
        raise ValueError(f"{path} has no metadata line")
    return Recording(Path(path), meta, marks, end)
