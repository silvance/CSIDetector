"""Parse CSI lines emitted by the receiver firmware.

Line format (single space separated):

    CSI <seq> <ts_us> <rssi> <noise> <ch> <bw> <mcs> <len> <base64(buf)>

`buf` is a sequence of int8 IQ pairs (imag, real, imag, real, ...). Some
subcarriers are reserved as guard / DC and are zeroed by the radio; they are
filtered out by `nonzero_subcarriers` before the detector consumes them.
"""

from __future__ import annotations

import base64
import dataclasses
import sys
import time
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclasses.dataclass
class CSISample:
    seq: int
    ts_us: int
    rssi: int
    noise: int
    channel: int
    bandwidth: int
    mcs: int
    csi: np.ndarray  # complex64, one entry per subcarrier

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.csi)


def parse_line(line: str) -> Optional[CSISample]:
    line = line.strip()
    if not line.startswith("CSI "):
        return None
    parts = line.split(" ")
    if len(parts) != 10:
        return None
    try:
        _, seq, ts, rssi, noise, ch, bw, mcs, length, b64 = parts
        raw = base64.b64decode(b64)
    except (ValueError, base64.binascii.Error):
        return None
    if len(raw) != int(length) or len(raw) % 2 != 0:
        return None
    iq = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    # Buffer layout is (imag, real) per subcarrier — see esp_wifi_types.h.
    imag = iq[0::2]
    real = iq[1::2]
    csi = (real + 1j * imag).astype(np.complex64)
    return CSISample(
        seq=int(seq),
        ts_us=int(ts),
        rssi=int(rssi),
        noise=int(noise),
        channel=int(ch),
        bandwidth=int(bw),
        mcs=int(mcs),
        csi=csi,
    )


def nonzero_subcarriers(samples: Iterable[CSISample], probe: int = 64) -> np.ndarray:
    """Return indices of subcarriers that carry signal in at least one of the
    first `probe` samples. The radio zeroes guard/DC bins, and the exact set
    depends on bandwidth and the lltf/htltf config; rather than hard-coding
    indices we discover them at startup."""
    seen = []
    for i, s in enumerate(samples):
        seen.append(s.amplitude > 0)
        if i + 1 >= probe:
            break
    if not seen:
        return np.array([], dtype=int)
    mask = np.any(np.stack(seen), axis=0)
    return np.flatnonzero(mask)


def iter_serial(port: str, baud: int = 921600) -> Iterator[CSISample]:
    import serial  # imported lazily so unit tests don't need pyserial

    with serial.Serial(port, baud, timeout=1) as ser:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            try:
                line = raw.decode("ascii", errors="replace")
            except UnicodeDecodeError:
                continue
            sample = parse_line(line)
            if sample is not None:
                yield sample


def iter_file(path: str) -> Iterator[CSISample]:
    with open(path, "r") as f:
        for line in f:
            sample = parse_line(line)
            if sample is not None:
                yield sample


def iter_stdin() -> Iterator[CSISample]:
    for line in sys.stdin:
        sample = parse_line(line)
        if sample is not None:
            yield sample


def open_source(source: str) -> Iterator[CSISample]:
    """Dispatch on a source spec: '-' for stdin, '/dev/ttyUSB0' for serial,
    or a path to a previously captured log file."""
    if source == "-":
        return iter_stdin()
    if source.startswith("/dev/") or source.lower().startswith("com"):
        return iter_serial(source)
    return iter_file(source)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
