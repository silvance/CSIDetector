"""Parse CSI lines emitted by the receiver firmware.

Line format (matches espressif/esp-csi conventions so their existing
tools/csi_data_read_parse.py parser also works against our captures):

    type,seq,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,
    aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,
    secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,
    "[Im0,Re0,Im1,Re1,...]"

The radio zeroes guard / DC subcarriers, and the exact set depends on
bandwidth and the lltf/htltf config; rather than hard-coding indices we
discover the active subcarriers at runtime via `nonzero_subcarriers`.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import sys
import time
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclasses.dataclass
class CSISample:
    seq: int
    ts_us: int
    mac: str
    rssi: int
    noise: int
    channel: int
    bandwidth: int
    sig_mode: int
    mcs: int
    ant: int
    csi: np.ndarray  # complex64, one entry per subcarrier

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.csi)


def parse_line(line: str) -> Optional[CSISample]:
    line = line.strip()
    if not line.startswith("CSI_DATA,"):
        return None
    # The data column is a quoted JSON array; csv handles the quoting.
    try:
        row = next(csv.reader(io.StringIO(line)))
    except (csv.Error, StopIteration):
        return None
    if len(row) != 25 or row[0] != "CSI_DATA":
        return None
    try:
        seq = int(row[1])
        mac = row[2]
        rssi = int(row[3])
        sig_mode = int(row[5])
        mcs = int(row[6])
        bandwidth = int(row[7])
        noise = int(row[14])
        channel = int(row[16])
        ts_us = int(row[18])
        ant = int(row[19])
        length = int(row[22])
        data = json.loads(row[24])
    except (ValueError, json.JSONDecodeError):
        return None
    if len(data) != length or len(data) % 2 != 0:
        return None
    iq = np.asarray(data, dtype=np.int8).astype(np.float32)
    # Layout per esp_wifi_types.h: (Im, Re) pairs.
    imag = iq[0::2]
    real = iq[1::2]
    csi = (real + 1j * imag).astype(np.complex64)
    return CSISample(
        seq=seq, ts_us=ts_us, mac=mac, rssi=rssi, noise=noise,
        channel=channel, bandwidth=bandwidth, sig_mode=sig_mode,
        mcs=mcs, ant=ant, csi=csi,
    )


def nonzero_subcarriers(samples: Iterable[CSISample], probe: int = 64) -> np.ndarray:
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
    if source == "-":
        return iter_stdin()
    if source.startswith("/dev/") or source.lower().startswith("com"):
        return iter_serial(source)
    return iter_file(source)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
