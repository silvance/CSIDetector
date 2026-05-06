"""Shared pytest fixtures.

Synthesizes CSI samples without needing real hardware. Each fixture is
a generator-style helper; tests can construct as much or as little as
they need.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


# Make the host/ directory importable as a package root, so tests run
# regardless of where pytest is invoked from.
_HOST = Path(__file__).resolve().parent.parent
if str(_HOST) not in sys.path:
    sys.path.insert(0, str(_HOST))


@pytest.fixture
def synthetic_amplitudes():
    """Factory: build (n_samples × n_subcarriers) amplitude arrays.

    ``noise=0.01`` gives a quiet still-room signal; bump it to simulate
    motion. Adds a small per-row bias so subcarriers are clearly nonzero
    (the parser's nonzero-subcarriers logic relies on that).
    """
    def _make(n: int = 200, n_sub: int = 64, noise: float = 0.01,
              seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        bias = 1.0 + rng.uniform(0.1, 0.5, size=n_sub)
        return bias[None, :] + rng.normal(0, noise, size=(n, n_sub))
    return _make


@pytest.fixture
def csi_data_line():
    """Factory: build a single ESP-CSI text-format line for parser tests."""
    def _make(seq: int = 0, mac: str = "ac:a7:04:2c:42:54",
              n_sub: int = 64, signed_byte: int = 5) -> str:
        # 25 columns matching parse_line; the data column is a JSON list.
        iq = json.dumps([signed_byte] * (n_sub * 2))   # Im,Re,Im,Re,...
        cols = [
            "CSI_DATA", str(seq), mac, "-50", "11",
            "1",       # sig_mode
            "0",       # mcs
            "0",       # bandwidth
            "1", "0", "1", "1", "1", "1",
            "-95",     # noise_floor
            "1",
            "11",      # channel
            "0",
            "100000",  # local_timestamp
            "0",       # ant
            "128",     # sig_len
            "0",
            str(n_sub * 2),  # length
            "0",
            iq,        # quoted JSON in CSV
        ]
        # csv.writer-style quoting on the last field.
        cols[-1] = '"' + cols[-1].replace('"', '""') + '"'
        return ",".join(cols)
    return _make


@pytest.fixture
def udp_packet():
    """Factory: build a binary UDP packet matching the firmware wire format."""
    def _make(rx_mac: str = "11:22:33:44:55:01",
              tx_mac: str = "ac:a7:04:2c:42:54",
              seq: int = 1,
              n_sub: int = 64,
              signed_byte: int = 5) -> bytes:
        from csidetector.core.collector import _UDP_HEADER
        rx_bytes = bytes.fromhex(rx_mac.replace(":", ""))
        tx_bytes = bytes.fromhex(tx_mac.replace(":", ""))
        length = n_sub * 2
        header = _UDP_HEADER.pack(
            1,        # version
            0,        # reserved
            rx_bytes,
            tx_bytes,
            seq,
            100000,   # ts_us
            -50,      # rssi
            -95,      # noise
            11,       # channel
            1,        # sig_mode
            0,        # mcs
            0,        # bandwidth
            length,
        )
        payload = bytes([signed_byte & 0xff] * length)
        return header + payload
    return _make
