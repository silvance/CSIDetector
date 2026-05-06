"""Smoke tests for csidetector.core.collector parsers."""

from __future__ import annotations

import numpy as np

from csidetector.core import collector


def test_parse_line_returns_csi_sample(csi_data_line):
    line = csi_data_line(seq=42, n_sub=64)
    sample = collector.parse_line(line)
    assert sample is not None
    assert sample.seq == 42
    assert sample.mac == "ac:a7:04:2c:42:54"
    assert sample.csi.shape == (64,)
    # signed_byte=5 → real=imag=5 → magnitude = 5*sqrt(2) ≈ 7.07
    assert np.allclose(sample.amplitude, 5 * np.sqrt(2), atol=0.01)


def test_parse_line_rejects_garbage():
    assert collector.parse_line("") is None
    assert collector.parse_line("not a csi line\n") is None
    assert collector.parse_line("CSI_DATA,too,few,cols") is None


def test_parse_udp_packet_round_trip(udp_packet):
    pkt = udp_packet(rx_mac="11:22:33:44:55:01",
                     tx_mac="ac:a7:04:2c:42:54", seq=7)
    sample = collector.parse_udp_packet(pkt)
    assert sample is not None
    assert sample.seq == 7
    assert sample.mac == "ac:a7:04:2c:42:54"
    assert sample.rx_id == "11:22:33:44:55:01"
    assert sample.csi.size == 64


def test_parse_udp_packet_rejects_short_payload():
    assert collector.parse_udp_packet(b"") is None
    # Header truncated — should reject without raising.
    assert collector.parse_udp_packet(b"\x01" * 10) is None


def test_open_source_dispatch():
    # Just verify routing — don't actually open anything that needs hardware.
    src = collector.open_source("/tmp/nonexistent.log")
    assert src is not None  # iter_file is lazy; doesn't fail until first next()
