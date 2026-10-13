"""The settings block: the setup app's encoder against the firmware's own
C parser (compiled with the system compiler; skipped where there's none)."""

from __future__ import annotations

import shutil
import subprocess
import zlib

import pytest

from csidetector.provision import board_settings as bs
from csidetector.provision import firmware as fw

COMPONENT = fw.repo_firmware_root() / "components" / "csi_settings"

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include "csi_settings.h"

int main(int argc, char **argv) {
    (void)argc;
    static uint8_t buf[8192];
    FILE *f = fopen(argv[1], "rb");
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    csi_settings_t s = {.role = "rx", .label = "default", .channel = 11, .rate_hz = 100,
                        .host_ip = "10.42.0.1", .host_port = 5566};
    int rc = csi_settings_parse(buf, n, &s);
    printf("rc=%d\nfrom_setup=%d\nrole=%s\nlabel=%s\nchannel=%d\nrate_hz=%d\n"
           "tx_macs=%s\nssid=%s\npassword=%s\nhost_ip=%s\nhost_port=%d\n",
           rc, s.from_setup, s.role, s.label, s.channel, s.rate_hz, s.tx_macs,
           s.ssid, s.password, s.host_ip, s.host_port);
    return 0;
}
"""


@pytest.fixture(scope="module")
def parser(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("csi_settings")
    (d / "harness.c").write_text(_HARNESS)
    exe = d / "harness"
    subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(COMPONENT / "include"),
                    str(d / "harness.c"), str(COMPONENT / "csi_settings_parse.c"), "-o", str(exe)],
                   check=True)

    def run(blob: bytes) -> dict:
        path = d / "blob.bin"
        path.write_bytes(blob)
        out = subprocess.run([str(exe), str(path)], capture_output=True, check=True).stdout
        return dict(line.split("=", 1) for line in out.decode("utf-8").splitlines())
    return run


def test_the_firmware_reads_what_the_setup_app_writes(parser):
    values = {"role": "rx", "label": "Küche — south wall", "channel": 6,
              "tx_macs": "aa:bb:cc:dd:ee:01, aa:bb:cc:dd:ee:02",
              "ssid": "Lab hotspot=2", "password": "p@ss=word with spaces",
              "host_ip": "192.168.137.1", "host_port": 5570}
    blob = bs.encode(values)
    assert len(blob) == bs.SIZE and blob.endswith(b"\xff")
    got = parser(blob)
    assert got["rc"] == "0" and got["from_setup"] == "1"
    for key, value in values.items():
        assert got[key] == str(value), key
    assert got["rate_hz"] == "100"                   # not in the block: default kept
    assert bs.decode(blob) == {k: str(v) for k, v in values.items()}


def test_erased_or_damaged_blocks_keep_the_defaults(parser):
    erased = parser(b"\xff" * bs.SIZE)
    assert erased["rc"] == "-1" and erased["from_setup"] == "0" and erased["label"] == "default"
    good = bytearray(bs.encode({"channel": 3, "label": "x"}))
    damaged = bytearray(good)
    damaged[bs.HEADER.size] ^= 0x01
    assert parser(bytes(damaged))["rc"] == "-4"
    newer = bytearray(good)
    newer[4] = 2
    assert parser(bytes(newer))["rc"] == "-2"


def test_a_bad_value_rejects_the_whole_block(parser):
    """All or nothing: a half-applied block could leave e.g. the right
    SSID with the default channel."""
    payload = b"label=new\nchannel=14\n"
    blob = bs.HEADER.pack(bs.MAGIC, 1, 0, len(payload), zlib.crc32(payload)) + payload
    got = parser(blob)
    assert got["rc"] == "-5" and got["label"] == "default" and got["channel"] == "11"
    payload = b"ssid=" + b"x" * 33 + b"\n"
    blob = bs.HEADER.pack(bs.MAGIC, 1, 0, len(payload), zlib.crc32(payload)) + payload
    assert parser(blob)["rc"] == "-5"


def test_keys_from_a_newer_setup_app_are_ignored(parser):
    blob = bs.encode({"channel": 1, "future_option": "on"})
    got = parser(blob)
    assert got["rc"] == "0" and got["channel"] == "1"


def test_encoder_rejects_what_the_board_cannot_hold():
    with pytest.raises(ValueError):
        bs.encode({"ssid": "x" * 33})
    with pytest.raises(ValueError):
        bs.encode({"label": "line\nbreak"})
    with pytest.raises(ValueError):
        bs.encode({"password": "x" * 65})
    assert len(bs.fit_label("é" * 40).encode()) <= bs.LABEL_MAX_BYTES
    assert bs.fit_label("  Kitchen  ") == "Kitchen"
    with pytest.raises(ValueError):
        bs.decode(b"\xff" * 16)
