"""A stand-in for firmware/prebuilt and for esptool, for the tests and the
GUI driver.

``make_prebuilt(dir)`` writes a manifest and small dummy images for every
supported chip (the setup app checks their sha256s). ``install_esptool(dir)``
writes a fake ``esptool`` package; with ``dir`` first on PYTHONPATH,
``python -m esptool`` runs it. It reads the simulated bench (boards.json,
control.json) and, for write-flash, records the settings block it was
given in bench/firmware/<mac>.json (the password redacted).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from csidetector.provision import firmware as fw

_BOOTLOADER_AT = {"esp32": 0x1000, "esp32c5": 0x2000}


def make_prebuilt(root: Path, targets=fw.TARGETS) -> Path:
    images = {}
    for target in targets:
        for project in fw.PROJECTS.values():
            name = f"{project}-{target}"
            (root / name).mkdir(parents=True, exist_ok=True)
            files = []
            for offset, fname in ((_BOOTLOADER_AT.get(target, 0), "bootloader.bin"),
                                  (0x8000, "partition-table.bin"),
                                  (0x10000, f"{project}.bin")):
                data = f"{name}/{fname}".encode()
                (root / name / fname).write_bytes(data)
                files.append({"offset": hex(offset), "path": f"{name}/{fname}",
                              "sha256": hashlib.sha256(data).hexdigest()})
            images[name] = {"project": project, "target": target, "chip": target,
                            "flash_mode": "dio", "flash_freq": "80m", "flash_size": "2MB",
                            "files": files, "settings": {"offset": "0xf000", "size": 4096},
                            "nvs": {"offset": "0x9000", "size": 0x5000}}
    (root / "manifest.json").write_text(json.dumps(
        {"format": fw.MANIFEST_FORMAT, "idf_version": "v5.5.1", "source_sha256": "test",
         "commit": "", "images": images}, indent=1))
    return root


_FAKE_ESPTOOL = r'''
import json, os, struct, subprocess, sys, time, zlib
BENCH = {bench!r}

def load(name, default):
    try:
        with open(os.path.join(BENCH, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default

CTL = load("control.json", {{}})
DELAY = float(CTL.get("line_delay", 0.01))

def out(line):
    print(line, flush=True)
    time.sleep(DELAY)

PREFIXES = ("C61", "S3", "S2", "C2", "C3", "C5", "C6", "H2", "P4")

def chip_target(chip):
    chip = chip.upper()
    for p in PREFIXES:
        if chip.startswith("ESP32-" + p):
            return "esp32" + p.lower()
    return "esp32" if chip.startswith("ESP32") else ""

def target_name(target):
    return "ESP32" + ("-" + target[5:].upper() if len(target) > 5 else "")

def hard_reset(port):
    """The chip resets: a native-USB board leaves the bus for a moment."""
    out("Hard resetting via RTS pin...")
    os.makedirs(os.path.join(BENCH, "reset"), exist_ok=True)
    with open(os.path.join(BENCH, "reset", port.replace("/", "_")), "w") as f:
        f.write(str(time.time()))

def opt(flag):
    return args[args.index(flag) + 1] if flag in args else ""

def settings_block(path):
    with open(path, "rb") as f:
        blob = f.read()
    magic, version, _, length, crc = struct.unpack_from("<4sBBHI", blob)
    payload = blob[12:12 + length]
    assert magic == b"CSIS" and zlib.crc32(payload) == crc, "bad settings block"
    values = dict(line.split("=", 1) for line in payload.decode().split("\n") if line)
    if values.get("password"):
        values["password"] = "<redacted>"
    return values

args = sys.argv[1:]
port = opt("--port")
cmd = next((a for a in args if a in ("read-mac", "write-flash")), "")
if not cmd:
    out("esptool: error: unknown command in %r" % (args,))
    sys.exit(2)
board = load("boards.json", {{}}).get(port)
behavior = board["behavior"] if board else "absent"
out("esptool v5.1.0")
out("Serial port " + port + ":")
if behavior == "busy":
    out("A fatal error occurred: Could not open %s, the port is busy or doesn't exist."
        % port)
    out("([Errno 16] could not open port %s: [Errno 16] Device or resource busy: '%s')"
        % (port, port))
    sys.exit(2)
out("Connecting....")
want = opt("--chip")
if behavior in ("absent", "no_connect"):
    out("A fatal error occurred: Failed to connect to %s: No serial data received."
        % (target_name(want) if want else "Espressif device"))
    sys.exit(2)
chip = board["chip"]
if want and chip_target(chip) != want:
    out("A fatal error occurred: This chip is %s, not %s. Wrong chip argument?"
        % (target_name(chip_target(chip)), target_name(want)))
    sys.exit(2)
out("Connected to %s on %s:" % (target_name(chip_target(chip)), port))
out("Chip type:          %s (revision v0.2)" % chip)
out("Features:           Wi-Fi, BT 5 (LE)")
out("MAC:                " + board["mac"])
if cmd == "read-mac":
    out("Stub flasher running.")
    hard_reset(port)
    sys.exit(0)

i = args.index("write-flash") + 1
pairs = []
while i < len(args):
    if args[i].startswith("--"):
        i += 2
        continue
    pairs.append((int(args[i], 0), args[i + 1]))
    i += 2
out("Stub flasher running.")
out("Changing baud rate to " + opt("--baud"))
if behavior == "hang":
    # A process esptool started, still running (python, so it exists on
    # Windows too): Cancel must take it down as well.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(os.path.join(BENCH, "children.txt"), "a") as f:
        f.write("%d\n" % child.pid)
    out("Writing at 0x00010000 [=>            ]  10.0% 81920/812345 bytes...")
    child.wait()
    sys.exit(1)
for offset, path in pairs:
    if not os.path.exists(path):
        out("A fatal error occurred: [Errno 2] No such file or directory: '%s'" % path)
        sys.exit(2)
    for pct in range(0, 101, 25):
        out("Writing at 0x%08x [%-20s] %5.1f%% ..." % (offset, "=" * (pct // 5), pct))
        if behavior == "write_fail" and offset >= 0x10000 and pct == 50:
            out("A fatal error occurred: Packet content transfer stopped (received 8 bytes)")
            sys.exit(2)
    out("Wrote %d bytes at 0x%08x in 0.1 seconds." % (os.path.getsize(path), offset))
out("Hash of data verified.")
hard_reset(port)

settings = {{}}
for offset, path in pairs:
    if offset == 0xf000:
        settings = settings_block(path)
role = settings.get("role", "")
os.makedirs(os.path.join(BENCH, "firmware"), exist_ok=True)
with open(os.path.join(BENCH, "firmware", board["mac"].replace(":", "") + ".json"), "w") as f:
    json.dump({{"project": {{"tx": "csi_transmitter", "rx": "csi_receiver"}}.get(role, ""),
               "target": want, "port": port, "config": settings,
               "files": [p for _, p in pairs]}}, f, indent=1)
'''


def install_esptool(pkg_root: Path, bench: Path) -> Path:
    """A fake ``esptool`` package under ``pkg_root`` (put it on PYTHONPATH)."""
    pkg = pkg_root / "esptool"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(_FAKE_ESPTOOL.format(bench=str(bench)))
    return pkg
