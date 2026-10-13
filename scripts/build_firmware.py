#!/usr/bin/env python3
"""Build the prebuilt firmware the setup app flashes (firmware/prebuilt/).

Runs in an ESP-IDF environment (CI: .github/workflows/firmware.yml, in the
espressif/idf container). Builds csi_transmitter and csi_receiver once per
chip — nothing per board is compiled in; the setup app writes each board's
settings into its csi_cfg partition when it flashes.

    python scripts/build_firmware.py            # build if the sources changed
    python scripts/build_firmware.py --check    # exit 1 if a rebuild is due
    python scripts/build_firmware.py --out DIR  # e.g. a developer's own build;
                                                # the setup app uses it with
                                                # CSIDETECTOR_FIRMWARE_DIR=DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIRMWARE = REPO / "firmware"
PROJECTS = ("csi_transmitter", "csi_receiver")
TARGETS = ("esp32s3", "esp32c3", "esp32c5", "esp32c6", "esp32")
SOURCES = ("partitions.csv", "components", *PROJECTS)
SETTINGS_LABEL = "csi_cfg"
FORMAT = "csidetector-firmware/1"
_SKIP = {"build", "sdkconfig", "sdkconfig.old", "managed_components",
         "dependencies.lock", "__pycache__"}


def idf_version() -> str:
    out = subprocess.run(["idf.py", "--version"], capture_output=True, text=True,
                         check=True).stdout.strip()
    return out.split()[-1] if out else "unknown"     # "ESP-IDF v5.5.1"


def source_fingerprint(idf: str) -> str:
    """sha256 over the firmware sources, this script and the ESP-IDF version."""
    h = hashlib.sha256()
    h.update(f"idf={idf}\n".encode())
    h.update(Path(__file__).read_bytes())
    for top in SOURCES:
        root = FIRMWARE / top
        paths = [root] if root.is_file() else sorted(
            p for p in root.rglob("*")
            if p.is_file() and not _SKIP.intersection(p.relative_to(root).parts))
        for p in paths:
            h.update(b"\0" + p.relative_to(FIRMWARE).as_posix().encode() + b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()


def partition(table: bytes, label: str) -> tuple[int, int]:
    """(offset, size) of the partition named ``label`` in a binary table."""
    for i in range(0, len(table) - 31, 32):
        magic, _type, _sub, offset, size, name, _flags = struct.unpack_from(
            "<2sBBII16sI", table, i)
        if magic != b"\xaa\x50":
            break
        if name.rstrip(b"\0").decode() == label:
            return offset, size
    raise SystemExit(f"partition {label!r} not in the partition table")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_one(project: str, target: str, work: Path, out: Path) -> dict:
    build = work / f"{project}-{target}"
    subprocess.run(["idf.py", "-C", str(FIRMWARE / project), "-B", str(build),
                    "-D", f"SDKCONFIG={build / 'sdkconfig'}",
                    "-D", f"IDF_TARGET={target}", "build"], check=True)
    args = json.loads((build / "flasher_args.json").read_text())
    name = f"{project}-{target}"
    dest = out / name
    dest.mkdir(parents=True)
    files = []
    for offset, rel in sorted(args["flash_files"].items(), key=lambda kv: int(kv[0], 0)):
        src = build / rel
        shutil.copyfile(src, dest / src.name)
        files.append({"offset": hex(int(offset, 0)), "path": f"{name}/{src.name}",
                      "sha256": _sha256(dest / src.name)})
    table = (build / args["partition-table"]["file"]).read_bytes()
    s_offset, s_size = partition(table, SETTINGS_LABEL)
    nvs_offset, nvs_size = partition(table, "nvs")
    settings = args["flash_settings"]
    return {
        "project": project, "target": target,
        "chip": args["extra_esptool_args"]["chip"],
        "flash_mode": settings["flash_mode"], "flash_freq": settings["flash_freq"],
        "flash_size": settings["flash_size"],
        "files": files,
        "settings": {"offset": hex(s_offset), "size": s_size},
        # Blanked on every flash: NVS left by another partition layout
        # can stop nvs_flash_init() at boot.
        "nvs": {"offset": hex(nvs_offset), "size": nvs_size},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=FIRMWARE / "prebuilt")
    ap.add_argument("--check", action="store_true",
                    help="only report whether the prebuilt firmware is out of date")
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = ap.parse_args()

    idf = idf_version()
    fingerprint = source_fingerprint(idf)
    manifest_path = args.out / "manifest.json"
    try:
        current = json.loads(manifest_path.read_text()).get("source_sha256")
    except (OSError, ValueError):
        current = None
    if current == fingerprint and not args.force:
        print(f"prebuilt firmware is up to date ({fingerprint[:12]})")
        return 0
    if args.check:
        print("prebuilt firmware is out of date")
        return 1

    work = Path(tempfile.mkdtemp(prefix="csid-fw-"))
    staging = work / "out"
    staging.mkdir()
    images, failed = {}, []
    for target in TARGETS:
        for project in PROJECTS:
            print(f"::group::{project} for {target}", flush=True)
            try:
                images[f"{project}-{target}"] = build_one(project, target, work, staging)
            except (subprocess.CalledProcessError, KeyError, OSError) as exc:
                failed.append(f"{project}-{target}: {exc}")
            print("::endgroup::", flush=True)
    if failed:
        print("Build failed:\n  " + "\n  ".join(failed), file=sys.stderr)
        return 1
    manifest = {"format": FORMAT, "idf_version": idf, "source_sha256": fingerprint,
                "commit": os.environ.get("GITHUB_SHA", ""), "images": images}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (staging / "README.md").write_text(
        "Built by scripts/build_firmware.py (CI: .github/workflows/firmware.yml)"
        " from the firmware sources; don't edit by hand.\n")
    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(staging, args.out)
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {len(images)} images to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
