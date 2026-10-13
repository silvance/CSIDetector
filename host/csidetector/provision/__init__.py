"""Provisioning: discover ESP32 boards, keep a registry of them, and flash
firmware with validated configuration — the plumbing behind
``python run.py setup``.

  devices   — USB serial enumeration, esptool identify parsing, and the
              ~/.csidetector/devices.json registry (MAC → label / role).
  firmware  — TX / RX config validation, sdkconfig-defaults generation,
              idf.py / esptool command construction, output parsing,
              and hotspot auto-detection.
  gui       — the Qt setup window (imports PyQt6; everything else here
              is importable without it).
"""
