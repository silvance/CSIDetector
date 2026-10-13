"""Provisioning: discover ESP32 boards, keep a registry of them, and flash
the prebuilt firmware with each board's validated settings — the plumbing
behind ``python run.py setup``.

  devices   — USB serial enumeration, esptool identify parsing, and the
              ~/.csidetector/devices.json registry (MAC → label / role).
  firmware  — TX / RX config validation, the prebuilt firmware manifest,
              esptool command construction, output parsing, and hotspot
              auto-detection.
  board_settings — the settings block written to a board's csi_cfg
              partition (read by the firmware at boot).
  gui       — the Qt setup window (imports PyQt6; everything else here
              is importable without it).
"""
