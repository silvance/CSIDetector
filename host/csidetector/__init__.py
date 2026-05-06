"""CSIDetector — CSI-based motion detection.

Two modes share a common signal-processing core:

  - ``csidetector.modes.localize`` — multi-RX/TX room-scale localization
    (heatmap, view3d, calibrate-links).
  - ``csidetector.modes.alert`` — single-TX/single-RX binary motion
    detection with notification + durable queue.

Old flat module names (``csi_collector``, ``heatmap``, ``detector``, …)
are kept as shim re-exports under ``host/`` so existing imports continue
to work; new code should import from this package.
"""

__version__ = "0.2.0"
