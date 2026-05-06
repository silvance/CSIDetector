"""Alert mode — single TX + single RX binary motion detection.

Phase 1 contains only the relocated single-stream calibrate/detect path.
Phase 2 adds the Notifier abstraction and Telegram sender. Phase 3 adds
the durable outbound queue with exponential-backoff retries.
"""
