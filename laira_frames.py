"""
Shared frame cache used by the brain (laira_brain.py) to snapshot what
LaiRA is currently seeing without contending with aiortc's WebRTC encode
pipeline.

USBVideoTrack.recv() writes the latest BGR numpy frame here after
producing each WebRTC frame. The brain reads from here on demand
(vision calls, SAMURAI track updates) at a much lower rate (≤10Hz), so
the aiortc pipeline isn't blocked on our JPEG encodes.

Design is deliberately minimal:
- Single-writer (USBVideoTrack) / multiple-reader (brain tasks).
- Reference swap is atomic in CPython, so no lock needed for the happy path.
- Lock only around the tuple-read so (bgr, wh, ts) stay coherent for callers
  that need all three fields together.
- JPEG encode runs synchronously in the caller; the brain already awaits
  its encodes off the main event loop path.
"""

import base64
import threading
import time

import cv2


class FrameCache:
    """Holds the most recent BGR frame from the video track."""

    def __init__(self):
        self._latest = None  # tuple (bgr_ndarray, (width, height), ts)
        self._lock = threading.Lock()

    def put(self, bgr):
        """Called by USBVideoTrack on each recv() to stash the latest frame."""
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        with self._lock:
            # Keep a reference to the original array; numpy shares buffer
            # with whatever the camera returned. If a future writer mutates
            # it in place we'd get torn reads, but cv2.VideoCapture.read()
            # allocates a fresh ndarray each call, so we're safe.
            self._latest = (bgr, (w, h), time.time())

    def get_latest(self):
        """Return (bgr, (w, h), ts) or None. Caller should not mutate bgr."""
        with self._lock:
            return self._latest

    def grab_jpeg_b64(self, quality=85):
        """
        Encode the latest frame as base64-encoded JPEG.

        Returns (b64_string, (width, height)) or (None, None) if no frame
        has been captured yet or encoding failed. Quality 85 matches
        `/smart_laira_test`'s current grabFrameJpegB64 default (0.85).
        """
        entry = self.get_latest()
        if entry is None:
            return None, None
        bgr, wh, _ts = entry
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            return None, None
        return base64.b64encode(buf.tobytes()).decode('ascii'), wh

    def age_seconds(self):
        """How stale is the latest frame? Useful for sanity checks. math.inf if none."""
        entry = self.get_latest()
        if entry is None:
            return float('inf')
        return time.time() - entry[2]


# Module-level singleton. laira_brain.py and laira_go_wsprotected.py both
# import this same instance.
cache = FrameCache()
