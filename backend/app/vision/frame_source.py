
"""Frame sources — video file, webcam, and HTTP streams via OpenCV.

Clean lifecycle (open/read/close), frame iteration with index + timestamp,
and measured source FPS.
"""
from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np


class FrameSourceError(Exception):
    """Raised when a source cannot be opened or read."""


@dataclass
class Frame:
    image: np.ndarray  # BGR, uint8
    index: int
    timestamp: float  # seconds since source opened


class FrameSource(abc.ABC):
    """Interface every camera source must implement."""

    @abc.abstractmethod
    def open(self) -> None:
        """Open the underlying capture. Raises FrameSourceError."""

    @abc.abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or None at end-of-stream."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release all resources. Safe to call multiple times."""

    @property
    @abc.abstractmethod
    def source_fps(self) -> float:
        """Nominal FPS reported by the source (0 if unknown)."""

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class OpenCVFrameSource(FrameSource):
    """File, webcam, or HTTP stream backed by cv2.VideoCapture.

    ``loop=True`` applies only to file sources.
    """

    def __init__(
        self,
        source_type: str,
        source_url: str,
        loop: bool = False,
    ):
        if source_type not in ("file", "webcam", "http"):
            raise FrameSourceError(
                f"unsupported source_type: {source_type}"
            )

        self.source_type = source_type
        self.loop = loop and source_type == "file"

        # Webcam sources use a numeric device index; other sources use URLs.
        self.target: str | int = (
            int(source_url) if source_type == "webcam" else source_url
        )

        self._cap: cv2.VideoCapture | None = None
        self._index = 0
        self._t0 = 0.0
        self._opened = False
        self.loops_completed = 0
        self._next_frame_time = 0.0
        self._frame_interval = 0.0

    def open(self) -> None:
        if self._opened:
            return

        if self.source_type == "file":
            if not os.path.isfile(self.target):
                raise FrameSourceError(
                    f"video file not found: {self.target}"
                )

        cap = cv2.VideoCapture(self.target)

        if not cap.isOpened():
            cap.release()
            raise FrameSourceError(
                f"could not open {self.source_type} source: {self.target}"
            )

        self._cap = cap
        self._index = 0
        self._t0 = time.time()

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._frame_interval = (
            1.0 / fps if self.source_type == "file" and fps > 0 else 0.0
        )
        self._next_frame_time = time.perf_counter()
        self._opened = True

    def read(self) -> Frame | None:
        if not self._opened or self._cap is None:
            raise FrameSourceError("source is not open")

        # Throttle file playback to its reported FPS.
        if self.source_type == "file" and self._frame_interval > 0:
            now = time.perf_counter()
            delay = self._next_frame_time - now

            if delay > 0:
                time.sleep(delay)

            self._next_frame_time = max(
                self._next_frame_time + self._frame_interval,
                time.perf_counter(),
            )

        ok, image = self._cap.read()

        if (not ok or image is None) and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._cap.read()

            if ok and image is not None:
                self.loops_completed += 1
                self._index = 0
                self._t0 = time.time()
                self._next_frame_time = time.perf_counter()

        if not ok or image is None:
            return None

        frame = Frame(
            image=image,
            index=self._index,
            timestamp=time.time() - self._t0,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._opened = False

    @property
    def source_fps(self) -> float:
        if self._cap is None:
            return 0.0

        return float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

    @property
    def is_open(self) -> bool:
        return self._opened


def create_frame_source(
    source_type: str,
    source_url: str,
    loop: bool = False,
) -> FrameSource:
    """Create a supported video file, webcam, or HTTP stream source."""
    if source_type in ("file", "webcam", "http"):
        return OpenCVFrameSource(
            source_type, source_url, loop=loop
        )

    raise FrameSourceError(
        f"unsupported source_type: {source_type}"
    )
