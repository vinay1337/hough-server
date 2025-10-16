from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

if TYPE_CHECKING:
    from numpy.typing import NDArray

LEN = struct.Struct("!Q")  # 8-byte big-endian unsigned length

SOCKET_PATH = Path("/tmp/hough_circles.sock")


### Socket framing helpers ###


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed mid-read")
        buf.extend(chunk)
    return bytes(buf)


def send_json(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sock.sendall(LEN.pack(len(data)) + data)


def recv_json(sock: socket.socket) -> dict[str, Any]:
    raw_len = recv_exact(sock, LEN.size)
    (num_bytes,) = LEN.unpack(raw_len)
    data = recv_exact(sock, num_bytes)

    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj


def send_msg(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(LEN.pack(len(payload)) + payload)


def recv_msg(sock: socket.socket) -> bytes:
    (nbytes,) = LEN.unpack(recv_exact(sock, LEN.size))
    return recv_exact(sock, nbytes)


### Data models ###


@dataclass
class ROIRequest:
    """Request to detect a circle within a region of interest (ROI).

    Container for passing ROI image data and detection parameters to the
    circle detection algorithm. Each ROI is processed independently to find
    the best-fit circle within the specified radius range.

    Attributes:
        id: Unique identifier for this ROI (e.g., hole name)
        roi: Grayscale image array of the ROI (must be uint8, 2D)
        min_radius_px: Minimum radius to search for circles (in pixels)
        max_radius_px: Maximum radius to search for circles (in pixels)
    """

    id: str
    roi: NDArray
    min_radius_px: int
    max_radius_px: int


class DetectRequest(BaseModel):
    type: Literal["detect"] = "detect"
    params: DetectParams
    roi_specs: list[ROISpec] = Field(..., min_length=1)


class DetectParams(BaseModel):
    canny_low: int = Field(..., ge=0, le=255)
    canny_high: int = Field(..., ge=0, le=255)


class ROISpec(BaseModel):
    id: str
    height: int = Field(..., gt=0)
    width: int = Field(..., gt=0)
    num_bytes: int = Field(..., gt=0)
    min_radius_px: int = Field(..., gt=0)
    max_radius_px: int = Field(..., gt=0)

    @field_validator("max_radius_px")
    @classmethod
    def _radii_ok(cls, v: int, info: ValidationInfo) -> int:
        mr = info.data.get("min_radius_px")
        if mr is not None and v <= mr:
            raise ValueError("max_radius_px must be > min_radius_px")
        return v


class DetectResponse(BaseModel):
    type: Literal["detect"] = "detect"
    ok: bool
    results: list[ROIResult] = Field(default_factory=list)
    ms: float | None = None
    error: str | None = None


class ROIResult(BaseModel):
    id: str
    circle: Circle | None = None
    error: str | None = None


class Circle(BaseModel):
    x: float
    y: float
    r: float
