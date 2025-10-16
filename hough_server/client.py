import socket
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from hough_server.protocol import (
    DetectParams,
    DetectRequest,
    DetectResponse,
    ROIRequest,
    ROIResult,
    ROISpec,
    recv_json,
    send_json,
    send_msg,
)


def detect_circles_batch(
    socket_path: Path,
    rois: Iterable[ROIRequest],
    *,
    canny_low: int,
    canny_high: int,
    timeout: float = 60.0,
) -> list[ROIResult]:
    """Batch detect circles in multiple regions of interest using the HoughServer.

    Sends multiple ROI images to the HoughServer for parallel circle detection
    using the Hough transform algorithm. Each ROI is processed independently
    to find the best-fit circle within the specified radius range.

    Args:
        socket_path: Path to the Unix domain socket for HoughServer communication
        rois: Iterable of ROIRequest objects containing ROI images and detection parameters
        canny_low: Lower threshold for Canny edge detection (0-255)
        canny_high: Upper threshold for Canny edge detection (0-255)
        timeout: Maximum time to wait for server response in seconds (default: 60)

    Returns:
        List of ROIResult objects containing detected circles or error information,
        ordered to match the input ROIs

    Raises:
        TypeError: If ROI array is not uint8 grayscale (wrong dtype or shape)
        RuntimeError: If server returns an error response or invalid data
        socket.timeout: If server does not respond within timeout period
        ConnectionError: If unable to connect to HoughServer
    """
    specs: list[ROISpec] = []  # Build header with ROI specs
    roi_bytes_list: list[bytes] = []  # Convert ROI arrays to bytes
    for roi_req in rois:
        arr = roi_req.roi
        if arr.dtype != np.uint8 or arr.ndim != 2:
            raise TypeError(f"ROI {roi_req.id}: must be uint8 grayscale; got dtype={arr.dtype}, shape={arr.shape}")
        h, w = arr.shape
        b = arr.tobytes(order="C")
        specs.append(
            ROISpec(
                id=roi_req.id,
                height=h,
                width=w,
                num_bytes=len(b),
                min_radius_px=roi_req.min_radius_px,
                max_radius_px=roi_req.max_radius_px,
            )
        )
        roi_bytes_list.append(b)

    # Construct detection request header
    header = DetectRequest(params=DetectParams(canny_low=canny_low, canny_high=canny_high), roi_specs=specs)

    # Connect and send
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(socket_path))
        send_json(sock, header.model_dump())
        for roi_bytes in roi_bytes_list:
            send_msg(sock, roi_bytes)
        raw = recv_json(sock)

    # Validate response
    try:
        resp = DetectResponse.model_validate(raw)
    except ValidationError as e:
        raise RuntimeError(f"Bad response: {e}\nraw={raw!r}") from e

    if not resp.ok:
        raise RuntimeError(resp.error or "Server returned ok=false")

    return resp.results
