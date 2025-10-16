import signal
import socket
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from logging import getLogger
from pathlib import Path

import numpy as np

from hough_server.cpu_worker import detect_one_circle_scikit
from hough_server.protocol import (
    SOCKET_PATH,
    DetectRequest,
    DetectResponse,
    ROIRequest,
    ROIResult,
    recv_json,
    recv_msg,
    send_json,
)

LOG = getLogger(__name__)


class HoughServer:
    """Multi-process server for parallel circle detection using Hough transform.

    Provides a Unix domain socket interface for batch processing of circle
    detection requests. Uses a process pool to parallelize the computationally
    intensive Hough transform across multiple CPU cores.

    The server accepts batches of ROI images, processes them in parallel using
    scikit-image's Hough circle detection, and returns the best-fit circle for
    each ROI.

    Attributes:
        _socket_path: Path to Unix domain socket file for client connections
        _executor: ProcessPoolExecutor for parallel circle detection
        _install_signal_handler: Whether to install SIGINT handler for graceful shutdown
        _srv: Server socket instance (None until serve() is called)
        _shutdown: Threading event to coordinate shutdown
    """

    def __init__(
        self,
        socket_path: Path = SOCKET_PATH,
        max_workers: int | None = None,
        install_signal_handler: bool = False,
    ):
        """Initialize the HoughServer.

        Args:
            socket_path: Path for the Unix domain socket (default: /tmp/hough_circles.sock)
            max_workers: Number of worker processes for parallel processing.
                         If None, uses system CPU count.
            install_signal_handler: If True, installs SIGINT handler for Ctrl-C support.
                                    Set to False when embedding in larger applications.
        """
        self._socket_path = socket_path
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._install_signal_handler = install_signal_handler
        self._srv: socket.socket | None = None
        self._shutdown = threading.Event()

    def shutdown(self) -> None:
        """Signal the server to shut down.

        Sets the shutdown event which causes the serve() loop to exit and
        triggers cleanup of client connections. Safe to call multiple times.
        """
        if not self._shutdown.is_set():
            self._shutdown.set()

    def handle_client(self, conn: socket.socket) -> None:
        """Handle requests from a single client connection.

        Processes detection requests in a loop until the client disconnects
        or the server shuts down. Each request contains ROI specifications
        followed by the actual image data.

        Args:
            conn: Connected client socket

        Protocol flow:
            1. Receive JSON header with DetectRequest
            2. Receive binary ROI data for each ROI specified
            3. Process ROIs in parallel using worker pool
            4. Send JSON DetectResponse with results
        """
        try:
            conn.settimeout(30.0)
            while not self._shutdown.is_set():
                try:
                    header = recv_json(conn)
                except Exception:
                    break  # drop client

                try:
                    req = DetectRequest.model_validate(header)
                except Exception as e:
                    res = DetectResponse(ok=False, error=f"Bad request: {e!r}")
                    send_json(conn, res.model_dump())
                    continue

                # Read ROI frames described by header
                roi_reqs: list[ROIRequest] = []
                try:
                    for spec in req.roi_specs:
                        blob = recv_msg(conn)
                        if len(blob) != spec.num_bytes:
                            raise ValueError(
                                f"ROI bytes mismatch for roi_id={spec.id}: expected {spec.num_bytes}, got {len(blob)}"
                            )
                        arr = np.frombuffer(blob, dtype=np.uint8)
                        if arr.size != spec.height * spec.width:
                            raise ValueError(
                                f"ROI size mismatch for roi_id={spec.id}: {arr.size} != {spec.height * spec.width}"
                            )
                        roi_reqs.append(
                            ROIRequest(
                                spec.id,
                                arr.reshape((spec.height, spec.width)).copy(),
                                spec.min_radius_px,
                                spec.max_radius_px,
                            )
                        )
                except Exception as e:
                    res = DetectResponse(ok=False, error=f"ROI decode failed: {e!r}")
                    send_json(conn, res.model_dump())
                    continue

                # One ROI per process
                t0 = time.perf_counter()
                futures = [
                    self._executor.submit(
                        detect_one_circle_scikit,
                        roi_req.roi,
                        roi_req.min_radius_px,
                        roi_req.max_radius_px,
                        req.params.canny_low,
                        req.params.canny_high,
                    )
                    for roi_req in roi_reqs
                ]

                # Collect results in same order as the header
                results: list[ROIResult] = []
                for roi_req, future in zip(roi_reqs, futures, strict=True):
                    try:
                        circle = future.result()
                        results.append(ROIResult(id=roi_req.id, circle=circle))
                    except Exception as e:
                        results.append(ROIResult(id=roi_req.id, circle=None, error=repr(e)))

                ms = round((time.perf_counter() - t0) * 1000, 2)
                response = DetectResponse(ok=True, results=results, ms=ms)
                send_json(conn, response.model_dump())

        finally:
            try:
                conn.close()
            except Exception:
                pass

    def serve(self) -> None:
        """Start the server and listen for client connections.

        Binds to the Unix domain socket and accepts client connections in a loop.
        Each client is handled in a separate thread. Continues until shutdown()
        is called or SIGINT is received (if signal handler is installed).

        The socket file is removed on startup (if it exists) and on shutdown.
        """
        # Clean up stale socket file
        self._socket_path.unlink(missing_ok=True)

        # Install signal handlers if requested
        if self._install_signal_handler:
            signal.signal(signal.SIGINT, lambda sig, frame: self.shutdown())

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
            self._srv = srv
            srv.bind(str(self._socket_path))
            srv.listen(64)
            srv.settimeout(0.5)
            LOG.debug(f"Listening on {self._socket_path}")

            while not self._shutdown.is_set():
                try:
                    # Handle client connections in a new thread
                    conn, _ = srv.accept()
                    threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()
                except TimeoutError:
                    continue
                except OSError:
                    if self._shutdown.is_set():
                        break
                    else:
                        raise

        # Shutdown executor and clean up socket file
        self._executor.shutdown()
        self._socket_path.unlink(missing_ok=True)


def main() -> None:
    """Entry point for running HoughServer as a standalone process.

    Starts the server with SIGINT handling enabled. The server runs until interrupted with Ctrl-C.
    """
    server = HoughServer(install_signal_handler=True)
    server.serve()
