from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks

from hough_server.protocol import Circle

if TYPE_CHECKING:
    from numpy.typing import NDArray


def detect_one_circle_scikit(
    gray_roi: NDArray,
    min_radius_px: int,
    max_radius_px: int,
    canny_low: int,
    canny_high: int,
) -> Circle | None:
    """Detect exactly one circle in a grayscale ROI using Canny + Hough Circle.

    This function applies OpenCV's Canny edge detector to a 2D grayscale image and then runs scikit-image's
    Hough Circle transform (`hough_circle` / `hough_circle_peaks`) over a range of radii.
    If **exactly one** circle is identified, that circle is returned; otherwise, `None` is returned.

    Args:
        gray_roi (NDArray): 2D grayscale ROI array of shape (H, W). Typically `uint8`
        min_radius_px (int): Inclusive minimum circle radius to consider, in pixels.
        max_radius_px (int): Exclusive maximum circle radius to consider, in pixels.
        canny_low (int): Lower hysteresis threshold for Canny edge detection.
        canny_high (int): Upper hysteresis threshold for Canny edge detection.

    Returns:
        Circle | None: A `Circle(x, y, r)` in pixel coordinates (center x, center y, radius)
                       relative to `gray_roi` if exactly one circle is detected; otherwise `None`.

    Notes:
        - radii are generated with `np.arange(min_radius_px, max_radius_px, 1)`, so `max_radius_px` is **not** included.
    """
    detected_circles_px: list[Circle] = []

    # Apply Canny edge detection algorithm
    roi_edges = cv2.Canny(gray_roi, canny_low, canny_high)

    # Create range of circle radii to search for
    search_radii_px = np.arange(min_radius_px, max_radius_px, 1)

    # Perform Hough circle transform on edge-detected region of interest
    hough_accumulator = hough_circle(roi_edges, search_radii_px)

    # Extract most likely circle from Hough accumulator
    accumulators, center_x_coords, center_y_coords, radii = hough_circle_peaks(
        hough_accumulator, search_radii_px, total_num_peaks=1
    )

    # Collect all detected circle centers
    for _, center_x, center_y, radius in zip(accumulators, center_x_coords, center_y_coords, radii, strict=False):
        detected_circles_px.append(Circle(x=center_x, y=center_y, r=radius))

    # Ensure one circle found
    if len(detected_circles_px) != 1:
        return None
    return detected_circles_px[0]
