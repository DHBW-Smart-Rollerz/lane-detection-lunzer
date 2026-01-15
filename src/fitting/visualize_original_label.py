"""
Utilities for visualizing ground-truth lane annotations on images.

This module loads images, parses CVAT polyline annotations, draws lane
points and connecting lines using OpenCV, and optionally streams the
resulting frames to a Push2View image server.
"""

import re
import time
import urllib.request
from typing import Iterable, Tuple

import cv2
import numpy as np
from .load_data import get_current_data


def send_frame_to_server(vis_bgr, url="http://localhost:8000/push", quality=90):
    """
    Encode an image as JPEG and send it to a Push2View streaming server.

    The image is encoded as a JPEG for efficient transfer and sent as raw
    bytes via HTTP POST.

    Args:
        vis_bgr (numpy.ndarray):
            Image in BGR color format (OpenCV convention).
        url (str, optional):
            Endpoint URL of the image stream server. Defaults to
            "http://localhost:8000/push".
        quality (int, optional):
            JPEG quality (0–100). Higher values improve quality at the cost
            of bandwidth. Defaults to 90.

    Returns:
        bool:
            True if the frame was successfully sent (HTTP 200), False otherwise.
    """
    # als JPEG kodieren (klein & schnell für Stream)
    ok, buf = cv2.imencode(".jpg", vis_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    data = buf.tobytes()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(data)),
        },
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
        return resp.status == 200


def load_image(path):
    """
    Load an image from disk using OpenCV.

    Args:
        path (str):
            Path to the image file.

    Returns:
        numpy.ndarray | None:
            Loaded image in BGR format, or None if loading fails.
    """
    img = cv2.imread(path)
    return img


def split_point_string_to_points(string):
    """
    Parse a CVAT polyline point string into a list of (x, y) coordinates.

    The expected format is a sequence of comma-separated coordinate pairs,
    e.g. "x1,y1 x2,y2 ...". Floating-point and negative values are supported.

    Args:
        string (str):
            Polyline point string as stored in the annotation file.

    Returns:
        list[tuple[float, float]]:
            List of (x, y) coordinate pairs. Returns an empty list if the
            input is invalid or empty.
    """
    _point_re = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")

    if not string or not isinstance(string, str):
        return []

    pts = _point_re.findall(
        string
    )  # Liste von Strings: [('x1','y1'), ('x2','y2'), ...]
    return [(float(x), float(y)) for x, y in pts]


def draw_points(
    img: np.ndarray,
    points: Iterable[Tuple[float, float]],
    color: str,
    radius: int = 4,
    thickness: int = -1,
    clip: bool = True,
) -> np.ndarray:
    """
    Draw points and connecting lines onto an image.

    Each point is rendered as a circle, and consecutive points are connected
    with straight line segments. Coordinates may optionally be clipped to
    the image bounds.

    Args:
        img (numpy.ndarray):
            Target image in BGR format.
        points (Iterable[tuple[float, float]]):
            Iterable of (x, y) coordinates to draw.
        color (str):
            Color name for drawing. Must be one of: "blue", "green", "red".
        radius (int, optional):
            Radius of drawn points in pixels. Defaults to 4.
        thickness (int, optional):
            Line thickness. Use -1 to draw filled circles. Defaults to -1.
        clip (bool, optional):
            If True, points outside the image bounds are skipped. Defaults to True.

    Raises:
        ValueError:
            If the image is invalid or the color is unsupported.

    Returns:
        numpy.ndarray:
            Image with drawn points and lines.
    """
    if img is None or not isinstance(img, np.ndarray) or img.ndim < 2:
        raise ValueError("img muss ein gültiges OpenCV-Array sein")

    if color == "blue":
        bgr = (255, 0, 0)
    elif color == "green":
        bgr = (0, 255, 0)
    elif color == "red":
        bgr = (0, 0, 255)
    else:
        raise ValueError("Color muss einer der folgenden strings sein: blue,green,red")

    h, w = img.shape[:2]
    last_xi = None
    last_yi = None
    for xy in points:
        if xy is None or len(xy) != 2:
            continue
        x, y = float(xy[0]), float(xy[1])
        xi, yi = int(round(x)), int(round(y))

        if clip:
            if xi < 0 or yi < 0 or xi > w or yi > h:
                continue
        # Anti-aliased Kreis (AA wirkt v.a. bei dünnen Linien)
        cv2.circle(img, (xi, yi), radius, bgr, thickness, lineType=cv2.LINE_AA)
        if last_xi and last_yi != None:
            cv2.line(img, (last_xi, last_yi), (xi, yi), bgr, 2, lineType=cv2.LINE_AA)
        last_xi = xi
        last_yi = yi

    return img


def draw_lanes(img, data_row):
    """
    Draw left, center, and right lane annotations onto an image.

    Lane polylines are extracted from the dataset row and rendered using
    fixed color conventions:
    - Left lane: red
    - Center lane: green
    - Right lane: blue

    Args:
        img (numpy.ndarray):
            Input image in BGR format.
        data_row:
            Pandas Series or dict-like object containing lane label fields
            ("left lane", "center lane", "right lane").

    Returns:
        numpy.ndarray:
            Image with all available lane annotations drawn.
    """
    left_lane_data = split_point_string_to_points(data_row["left lane"])
    center_lane_data = split_point_string_to_points(data_row["center lane"])
    right_lane_data = split_point_string_to_points(data_row["right lane"])

    img = draw_points(img, left_lane_data, "red")
    img = draw_points(img, center_lane_data, "green")
    img = draw_points(img, right_lane_data, "blue")
    return img


if __name__ == "__main__":
    data = get_current_data()
    sample_data = data.sample(n=100)

    for index, row in sample_data.iterrows():
        img = load_image(row["image_path"])
        img = draw_lanes(img, row)
        send_frame_to_server(img)
        print(row["image_path"])
        time.sleep(0.4)
