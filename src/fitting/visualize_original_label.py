import re
import time
import urllib.request
from typing import Iterable, Tuple

import cv2
import numpy as np
from load_data import get_current_data


def send_frame_to_server(vis_bgr, url="http://localhost:8000/push", quality=90):
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
    img = cv2.imread(path)
    return img


def split_point_string_to_points(string):
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
