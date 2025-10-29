from load_data import DataParser
import urllib.request
import time
import cv2
import re
import numpy as np
from typing import Iterable, Tuple, Union

def send_frame_to_server(vis_bgr, url="http://localhost:8000/push", quality=90):
    # als JPEG kodieren (klein & schnell für Stream)
    ok, buf = cv2.imencode(".jpg", vis_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    data = buf.tobytes()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/octet-stream",
                                          "Content-Length": str(len(data))})
    with urllib.request.urlopen(req, timeout=2) as resp:
        return resp.status == 200
    
def load_image(path):
    img = cv2.imread(path)
    return img

def split_point_string_to_points(string):
    _point_re = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
    
    if not string or not isinstance(string, str):
        return []
    
    pts = _point_re.findall(string)  # Liste von Strings: [('x1','y1'), ('x2','y2'), ...]
    return [(float(x), float(y)) for x, y in pts]

def draw_points(
    img: np.ndarray,
    points: Iterable[Tuple[float, float]],
    color: Union[Tuple[int, int, int], str] = (0, 255, 0),
    radius: int = 4,
    thickness: int = -1,
    assume_rgb: bool = False,
    clip: bool = True,
) -> np.ndarray:
    
    if img is None or not isinstance(img, np.ndarray) or img.ndim < 2:
        raise ValueError("img muss ein gültiges OpenCV-Array sein")

    # Farbe in BGR umwandeln
    def to_bgr(c: Color) -> Tuple[int, int, int]:
        if isinstance(c, str):
            c = c.strip()
            if c.startswith("#") and len(c) == 7:
                r = int(c[1:3], 16)
                g = int(c[3:5], 16)
                b = int(c[5:7], 16)
                return (b, g, r)
            else:
                raise ValueError("Hex-Farbe muss Format '#RRGGBB' haben")
        if isinstance(c, (tuple, list)) and len(c) == 3:
            r, g, b = c if assume_rgb else (c[2], c[1], c[0])  # falls BGR übergeben
            return (b, g, r) if assume_rgb else tuple(c)  # Ziel immer BGR
        raise ValueError("Farbe als (B,G,R), (R,G,B mit assume_rgb=True) oder '#RRGGBB' angeben")

    bgr = to_bgr(color)

    h, w = img.shape[:2]
    for xy in points:
        if xy is None or len(xy) != 2:
            continue
        x, y = float(xy[0]), float(xy[1])
        xi, yi = int(round(x)), int(round(y))

        if clip:
            if xi < 0 or yi < 0 or xi >= w or yi >= h:
                continue
        # Anti-aliased Kreis (AA wirkt v.a. bei dünnen Linien)
        cv2.circle(img, (xi, yi), radius, bgr, thickness, lineType=cv2.LINE_AA)

    return img

task_names_to_load = ["2023-05", "2023-06", "2023-10", "2023-12", "2025-05"]
data = DataParser(task_names_to_load).get_data()

sample_data = data.sample(n=100)

for index, row in sample_data.iterrows():
    print(index)
    print(split_point_string_to_points(row["left lane"]))
    print(row["image_path"])
    
    img = load_image(row["image_path"])
    send_frame_to_server(img)
    time.sleep(2)
    
