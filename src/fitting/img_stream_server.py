#!/usr/bin/env python3
import argparse
import threading
import time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import traceback

# Shared state
LAST_FRAME = {
    "bytes": None,
    "ts": 0.0,
    "counter": 0,
}
LOCK = threading.Lock()

HTML_PAGE = """<!doctype html>
<html lang="de">
<meta charset="utf-8">
<title>Live Image Stream</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { color-scheme: dark light; }
  body { margin:0; font: 15px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#111; color:#eee; }
  .wrap { width:min(96vw,1200px); margin:0 auto; padding:16px; }
  h1 { font-size:1.1rem; margin:.3rem 0 1rem; opacity:.9 }
  .row { display:flex; gap:16px; align-items:center; justify-content:space-between; flex-wrap:wrap; }
  .card { background:#1b1b1b; border-radius:10px; padding:10px; }
  img { display:block; width:100%; height:auto; background:#222; object-fit:contain; border-radius:8px; }
  a.btn { color:#fff; text-decoration:none; padding:8px 12px; border:1px solid #444; border-radius:8px; }
  .muted { opacity:.7; font-size:.9rem; }
</style>
<div class="wrap">
  <div class="row">
    <h1>Live Image Stream (MJPEG)</h1>
    <div class="muted">Pushes: <span id="cnt">0</span> • <a class="btn" href="/latest.jpg" target="_blank" rel="noreferrer">/latest.jpg</a> • <a class="btn" href="/meta" target="_blank" rel="noreferrer">/meta</a></div>
  </div>
  <div class="card">
    <img id="view" src="/mjpg" alt="waiting for frames…"/>
  </div>
</div>
<script>
(async function pollMeta(){
  try {
    const r = await fetch('/meta?t=' + Date.now());
    if (r.ok) { const j = await r.json(); document.getElementById('cnt').textContent = j.counter; }
  } catch(e) {}
  setTimeout(pollMeta, 500);
})();
</script>
"""

def _write_fixed(handler, status: int, headers: dict, body_bytes: bytes, head_only=False):
    handler.send_response(status)
    for k, v in headers.items():
        handler.send_header(k, v)
    handler.send_header("Content-Length", str(len(body_bytes)))
    handler.end_headers()
    if not head_only:
        handler.wfile.write(body_bytes)

class ImageStreamHandler(BaseHTTPRequestHandler):
    server_version = "ImageStreamServer/2.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          fmt % args))

    # --------- GET / HEAD ----------
    def do_HEAD(self):
        try:
            self._route(head_only=True)
        except Exception:
            traceback.print_exc()
            _write_fixed(self, 500, {"Content-Type": "text/plain; charset=utf-8",
                                     "Cache-Control": "no-store"}, b"", head_only=True)

    def do_GET(self):
        try:
            self._route(head_only=False)
        except Exception:
            traceback.print_exc()
            _write_fixed(self, 500, {"Content-Type": "text/plain; charset=utf-8",
                                     "Cache-Control": "no-store"}, b"Internal Server Error")

    def _route(self, head_only: bool):
        path = self.path.split('?', 1)[0]

        if path == "/" or path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            _write_fixed(self, 200,
                         {"Content-Type": "text/html; charset=utf-8",
                          "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
                         b"" if head_only else body)
            return

        if path == "/latest.jpg":
            with LOCK:
                buf = LAST_FRAME["bytes"]
            if buf is None:
                _write_fixed(self, 404,
                             {"Content-Type": "text/plain; charset=utf-8",
                              "Cache-Control": "no-store"},
                             b"No frame yet", head_only=head_only)
                return
            _write_fixed(self, 200,
                         {"Content-Type": "image/jpeg",
                          "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                          "Pragma": "no-cache",
                          "Expires": "0"},
                         b"" if head_only else buf)
            return

        if path == "/mjpg":
            # Multipart MJPEG stream. Each part is a full JPEG of the latest frame.
            boundary = "frameboundary"
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()

            last_seen_counter = -1
            # Stream loop
            while True:
                with LOCK:
                    cnt = LAST_FRAME["counter"]
                    buf = LAST_FRAME["bytes"]
                if buf is not None and cnt != last_seen_counter:
                    last_seen_counter = cnt
                    try:
                        part_head = (
                            f"--{boundary}\r\n"
                            "Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(buf)}\r\n"
                            "\r\n"
                        ).encode("utf-8")
                        self.wfile.write(part_head)
                        self.wfile.write(buf)
                        self.wfile.write(b"\r\n")
                    except BrokenPipeError:
                        break
                    except ConnectionResetError:
                        break
                # kleine Pause, um CPU zu schonen
                time.sleep(0.01)
            return

        if path == "/meta":
            with LOCK:
                cnt = LAST_FRAME["counter"]
                ts = LAST_FRAME["ts"]
            age_ms = int(1000 * (time.monotonic() - ts)) if ts else -1
            body = json.dumps({"counter": cnt, "age_ms": age_ms}).encode("utf-8")
            _write_fixed(self, 200,
                         {"Content-Type": "application/json; charset=utf-8",
                          "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
                         b"" if head_only else body)
            return

        if path == "/health":
            _write_fixed(self, 200,
                         {"Content-Type": "text/plain; charset=utf-8",
                          "Cache-Control": "no-store"},
                         b"" if head_only else b"OK")
            return

        _write_fixed(self, 404,
                     {"Content-Type": "text/plain; charset=utf-8",
                      "Cache-Control": "no-store"},
                     b"Not Found", head_only=head_only)

    # ---------- POST ----------
    def do_POST(self):
        try:
            path = self.path.split('?', 1)[0]
            if path != "/push":
                _write_fixed(self, 404,
                             {"Content-Type": "text/plain; charset=utf-8",
                              "Cache-Control": "no-store"},
                             b"Not Found")
                return

            length = self.headers.get('Content-Length')
            if not length:
                _write_fixed(self, 411,
                             {"Content-Type": "text/plain; charset=utf-8",
                              "Cache-Control": "no-store"},
                             b"Length Required")
                return

            try:
                n = int(length)
            except ValueError:
                _write_fixed(self, 400,
                             {"Content-Type": "text/plain; charset=utf-8",
                              "Cache-Control": "no-store"},
                             b"Bad Content-Length")
                return

            data = self.rfile.read(n)
            if not data:
                _write_fixed(self, 400,
                             {"Content-Type": "text/plain; charset=utf-8",
                              "Cache-Control": "no-store"},
                             b"Empty body")
                return

            # Light JPEG check (not strict)
            if not (len(data) >= 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"):
                # akzeptieren trotzdem (falls du PNG o.ä. schicken würdest, würde MJPEG nicht gehen)
                pass

            with LOCK:
                LAST_FRAME["bytes"] = data
                LAST_FRAME["ts"] = time.monotonic()
                LAST_FRAME["counter"] += 1

            _write_fixed(self, 200,
                         {"Content-Type": "text/plain; charset=utf-8",
                          "Cache-Control": "no-store"},
                         b"OK")
        except Exception:
            traceback.print_exc()
            _write_fixed(self, 500, {"Content-Type": "text/plain; charset=utf-8",
                                     "Cache-Control": "no-store"}, b"Internal Server Error")

def main():
    ap = argparse.ArgumentParser(description="Tiny push-based image stream server (MJPEG).")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ImageStreamHandler)
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl+C to quit)")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Server stopped.")

if __name__ == "__main__":
    main()


