# img_stream_server.py
import io, time, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

LATEST = {"bytes": None, "ts": 0.0}
NEW_FRAME = threading.Event()

class Handler(BaseHTTPRequestHandler):
    server_version = "ImgStream/0.1"

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/view"):
            self._serve_html(); return
        if self.path.startswith("/latest"):
            self._serve_latest(); return
        if self.path.startswith("/stream"):
            self._serve_mjpeg(); return
        self.send_error(404, "Not found")

    def do_POST(self):
        # Push endpoint: POST /push  (Body = JPEG/PNG bytes)
        if not self.path.startswith("/push"):
            self.send_error(404, "Not found"); return
        cl = int(self.headers.get("Content-Length", "0"))
        if cl <= 0:
            self.send_error(411, "No Content-Length"); return
        data = self.rfile.read(cl)
        if not data:
            self.send_error(400, "Empty body"); return
        # Update global frame
        LATEST["bytes"] = data
        LATEST["ts"] = time.time()
        NEW_FRAME.set()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def _serve_html(self):
        html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Live Stream</title>
<style>
body{{background:#111;color:#eee;margin:0;font-family:system-ui,sans-serif}}
header{{padding:10px 14px;opacity:.85}}img{{display:block;margin:auto;max-width:98vw;max-height:92vh}}
small{{opacity:.7}}
</style></head><body>
<header>Live Stream &nbsp;|&nbsp; <small>/stream (MJPEG) or /latest</small></header>
<img src="/stream" alt="live stream"/>
</body></html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_latest(self):
        if not LATEST["bytes"]:
            self.send_error(503, "No frame yet"); return
        # Heuristik: Content-Type raten (default JPEG)
        ctype = "image/jpeg"
        if LATEST["bytes"][:8].startswith(b"\x89PNG"):
            ctype = "image/png"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(LATEST["bytes"])

    def _serve_mjpeg(self):
        boundary = "frameboundary"
        self.send_response(200)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()

        # Stream-Schleife: schickt jedes neue Frame sofort raus
        try:
            last_sent_ts = 0.0
            while True:
                if not LATEST["bytes"]:
                    # warten bis erstes Bild da ist
                    NEW_FRAME.wait(timeout=1.0)
                    NEW_FRAME.clear()
                    continue

                # Wenn kein neues Frame vorhanden ist, kurz warten
                if LATEST["ts"] <= last_sent_ts:
                    NEW_FRAME.wait(timeout=0.5)
                    NEW_FRAME.clear()
                    continue

                frame = LATEST["bytes"]
                last_sent_ts = LATEST["ts"]

                self.wfile.write(b"--" + boundary.encode() + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client hat geschlossen – Stream endet
            return

def serve(host="0.0.0.0", port=8000):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}  (Endpoints: /view, /stream, /latest, POST /push)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    serve()
