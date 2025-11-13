import asyncio
from datetime import datetime
from typing import Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

app = FastAPI(title="Push2View (SSE + latest)")

# --- Shared state ---
latest_frame: Optional[bytes] = None
latest_mime: str = "image/jpeg"
frame_seq: int = 0


client_queues: Set[asyncio.Queue] = set()
queues_lock = asyncio.Lock()


async def _broadcast(seq: int):
    async with queues_lock:
        dead = []
        for q in client_queues:
            if q.full():
                try:
                    q.get_nowait()
                    q.task_done()
                except Exception:
                    pass
            try:
                q.put_nowait(seq)
            except asyncio.QueueFull:
                pass
            except Exception:
                dead.append(q)
        for q in dead:
            client_queues.discard(q)


@app.post("/push")
async def push(request: Request):
    """
    Akzeptiert raw JPEG (application/octet-stream) und triggert ein Update-Signal.
    """
    global latest_frame, frame_seq, latest_mime

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    ct = request.headers.get("content-type", "application/octet-stream").lower()
    latest_mime = (
        "image/jpeg" if "jpeg" in ct or "jpg" in ct else "application/octet-stream"
    )

    latest_frame = body
    frame_seq += 1

    await _broadcast(frame_seq)

    return PlainTextResponse("ok", status_code=200)


@app.get("/latest")
async def latest(seq: Optional[int] = None):
    """
    Gibt das letzte Bild zurück. 'seq' dient nur zum Cache-Busting in der URL.
    """
    if latest_frame is None:
        raise HTTPException(status_code=404, detail="No frame yet")
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Seq": str(frame_seq),
        "X-Time": f"{datetime.utcnow().isoformat()}Z",
    }
    return Response(latest_frame, media_type=latest_mime, headers=headers)


@app.get("/events")
async def sse():
    """
    Server-Sent Events: schickt nur die aktuelle frame_seq bei Änderungen.
    Extrem leichtgewichtig.
    """
    q: asyncio.Queue[int] = asyncio.Queue(maxsize=1)

    async def gen():
        if latest_frame is not None:
            try:
                # initiale Nachricht
                yield f"data: {frame_seq}\n\n".encode("utf-8")
            except asyncio.CancelledError:
                return

        # registrieren
        async with queues_lock:
            client_queues.add(q)

        try:
            while True:
                seq = await q.get()
                try:
                    yield f"data: {seq}\n\n".encode("utf-8")
                finally:
                    q.task_done()
        except asyncio.CancelledError:
            pass
        finally:
            async with queues_lock:
                client_queues.discard(q)

    headers = {
        "Cache-Control": "no-store",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",  # Nginx: kein Buffering für SSE
    }
    return StreamingResponse(gen(), headers=headers, media_type="text/event-stream")


@app.get("/view")
async def view():
    """
    Simple Viewer: hält eine SSE-Verbindung und setzt das <img> nur bei neuen Frames neu.
    Kein Polling, minimale Last.
    """
    html = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Push2View – SSE</title>
  <style>
    :root {{ color-scheme: dark light; }}
    body {{
      margin:0; min-height:100vh; display:grid; place-items:center;
      background:#0b0b0b; color:#ddd; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial;
    }}
    .card {{ width:min(95vw,1200px); padding:16px; background:#161616; border-radius:16px; }}
    img#img {{ width:100%; height:auto; background:#000; border-radius:12px; outline:1px solid rgba(255,255,255,.05); }}
    .row {{ display:flex; justify-content:space-between; align-items:center; margin-top:10px; gap:12px; }}
    button {{
      border:none; border-radius:999px; padding:8px 14px; background:#1f7ae0; color:#fff; cursor:pointer;
    }}
    code {{ opacity:.8 }}
  </style>
</head>
<body>
  <div class="card">
    <h1 style="margin:0 0 12px 0; font-size:1.1rem; color:#9ad">Push2View – Live</h1>
    <img id="img" alt="Live Image"/>
    <div class="row">
      <div>Events: <code>/events</code> · Einzelbild: <code>/latest?seq=...</code></div>
      <div><button onclick="forceReload()">Neu verbinden</button></div>
    </div>
  </div>
  <script>
    const img = document.getElementById('img');
    let es;

    function connect() {{
      if (es) es.close();
      es = new EventSource('/events', {{ withCredentials: false }});
      es.onmessage = (ev) => {{
        const seq = ev.data.trim();
        // Cache-Busting mit seq
        const u = new URL('/latest', location.origin);
        u.searchParams.set('seq', seq);
        img.src = u.toString();
      }};
      es.onerror = () => {{
        // kurz warten, dann reconnect (EventSource reconnectt oft auch selbst)
        setTimeout(connect, 1000);
      }};
    }}

    function forceReload() {{
      connect();
    }}

    connect();
  </script>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/")
async def root():
    return HTMLResponse(
        '<!doctype html><meta charset="utf-8">'
        '<p>Öffne <a href="/view">/view</a>. Sende Frames an <code>POST /push</code> (raw JPEG bytes).</p>'
    )


if __name__ == "__main__":
    # EIN Worker, weil In-Memory-State. Port nach außen mappen: -p 8000:8000
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
