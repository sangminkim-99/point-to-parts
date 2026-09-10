"""Read-only Gaussian image panel, served separately from the Viser scene."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

PAGE = '''<!doctype html><meta charset="utf-8"><title>Gaussian rendering</title>
<style>body{background:#14181e;color:#eef2f7;font:15px system-ui;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}article{background:#202731;padding:14px;border-radius:12px}img{width:100%;image-rendering:auto}small{color:#aab7c8}h2{font-size:17px}</style>
<h1>Gaussian rendering</h1><p id="status">Enable Render diagnostics in Viser. Waiting for a render.</p>
<small>Read-only camera-view renders. Part images share the camera frame; black means no rendered contribution. Overall image uses approximate nearest-depth compositing. Held poses are not new observations.</small><main id="cards"></main>
<script>let previous='';async function update(){try{let s=await(await fetch('/state',{cache:'no-store'})).json();document.querySelector('#status').textContent=s.status;if(s.revision!==previous){previous=s.revision;let root=document.querySelector('#cards');root.replaceChildren();for(let item of s.images){let card=document.createElement('article'),title=document.createElement('h2'),img=document.createElement('img');title.textContent=item.title;img.src=item.src;card.append(title,img);root.append(card);}}}catch(e){document.querySelector('#status').textContent='Disconnected — last images may be stale.'}}setInterval(update,1000);update();</script>'''

class RenderDock:
    def __init__(self, port=0):
        self._lock = threading.Lock()
        self._state = dict(revision=0, status='Enable Render diagnostics in Viser. No render yet.', images=[])
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/state':
                    with owner._lock:
                        data = json.dumps(owner._state).encode()
                    mime = 'application/json'
                elif self.path == '/':
                    data, mime = PAGE.encode(), 'text/html; charset=utf-8'
                else:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data)
            def log_message(self, *args):
                pass
        try:
            self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        except OSError:
            self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}/'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, frame, images, status):
        encoded = []
        for title, rgb in images:
            if rgb is None:
                continue
            ok, blob = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if ok:
                encoded.append(dict(title=title, src='data:image/jpeg;base64,' + base64.b64encode(blob).decode()))
        with self._lock:
            self._state = dict(revision=self._state['revision'] + 1,
                               status=f'Render frame {frame} · {status}', images=encoded)

    def status(self, message):
        with self._lock:
            self._state['status'] = message

    def close(self):
        self.server.shutdown()
        self.server.server_close()
