import json
import os
import sys
import threading
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import io
import base64
from datetime import datetime

WHITEBOARD_DIR = os.path.join(os.path.dirname(__file__), "whiteboard")
DISCORD_WEBHOOK_URL = os.environ.get("WHITEBOARD_WEBHOOK_URL", "")
WHITEBOARD_PORT = int(os.environ.get("WHITEBOARD_PORT", "8080"))

rooms = {}

class WhiteboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/whiteboard":
            self._serve_static("index.html", "text/html; charset=utf-8")
        elif path == "/style.css":
            self._serve_static("style.css", "text/css; charset=utf-8")
        elif path == "/script.js":
            self._serve_static("script.js", "application/javascript; charset=utf-8")
        elif path == "/api/rooms":
            self._send_json({"rooms": list(rooms.keys())})
        elif path.startswith("/whiteboard/") and len(path.split("/")) == 3:
            room = path.split("/")[2]
            self._serve_static("index.html", "text/html; charset=utf-8")
        else:
            self._serve_static(path.lstrip("/"), self._guess_type(path))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/create_room":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len else b"{}"
            data = json.loads(body) if body else {}
            room_name = data.get("name", f"room-{uuid.uuid4().hex[:6]}")
            rooms[room_name] = {"created": datetime.now().isoformat(), "clients": 0}
            self._send_json({"room": room_name, "url": f"/whiteboard/{room_name}"})

        elif path == "/api/save":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len else b"{}"
            data = json.loads(body) if body else {}
            image_data = data.get("image", "")
            room = data.get("room", "default")

            if not image_data:
                self._send_json({"success": False, "error": "No image data"}, 400)
                return

            if not DISCORD_WEBHOOK_URL:
                self._send_json({"success": False, "error": "Discord webhook not configured"}, 500)
                return

            try:
                self._post_to_discord(image_data, room)
                self._send_json({"success": True, "message": "Drawing saved to Discord!"})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, 500)

        else:
            self._send_json({"error": "Not found"}, 404)

    def _serve_static(self, filename, content_type):
        filepath = os.path.join(WHITEBOARD_DIR, "static", filename)
        if not os.path.exists(filepath):
            self.send_error(404, "File not found")
            return
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _guess_type(self, path):
        ext = path.split(".")[-1].lower() if "." in path else ""
        return {
            "html": "text/html; charset=utf-8",
            "css": "text/css; charset=utf-8",
            "js": "application/javascript; charset=utf-8",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "svg": "image/svg+xml",
            "ico": "image/x-icon",
            "json": "application/json",
        }.get(ext, "application/octet-stream")

    def _post_to_discord(self, image_data_url, room):
        import urllib.request
        image_data = base64.b64decode(image_data_url.split(",")[1] if "," in image_data_url else image_data_url)
        boundary = uuid.uuid4().hex
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="content"\r\n\r\n'
            f"New drawing from **{room}** at {ts}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="drawing.png"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + image_data + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=15)

    def log_message(self, format, *args):
        print(f"[WHITEBOARD] {args[0]} {args[1]} {args[2]}")


def run_server(port=None, webhook_url=None):
    global DISCORD_WEBHOOK_URL, WHITEBOARD_PORT
    if webhook_url:
        DISCORD_WEBHOOK_URL = webhook_url
    if port:
        WHITEBOARD_PORT = port
    server = HTTPServer(("0.0.0.0", WHITEBOARD_PORT), WhiteboardHandler)
    print(f"[WHITEBOARD] Server running at http://0.0.0.0:{WHITEBOARD_PORT}")
    print(f"[WHITEBOARD] Webhook: {'configured' if DISCORD_WEBHOOK_URL else 'NOT configured'}")
    server.serve_forever()


def start_server_thread(port=None, webhook_url=None):
    t = threading.Thread(target=run_server, args=(port, webhook_url), daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    run_server()
