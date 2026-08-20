import json
import subprocess
import tempfile
import urllib.parse
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .ffmpeg import require_tools
from .sync import analyze_files

ROOT = Path(__file__).parent / "web"
OUTPUTS = {}


def parse_upload(request):
    content_type = request.headers.get("Content-Type", "")
    if not content_type.startswith("multipart/form-data;") or "boundary=" not in content_type:
        raise ValueError("Expected multipart/form-data upload")
    body = request.rfile.read(int(request.headers.get("Content-Length", 0)))
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )
    files = {}
    for part in message.iter_attachments():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        if name and filename:
            files[name] = (filename, part.get_payload(decode=True))
    return files


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path.startswith("/api/download/"):
            name = path.removeprefix("/api/download/")
            output = OUTPUTS.pop(name, None)
            if output and output.exists():
                self._send(200, output.read_bytes(), "video/x-matroska")
                output.unlink(missing_ok=True)
                return
        self._send(404, json.dumps({"error": "Not found"}))

    def do_POST(self):
        target = source = None
        try:
            require_tools()
            path = urllib.parse.urlparse(self.path).path
            files = parse_upload(self)
            target_name, target_data = files["target"]
            source_name, source_data = files["source"]
            target = Path(tempfile.mkstemp(suffix=Path(target_name).suffix)[1])
            source = Path(tempfile.mkstemp(suffix=Path(source_name).suffix)[1])
            target.write_bytes(target_data)
            source.write_bytes(source_data)
            report = analyze_files(target, source)
            if path == "/api/analyze":
                self._send(200, json.dumps(report))
                return
            if path == "/api/mux":
                output = Path(tempfile.mkstemp(suffix=".mkv")[1])
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(target), "-itsoffset", str(-report["offset"]), "-i", str(source), "-map", "0:v?", "-map", "0:s?", "-map", "0:d?", "-map", "1:a:0", "-map_metadata", "0", "-c:v", "copy", "-c:s", "copy", "-c:d", "copy", "-c:a", "aac", "-shortest", str(output)], check=True)
                OUTPUTS[output.name] = output
                self._send(200, json.dumps({"report": report, "download": f"/api/download/{output.name}"}))
                return
            self._send(404, json.dumps({"error": "Not found"}))
        except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
            self._send(400, json.dumps({"error": str(error)}))
        except Exception as error:
            self._send(500, json.dumps({"error": "Request failed: " + str(error)}))
        finally:
            for path in (target, source):
                if path:
                    path.unlink(missing_ok=True)

    def log_message(self, *_):
        return


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("AudioSync Web UI: http://127.0.0.1:8765", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
