import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .cli import episode_tag, mux, output_path
from .ffmpeg import require_tools
from .sync import analyze_files

ROOT = Path(__file__).parent / "web"
# /tmp is often tmpfs, so multi-GB uploads would eat RAM and swap.
TMP_DIR = os.environ.get("AUDIOSYNC_TMP", "/var/tmp")


def _tmp(suffix: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=suffix, dir=TMP_DIR)
    os.close(fd)
    return Path(name)


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix
    suffix = "".join(c for c in suffix if c.isalnum() or c in (".", "-", "_"))[:16]
    if not suffix.startswith(".") or len(suffix) < 2:
        return ".bin"
    return suffix


def _field_name(disposition: str) -> tuple[str | None, str | None]:
    name = re.search(r'name="([^"]*)"', disposition)
    filename = re.search(r'filename="([^"]*)"', disposition)
    star = re.search(r"filename\*\s*=\s*([^;\s]+)", disposition)
    result = filename.group(1) if filename else None
    if result is None and star:
        value = star.group(1).strip("'\"")
        if "''" in value:
            _, _, value = value.partition("''")
        result = urllib.parse.unquote(value)
    return (name.group(1) if name else None), (result or None)


def parse_upload(request):
    # Streams file parts straight to disk in chunks so GB uploads never sit in RAM.
    content_type = request.headers.get("Content-Type", "")
    if not content_type.startswith("multipart/form-data;") or "boundary=" not in content_type:
        raise ValueError("Expected multipart/form-data upload")
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    if not boundary:
        raise ValueError("Missing multipart boundary")
    delimiter = b"--" + boundary.encode()
    try:
        length = int(request.headers.get("Content-Length", 0))
    except ValueError:
        raise ValueError("Invalid Content-Length")
    if length <= 0:
        raise ValueError("Empty upload")

    remaining = length
    buf = bytearray()
    created: list[Path] = []

    def fill(min_bytes: int = 1) -> None:
        nonlocal remaining
        while len(buf) < min_bytes and remaining > 0:
            chunk = request.rfile.read(min(65536, remaining))
            if not chunk:
                break
            buf.extend(chunk)
            remaining -= len(chunk)

    def readline() -> bytes:
        while True:
            index = buf.find(b"\r\n")
            if index >= 0:
                line = bytes(buf[:index + 2])
                del buf[:index + 2]
                return line
            if remaining <= 0:
                if buf:
                    line = bytes(buf)
                    buf.clear()
                    return line
                raise ValueError("Truncated upload")
            fill(len(buf) + 65536)

    try:
        fill(2)
        first = readline()
        if not first.startswith(delimiter):
            raise ValueError("Malformed multipart body")

        files: dict[str, tuple[str, Path]] = {}
        tail = len(delimiter) + 8
        while True:
            headers: list[str] = []
            while True:
                line = readline().decode("latin-1")
                if line in ("", "\r\n"):
                    break
                headers.append(line)
            disposition = next((h for h in headers if h.lower().startswith("content-disposition:")), "")
            name, filename = _field_name(disposition)
            if name and filename:
                path = _tmp(_safe_suffix(filename))
                created.append(path)
                with path.open("wb") as handle:
                    while True:
                        fill(tail + 1)
                        index = buf.find(b"\r\n" + delimiter)
                        if index >= 0:
                            handle.write(buf[:index])
                            del buf[:index + 2]
                            break
                        if len(buf) > tail:
                            handle.write(buf[:-tail])
                            del buf[:-tail]
                        if remaining <= 0 and not buf:
                            raise ValueError("Truncated upload")
                        if remaining <= 0:
                            # No boundary found before EOF.
                            raise ValueError("Truncated upload")
                files[name] = (filename, path)
            else:
                # Skip non-file fields without buffering them fully.
                while True:
                    fill(tail + 1)
                    index = buf.find(b"\r\n" + delimiter)
                    if index >= 0:
                        del buf[:index + 2]
                        break
                    if len(buf) > tail:
                        del buf[:-tail]
                    if remaining <= 0:
                        raise ValueError("Truncated upload")
            line = readline()
            if line.startswith(delimiter + b"--"):
                break
            if not line.startswith(delimiter):
                raise ValueError("Malformed multipart body")
        return files
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        # Closing with unread body makes browser report a network error instead of our JSON.
        while remaining > 0:
            chunk = request.rfile.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        raise


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else body.encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/favicon.ico":
            self._send(204, b"")
            return
        self._send(404, json.dumps({"error": "Not found"}))

    def do_POST(self):
        target = source = None
        files: dict = {}
        try:
            require_tools()
            path = urllib.parse.urlparse(self.path).path
            files = parse_upload(self)
            target_name, target = files["target"]
            source_name, source = files["source"]
            target_tag, source_tag = episode_tag(target_name), episode_tag(source_name)
            if target_tag is not None and source_tag is not None and target_tag != source_tag:
                raise ValueError(f"Episode mismatch: HQ is {target_tag} but dubbed is {source_tag}")
            report = analyze_files(target, source)
            if path == "/api/analyze":
                self._send(200, json.dumps(report))
                return
            if path == "/api/mux":
                # Name the output after the HQ upload: S02E03 in the name
                # becomes muxes/S02E03.mkv, otherwise muxes/<hq>.dubbed.mkv.
                output = output_path(Path(target_name), None)
                mux(target, source, output, report["offset"])
                self._send(200, json.dumps({"report": report, "saved": f"muxes/{output.name}"}))
                return
            self._send(404, json.dumps({"error": "Not found"}))
        except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
            self._send(400, json.dumps({"error": str(error)}))
        except Exception as error:
            self._send(500, json.dumps({"error": "Request failed: " + str(error)}))
        finally:
            seen = set()
            for value in list(files.values()) + [target, source]:
                path = value[1] if isinstance(value, tuple) else value
                if path and path not in seen:
                    seen.add(path)
                    Path(path).unlink(missing_ok=True)

    def log_message(self, *_):
        return


class Server(ThreadingHTTPServer):
    # Client aborts (tab closed, fetch cancelled, favicon prefetch) surface as
    # BrokenPipe/ConnectionReset inside handler threads. Not worth a traceback.
    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main():
    server = Server(("127.0.0.1", 8765), Handler)
    print("AudioSync Web UI: http://127.0.0.1:8765", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
