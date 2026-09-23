"""HTTPS/WebSocket camera server. Does not start ZRT, Qwen, or a cloud API."""
import argparse
import asyncio
import json
import logging
import math
import os
from pathlib import Path
import ssl
import subprocess
import time
from urllib.parse import urlsplit

from aiohttp import web, WSMsgType

from .direction import AlignmentController, CLASSES
from .engine import SharedDetector, YOLOBackend

LOG = logging.getLogger("yolo_live")
STATIC = Path(__file__).resolve().parent / "static"
DETECTOR = web.AppKey("detector", SharedDetector)
SOCKETS = web.AppKey("sockets", set)
INFO = web.AppKey("info", dict)


async def index(request):
    return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


async def health(request):
    return web.json_response({"ready": True, **request.app[INFO]})


def parse_settings(data):
    target = data.get("target", "whiteboard")
    confidence = float(data.get("confidence", 0.25))
    revision = data.get("revision")
    if target not in CLASSES:
        raise ValueError("Select one of the configured target classes.")
    if not math.isfinite(confidence) or not 0.1 <= confidence <= 0.9:
        raise ValueError("Confidence must be between 0.1 and 0.9.")
    if type(revision) is not int or revision < 0:
        raise ValueError("Invalid configuration revision.")
    return target, confidence, revision


async def camera_socket(request):
    # Camera clients originate from this page, including localhost through SSH forwarding.
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc.lower() != request.host.lower():
        raise web.HTTPForbidden(text="Open the camera page on this same host and port.")
    if len(request.app[SOCKETS]) >= 4:
        raise web.HTTPServiceUnavailable(text="Four camera sessions are already connected.")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=2_500_000)
    await ws.prepare(request)
    request.app[SOCKETS].add(ws)
    controller = AlignmentController()
    confidence, revision, metadata, count = 0.25, -1, None, 0
    try:
        await ws.send_json({"type": "ready", **request.app[INFO], "classes": CLASSES})
        async for message in ws:
            if message.type == WSMsgType.TEXT:
                try:
                    data = json.loads(message.data)
                    if not isinstance(data, dict):
                        raise ValueError("Expected an object.")
                    if data.get("type") == "configure":
                        target, confidence, revision = parse_settings(data)
                        controller = AlignmentController(target=target)
                        metadata = None
                        await ws.send_json({"type": "configured", "revision": revision})
                    elif data.get("type") == "frame":
                        if type(data.get("id")) is not int or data["id"] < 1:
                            raise ValueError("Invalid frame number.")
                        metadata = {"id": data["id"], "revision": data.get("revision")}
                    else:
                        raise ValueError("Unknown message type.")
                except (ValueError, TypeError) as error:
                    metadata = None
                    await ws.send_json({"type": "error", "message": str(error)})
            elif message.type == WSMsgType.BINARY:
                frame, metadata = metadata, None
                if frame is None or revision < 0:
                    await ws.send_json({"type": "error", "message": "Configure the camera before sending a frame."})
                    continue
                if frame["revision"] != revision:
                    await ws.send_json({"type": "discarded", **frame})
                    continue
                started = time.perf_counter()
                try:
                    result = await request.app[DETECTOR].try_infer(message.data, confidence)
                    if result is None:
                        await ws.send_json({"type": "busy", **frame})
                        continue
                    direction = controller.update(result["detections"], time.monotonic())
                    count += 1
                    LOG.info("Frame %s | %s | %s | %.1f ms", frame["id"], controller.target,
                             direction["text"], result["detector_ms"])
                    await ws.send_json({"type": "result", **frame, **result, "direction": direction,
                                        "count": count, "server_ms": round((time.perf_counter() - started) * 1000, 1)})
                except (ValueError, OSError) as error:
                    await ws.send_json({"type": "error", **frame, "message": str(error)})
                except Exception:
                    LOG.exception("YOLO inference failed")
                    await ws.send_json({"type": "error", **frame,
                                        "message": "YOLO inference failed. Check the GB10 terminal for details."})
            elif message.type == WSMsgType.ERROR:
                LOG.warning("Camera socket error: %s", ws.exception())
    finally:
        request.app[SOCKETS].discard(ws)
    return ws


def create_app(detector, info=None):
    app = web.Application(client_max_size=2_500_000)
    app[DETECTOR], app[SOCKETS], app[INFO] = detector, set(), info or {}
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/ws", camera_socket)
    app.router.add_static("/static/", STATIC, show_index=False)

    async def lifetime(app):
        try:
            await app[DETECTOR].load()
            LOG.info("YOLO is ready. Open the URL printed below; allow browser camera access.")
            yield
        finally:
            await app[DETECTOR].close()

    async def shutdown(app):
        await asyncio.gather(*(ws.close(code=1001, message=b"Server shutting down")
                               for ws in list(app[SOCKETS])), return_exceptions=True)

    app.cleanup_ctx.append(lifetime)
    app.on_shutdown.append(shutdown)
    return app


def ssl_context(cert_arg=None, key_arg=None):
    if bool(cert_arg) != bool(key_arg):
        raise ValueError("Supply both --cert and --key.")
    if cert_arg:
        cert, key = Path(cert_arg).expanduser(), Path(key_arg).expanduser()
    else:
        config = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "live-yolo-webui"
        config.mkdir(parents=True, exist_ok=True)
        cert, key = config / "cert.pem", config / "key.pem"
        if not (cert.is_file() and key.is_file()):
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                            "-keyout", str(key), "-out", str(cert), "-days", "365",
                            "-subj", "/CN=localhost", "-addext",
                            "subjectAltName=DNS:localhost,IP:127.0.0.1"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            key.chmod(0o600)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert), str(key))
    return context


def main():
    parser = argparse.ArgumentParser(description="Live YOLO-World camera alignment on your existing GPU.")
    parser.add_argument("--host", default="127.0.0.1", help="Default: local + SSH port forwarding")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--weights", default="yolov8s-worldv2.pt", help="Existing Ultralytics .pt file or asset name")
    parser.add_argument("--device", default="0", help="CUDA index (default 0), or cpu for diagnostics")
    parser.add_argument("--imgsz", type=int, choices=(320, 480, 640, 960, 1280), default=640)
    parser.add_argument("--cert")
    parser.add_argument("--key")
    parser.add_argument("--http", action="store_true", help="Plain HTTP, loopback only")
    args = parser.parse_args()
    if args.http and args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("--http is limited to loopback. Use HTTPS for access from another device.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    try:
        context = None if args.http else ssl_context(args.cert, args.key)
        device = int(args.device) if args.device.isdecimal() else args.device
        detector = SharedDetector(YOLOBackend(str(Path(args.weights).expanduser()), device, args.imgsz))
        app = create_app(detector, {"model": Path(args.weights).name, "device": str(device), "imgsz": args.imgsz})
        LOG.info("Loading YOLO once and warming up. First use may download weights/text-encoder assets.")
        LOG.info("Browser: %s://localhost:%d (forward this port from GB10)",
                 "http" if args.http else "https", args.port)
        web.run_app(app, host=args.host, port=args.port, ssl_context=context, access_log=None)
    except (OSError, RuntimeError, ImportError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(f"Could not start the YOLO UI: {error}") from error
