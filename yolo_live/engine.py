"""Shared YOLO, appearance extraction, and optional OCR."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import time

from .appearance import AppearanceEncoder
from .direction import CLASSES, position


LOG = logging.getLogger("yolo_live")


def decode_jpeg(data):
    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(data)) as source:
        width, height = source.size
        if (
            source.format != "JPEG"
            or min(width, height) < 32
            or max(width, height) > 2048
            or width * height > 2_100_000
        ):
            raise ValueError(
                "Send a JPEG of at most 2048 pixels per side "
                "and 2.1 megapixels."
            )

        return np.asarray(source.convert("RGB"))[:, :, ::-1].copy()


class YOLOBackend:
    def __init__(self, weights, device, imgsz, ocr=None):
        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.ocr = ocr
        self.model = None
        self.appearance = AppearanceEncoder(device)

    def load(self):
        import numpy as np
        import torch
        from ultralytics import YOLOE

        if self.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Activate the working indoor-nav environment."
            )

        self.model = YOLOE(self.weights)
        self.model.set_classes(list(CLASSES))
        self.model.predict(
            np.zeros((480, 640, 3), dtype=np.uint8),
            device=self.device,
            imgsz=self.imgsz,
            conf=0.25,
            verbose=False,
        )

        LOG.info("Loading appearance model: %s", self.appearance.model_id)
        self.appearance.load()

        if self.ocr is not None:
            LOG.info("Loading and warming up OCR...")
            self.ocr.load()

    def infer(self, jpeg, confidence, run_ocr=False, target=None):
        started = time.perf_counter()
        image = decode_jpeg(jpeg)
        height, width = image.shape[:2]

        result = self.model.predict(
            source=image,
            device=self.device,
            imgsz=self.imgsz,
            conf=confidence,
            max_det=60,
            verbose=False,
        )[0]

        detections = []
        for row in result.boxes.data.cpu().tolist():
            x1, y1, x2, y2, score, class_id = row[:6]
            box = [
                max(0, min(1, x1 / width)),
                max(0, min(1, y1 / height)),
                max(0, min(1, x2 / width)),
                max(0, min(1, y2 / height)),
            ]
            detections.append({
                "label": result.names[int(class_id)],
                "confidence": round(score, 4),
                "box": box,
                "position": position(box),
            })

        appearance_started = time.perf_counter()
        features = self.appearance.encode(image, detections, target)
        appearance_ms = (time.perf_counter() - appearance_started) * 1000

        response = {
            "detections": detections,
            "width": width,
            "height": height,
            "detector_ms": round((time.perf_counter() - started) * 1000, 1),
            "model_inference_ms": round(
                float(result.speed.get("inference", 0)), 1,
            ),
            "appearance_ms": round(appearance_ms, 1),

            # Server-only data: removed before the WebSocket response.
            "_appearance": features,
            "ocr": {
                "status": "skipped" if self.ocr is not None else "disabled",
            },
        }

        if run_ocr and self.ocr is not None:
            try:
                response["ocr"] = self.ocr.infer(image, detections)
            except Exception:
                LOG.exception("OCR failed; returning YOLO detections")
                response["ocr"] = {
                    "status": "error",
                    "message": "OCR failed. Check the GB10 terminal.",
                }

        return response


class SharedDetector:
    def __init__(self, backend):
        self.backend = backend
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="yolo-gpu",
        )

    async def load(self):
        await asyncio.get_running_loop().run_in_executor(
            self.executor, self.backend.load,
        )

    async def try_infer(
        self, jpeg, confidence, run_ocr=False, target=None,
    ):
        if self.lock.locked():
            return None

        async with self.lock:
            job = asyncio.get_running_loop().run_in_executor(
                self.executor,
                self.backend.infer,
                jpeg,
                confidence,
                run_ocr,
                target,
            )
            try:
                return await asyncio.shield(job)
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(job)
                finally:
                    raise

    async def close(self):
        await asyncio.to_thread(
            self.executor.shutdown,
            wait=True,
            cancel_futures=True,
        )