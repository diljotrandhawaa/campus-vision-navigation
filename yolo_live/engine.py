"""One shared model, serialized GPU calls, and no inference queue."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import time

from .direction import CLASSES, position


def decode_jpeg(data):
    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(data)) as source:
        w, h = source.size
        if source.format != "JPEG" or min(w, h) < 32 or max(w, h) > 2048 or w * h > 2_100_000:
            raise ValueError("Send a JPEG of at most 2048 pixels per side and 2.1 megapixels.")
        return np.asarray(source.convert("RGB"))[:, :, ::-1].copy()  # Ultralytics: BGR


class YOLOBackend:
    def __init__(self, weights, device, imgsz):
        self.weights, self.device, self.imgsz = weights, device, imgsz
        self.model = None

    def load(self):
        import numpy as np
        import torch
        from ultralytics import YOLOE

        if self.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment. Activate your "
                               "working indoor-nav environment and check its PyTorch installation.")
        self.model = YOLOE(self.weights)
        self.model.set_classes(list(CLASSES))
        # Warm up before accepting a camera connection; surfaces unsupported CUDA builds.
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device=self.device,
                           imgsz=self.imgsz, conf=0.25, verbose=False)

    def infer(self, jpeg, confidence):
        started = time.perf_counter()
        image = decode_jpeg(jpeg)
        height, width = image.shape[:2]
        result = self.model.predict(source=image, device=self.device, imgsz=self.imgsz,
                                    conf=confidence, max_det=60, verbose=False)[0]
        detections = []
        # Moving results to CPU also waits for the GPU's results before the wall timer ends.
        for row in result.boxes.data.cpu().tolist():
            x1, y1, x2, y2, score, class_id = row[:6]
            box = [max(0, min(1, x1 / width)), max(0, min(1, y1 / height)),
                   max(0, min(1, x2 / width)), max(0, min(1, y2 / height))]
            detections.append({"label": result.names[int(class_id)], "confidence": round(score, 4),
                               "box": box, "position": position(box)})
        return {"detections": detections, "width": width, "height": height,
                "detector_ms": round((time.perf_counter() - started) * 1000, 1),
                "model_inference_ms": round(float(result.speed.get("inference", 0)), 1)}


class SharedDetector:
    def __init__(self, backend):
        self.backend = backend
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-gpu")

    async def load(self):
        await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.load)

    async def try_infer(self, jpeg, confidence):
        if self.lock.locked():
            return None  # Other tabs retry with a NEW frame, rather than queue old frames.
        async with self.lock:
            job = asyncio.get_running_loop().run_in_executor(
                self.executor, self.backend.infer, jpeg, confidence)
            try:
                return await asyncio.shield(job)
            except asyncio.CancelledError:
                # Cancelling an HTTP handler cannot cancel a running GPU operation.
                # Keep ownership until the thread finishes, so another tab can't overlap it.
                try:
                    await job
                finally:
                    raise

    async def close(self):
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
