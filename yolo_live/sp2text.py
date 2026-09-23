"""Local speech transcription and an aiohttp voice endpoint."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import os
import time
from urllib.parse import urlsplit

from aiohttp import web

from .targets import interpret_text


LOG = logging.getLogger("yolo_live.voice")
VOICE = web.AppKey("voice_service", object)

SAMPLE_RATE = 16000
MAX_SECONDS = 12
MAX_BYTES = 2_000_000

FORMATS = {
    "audio/webm": "matroska",
    "audio/ogg": "ogg",
    "audio/mp4": "mov",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


class VoiceBusy(Exception):
    pass


def decode_audio(payload, container_format):
    import av
    import numpy as np

    pieces = []
    samples = 0
    decoded_seconds = 0.0

    def append(frame):
        nonlocal samples
        values = frame.to_ndarray().reshape(-1)
        samples += values.size
        if samples > MAX_SECONDS * SAMPLE_RATE:
            raise ValueError("Record a command shorter than 12 seconds.")
        pieces.append(values.copy())

    try:
        with av.open(BytesIO(payload), format=container_format) as container:
            if not container.streams.audio:
                raise ValueError("The upload contains no audio.")

            resampler = av.AudioResampler(
                format="fltp", layout="mono", rate=SAMPLE_RATE,
            )

            for frame in container.decode(audio=0):
                if not frame.sample_rate or frame.sample_rate > 96000:
                    raise ValueError("Unsupported audio sample rate.")
                if len(frame.layout.channels) > 2:
                    raise ValueError("Send mono or stereo microphone audio.")

                decoded_seconds += frame.samples / frame.sample_rate
                if decoded_seconds > MAX_SECONDS:
                    raise ValueError("Record a command shorter than 12 seconds.")

                for converted in resampler.resample(frame):
                    append(converted)

            for converted in resampler.resample(None):
                append(converted)

    except av.error.FFmpegError as error:
        raise ValueError("The microphone recording could not be decoded.") from error

    if not pieces:
        raise ValueError("The microphone recording is empty.")

    audio = np.concatenate(pieces).astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        raise ValueError("The recording contains invalid audio samples.")

    return audio


class SpeechBackend:
    def __init__(self):
        self.model_id = os.environ.get(
            "VOICE_MODEL", "openai/whisper-large-v3-turbo",
        )
        self.device = os.environ.get("VOICE_DEVICE", "cuda:0")

    def load(self):
        import torch
        from silero_vad import load_silero_vad
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Voice CUDA is unavailable. Use the working GB10 environment "
                "or set VOICE_DEVICE=cpu."
            )

        self.dtype = (
            torch.float16 if self.device.startswith("cuda") else torch.float32
        )

        LOG.info("Loading local speech models: %s", self.model_id)
        self.vad = load_silero_vad().eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
            attn_implementation="sdpa",
        ).to(self.device).eval()
        LOG.info("Speech models ready on %s", self.device)

    def interpret(self, payload, container_format, language):
        import torch
        from silero_vad import get_speech_timestamps

        started = time.perf_counter()
        audio = decode_audio(payload, container_format)

        with torch.inference_mode():
            spans = get_speech_timestamps(
                torch.from_numpy(audio),
                self.vad,
                sampling_rate=SAMPLE_RATE,
                min_speech_duration_ms=100,
                min_silence_duration_ms=300,
                speech_pad_ms=150,
            )

            if not spans:
                result = interpret_text("")
            else:
                # Trim only the leading/trailing silence. Preserve internal
                # pauses so a correction or negation is not split away.
                audio = audio[spans[0]["start"]:spans[-1]["end"]]

                features = self.processor(
                    audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                features = {
                    key: value.to(
                        device=self.device,
                        dtype=self.dtype if value.is_floating_point() else value.dtype,
                    )
                    for key, value in features.items()
                }

                tokens = self.model.generate(
                    **features,
                    task="transcribe",
                    language=None if language == "auto" else "en",
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=128,
                    condition_on_prev_tokens=False,
                )

                transcript = self.processor.batch_decode(
                    tokens, skip_special_tokens=True,
                )[0].strip()
                result = interpret_text(transcript)

        result["language_mode"] = language
        result["processing_ms"] = round(
            (time.perf_counter() - started) * 1000, 1,
        )
        return result


class VoiceService:
    def __init__(self):
        self.backend = SpeechBackend()
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="speech",
        )

    async def load(self):
        await asyncio.get_running_loop().run_in_executor(
            self.executor, self.backend.load,
        )

    async def interpret(self, payload, container_format, language):
        if self.lock.locked():
            raise VoiceBusy()

        async with self.lock:
            job = asyncio.get_running_loop().run_in_executor(
                self.executor,
                self.backend.interpret,
                payload,
                container_format,
                language,
            )
            try:
                return await asyncio.shield(job)
            except asyncio.CancelledError:
                # Cancelling a request cannot stop an active GPU operation.
                # Keep ownership until it actually finishes.
                try:
                    await asyncio.shield(job)
                finally:
                    raise

    async def close(self):
        await asyncio.to_thread(
            self.executor.shutdown, wait=True, cancel_futures=True,
        )


def json_reply(payload, status=200):
    return web.json_response(
        payload, status=status, headers={"Cache-Control": "no-store"},
    )


async def interpret_request(request):
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc.lower() != request.host.lower():
        raise web.HTTPForbidden(text="Use the microphone on this app's page.")

    # Custom header plus no CORS approval prevents cross-origin simple forms
    # from submitting recordings to this endpoint.
    if request.headers.get("X-Voice-Request") != "1":
        raise web.HTTPForbidden(text="Missing voice request header.")

    request_id = request.headers.get("X-Request-ID", "")
    if not request_id or len(request_id) > 80:
        raise web.HTTPBadRequest(text="Invalid voice request ID.")

    language = request.query.get("language", "en")
    if language not in ("en", "auto"):
        raise web.HTTPBadRequest(text="Choose English or automatic transcription.")

    container_format = FORMATS.get(request.content_type)
    if container_format is None:
        return json_reply({
            "request_id": request_id,
            "status": "invalid_audio",
            "message": "Use a WebM, Ogg, MP4, or WAV microphone recording.",
        }, 415)

    try:
        payload = bytearray()
        async for chunk in request.content.iter_chunked(65536):
            payload.extend(chunk)
            if len(payload) > MAX_BYTES:
                return json_reply({
                    "request_id": request_id,
                    "status": "invalid_audio",
                    "message": "The recording is too large. Use a shorter command.",
                }, 413)

        if not payload:
            raise ValueError("The recording is empty.")

        result = await request.app[VOICE].interpret(
            bytes(payload), container_format, language,
        )
        return json_reply({"request_id": request_id, **result})

    except VoiceBusy:
        return json_reply({
            "request_id": request_id,
            "status": "busy",
            "message": "Speech processing is busy. Please try again shortly.",
        }, 429)
    except ValueError as error:
        return json_reply({
            "request_id": request_id,
            "status": "invalid_audio",
            "message": str(error),
        }, 400)
    except Exception:
        LOG.exception("Speech processing failed")
        return json_reply({
            "request_id": request_id,
            "status": "error",
            "message": "Speech processing failed. Check the server terminal.",
        }, 500)


def install_voice(app):
    app[VOICE] = VoiceService()
    app.router.add_post("/api/voice/interpret", interpret_request)

    async def lifecycle(app):
        try:
            await app[VOICE].load()
            yield
        finally:
            await app[VOICE].close()

    app.cleanup_ctx.append(lifecycle)