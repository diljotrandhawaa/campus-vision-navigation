"use strict";
const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("result"), context = canvas.getContext("2d");
const MAX_AGE_MS = 1500;
let generation = 0, running = false, socket = null, stream = null, pending = null;
let revision = 0, configuredRevision = -1, frameId = 0, nextTimer = null, frameCallback = null;
let lastCameraFrame = 0, lastVideoTime = -1, lastCapture = 0, lastResult = null;
let sampleCount = 0, latencyTotal = 0, arrivals = [], hasSnapshot = false;

function status(text, active = false) {
  $("status").textContent = text;
  $("status").classList.toggle("active", active);
}
function instruction(state, text, detail = "") {
  $("direction-card").dataset.state = state;
  $("instruction").textContent = text;
  $("direction-icon").textContent = ({left: "←", right: "→", centered: "◎", lost: "⌕", stale: "◷"})[state] || "◎";
  $("target-detail").textContent = detail || `Following: ${$("target").value}`;
  if (!["left", "right", "centered"].includes(state)) {
    $("marker").hidden = true;
    $("offset").textContent = "—";
  }
}
function clearAnalysis(text = "Waiting for a fresh analysis") {
  lastResult = null;
  $("age").textContent = "—";
  instruction("stale", text);
  canvas.classList.add("stale");
  $("expired").hidden = !hasSnapshot;
  $("expired").textContent = text;
  $("save").disabled = true;
  $("detections").replaceChildren();
  $("object-count").textContent = "—";
  $("frame-state").textContent = text;
}
function stop(message = "Camera stopped") {
  ++generation;
  running = false;
  clearTimeout(nextTimer);
  if (frameCallback !== null && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(frameCallback);
  frameCallback = null;
  const oldSocket = socket;
  socket = null;
  if (oldSocket) oldSocket.close();
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  $("preview-wrap").hidden = true;
  pending = null;
  configuredRevision = -1;
  $("start").disabled = false;
  $("stop").disabled = true;
  clearAnalysis(message);
  instruction("idle", message);
  status(message);
  $("fps").textContent = "—";
}

async function refreshCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const selected = stream?.getVideoTracks()[0]?.getSettings().deviceId || $("camera").value;
  const options = [new Option("Default camera", "")];
  devices.filter(d => d.kind === "videoinput").forEach((d, i) => {
    options.push(new Option(d.label || `Camera ${i + 1}`, d.deviceId));
  });
  $("camera").replaceChildren(...options);
  if (options.some(o => o.value === selected)) $("camera").value = selected;
}

function watchVideoFrames(token) {
  if (!video.requestVideoFrameCallback) return;
  frameCallback = video.requestVideoFrameCallback(() => {
    if (!running || token !== generation) return;
    lastCameraFrame = performance.now();
    watchVideoFrames(token);
  });
}

function configure() {
  revision++;
  clearAnalysis("Finding target…");
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({
    type: "configure", target: $("target").value,
    confidence: Number($("confidence").value), revision
  }));
}

async function start() {
  stop();
  const token = generation;
  running = true;
  $("start").disabled = true;
  $("stop").disabled = false;
  $("error").textContent = "";
  status("Opening camera…");
  instruction("idle", "Opening camera…");
  sampleCount = 0; latencyTotal = 0; arrivals = []; frameId = 0;
  lastCameraFrame = 0; lastVideoTime = -1; lastCapture = 0;
  for (const key of ["latency", "average", "detector", "fps", "age"]) $(key).textContent = "—";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access requires HTTPS or http://localhost. Open the forwarded local address.");
    const choice = $("camera").value;
    const cameraStream = await navigator.mediaDevices.getUserMedia({audio: false, video: {
      ...(choice ? {deviceId: {exact: choice}} : {}), width: {ideal: 1280},
      height: {ideal: 720}, frameRate: {ideal: 30, max: 30}
    }});
    if (token !== generation) { cameraStream.getTracks().forEach(t => t.stop()); return; }
    stream = cameraStream;
    video.srcObject = stream;
    stream.getVideoTracks()[0].addEventListener("ended", () => {
      if (token === generation) stop("Camera disconnected");
    });
    await video.play();
    if (token !== generation) return;
    $("preview-wrap").hidden = false;
    watchVideoFrames(token);
    refreshCameras().catch(() => {});
    status("Connecting to YOLO…");
    const connection = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`);
    socket = connection;
    connection.addEventListener("message", event => {
      if (token !== generation) return;
      try { handleMessage(JSON.parse(event.data), token); }
      catch (error) { $("error").textContent = error.message; stop("Invalid server response"); }
    });
    connection.addEventListener("close", () => {
      if (token === generation) {
        $("error").textContent = "Connection closed. Check the YOLO terminal and port forwarding, then start again.";
        stop("Disconnected");
      }
    });
    connection.addEventListener("error", () => {
      if (token === generation) $("error").textContent = "Could not reach the YOLO server on this page's port.";
    });
  } catch (error) {
    if (token !== generation) return;
    const explanations = {
      NotAllowedError: "Camera access was denied. Allow camera access for this page in the browser.",
      NotFoundError: "No camera found. Connect the USB camera to the computer running this browser.",
      NotReadableError: "The camera could not be opened. Close other apps using it and try again.",
      OverconstrainedError: "That camera is unavailable. Select Default camera and try again."
    };
    $("error").textContent = explanations[error.name] || error.message;
    stop("Camera unavailable");
  }
}

function schedule(delay = 0) {
  clearTimeout(nextTimer);
  if (running) nextTimer = setTimeout(capture, Math.max(0, delay));
}

async function capture() {
  if (!running || pending || socket?.readyState !== WebSocket.OPEN || configuredRevision !== revision) return;
  if (document.hidden || video.readyState < 2 || !video.videoWidth || performance.now() - lastCameraFrame > 800) {
    schedule(150); return;
  }
  const token = generation;
  const image = document.createElement("canvas");
  const scale = Math.min(1, Number($("size").value) / Math.max(video.videoWidth, video.videoHeight));
  image.width = Math.round(video.videoWidth * scale);
  image.height = Math.round(video.videoHeight * scale);
  const frame = {id: ++frameId, revision, captured: performance.now(), image};
  pending = frame;  // Includes JPEG encoding time: never allow a second in-flight frame.
  lastCapture = frame.captured;
  image.getContext("2d").drawImage(video, 0, 0, image.width, image.height);
  const jpeg = await new Promise(resolve => image.toBlob(resolve, "image/jpeg", 0.8));
  if (token !== generation || !running || pending !== frame) return;
  if (frame.revision !== revision) { pending = null; schedule(); return; }
  if (!jpeg) { $("error").textContent = "Could not encode a camera frame."; stop("Camera error"); return; }
  if (socket?.readyState !== WebSocket.OPEN) { stop("Disconnected"); return; }
  socket.send(JSON.stringify({type: "frame", id: frame.id, revision: frame.revision}));
  socket.send(jpeg);
}

function handleMessage(data) {
  if (data.type === "ready") {
    $("model").textContent = data.model || "YOLO-World";
    $("device").textContent = `Device ${data.device} · inference size ${data.imgsz}`;
    configure();
    return;
  }
  if (data.type === "configured") {
    if (data.revision === revision) { configuredRevision = revision; schedule(); }
    return;
  }
  if (data.type === "error") {
    $("error").textContent = data.message || "Inference error";
    stop("Analysis stopped");
    return;
  }
  if (!["result", "busy", "discarded"].includes(data.type) || !pending || pending.id !== data.id) return;
  const frame = pending;
  pending = null;
  if (data.type === "busy") {
    status("GPU busy · retrying");
    schedule(80 + Math.random() * 160);
    return;
  }
  const now = performance.now();
  if (data.type === "result" && data.revision === revision && frame.revision === revision && !document.hidden) {
    if (now - frame.captured <= MAX_AGE_MS && now - lastCameraFrame <= 800) {
      render(data, frame);
      const ms = performance.now() - frame.captured;
      sampleCount++; latencyTotal += ms;
      arrivals.push(performance.now());
      arrivals = arrivals.filter(t => performance.now() - t < 5000);
      $("latency").textContent = `${Math.round(ms)} ms`;
      $("average").textContent = `${Math.round(latencyTotal / sampleCount)} ms`;
      $("detector").textContent = `${data.detector_ms} ms`;
      lastResult = {captured: frame.captured};
      status("Streaming", true);
    } else clearAnalysis("Result too old — waiting");
  }
  schedule(Math.max(0, 1000 / Number($("rate").value) - (performance.now() - lastCapture)));
}

function boxLabel(text, left, top, color, fontSize) {
  context.font = `600 ${fontSize}px system-ui`;
  const width = Math.min(canvas.width, context.measureText(text).width + 10);
  left = Math.max(0, Math.min(left, canvas.width - width));
  top = Math.max(0, Math.min(top, canvas.height - fontSize - 9));
  context.fillStyle = color;
  context.fillRect(left, top, width, fontSize + 9);
  context.fillStyle = "#07120b";
  context.fillText(text, left + 5, top + fontSize + 1, width - 10);
}

function render(data, frame) {
  if (data.width !== frame.image.width || data.height !== frame.image.height) throw new Error("Frame dimensions do not match the result.");
  canvas.width = frame.image.width; canvas.height = frame.image.height;
  canvas.hidden = false;
  $("placeholder").hidden = true;
  canvas.classList.remove("stale"); $("expired").hidden = true;
  context.drawImage(frame.image, 0, 0);
  const w = canvas.width, h = canvas.height, direction = data.direction;
  const band = direction.state === "centered" ? direction.exit_band : direction.enter_band;
  const fontSize = Math.max(12, Math.round(w / 70));
  context.fillStyle = "rgba(244,216,117,0.12)";
  context.fillRect(w * (0.5 - band), 0, 2 * band * w, h);
  context.strokeStyle = "#f4d875"; context.lineWidth = 2;
  context.beginPath(); context.moveTo(w / 2, 0); context.lineTo(w / 2, h); context.stroke();
  data.detections.forEach((d, i) => {
    const selected = i === direction.target_index;
    const color = selected ? "#6edcdb" : "#b4eb6d";
    const [x1, y1, x2, y2] = d.box;
    context.strokeStyle = color; context.lineWidth = selected ? 3 : 2;
    context.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
    boxLabel(`${d.label} ${d.confidence.toFixed(2)} | ${d.position}`, x1 * w, y1 * h - fontSize - 9, color, fontSize);
    if (selected) {
      const cx = (x1 + x2) / 2 * w, cy = (y1 + y2) / 2 * h;
      context.beginPath(); context.moveTo(cx - 8, cy); context.lineTo(cx + 8, cy);
      context.moveTo(cx, cy - 8); context.lineTo(cx, cy + 8); context.stroke();
    }
  });
  boxLabel("CENTER", w / 2 - 30, 8, "#f4d875", fontSize);
  const bannerSize = Math.max(15, Math.round(w / 44));
  context.fillStyle = "rgba(9,15,8,.85)";
  context.fillRect(0, h - bannerSize - 21, w, bannerSize + 21);
  context.fillStyle = direction.state === "centered" ? "#b4eb6d" : "#fff";
  context.font = `600 ${bannerSize}px system-ui`;
  context.fillText(direction.text, 14, h - 13);
  instruction(direction.state, direction.text,
    `Following: ${$("target").value}${direction.matches > 1 ? ` · ${direction.matches} matches; following one` : ""}`);
  if (direction.raw_offset !== null) {
    const offset = direction.raw_offset;
    $("marker").hidden = false;
    $("marker").style.left = `${Math.max(0, Math.min(100, (offset + 0.5) * 100))}%`;
    $("offset").textContent = `${Math.abs(offset * 100).toFixed(1)}% ${offset < 0 ? "left" : "right"}`;
  }
  const rows = data.detections.map((d, i) => {
    const row = document.createElement("div");
    row.className = `detection${i === direction.target_index ? " selected" : ""}`;
    const label = document.createElement("span"), score = document.createElement("strong"), pos = document.createElement("small");
    label.textContent = d.label + (i === direction.target_index ? " · target" : "");
    score.textContent = d.confidence.toFixed(2); pos.textContent = d.position;
    row.append(label, score, pos); return row;
  });
  if (!rows.length) {
    const message = document.createElement("p"); message.className = "hint";
    message.textContent = "No configured objects detected in this frame."; rows.push(message);
  }
  $("detections").replaceChildren(...rows);
  $("object-count").textContent = String(data.detections.length);
  $("frame-state").textContent = `Frame ${frame.id}`;
  hasSnapshot = true;
  $("save").disabled = false;
}

setInterval(() => {
  if (!running) return;
  const now = performance.now();
  if (!video.requestVideoFrameCallback && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime; lastCameraFrame = now;
  }
  if (lastResult) {
    const age = now - lastResult.captured;
    $("age").textContent = `${Math.round(age)} ms`;
    if (age > MAX_AGE_MS) clearAnalysis("Analysis is stale");
  }
  if (lastCameraFrame && now - lastCameraFrame > 800) clearAnalysis("Camera paused — waiting");
  if (pending && now - pending.captured > 10000) {
    $("error").textContent = "No result for 10 seconds. Check the GB10 terminal, then restart the camera.";
    stop("Response timed out");
  }
  arrivals = arrivals.filter(t => now - t < 5000);
  $("fps").textContent = arrivals.length > 1 ? `${((arrivals.length - 1) / ((now - arrivals[0]) / 1000)).toFixed(1)} / s` : "—";
}, 200);

$("start").addEventListener("click", start);
$("stop").addEventListener("click", () => stop());
$("camera").addEventListener("change", () => { if (running) start(); });
$("target").addEventListener("change", configure);
$("confidence").addEventListener("input", () => { $("confidence-value").textContent = Number($("confidence").value).toFixed(2); });
$("confidence").addEventListener("change", configure);
$("save").addEventListener("click", () => {
  if (!lastResult || performance.now() - lastResult.captured > MAX_AGE_MS) return;
  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `yolo-frame-${Date.now()}.jpg`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, "image/jpeg", 0.95);
});
document.addEventListener("visibilitychange", () => {
  if (!running) return;
  if (document.hidden) clearAnalysis("Tab paused"); else schedule();
});
window.addEventListener("beforeunload", () => stop());
navigator.mediaDevices?.addEventListener?.("devicechange", () => refreshCameras().catch(() => {}));
refreshCameras().catch(() => {});
