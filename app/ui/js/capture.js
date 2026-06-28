// Capture lifecycle controls: Start / Pause / Stop. Each button invokes the
// backend `set_capture` command; the buttons' enabled/disabled state follows the
// current capture state, which we also resync from `status` events (so the
// pipeline stays the source of truth, e.g. after a webview reload).
import { invoke, listen } from "./tauri.js";

const startBtn = document.getElementById("cap-start");
const pauseBtn = document.getElementById("cap-pause");
const stopBtn = document.getElementById("cap-stop");

// Reflect a capture state ("running" | "paused" | "stopped") in the buttons.
function reflect(state) {
  const running = state === "running";
  const paused = state === "paused";
  const stopped = state === "stopped";
  // Start doubles as Resume; only meaningful when not already running.
  startBtn.disabled = running;
  pauseBtn.disabled = !running;
  stopBtn.disabled = stopped;
  startBtn.classList.toggle("active", running);
  pauseBtn.classList.toggle("active", paused);
}

async function set(mode, optimistic) {
  reflect(optimistic);
  try {
    await invoke("set_capture", { mode });
  } catch (e) {
    console.error("set_capture failed", e);
  }
}

// Map the pipeline's status states onto capture states so the buttons track the
// backend even when it changes on its own (errors, app-driven transitions).
const STATUS_TO_CAPTURE = { listening: "running", paused: "paused", idle: "stopped" };

export function initCapture() {
  startBtn.addEventListener("click", () => set("start", "running"));
  pauseBtn.addEventListener("click", () => set("pause", "paused"));
  stopBtn.addEventListener("click", () => set("stop", "stopped"));

  listen("status", (event) => {
    const mapped = STATUS_TO_CAPTURE[event.payload.state];
    if (mapped) reflect(mapped);
  });

  // Sync to the real backend state on load (handles webview reloads).
  invoke("get_capture")
    .then((state) => reflect(state || "running"))
    .catch(() => reflect("running"));
}
