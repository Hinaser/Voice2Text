// The status line (dot + text) and transient "flash" messages shared by the
// export/summary actions.
import { listen } from "./tauri.js";

const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

let flashTimer = null;

/// Briefly show a message in the status line, then clear it.
export function flash(msg) {
  statusText.textContent = msg;
  if (flashTimer) clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { statusText.textContent = ""; }, 4000);
}

export function initStatus() {
  listen("status", (event) => {
    const { state, detail } = event.payload;
    statusText.textContent = detail;
    statusDot.className = "dot " + state;
  });
}
