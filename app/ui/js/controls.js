// Title-bar controls: font size, opacity, always-on-top pin, clear, close.
import { appWindow } from "./tauri.js";
import { config, persist } from "./state.js";
import { clearTranscript } from "./render.js";

export function applyFont() {
  document.documentElement.style.setProperty("--font-size", config.font_size + "px");
}
export function applyOpacity() {
  document.documentElement.style.setProperty("--bg-opacity", config.opacity);
}

const pinBtn = document.getElementById("pin");
export async function applyPin() {
  await appWindow.setAlwaysOnTop(config.always_on_top);
  pinBtn.classList.toggle("off", !config.always_on_top);
}

const OPACITIES = [0.82, 0.6, 0.4, 1.0];

export function initControls() {
  document.getElementById("font-dec").addEventListener("click", () => {
    config.font_size = Math.max(11, config.font_size - 2);
    applyFont();
    persist();
  });
  document.getElementById("font-inc").addEventListener("click", () => {
    config.font_size = Math.min(40, config.font_size + 2);
    applyFont();
    persist();
  });
  document.getElementById("opacity").addEventListener("click", () => {
    const idx = OPACITIES.indexOf(config.opacity);
    config.opacity = OPACITIES[(idx + 1) % OPACITIES.length];
    applyOpacity();
    persist();
  });
  pinBtn.addEventListener("click", async () => {
    config.always_on_top = !config.always_on_top;
    await applyPin();
    persist();
  });
  document.getElementById("clear").addEventListener("click", clearTranscript);
  document.getElementById("close").addEventListener("click", () => appWindow.close());
}
