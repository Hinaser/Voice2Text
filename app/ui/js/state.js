// Shared UI state: the config mirror (persisted to the backend) and the current
// session's finalized lines (used for export/summary).
import { invoke } from "./tauri.js";

// Mirror of the persisted config; mutate fields in place and call persist().
export const config = {
  save_transcript: true,
  save_dir: "",
  punctuation: true,
  diarization: true,
  output_device: "",
  input_device: "",
  mic_capture: true,
  echo_suppression: true,
  whisper_transcript: true,
  summary_model: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
  font_size: 18,
  opacity: 0.82,
  always_on_top: true,
  models_dir: "",
  hotkey: "Alt+Shift+V",
};

export const history = [];

export async function persist() {
  try {
    await invoke("set_config", { new: config });
  } catch (e) {
    console.error("set_config failed", e);
  }
}

export async function loadConfig() {
  try {
    const loaded = await invoke("get_config");
    if (loaded) Object.assign(config, loaded);
  } catch (e) {
    console.error("get_config failed", e);
  }
}
