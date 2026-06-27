// Settings panel: reflects and edits the config mirror, persisting on change.
import { invoke } from "./tauri.js";
import { config, persist } from "./state.js";

const panel = document.getElementById("settings");
const el = {
  save: document.getElementById("cfg-save"),
  saveDir: document.getElementById("cfg-savedir"),
  whisper: document.getElementById("cfg-whisper"),
  punct: document.getElementById("cfg-punct"),
  diar: document.getElementById("cfg-diar"),
  mic: document.getElementById("cfg-mic"),
  echo: document.getElementById("cfg-echo"),
  models: document.getElementById("cfg-models"),
  hotkey: document.getElementById("cfg-hotkey"),
};

export function syncSettingsForm() {
  el.save.checked = config.save_transcript;
  el.saveDir.value = config.save_dir;
  el.whisper.checked = config.whisper_transcript;
  el.punct.checked = config.punctuation;
  el.diar.checked = config.diarization;
  el.mic.checked = config.mic_capture;
  el.echo.checked = config.echo_suppression;
  el.models.value = config.models_dir;
  el.hotkey.value = config.hotkey;
}

// Bind a checkbox/text input to a config field; persist on change.
function bindBool(input, key) {
  input.addEventListener("change", () => { config[key] = input.checked; persist(); });
}
function bindText(input, key) {
  input.addEventListener("change", () => { config[key] = input.value.trim(); persist(); });
}

export function initSettings() {
  document.getElementById("settings-btn").addEventListener("click", () => {
    syncSettingsForm();
    panel.classList.toggle("hidden");
  });
  document.getElementById("settings-close").addEventListener("click", () => panel.classList.add("hidden"));

  bindBool(el.save, "save_transcript");
  bindBool(el.whisper, "whisper_transcript");
  bindBool(el.punct, "punctuation");
  bindBool(el.diar, "diarization");
  bindBool(el.mic, "mic_capture");
  bindBool(el.echo, "echo_suppression");
  bindText(el.saveDir, "save_dir");
  bindText(el.models, "models_dir");
  bindText(el.hotkey, "hotkey");

  document.getElementById("cfg-opendir").addEventListener("click", () => {
    config.save_dir = el.saveDir.value.trim();
    persist().then(() => invoke("open_save_dir").catch((e) => console.error(e)));
  });
}
