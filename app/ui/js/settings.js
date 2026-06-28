// Settings panel: reflects and edits the config mirror, persisting on change.
import { invoke } from "./tauri.js";
import { config, persist } from "./state.js";

const panel = document.getElementById("settings");
const el = {
  save: document.getElementById("cfg-save"),
  saveDir: document.getElementById("cfg-savedir"),
  whisper: document.getElementById("cfg-whisper"),
  language: document.getElementById("cfg-language"),
  punct: document.getElementById("cfg-punct"),
  diar: document.getElementById("cfg-diar"),
  mic: document.getElementById("cfg-mic"),
  echo: document.getElementById("cfg-echo"),
  output: document.getElementById("cfg-output"),
  input: document.getElementById("cfg-input"),
  summaryModel: document.getElementById("cfg-summary-model"),
  models: document.getElementById("cfg-models"),
  hotkey: document.getElementById("cfg-hotkey"),
};

// Fill the summary-model <select> with the .gguf files found in the models
// folder, keeping the configured one selectable even if it isn't there yet.
function fillModelSelect(select, models, selectedId) {
  const options = [...models];
  if (selectedId && !options.includes(selectedId)) options.push(selectedId);
  select.innerHTML = "";
  if (options.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no .gguf models found)";
    select.appendChild(opt);
    return;
  }
  for (const name of options) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  }
  select.value = selectedId || "";
}

// Query the backend for .gguf models and (re)populate the summary-model dropdown.
async function refreshModels() {
  try {
    const models = await invoke("list_gguf_models");
    fillModelSelect(el.summaryModel, models, config.summary_model);
  } catch (e) {
    console.error("list_gguf_models failed", e);
  }
}

// Show the effective (resolved) folders as placeholders so "default" /
// "auto-detect" aren't opaque — users can see where files actually go.
async function refreshPaths() {
  try {
    const sd = await invoke("save_dir");
    if (sd) el.saveDir.placeholder = sd;
    const md = await invoke("models_dir");
    if (md) el.models.placeholder = md;
  } catch (e) {
    console.error("resolving folders failed", e);
  }
}

// Open a native folder picker (parented to the window), returning the chosen
// path or null if cancelled.
async function pickFolder(defaultPath) {
  try {
    const picked = await invoke("plugin:dialog|open", {
      options: { directory: true, multiple: false, title: "Choose folder", defaultPath: defaultPath || undefined },
    });
    if (!picked) return null;
    if (typeof picked === "string") return picked;
    if (Array.isArray(picked)) return picked[0] || null;
    if (picked.path) return picked.path;
    return null;
  } catch (e) {
    console.error("folder picker failed", e);
    return null;
  }
}

// Fill a <select> with a "System default" entry plus one <option> per device,
// preserving the currently-configured id even if enumeration hasn't run yet.
function fillDeviceSelect(select, devices, selectedId) {
  const options = [{ id: "", name: "System default" }, ...devices];
  if (selectedId && !options.some((d) => d.id === selectedId)) {
    options.push({ id: selectedId, name: "(saved device — not connected)" });
  }
  select.innerHTML = "";
  for (const d of options) {
    const opt = document.createElement("option");
    opt.value = d.id;
    opt.textContent = d.name;
    select.appendChild(opt);
  }
  select.value = selectedId || "";
}

// Query the backend for endpoints and (re)populate both device dropdowns.
async function refreshDevices() {
  try {
    const { output, input } = await invoke("list_audio_devices");
    fillDeviceSelect(el.output, output, config.output_device);
    fillDeviceSelect(el.input, input, config.input_device);
  } catch (e) {
    console.error("list_audio_devices failed", e);
  }
}

export function syncSettingsForm() {
  el.save.checked = config.save_transcript;
  el.saveDir.value = config.save_dir;
  el.whisper.checked = config.whisper_transcript;
  el.language.value = config.language;
  el.punct.checked = config.punctuation;
  el.diar.checked = config.diarization;
  el.mic.checked = config.mic_capture;
  el.echo.checked = config.echo_suppression;
  el.output.value = config.output_device;
  el.input.value = config.input_device;
  el.summaryModel.value = config.summary_model;
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
    if (!panel.classList.contains("hidden")) {
      refreshDevices();
      refreshModels();
      refreshPaths();
    }
  });
  document.getElementById("settings-close").addEventListener("click", () => panel.classList.add("hidden"));

  bindBool(el.save, "save_transcript");
  bindBool(el.whisper, "whisper_transcript");
  bindText(el.language, "language");
  bindBool(el.punct, "punctuation");
  bindBool(el.diar, "diarization");
  bindBool(el.mic, "mic_capture");
  bindBool(el.echo, "echo_suppression");
  bindText(el.output, "output_device");
  bindText(el.input, "input_device");
  bindText(el.summaryModel, "summary_model");
  bindText(el.saveDir, "save_dir");
  bindText(el.models, "models_dir");
  bindText(el.hotkey, "hotkey");

  // Transcript folder: pick (Browse…) or reveal (Open) in Explorer.
  document.getElementById("cfg-browsedir").addEventListener("click", async () => {
    const dir = await pickFolder(el.saveDir.value.trim() || el.saveDir.placeholder);
    if (dir) { el.saveDir.value = dir; config.save_dir = dir; persist(); }
  });
  document.getElementById("cfg-opendir").addEventListener("click", () => {
    config.save_dir = el.saveDir.value.trim();
    persist().then(() => invoke("open_save_dir").catch((e) => console.error(e)));
  });

  // Models folder: pick (Browse…) or reveal (Open) the effective folder.
  document.getElementById("cfg-browsemodels").addEventListener("click", async () => {
    const dir = await pickFolder(el.models.value.trim() || el.models.placeholder);
    if (dir) { el.models.value = dir; config.models_dir = dir; persist(); }
  });
  document.getElementById("cfg-openmodels").addEventListener("click", () => {
    config.models_dir = el.models.value.trim();
    persist().then(() => invoke("open_models_dir").catch((e) => console.error(e)));
  });
}
