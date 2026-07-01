// Persistent bottom status bar: shows which audio sources are being captured
// (left) and where the transcript is being logged (right). Driven by two backend
// events: `capture` (sources + save target, emitted once at startup) and
// `saving` (the actual file path, emitted when the transcript file is opened).
import { listen } from "./tauri.js";

const sourcesEl = document.getElementById("sb-sources");
const saveEl = document.getElementById("sb-save");

let saving = false;
let saveDir = "";
let savePath = "";

function renderSave() {
  if (!saving) {
    saveEl.textContent = "Not saving";
    saveEl.title = "Transcript saving is off (enable it in Settings)";
    return;
  }
  if (savePath) {
    saveEl.textContent = "💾 " + savePath;
    saveEl.title = "Logging to " + savePath;
  } else {
    saveEl.textContent = "💾 " + saveDir + " — waiting for first line…";
    saveEl.title = "Will log into " + saveDir;
  }
}

// Reflect a live Settings change (save on/off, folder) immediately, without
// waiting for the next written line — otherwise a toggle looks like it did
// nothing. Leaves `savePath` intact so re-enabling still shows the live file.
export function setSaving(isSaving, dir) {
  saving = isSaving;
  if (dir) saveDir = dir;
  renderSave();
}

export function initStatusbar() {
  listen("capture", (event) => {
    const { sources, saving: s, save_dir } = event.payload;
    saving = s;
    saveDir = save_dir;
    const n = sources.length;
    const detail = sources.map((x) => `${x.role}: ${x.name}`).join(" · ");
    sourcesEl.textContent = `🎙 ${n} source${n === 1 ? "" : "s"}` + (detail ? ` — ${detail}` : "");
    sourcesEl.title = detail;
    renderSave();
  });

  listen("saving", (event) => {
    savePath = event.payload.path;
    saving = true;
    renderSave();
  });
}
