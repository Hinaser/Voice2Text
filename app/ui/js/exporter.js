// Transcript export: Copy / .txt / .md / .srt of the current session.
import { invoke } from "./tauri.js";
import { history } from "./state.js";
import { flash } from "./status.js";

function pad(n, w) { return String(n).padStart(w, "0"); }

function srtTime(hms, addSec) {
  const [h, m, s] = hms.split(":").map(Number);
  const total = h * 3600 + m * 60 + s + (addSec || 0);
  return `${pad(Math.floor(total / 3600), 2)}:${pad(Math.floor((total % 3600) / 60), 2)}:${pad(total % 60, 2)},000`;
}

export function buildPlain() {
  return history.map((e) => `[${e.time}] ${e.speaker ? e.speaker + ": " : ""}${e.text}`).join("\r\n");
}

function buildMarkdown() {
  const head = "# Voice2Text transcript\r\n\r\n";
  return head + history.map((e) => `**${e.time}${e.speaker ? " · " + e.speaker : ""}:** ${e.text}`).join("\r\n\r\n");
}

function buildSrt() {
  return history.map((e, i) => {
    const start = srtTime(e.time, 0);
    const end = i + 1 < history.length ? srtTime(history[i + 1].time, 0) : srtTime(e.time, 3);
    const who = e.speaker ? e.speaker + ": " : "";
    return `${i + 1}\r\n${start} --> ${end}\r\n${who}${e.text}\r\n`;
  }).join("\r\n");
}

// Name exports after the first line's time; fall back to a generic name.
export function stamp() {
  return history.length ? history[0].time.replace(/:/g, "") : "export";
}

async function doExport(ext, content) {
  if (!history.length) { flash("Nothing to export yet"); return; }
  try {
    const path = await invoke("export_transcript", { filename: `transcript-${stamp()}.${ext}`, content });
    flash("Saved " + path);
  } catch (e) {
    flash("Export failed: " + e);
  }
}

export function initExport() {
  document.getElementById("exp-txt").addEventListener("click", () => doExport("txt", buildPlain()));
  document.getElementById("exp-md").addEventListener("click", () => doExport("md", buildMarkdown()));
  document.getElementById("exp-srt").addEventListener("click", () => doExport("srt", buildSrt()));
  document.getElementById("exp-copy").addEventListener("click", async () => {
    if (!history.length) { flash("Nothing to copy yet"); return; }
    try { await navigator.clipboard.writeText(buildPlain()); flash("Copied transcript"); }
    catch (e) { flash("Copy failed: " + e); }
  });
}
