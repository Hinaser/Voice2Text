// Meeting summary panel (local LLM via the summarize command).
import { invoke } from "./tauri.js";
import { history } from "./state.js";
import { flash } from "./status.js";
import { buildPlain, stamp } from "./exporter.js";

const panel = document.getElementById("summary");
const body = document.getElementById("summary-body");
let lastSummary = "";
let summarizing = false;

export function initSummary() {
  document.getElementById("summarize-btn").addEventListener("click", async () => {
    if (summarizing) return;
    if (!history.length) { flash("Nothing to summarize yet"); return; }
    panel.classList.remove("hidden");
    body.textContent = "Summarizing… (the first run loads the model, ~a few seconds)";
    summarizing = true;
    try {
      lastSummary = await invoke("summarize", { transcript: buildPlain() });
      body.textContent = lastSummary;
    } catch (e) {
      body.textContent = "Summary failed: " + e;
      lastSummary = "";
    } finally {
      summarizing = false;
    }
  });

  document.getElementById("summary-close").addEventListener("click", () => panel.classList.add("hidden"));

  document.getElementById("summary-copy").addEventListener("click", async () => {
    if (!lastSummary) return;
    try { await navigator.clipboard.writeText(lastSummary); flash("Copied summary"); }
    catch (e) { flash("Copy failed: " + e); }
  });

  document.getElementById("summary-save").addEventListener("click", async () => {
    if (!lastSummary) return;
    const content = "# Meeting summary\r\n\r\n" + lastSummary.replace(/\n/g, "\r\n");
    try {
      const path = await invoke("export_transcript", { filename: `summary-${stamp()}.md`, content });
      flash("Saved " + path);
    } catch (e) { flash("Save failed: " + e); }
  });
}
