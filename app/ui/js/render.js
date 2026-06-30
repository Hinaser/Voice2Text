// Transcript rendering: finalized lines (speaker-colored) and the two live
// partial lines (others / me).
import { listen } from "./tauri.js";
import { history } from "./state.js";

const transcript = document.getElementById("transcript");
const partialOthers = document.getElementById("partial-others");
const partialMe = document.getElementById("partial-me");

const PALETTE = [
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24",
  "#a78bfa", "#fb923c", "#22d3ee", "#a3e635",
];
const DEFAULT_COLOR = "#eef1f6";
const YOU_COLOR = "#fca5a5";
// "You" gets a fixed, distinct color; attendees draw from the palette.
const speakerColors = { You: YOU_COLOR };
let nextColor = 0;
function colorFor(speaker) {
  if (!speaker) return DEFAULT_COLOR;
  if (!(speaker in speakerColors)) {
    speakerColors[speaker] = PALETTE[nextColor % PALETTE.length];
    nextColor += 1;
  }
  return speakerColors[speaker];
}

// Auto-scroll only when already near the bottom, so scrolling up to re-read
// older lines isn't yanked back down.
function nearBottom() {
  return transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 40;
}
function scrollToBottom() {
  transcript.scrollTop = transcript.scrollHeight;
}

let lastSpeaker = null;
// id → { entry (history record), tx (text span) } so a later Whisper/LLM
// refinement can replace the line's text in place.
const lines = new Map();

function renderFinal({ id, time, speaker, text, source }) {
  const entry = { time, speaker, text };
  history.push(entry);
  const stick = nearBottom();

  const line = document.createElement("div");
  line.className = "line";

  const tm = document.createElement("span");
  tm.className = "time";
  tm.textContent = time;
  line.appendChild(tm);

  // Print the speaker label only when it changes, to reduce clutter.
  if (speaker && speaker !== lastSpeaker) {
    const sp = document.createElement("span");
    sp.className = "spk";
    sp.textContent = speaker + ": ";
    sp.style.color = colorFor(speaker);
    line.appendChild(sp);
    lastSpeaker = speaker;
  }

  const tx = document.createElement("span");
  tx.textContent = text;
  tx.style.color = colorFor(speaker);
  line.appendChild(tx);

  transcript.appendChild(line);
  if (id !== undefined) lines.set(id, { entry, tx });
  if (source === "me") partialMe.textContent = "";
  else partialOthers.textContent = "";
  if (stick) scrollToBottom();
}

// Swap a finalized line's text for a refined version (Whisper clean, then
// LLM-polished), keeping its place, speaker, and color.
function renderReplace({ id, text }) {
  const rec = lines.get(id);
  if (!rec) return;
  const stick = nearBottom();
  rec.entry.text = text;
  rec.tx.textContent = text;
  if (stick) scrollToBottom();
}

function renderPartial({ source, text }) {
  const stick = nearBottom();
  if (source === "me") {
    partialMe.textContent = text ? "You: " + text : "";
  } else {
    partialOthers.textContent = text;
  }
  if (stick) scrollToBottom();
}

export function initTranscript() {
  listen("final", (e) => renderFinal(e.payload));
  listen("replace", (e) => renderReplace(e.payload));
  listen("partial", (e) => renderPartial(e.payload));
}

export function clearTranscript() {
  transcript.innerHTML = "";
  partialOthers.textContent = "";
  partialMe.textContent = "";
  history.length = 0;
  lines.clear();
  lastSpeaker = null;
}
