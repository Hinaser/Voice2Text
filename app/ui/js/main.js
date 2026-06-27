// Entry point: wire up every UI module, then load + apply the persisted config.
import { loadConfig } from "./state.js";
import { initStatus } from "./status.js";
import { initTranscript } from "./render.js";
import { initControls, applyFont, applyOpacity, applyPin } from "./controls.js";
import { initSettings, syncSettingsForm } from "./settings.js";
import { initExport } from "./exporter.js";
import { initSummary } from "./summary.js";

initStatus();
initTranscript();
initControls();
initSettings();
initExport();
initSummary();

(async () => {
  await loadConfig();
  applyFont();
  applyOpacity();
  await applyPin();
  syncSettingsForm();
})();
