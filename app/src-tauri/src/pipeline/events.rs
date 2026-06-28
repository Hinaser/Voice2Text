//! UI event payloads and a thin emitter that isolates the Tauri dependency from
//! the rest of the pipeline.

use tauri::{AppHandle, Emitter};

use super::Source;

#[derive(Clone, serde::Serialize)]
struct Partial {
    source: &'static str,
    text: String,
}

#[derive(Clone, serde::Serialize)]
struct Final {
    time: String,
    source: &'static str,
    speaker: String,
    text: String,
}

#[derive(Clone, serde::Serialize)]
struct Status {
    state: &'static str,
    detail: String,
}

#[derive(Clone, serde::Serialize)]
struct CaptureSource {
    role: &'static str,
    name: String,
}

/// Persistent capture info for the status bar: which endpoints are being
/// captured and where (if anywhere) the transcript is saved.
#[derive(Clone, serde::Serialize)]
struct Capture {
    sources: Vec<CaptureSource>,
    saving: bool,
    save_dir: String,
}

#[derive(Clone, serde::Serialize)]
struct Saving {
    path: String,
}

/// Wraps the Tauri `AppHandle` so the pipeline talks in domain terms
/// (partial / final / status) instead of stringly-typed `emit` calls.
#[derive(Clone)]
pub struct Ui {
    app: AppHandle,
}

impl Ui {
    pub fn new(app: AppHandle) -> Self {
        Self { app }
    }

    pub fn partial(&self, source: Source, text: String) {
        let _ = self.app.emit("partial", Partial { source: source.tag(), text });
    }

    pub fn clear_partial(&self, source: Source) {
        self.partial(source, String::new());
    }

    pub fn final_line(&self, time: String, source: Source, speaker: String, text: String) {
        let _ = self.app.emit("final", Final { time, source: source.tag(), speaker, text });
    }

    pub fn status(&self, state: &'static str, detail: impl Into<String>) {
        let _ = self.app.emit("status", Status { state, detail: detail.into() });
    }

    /// Emit the persistent capture summary (which sources, and the save target).
    pub fn capture(&self, sources: Vec<(&'static str, String)>, saving: bool, save_dir: String) {
        let sources = sources.into_iter().map(|(role, name)| CaptureSource { role, name }).collect();
        let _ = self.app.emit("capture", Capture { sources, saving, save_dir });
    }

    /// Emit the resolved transcript file path once it's known (file opened).
    pub fn saving(&self, path: String) {
        let _ = self.app.emit("saving", Saving { path });
    }
}
