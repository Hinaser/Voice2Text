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
}
