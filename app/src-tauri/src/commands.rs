//! Tauri command handlers — the JS ↔ Rust bridge — plus the shared app state
//! they read.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tauri::Manager;

use crate::audio;
use crate::config::Config;
use crate::pipeline::{capture_state, CaptureControl};
use crate::summary;

#[derive(serde::Serialize)]
pub struct AudioDevice {
    id: String,
    name: String,
}

#[derive(serde::Serialize)]
pub struct AudioDevices {
    output: Vec<AudioDevice>,
    input: Vec<AudioDevice>,
}

/// List active speaker-output and microphone endpoints for the settings UI.
#[tauri::command]
pub fn list_audio_devices() -> Result<AudioDevices, String> {
    let (render, capture) = audio::list_devices().map_err(|e| e.to_string())?;
    let map = |v: Vec<(String, String)>| v.into_iter().map(|(id, name)| AudioDevice { id, name }).collect();
    Ok(AudioDevices { output: map(render), input: map(capture) })
}

/// Shared state handed to every command.
pub struct AppState {
    pub config: Arc<Mutex<Config>>,
    pub config_path: PathBuf,
    pub capture: Arc<CaptureControl>,
}

/// Set the capture-session state from the UI's Start / Pause / Stop buttons.
/// The pipeline picks the change up on its next loop (well under a second).
#[tauri::command]
pub fn set_capture(state: tauri::State<AppState>, mode: String) -> Result<(), String> {
    let value = match mode.as_str() {
        "start" => capture_state::RUNNING,
        "pause" => capture_state::PAUSED,
        "stop" => capture_state::STOPPED,
        other => return Err(format!("unknown capture mode: {other}")),
    };
    state.capture.set(value);
    Ok(())
}

/// Current capture state as a UI-facing string, so the buttons can sync on load
/// (e.g. after a webview reload). One of "running" / "paused" / "stopped".
#[tauri::command]
pub fn get_capture(state: tauri::State<AppState>) -> &'static str {
    match state.capture.get() {
        capture_state::RUNNING => "running",
        capture_state::PAUSED => "paused",
        _ => "stopped",
    }
}

#[tauri::command]
pub fn get_config(state: tauri::State<AppState>) -> Config {
    state.config.lock().unwrap().clone()
}

#[tauri::command]
pub fn set_config(new: Config, state: tauri::State<AppState>) -> Result<(), String> {
    *state.config.lock().unwrap() = new.clone();
    new.save(&state.config_path).map_err(|e| e.to_string())
}

/// The effective transcript folder, for display.
#[tauri::command]
pub fn save_dir(state: tauri::State<AppState>) -> String {
    state.config.lock().unwrap().resolved_save_dir().to_string_lossy().into_owned()
}

/// Write a UI-built export (md/srt/txt) into the save folder; returns its path.
#[tauri::command]
pub fn export_transcript(state: tauri::State<AppState>, filename: String, content: String) -> Result<String, String> {
    let dir = state.config.lock().unwrap().resolved_save_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let safe: String = filename
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | '_'))
        .collect();
    let name = if safe.is_empty() { "transcript-export.txt".to_string() } else { safe };
    let path = dir.join(name);
    std::fs::write(&path, content).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

/// Open the transcript folder in Explorer (creating it if needed).
#[tauri::command]
pub fn open_save_dir(state: tauri::State<AppState>) -> Result<(), String> {
    let dir = state.config.lock().unwrap().resolved_save_dir();
    let _ = std::fs::create_dir_all(&dir);
    open_in_explorer(&dir)
}

/// The effective models folder (the configured override or the auto-detected
/// location), for display so "auto-detect" isn't opaque. Normalized (the dev
/// fallback contains `..` segments) and without the `\\?\` verbatim prefix.
#[tauri::command]
pub fn models_dir(state: tauri::State<AppState>) -> String {
    let override_dir = state.config.lock().unwrap().models_dir.clone();
    let dir = crate::paths::models_dir(&override_dir);
    let display = std::fs::canonicalize(&dir).unwrap_or(dir);
    let s = display.to_string_lossy();
    s.strip_prefix(r"\\?\").unwrap_or(&s).to_string()
}

/// Open the effective models folder in Explorer.
#[tauri::command]
pub fn open_models_dir(state: tauri::State<AppState>) -> Result<(), String> {
    let override_dir = state.config.lock().unwrap().models_dir.clone();
    open_in_explorer(&crate::paths::models_dir(&override_dir))
}

fn open_in_explorer(dir: &std::path::Path) -> Result<(), String> {
    std::process::Command::new("explorer")
        .arg(dir)
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Summarize the session transcript with the local LLM (runs off-thread).
#[tauri::command]
pub async fn summarize(app: tauri::AppHandle, transcript: String) -> Result<String, String> {
    let (models_override, model_file) = {
        let state = app.state::<AppState>();
        let cfg = state.config.lock().unwrap();
        (cfg.models_dir.clone(), cfg.summary_model.clone())
    };
    tauri::async_runtime::spawn_blocking(move || summary::run(&models_override, &model_file, &transcript))
        .await
        .map_err(|e| e.to_string())?
}

/// List the GGUF models available in the models folder, for the summary-model
/// dropdown. Sorted; empty if the folder is missing or has none.
#[tauri::command]
pub fn list_gguf_models(state: tauri::State<AppState>) -> Vec<String> {
    let dir = {
        let cfg = state.config.lock().unwrap();
        crate::paths::models_dir(&cfg.models_dir)
    };
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    out.push(name.to_string());
                }
            }
        }
    }
    out.sort();
    out
}
