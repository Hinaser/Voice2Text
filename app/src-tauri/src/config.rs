//! User-configurable settings, persisted as JSON in the app config dir
//! (%APPDATA%\com.voice2text.overlay\config.json on Windows). Shared between
//! the Tauri command layer (read/write from the settings UI) and the pipeline
//! thread (which reads the live values each finalized utterance).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Append finalized lines to a transcript file.
    pub save_transcript: bool,
    /// Folder for transcript files. Empty => Documents\Voice2Text.
    pub save_dir: String,
    /// Add punctuation + casing to finalized text.
    pub punctuation: bool,
    /// Label utterances by detected speaker.
    pub diarization: bool,
    /// Capture the microphone too, labeled "You". Applies on restart.
    pub mic_capture: bool,
    /// Drop mic lines that echo attendees' audio coming from the speakers.
    pub echo_suppression: bool,
    /// Re-transcribe each utterance with Whisper (GPU) for a clean saved
    /// transcript; live captions stay streaming. Applies on restart.
    pub whisper_transcript: bool,
    /// Overlay text size (px). Persisted so the UI restores it.
    pub font_size: u32,
    /// Overlay background opacity 0.0–1.0.
    pub opacity: f32,
    /// Keep the window above others.
    pub always_on_top: bool,
    /// Override the models directory. Empty => auto-detect. Applies on restart.
    pub models_dir: String,
    /// Global hotkey to show/hide the overlay (e.g. "Alt+Shift+V"). Empty =
    /// disabled. Applies on restart.
    pub hotkey: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            save_transcript: true,
            save_dir: String::new(),
            punctuation: true,
            diarization: true,
            mic_capture: true,
            echo_suppression: true,
            whisper_transcript: true,
            font_size: 18,
            opacity: 0.82,
            always_on_top: true,
            models_dir: String::new(),
            hotkey: "Alt+Shift+V".to_string(),
        }
    }
}

impl Config {
    /// Load from disk, falling back to defaults if missing or malformed.
    pub fn load(path: &PathBuf) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Write to disk, creating the parent directory if needed.
    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let json = serde_json::to_string_pretty(self).unwrap_or_default();
        std::fs::write(path, json)
    }

    /// Default transcript folder: Documents\Voice2Text.
    pub fn default_save_dir() -> PathBuf {
        std::env::var("USERPROFILE")
            .map(|p| PathBuf::from(p).join("Documents").join("Voice2Text"))
            .unwrap_or_else(|_| PathBuf::from("."))
    }

    /// The effective transcript folder (configured value or the default).
    pub fn resolved_save_dir(&self) -> PathBuf {
        if self.save_dir.trim().is_empty() {
            Self::default_save_dir()
        } else {
            PathBuf::from(self.save_dir.trim())
        }
    }
}
