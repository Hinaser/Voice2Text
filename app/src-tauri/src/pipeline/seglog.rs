//! Opt-in segment-boundary diagnostics for chasing "first words dropped at a
//! speaker change". Disabled unless `VOICE2TEXT_DEBUG_SEGMENTS` is set; then each
//! finalized utterance logs (correlated by line id) its gap, duration and the
//! streaming text, plus the Whisper/polished text from the sidecar thread — so
//! we can see whether the start is lost only in the live caption (streaming
//! warm-up) or in Whisper too (real audio/segmentation loss).
//!
//! Output goes to `%TEMP%\voice2text-segments.log` (override with
//! `VOICE2TEXT_DEBUG_SEGMENTS_FILE`). One line per event; cheap no-op when off.

use std::fs::OpenOptions;
use std::io::Write;

/// True if segment logging is enabled (checked per call — utterances are
/// infrequent, so the env lookup cost is irrelevant).
pub fn enabled() -> bool {
    std::env::var_os("VOICE2TEXT_DEBUG_SEGMENTS").is_some()
}

/// Append one diagnostic line (no-op unless enabled).
pub fn log(msg: &str) {
    if !enabled() {
        return;
    }
    let path = std::env::var_os("VOICE2TEXT_DEBUG_SEGMENTS_FILE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("voice2text-segments.log"));
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{msg}");
    }
}
