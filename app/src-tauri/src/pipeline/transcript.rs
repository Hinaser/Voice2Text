//! Transcript file writing. A single `TranscriptWriter` is used by both the
//! Whisper sidecar (clean text) and the streaming fallback, so there's one place
//! that knows the file format and lazy-open behavior.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use chrono::Local;

pub struct TranscriptWriter {
    dir: PathBuf,
    writer: Option<BufWriter<File>>,
}

impl TranscriptWriter {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir, writer: None }
    }

    /// Point future lines at the (possibly changed) folder. If it differs from
    /// where the current file lives, that file is closed and the next line
    /// opens a fresh one in the new folder — so changing the save folder in
    /// Settings takes effect on the next line instead of being silently
    /// ignored for the rest of the session.
    pub fn retarget(&mut self, dir: PathBuf) {
        if dir != self.dir {
            self.dir = dir;
            self.writer = None;
        }
    }

    /// Append one finalized line, opening a fresh timestamped file on first use.
    /// Returns the file path the first time it opens (so the caller can surface
    /// a "Saving to …" status), and `None` thereafter.
    pub fn write_line(&mut self, time: &str, label: &str, text: &str) -> io::Result<Option<PathBuf>> {
        let mut opened = None;
        if self.writer.is_none() {
            std::fs::create_dir_all(&self.dir)?;
            let path = self.dir.join(format!("transcript-{}.txt", Local::now().format("%Y%m%d-%H%M%S")));
            let file = OpenOptions::new().create(true).append(true).open(&path)?;
            self.writer = Some(BufWriter::new(file));
            opened = Some(path);
        }
        let w = self.writer.as_mut().expect("writer just set");
        writeln!(w, "{}", format_line(time, label, text))?;
        w.flush()?;
        Ok(opened)
    }
}

/// `"[HH:MM:SS] Speaker 1: text"` (or no label when empty).
pub fn format_line(time: &str, label: &str, text: &str) -> String {
    format!("[{time}] {label}{text}")
}

/// Render a speaker into a transcript/UI label: `"Speaker 1: "` or `""`.
pub fn speaker_label(speaker: &str) -> String {
    if speaker.is_empty() {
        String::new()
    } else {
        format!("{speaker}: ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_line_with_and_without_label() {
        assert_eq!(format_line("10:00:01", "Speaker 1: ", "hi"), "[10:00:01] Speaker 1: hi");
        assert_eq!(format_line("10:00:01", "", "hi"), "[10:00:01] hi");
    }

    #[test]
    fn speaker_label_empty_for_no_speaker() {
        assert_eq!(speaker_label(""), "");
        assert_eq!(speaker_label("You"), "You: ");
    }
}
