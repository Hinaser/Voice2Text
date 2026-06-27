//! The optional Whisper-CUDA sidecar: a separate GPU process that re-transcribes
//! each finalized utterance into the clean saved transcript. The main app stays
//! CPU-only. Owns spawning, the readiness handshake, the wire protocol, and the
//! background manager thread that writes results to the transcript file.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::config::Config;
use crate::paths;

use super::events::Ui;
use super::transcript::TranscriptWriter;

/// Bound on queued utterances. Whisper runs far faster than real time, so this
/// is only a safety valve against unbounded memory growth if it ever stalls.
const QUEUE_CAP: usize = 64;

const SIDECAR_EXE: &str = "whisper-sidecar.exe";

/// A finalized utterance handed to the sidecar for clean re-transcription.
pub struct WhisperJob {
    pub audio: Vec<f32>,
    pub time: String,
    pub label: String,
}

pub struct WhisperSidecar {
    tx: SyncSender<WhisperJob>,
}

impl WhisperSidecar {
    /// Spawn the sidecar and start its manager thread. Returns `None` (after
    /// emitting a status) if the process can't start or fails the handshake, so
    /// the caller falls back to writing streaming text.
    pub fn spawn(model: &Path, config: Arc<Mutex<Config>>, ui: Ui) -> Option<Self> {
        let (mut child, stdin, mut reader) = match spawn_process(model) {
            Ok(io) => io,
            Err(e) => {
                ui.status("error", format!("Whisper sidecar failed, saving streaming text: {e}"));
                return None;
            }
        };

        let mut first = String::new();
        let ready = reader.read_line(&mut first).is_ok() && first.trim() == "READY";
        if !ready {
            let msg = first.trim().trim_start_matches("ERROR").trim();
            ui.status("error", format!("Whisper unavailable, saving streaming text ({msg})"));
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }

        let (tx, rx) = mpsc::sync_channel(QUEUE_CAP);
        thread::spawn(move || manager(child, stdin, reader, config, ui, rx));
        Some(Self { tx })
    }

    /// Queue a job. Drops it (rather than blocking the audio loop) if the
    /// sidecar has fallen far behind.
    pub fn send(&self, job: WhisperJob) {
        match self.tx.try_send(job) {
            Ok(()) | Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => {}
        }
    }
}

type SidecarIo = (Child, ChildStdin, BufReader<ChildStdout>);

/// Spawn the sidecar with piped stdin/stdout (binary jobs / text results) and
/// inherited stderr (whisper.cpp logs go to the console).
fn spawn_process(model: &Path) -> std::io::Result<SidecarIo> {
    let exe = paths::sidecar_exe(SIDECAR_EXE);
    let mut child = Command::new(&exe)
        .arg(model)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdin = child.stdin.take().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "no stdin"))?;
    let stdout = child.stdout.take().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "no stdout"))?;
    Ok((child, stdin, BufReader::new(stdout)))
}

/// Drive the sidecar synchronously: one job in, one transcript line out, in
/// order. Owns the child so it lives for the app's lifetime.
fn manager(
    mut child: Child,
    mut stdin: ChildStdin,
    mut reader: BufReader<ChildStdout>,
    config: Arc<Mutex<Config>>,
    ui: Ui,
    rx: Receiver<WhisperJob>,
) {
    let mut writer: Option<TranscriptWriter> = None;
    while let Ok(job) = rx.recv() {
        let cfg = config.lock().unwrap().clone();
        if stdin.write_all(&encode_frame(&job.audio)).and_then(|_| stdin.flush()).is_err() {
            ui.status("error", "Whisper sidecar stopped; clean transcript disabled");
            break;
        }
        // Exactly one result line per job (FIFO).
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => {
                ui.status("error", "Whisper sidecar closed; clean transcript disabled");
                break;
            }
            Ok(_) => {}
        }
        let text = line.trim();
        if !cfg.save_transcript || text.is_empty() {
            continue;
        }
        let w = writer.get_or_insert_with(|| TranscriptWriter::new(cfg.resolved_save_dir()));
        match w.write_line(&job.time, &job.label, text) {
            Ok(Some(path)) => ui.status("listening", format!("Saving to {}", path.display())),
            Ok(None) => {}
            Err(e) => ui.status("error", format!("Save failed: {e}")),
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Wire frame: u32 LE sample-count, then that many f32 LE samples.
fn encode_frame(audio: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + audio.len() * 4);
    buf.extend_from_slice(&(audio.len() as u32).to_le_bytes());
    for s in audio {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_count_then_samples_little_endian() {
        let frame = encode_frame(&[1.0_f32, -1.0_f32]);
        assert_eq!(&frame[0..4], &2u32.to_le_bytes());
        assert_eq!(&frame[4..8], &1.0f32.to_le_bytes());
        assert_eq!(&frame[8..12], &(-1.0f32).to_le_bytes());
        assert_eq!(frame.len(), 4 + 2 * 4);
    }

    #[test]
    fn encodes_empty_audio_as_zero_count() {
        assert_eq!(encode_frame(&[]), 0u32.to_le_bytes().to_vec());
    }
}
