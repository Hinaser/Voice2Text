//! live-stt — vertical-slice demo: show system audio as live text.
//!
//! Captures the system **loopback** (whatever plays through the speakers —
//! YouTube, Zoom, Google Meet), segments it into utterances with a simple energy
//! gate, transcribes each with whisper-rs on the GPU, and prints the text as it
//! is recognized.
//!
//! This is a DEMO, not the final app: the segmenter is a basic energy VAD (M2
//! will use Silero), there is no UI yet (M4), and only the loopback track is
//! transcribed. It exists to show speech -> text working end to end.
//!
//! Usage:  live-stt [seconds] [model.bin] [transcript.txt]
//!   defaults: 120 seconds, models/ggml-large-v3-q5_0.bin,
//!             transcripts/transcript-<YYYYMMDD-HHMMSS>.txt

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use chrono::Local;
use wasapi::{DeviceEnumerator, Direction, Role, SampleType, StreamMode, WaveFormat};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const RATE: usize = 16_000;
const FRAME: usize = 480; // 30 ms @ 16 kHz

// Energy-VAD tuning (hysteresis so noise doesn't toggle it).
const RMS_ON: f32 = 0.015; // start an utterance above this
const RMS_OFF: f32 = 0.008; // a frame below this counts as silence
const HANG_FRAMES: usize = 20; // ~600 ms of trailing silence closes the utterance
const MIN_SPEECH_FRAMES: usize = 12; // ignore <~0.36 s blips
const MAX_UTT_SAMPLES: usize = RATE * 15; // hard cap to bound latency

// Interim-partial cadence: refresh the evolving line ~3x/sec, transcribing at
// most the last 10 s so long turns don't slow the refresh.
const PARTIAL_INTERVAL: Duration = Duration::from_millis(300);
const PARTIAL_WINDOW: usize = RATE * 10;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    if let Err(e) = run() {
        eprintln!("\nERROR: {e}");
        std::process::exit(1);
    }
}

fn run() -> Res<()> {
    let mut args = std::env::args().skip(1);
    let secs: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(120);
    let model: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models").join("ggml-large-v3-q5_0.bin")
    });
    let transcript_path: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| {
        let stamp = Local::now().format("%Y%m%d-%H%M%S");
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("transcripts").join(format!("transcript-{stamp}.txt"))
    });

    // Open the transcript file up front; lines are appended + flushed as soon as
    // each utterance is recognized, so the file is a live, crash-safe record.
    if let Some(parent) = transcript_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut transcript = BufWriter::new(File::create(&transcript_path)?);
    writeln!(transcript, "# Voice2Text transcript — started {}", Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    transcript.flush()?;

    eprintln!("Loading model: {} ...", model.display());
    let ctx = WhisperContext::new_with_params(
        model.to_str().ok_or("bad model path")?,
        WhisperContextParameters::default(),
    )?;
    let mut state = ctx.create_state()?;
    eprintln!("Model loaded. Listening to system audio for {secs}s.");
    eprintln!("Saving transcript to: {}", transcript_path.display());
    eprintln!("Play a YouTube talk / join a meeting through your speakers — text appears below.\n");

    // Capture thread -> channel of f32 sample chunks.
    let running = Arc::new(AtomicBool::new(true));
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let cap = {
        let running = running.clone();
        thread::spawn(move || {
            if let Err(e) = capture_loopback(&running, &tx) {
                eprintln!("capture error: {e}");
            }
        })
    };

    let start = Instant::now();
    let mut seg = Segmenter::new();
    let mut pending: Vec<f32> = Vec::new();

    // Interim-partial state: while an utterance is open, periodically re-transcribe
    // the audio-so-far and show evolving text (overwriting the current line), so
    // the user sees words appear well before the pause-triggered final.
    let mut last_partial_len = 0usize;
    let mut last_partial_at = Instant::now();

    while start.elapsed().as_secs() < secs {
        // Pull available audio (block briefly so we don't busy-spin).
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(chunk) => pending.extend(chunk),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(_) => break,
        }
        while let Ok(chunk) = rx.try_recv() {
            pending.extend(chunk);
        }
        // Process whole 30 ms frames; keep the remainder for next time.
        let n_frames = pending.len() / FRAME;
        for i in 0..n_frames {
            let frame = &pending[i * FRAME..(i + 1) * FRAME];
            if let Some(utt) = seg.push(frame) {
                emit(&mut state, &utt, &mut transcript); // final line (file + console)
                last_partial_len = 0;
                last_partial_at = Instant::now();
            }
        }
        pending.drain(0..n_frames * FRAME);

        // Interim partial: re-transcribe the open utterance every PARTIAL_INTERVAL
        // once there's enough new audio. Bounded to the last PARTIAL_WINDOW so
        // latency stays flat on long turns.
        if seg.active() {
            let cur = seg.current();
            let enough = cur.len() >= RATE * 6 / 10; // >= 0.6 s of speech
            let new_audio = cur.len().saturating_sub(last_partial_len) >= RATE * 3 / 10; // >= 0.3 s new
            if enough && new_audio && last_partial_at.elapsed() >= PARTIAL_INTERVAL {
                let from = cur.len().saturating_sub(PARTIAL_WINDOW);
                if let Some(text) = transcribe(&mut state, &cur[from..]) {
                    print_partial(&text);
                }
                last_partial_len = cur.len();
                last_partial_at = Instant::now();
            }
        }
    }

    running.store(false, Ordering::SeqCst);
    let _ = cap.join();
    // Flush a final in-progress utterance, if any.
    if let Some(utt) = seg.flush() {
        emit(&mut state, &utt, &mut transcript);
    }
    transcript.flush().ok();
    eprintln!("\nDone. Transcript saved to: {}", transcript_path.display());
    Ok(())
}

/// Simple energy-based utterance segmenter with hysteresis + silence hangover.
struct Segmenter {
    in_speech: bool,
    utt: Vec<f32>,
    silence: usize,
    speech_frames: usize,
}

impl Segmenter {
    fn new() -> Self {
        Self { in_speech: false, utt: Vec::new(), silence: 0, speech_frames: 0 }
    }

    /// Feed one 30 ms frame; returns a finished utterance when one closes.
    fn push(&mut self, frame: &[f32]) -> Option<Vec<f32>> {
        let rms = rms(frame);
        if !self.in_speech {
            if rms > RMS_ON {
                self.in_speech = true;
                self.utt.clear();
                self.utt.extend_from_slice(frame);
                self.silence = 0;
                self.speech_frames = 1;
            }
            return None;
        }
        self.utt.extend_from_slice(frame);
        if rms < RMS_OFF {
            self.silence += 1;
        } else {
            self.silence = 0;
            self.speech_frames += 1;
        }
        if self.silence >= HANG_FRAMES || self.utt.len() >= MAX_UTT_SAMPLES {
            return self.close();
        }
        None
    }

    fn close(&mut self) -> Option<Vec<f32>> {
        let enough = self.speech_frames >= MIN_SPEECH_FRAMES;
        let utt = std::mem::take(&mut self.utt);
        self.in_speech = false;
        self.silence = 0;
        self.speech_frames = 0;
        if enough {
            Some(utt)
        } else {
            None
        }
    }

    fn flush(&mut self) -> Option<Vec<f32>> {
        if self.in_speech {
            self.close()
        } else {
            None
        }
    }

    /// True while an utterance is open (used to drive interim partials).
    fn active(&self) -> bool {
        self.in_speech
    }

    /// The audio accumulated so far in the open utterance.
    fn current(&self) -> &[f32] {
        &self.utt
    }
}

fn rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let s: f32 = frame.iter().map(|x| x * x).sum();
    (s / frame.len() as f32).sqrt()
}

/// Transcribe one utterance and emit it to the console and the transcript file
/// (timestamped with wall-clock local time, flushed immediately).
fn emit(state: &mut whisper_rs::WhisperState, utt: &[f32], transcript: &mut BufWriter<File>) {
    if let Some(text) = transcribe(state, utt) {
        let ts = Local::now().format("%H:%M:%S");
        clear_line();
        println!("[{ts}] {text}"); // permanent final line
        std::io::stdout().flush().ok();
        // Append + flush so the file is current even if the app is killed.
        let _ = writeln!(transcript, "[{ts}] {text}");
        let _ = transcript.flush();
    }
}

/// Show an interim (not-yet-final) line, overwriting the current console line.
/// Partials are console-only — only finalized lines go to the file. The tail is
/// shown (most recent words) and kept to one line.
fn print_partial(text: &str) {
    const MAXW: usize = 100;
    let shown: String = if text.chars().count() > MAXW {
        let tail: String = text.chars().rev().take(MAXW).collect::<Vec<_>>().into_iter().rev().collect();
        format!("...{tail}")
    } else {
        text.to_string()
    };
    clear_line();
    print!("  ~ {shown}");
    std::io::stdout().flush().ok();
}

/// Erase the current console line (cp932-safe ASCII).
fn clear_line() {
    print!("\r{}\r", " ".repeat(120));
}

fn transcribe(state: &mut whisper_rs::WhisperState, utt: &[f32]) -> Option<String> {
    // whisper.cpp wants >=~1s; pad short utterances with silence.
    let mut audio = utt.to_vec();
    if audio.len() < RATE {
        audio.resize(RATE, 0.0);
    }

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_no_context(true);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_single_segment(false);

    state.full(params, &audio).ok()?;
    let n = state.full_n_segments();
    let mut text = String::new();
    for i in 0..n {
        if let Some(seg) = state.get_segment(i) {
            // Skip segments whisper itself thinks are non-speech (reduces music /
            // silence hallucinations).
            if seg.no_speech_probability() > 0.6 {
                continue;
            }
            if let Ok(t) = seg.to_str_lossy() {
                text.push_str(t.trim());
                text.push(' ');
            }
        }
    }
    let text = text.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Capture the default render device's loopback as 16 kHz mono f32.
fn capture_loopback(running: &AtomicBool, tx: &mpsc::Sender<Vec<f32>>) -> Res<()> {
    wasapi::initialize_mta().ok()?;
    let enumerator = DeviceEnumerator::new()?;
    let device = enumerator.get_default_device_for_role(&Direction::Render, &Role::Console)?;
    let mut client = device.get_iaudioclient()?;

    let fmt = WaveFormat::new(32, 32, &SampleType::Float, RATE, 1, None);
    let (_def, min_time) = client.get_device_period()?;
    let mode = StreamMode::EventsShared { autoconvert: true, buffer_duration_hns: min_time };
    client.initialize_client(&fmt, &Direction::Capture, &mode)?;

    let h_event = client.set_get_eventhandle()?;
    let capture = client.get_audiocaptureclient()?;

    let mut bytes: VecDeque<u8> = VecDeque::new();
    client.start_stream()?;
    while running.load(Ordering::SeqCst) {
        capture.read_from_device_to_deque(&mut bytes)?;
        if bytes.len() >= 4 {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            while bytes.len() >= 4 {
                let b = [
                    bytes.pop_front().unwrap(),
                    bytes.pop_front().unwrap(),
                    bytes.pop_front().unwrap(),
                    bytes.pop_front().unwrap(),
                ];
                out.push(f32::from_le_bytes(b));
            }
            if tx.send(out).is_err() {
                break;
            }
        }
        let _ = h_event.wait_for_event(200);
    }
    client.stop_stream()?;
    Ok(())
}
