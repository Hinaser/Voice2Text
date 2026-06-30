//! Audio → live captions + saved transcript pipeline.
//!
//! Threading model (so no non-Send FFI handle is shared across threads):
//!   - one capture thread per source (system loopback + mic) → a tagged channel;
//!   - this processing thread owns both streaming recognizers, the punctuation
//!     model, the diarizer and the echo filter, and emits UI events;
//!   - the optional Whisper sidecar runs in its own process + manager thread.
//!
//! Each concern lives in its own submodule; this file is just orchestration.

mod aec;
mod capture;
mod corrector;
mod diarize;
mod echo;
mod events;
mod seglog;
mod text;
mod transcript;
mod whisper;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chrono::Local;
use sherpa_rs::punctuate::{Punctuation, PunctuationConfig};
use tauri::AppHandle;

use crate::config::Config;
use crate::paths;
use crate::streaming::Streamer;

use aec::Aec;
use diarize::Diarizer;
use echo::EchoFilter;
use events::Ui;
use transcript::TranscriptWriter;
use whisper::{WhisperJob, WhisperSidecar};

/// Capture/recognition sample rate (16 kHz mono).
pub const RATE: usize = 16_000;

/// Capture-session lifecycle, driven from the UI (Start / Pause / Stop). The
/// pipeline polls this each loop: capture threads are spawned on Running and
/// torn down on Stop (releasing the audio devices); Paused keeps them alive but
/// discards incoming audio.
pub mod capture_state {
    pub const STOPPED: u8 = 0;
    pub const RUNNING: u8 = 1;
    pub const PAUSED: u8 = 2;
}

/// Shared, atomically-updated capture state. Cheap to clone behind an `Arc`; the
/// UI mutates it via the `set_capture` command, the pipeline reads it each loop.
#[derive(Default)]
pub struct CaptureControl {
    state: std::sync::atomic::AtomicU8,
}

impl CaptureControl {
    pub fn new(initial: u8) -> Self {
        Self { state: std::sync::atomic::AtomicU8::new(initial) }
    }

    pub fn get(&self) -> u8 {
        self.state.load(Ordering::SeqCst)
    }

    pub fn set(&self, value: u8) {
        self.state.store(value, Ordering::SeqCst);
    }
}

/// Which side of the conversation an utterance came from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Source {
    System,
    Mic,
}

impl Source {
    /// UI/track tag: attendees vs the user.
    pub fn tag(self) -> &'static str {
        match self {
            Source::System => "others",
            Source::Mic => "me",
        }
    }
}

/// Per-source streaming state held on the processing thread.
struct Track {
    asr: Streamer,
    utt: Vec<f32>,
    last_partial: String,
}

impl Track {
    fn new(asr_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self { asr: Streamer::new(asr_dir)?, utt: Vec::new(), last_partial: String::new() })
    }

    fn reset(&mut self) {
        self.asr.reset();
        self.utt.clear();
        self.last_partial.clear();
    }
}

pub fn run(
    app: AppHandle,
    running: Arc<AtomicBool>,
    config: Arc<Mutex<Config>>,
    control: Arc<CaptureControl>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ui = Ui::new(app);
    let cfg0 = config.lock().unwrap().clone();
    let models = paths::models_dir(&cfg0.models_dir);
    let asr_dir = models.join("sherpa-onnx-streaming-zipformer-en-2023-06-26");
    let punct_model = models.join("sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12").join("model.onnx");
    let speaker_model = models.join("3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx");
    let whisper_model = models.join("ggml-large-v3-q5_0.bin");

    ui.status("loading", "Loading models…");
    let mut sys_track = Track::new(&asr_dir)?;
    let mut mic_track = if cfg0.mic_capture { Some(Track::new(&asr_dir)?) } else { None };
    let mut punct = load_punct(&punct_model);
    let mut diarizer = Diarizer::new(&speaker_model);

    // Optional Whisper sidecar; falls back to streaming text on any failure.
    let whisper = if cfg0.whisper_transcript && whisper_model.exists() {
        ui.status("loading", "Starting Whisper sidecar (GPU)…");
        WhisperSidecar::spawn(&whisper_model, config.clone(), ui.clone())
    } else {
        None
    };
    // Used only when the sidecar is off; lazily opened on first save so it picks
    // up the configured folder at that moment.
    let mut fallback_writer: Option<TranscriptWriter> = None;

    // Persistent status-bar summary: which endpoints we're capturing and where
    // the transcript goes. The file path follows later via `saving` once opened.
    let sources = crate::audio::capture_source_names(&cfg0.output_device, &cfg0.input_device, mic_track.is_some());
    ui.capture(sources, cfg0.save_transcript, cfg0.resolved_save_dir().to_string_lossy().into_owned());

    let mut echo = EchoFilter::new();
    // Signal-level echo cancellation: only meaningful when capturing the mic,
    // and only built when enabled. Reset per session below alongside the tracks.
    let mut aec = (cfg0.mic_capture && cfg0.acoustic_echo_cancel).then(Aec::new);
    // Monotonic id per finalized line, stable across capture sessions so a later
    // Whisper/LLM refinement can replace the right line in the UI.
    let mut line_id: u64 = 0;
    // For segment diagnostics: when the previous line (per source) finalized, so
    // we can log the inter-utterance gap.
    let mut last_final: std::collections::HashMap<&'static str, Instant> = std::collections::HashMap::new();

    // Outer session loop: idle until the UI starts capture, run a capture
    // session until it's stopped (or the app exits), then loop. Models above are
    // loaded once and reused across sessions; only the audio devices are
    // acquired/released per session.
    while running.load(Ordering::SeqCst) {
        // Idle until Started (or the app exits).
        if control.get() != capture_state::RUNNING {
            ui.status("idle", "Capture stopped — press ▶ to start");
            while running.load(Ordering::SeqCst) && control.get() != capture_state::RUNNING {
                thread::sleep(Duration::from_millis(100));
            }
            if !running.load(Ordering::SeqCst) {
                break;
            }
        }

        // Start a capture session: spawn the device threads onto a tagged channel.
        let session = Arc::new(AtomicBool::new(true));
        ui.status("listening", listening_detail(mic_track.is_some(), whisper.is_some()));
        let (tx, rx) = mpsc::channel::<(Source, Vec<f32>)>();
        let sys_cap = spawn_capture(Source::System, &cfg0.output_device, &session, &tx, &ui);
        let mic_cap = mic_track
            .as_ref()
            .map(|_| spawn_capture(Source::Mic, &cfg0.input_device, &session, &tx, &ui));
        drop(tx);

        let mut paused = false;
        while running.load(Ordering::SeqCst) && control.get() != capture_state::STOPPED {
            // Paused: keep the devices open but drop audio so nothing is
            // transcribed; reset the in-flight utterances so resume starts clean.
            if control.get() == capture_state::PAUSED {
                if !paused {
                    paused = true;
                    ui.status("paused", "Capture paused");
                    sys_track.reset();
                    ui.clear_partial(Source::System);
                    if let Some(t) = mic_track.as_mut() {
                        t.reset();
                    }
                    if let Some(a) = aec.as_mut() {
                        a.reset();
                    }
                    ui.clear_partial(Source::Mic);
                }
                // Drain and discard so the channel doesn't back up while paused.
                let _ = rx.recv_timeout(Duration::from_millis(100));
                continue;
            }
            if paused {
                paused = false;
                ui.status("listening", listening_detail(mic_track.is_some(), whisper.is_some()));
            }

            let (source, chunk) = match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(v) => v,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            };

            // Signal-level AEC: the loopback is the echo reference; the mic is
            // cleaned before it reaches the recognizer (and the saved audio).
            let chunk = match (source, aec.as_mut()) {
                (Source::System, Some(a)) => {
                    a.push_reference(&chunk);
                    chunk
                }
                (Source::Mic, Some(a)) => a.process_capture(&chunk),
                _ => chunk,
            };

            let track = match source {
                Source::System => &mut sys_track,
                Source::Mic => match mic_track.as_mut() {
                    Some(t) => t,
                    None => continue,
                },
            };

            track.asr.accept(&chunk);
            track.utt.extend_from_slice(&chunk);

            let text = track.asr.partial();
            if !text.is_empty() && text != track.last_partial {
                ui.partial(source, text::truecase_partial(&text));
                track.last_partial = text.clone();
                // Feed the system partial to the echo filter so a mic echo that
                // finalizes before the matching system line can still be caught.
                if source == Source::System {
                    echo.record_partial(Instant::now(), &text);
                }
            }

            if track.asr.is_endpoint() {
                if !text.is_empty() {
                    let cfg = config.lock().unwrap().clone();
                    let polished = text::finalize(&text, if cfg.punctuation { punct.as_mut() } else { None });
                    let now = Instant::now();
                    let time = Local::now().format("%H:%M:%S").to_string();

                    let (speaker, suppress) = classify(source, &polished, &cfg, &mut diarizer, &mut echo, now, &track.utt);

                    if !suppress {
                        let id = line_id;
                        line_id += 1;

                        if seglog::enabled() {
                            let gap_ms = last_final
                                .get(source.tag())
                                .map(|t| now.duration_since(*t).as_millis() as i64)
                                .unwrap_or(-1);
                            let utt_ms = track.utt.len() as u128 * 1000 / RATE as u128;
                            seglog::log(&format!(
                                "t={time} id={id} src={} gap_ms={gap_ms} utt_ms={utt_ms} stream={:?}",
                                source.tag(),
                                polished
                            ));
                            last_final.insert(source.tag(), now);
                        }

                        ui.final_line(id, time.clone(), source, speaker.clone(), polished.clone());
                        ui.clear_partial(source);

                        let label = transcript::speaker_label(&speaker);
                        if let Some(w) = whisper.as_ref() {
                            // Sidecar re-transcribes → replaces the live line with
                            // clean (then LLM-polished) text and writes the file.
                            // Sent regardless of save so the live caption is
                            // upgraded even when not saving; the manager gates the
                            // file write on save_transcript itself.
                            w.send(WhisperJob { id, audio: track.utt.clone(), time, label });
                        } else if cfg.save_transcript {
                            let writer = fallback_writer
                                .get_or_insert_with(|| TranscriptWriter::new(cfg.resolved_save_dir()));
                            match writer.write_line(&time, &label, &polished) {
                                Ok(Some(path)) => ui.saving(path.to_string_lossy().into_owned()),
                                Ok(None) => {}
                                Err(e) => ui.status("error", format!("Save failed: {e}")),
                            }
                        }
                    }
                }
                track.reset();
            }
        }

        // End the session: stop the device threads (releasing the endpoints),
        // then reset recognizer state so the next session starts fresh.
        session.store(false, Ordering::SeqCst);
        let _ = sys_cap.join();
        if let Some(h) = mic_cap {
            let _ = h.join();
        }
        sys_track.reset();
        ui.clear_partial(Source::System);
        if let Some(t) = mic_track.as_mut() {
            t.reset();
        }
        if let Some(a) = aec.as_mut() {
            a.reset();
        }
        ui.clear_partial(Source::Mic);
    }
    Ok(())
}

/// Decide the speaker label and whether the utterance should be suppressed as
/// echo. Also records system lines into the echo filter.
fn classify(
    source: Source,
    polished: &str,
    cfg: &Config,
    diarizer: &mut Diarizer,
    echo: &mut EchoFilter,
    now: Instant,
    utt: &[f32],
) -> (String, bool) {
    match source {
        Source::System => {
            let speaker = if cfg.diarization { diarizer.label(utt) } else { String::new() };
            echo.record_final(now, polished);
            (speaker, false)
        }
        Source::Mic => {
            let suppress = cfg.echo_suppression && echo.is_echo(now, polished);
            ("You".to_string(), suppress)
        }
    }
}

fn spawn_capture(
    source: Source,
    device_id: &str,
    running: &Arc<AtomicBool>,
    tx: &mpsc::Sender<(Source, Vec<f32>)>,
    ui: &Ui,
) -> thread::JoinHandle<()> {
    let (running, tx, ui, device_id) = (running.clone(), tx.clone(), ui.clone(), device_id.to_string());
    thread::spawn(move || {
        if let Err(e) = capture::run(source, &device_id, &running, &tx) {
            let which = if source == Source::Mic { "Microphone" } else { "System audio" };
            ui.status("error", format!("{which} capture stopped: {e}"));
        }
    })
}

fn load_punct(model: &Path) -> Option<Punctuation> {
    model
        .exists()
        .then(|| Punctuation::new(PunctuationConfig { model: model.to_string_lossy().into_owned(), ..Default::default() }).ok())
        .flatten()
}

fn listening_detail(mic: bool, whisper: bool) -> &'static str {
    match (mic, whisper) {
        (true, true) => "Listening (system + mic, clean transcript)",
        (true, false) => "Listening (system + mic)",
        (false, true) => "Listening (system, clean transcript)",
        (false, false) => "Listening to system audio",
    }
}
