//! live-stream — Google-style live captions from system audio.
//!
//! Pipeline: system loopback -> sherpa-onnx streaming Zipformer (word-by-word
//! partials) -> truecasing (live) + CT punctuation (finals) -> per-utterance
//! speaker diarization (voice embedding -> "Speaker N" label) -> a word-wrapped
//! scrolling transcript with the live partial pinned below. CPU-only. Finals are
//! saved to a transcript file.
//!
//! Usage:  live-stream [seconds] [asr_model_dir] [transcript.txt]

mod render;
mod streaming;

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
use render::Live;
use sherpa_rs::embedding_manager::EmbeddingManager;
use sherpa_rs::punctuate::{Punctuation, PunctuationConfig};
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};
use streaming::Streamer;
use wasapi::{DeviceEnumerator, Direction, Role, SampleType, StreamMode, WaveFormat};

const RATE: usize = 16_000;
/// Cosine threshold for matching a voice to a known speaker (CAM++/voxceleb).
const SPEAKER_THRESHOLD: f32 = 0.5;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    if let Err(e) = run() {
        eprintln!("\nERROR: {e}");
        std::process::exit(1);
    }
}

fn run() -> Res<()> {
    let mut args = std::env::args().skip(1);
    let secs: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(300);
    let models = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models");
    let model_dir: PathBuf = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| models.join("sherpa-onnx-streaming-zipformer-en-2023-06-26"));
    let transcript_path: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| {
        let stamp = Local::now().format("%Y%m%d-%H%M%S");
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("transcripts").join(format!("stream-{stamp}.txt"))
    });
    let punct_model = models.join("sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12").join("model.onnx");
    let speaker_model = models.join("3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx");

    if let Some(p) = transcript_path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let mut transcript = BufWriter::new(File::create(&transcript_path)?);
    writeln!(transcript, "# Voice2Text live-stream transcript — started {}", Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    transcript.flush()?;

    eprintln!("Loading models (streaming ASR + punctuation + speaker) ...");
    let mut asr = Streamer::new(&model_dir)?;
    let mut punct = punct_model.exists().then(|| {
        Punctuation::new(PunctuationConfig { model: punct_model.to_string_lossy().into_owned(), ..Default::default() }).ok()
    }).flatten();
    let mut diarizer = Diarizer::new(&speaker_model);
    eprintln!("Saving transcript to: {}", transcript_path.display());
    eprintln!("Listening {secs}s — play a YouTube talk / join a meeting through your speakers.");

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

    let mut live = Live::new();
    let mut utt_audio: Vec<f32> = Vec::new(); // raw audio of the current utterance (for diarization)
    let start = Instant::now();
    let mut last_partial = String::new();

    while start.elapsed().as_secs() < secs {
        let chunk = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(c) => c,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => break,
        };
        asr.accept(&chunk);
        utt_audio.extend_from_slice(&chunk);

        let text = asr.partial();
        if !text.is_empty() && text != last_partial {
            live.set_partial(&format!("> {}", truecase(&text)));
            last_partial = text.clone();
        }
        if asr.is_endpoint() {
            if !text.is_empty() {
                let polished = finalize(&text, punct.as_mut());
                let speaker = diarizer.label(&utt_audio);
                let ts = Local::now().format("%H:%M:%S");
                let line = format!("[{ts}] {speaker}: {polished}");
                live.commit(&line);
                let _ = writeln!(transcript, "{line}");
                let _ = transcript.flush();
            }
            asr.reset();
            utt_audio.clear();
            last_partial.clear();
        }
    }

    running.store(false, Ordering::SeqCst);
    let _ = cap.join();
    transcript.flush().ok();
    println!("\n");
    eprintln!("Done. Transcript saved to: {}", transcript_path.display());
    Ok(())
}

/// Assigns "Speaker N" labels to utterances by voice embedding similarity.
struct Diarizer {
    extractor: Option<EmbeddingExtractor>,
    manager: EmbeddingManager,
    next_id: usize,
    last: String,
}

impl Diarizer {
    fn new(model: &std::path::Path) -> Self {
        let extractor = if model.exists() {
            EmbeddingExtractor::new(ExtractorConfig {
                model: model.to_string_lossy().into_owned(),
                provider: None,
                num_threads: Some(1),
                debug: false,
            })
            .ok()
        } else {
            None
        };
        let dim = extractor.as_ref().map(|e| e.embedding_size as i32).unwrap_or(512);
        Self { extractor, manager: EmbeddingManager::new(dim), next_id: 1, last: "Speaker 1".into() }
    }

    /// Identify the speaker of an utterance; falls back to the previous speaker
    /// for clips too short to embed reliably or if diarization is unavailable.
    fn label(&mut self, audio: &[f32]) -> String {
        let Some(ex) = self.extractor.as_mut() else { return "Speaker 1".into() };
        if audio.len() < RATE / 2 {
            return self.last.clone();
        }
        match ex.compute_speaker_embedding(audio.to_vec(), RATE as u32) {
            Ok(mut emb) => {
                let name = self.manager.search(&emb, SPEAKER_THRESHOLD).unwrap_or_else(|| {
                    let n = format!("Speaker {}", self.next_id);
                    self.next_id += 1;
                    let _ = self.manager.add(n.clone(), &mut emb);
                    n
                });
                self.last = name.clone();
                name
            }
            Err(_) => self.last.clone(),
        }
    }
}

/// Finalize an utterance: punctuation -> ASCII-normalize -> truecase.
fn finalize(caps_text: &str, punct: Option<&mut Punctuation>) -> String {
    let with_punct = match punct {
        Some(p) => p.add_punctuation(&caps_text.to_lowercase()),
        None => caps_text.to_string(),
    };
    truecase(&normalize_punct(&with_punct))
}

/// Map the zh-en model's full-width CJK punctuation to ASCII + tidy spacing.
fn normalize_punct(s: &str) -> String {
    let mut t = s
        .replace('，', ", ").replace('、', ", ").replace('。', ". ")
        .replace('？', "? ").replace('！', "! ").replace('：', ": ").replace('；', "; ");
    t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    for p in [",", ".", "?", "!", ":", ";"] {
        t = t.replace(&format!(" {p}"), p);
    }
    t.trim().to_string()
}

/// Cheap truecasing: lowercase, capitalize sentence starts and standalone "I".
fn truecase(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut sentence_start = true;
    for c in lower.chars() {
        if sentence_start && c.is_alphabetic() {
            out.extend(c.to_uppercase());
            sentence_start = false;
        } else {
            out.push(c);
        }
        if c == '.' || c == '!' || c == '?' {
            sentence_start = true;
        }
    }
    out.split(' ')
        .map(|w| if w == "i" || w.starts_with("i'") { format!("I{}", &w[1..]) } else { w.to_string() })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Capture default render loopback as 16 kHz mono f32.
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
                let b = [bytes.pop_front().unwrap(), bytes.pop_front().unwrap(), bytes.pop_front().unwrap(), bytes.pop_front().unwrap()];
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
