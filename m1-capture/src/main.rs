//! M1 — Windows dual-track audio capture stress test.
//!
//! Captures the **microphone** and the **system loopback** (what comes out of the
//! speakers = the remote attendees) simultaneously, at 16 kHz mono, and writes
//! each to its own WAV. Prints live level meters and a drift/dropout summary.
//!
//! Goal (per DESIGN.md M1): prove we can reliably grab both tracks off real
//! Zoom/Meet through speakers before building VAD/STT/UI on top. The recorded
//! WAVs become the corpus for the q5_0-vs-large-v3 accuracy and echo decisions.
//!
//! Usage:
//!   m1-capture [seconds] [out_dir]
//!   (defaults: 20 seconds, ../models/m1)

use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use wasapi::{
    AudioClientProperties, Device, DeviceEnumerator, Direction, Role, SampleType, StreamCategory,
    StreamMode, WaveFormat,
};

const TARGET_RATE: u32 = 16_000;
const TARGET_CH: u16 = 1;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A periodic level update sent from a capture thread to the UI thread.
struct Meter {
    label: &'static str,
    rms: f32,
    peak: f32,
}

/// What a capture thread reports back when it stops. The error is flattened to a
/// `String` because `Box<dyn Error>` is not `Send` and so can't cross `join`.
struct CaptureOutcome {
    result: Result<(), String>,
    silent: bool,
    frames: u64,
    note: String,
}

enum CaptureKind {
    /// Raw mic, no processing — the "me" track unmodified.
    Mic,
    /// Same physical mic, but opened as a Communications stream with Windows AEC
    /// enabled (speaker output as the echo reference). Directly comparable to Mic.
    MicAec,
    /// System loopback — the remote attendees coming out of the speakers.
    Loopback,
}

impl CaptureKind {
    fn label(&self) -> &'static str {
        match self {
            CaptureKind::Mic => "mic",
            CaptureKind::MicAec => "mic_aec",
            CaptureKind::Loopback => "loopback",
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("\nERROR: {e}");
        std::process::exit(1);
    }
}

fn run() -> Res<()> {
    let mut args = std::env::args().skip(1);
    let secs: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let out_dir: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models").join("m1")
    });
    std::fs::create_dir_all(&out_dir)?;

    wasapi::initialize_mta().ok()?;
    print_devices()?;

    let mic_path = out_dir.join("mic_16k.wav");
    let aec_path = out_dir.join("mic_aec_16k.wav");
    let loop_path = out_dir.join("loopback_16k.wav");
    println!(
        "\nRecording {secs}s ->\n  mic (raw): {}\n  mic (AEC): {}\n  loopback : {}",
        mic_path.display(), aec_path.display(), loop_path.display()
    );
    println!("\nTip: start your Zoom/Meet audio (through speakers) and speak, so the meters move.\n");

    let running = Arc::new(AtomicBool::new(true));
    let (tx, rx) = mpsc::channel::<Meter>();

    let spawn = |kind, path: PathBuf| {
        let (running, tx) = (running.clone(), tx.clone());
        thread::spawn(move || capture_thread(kind, running, tx, path))
    };
    let mic_thread = spawn(CaptureKind::Mic, mic_path);
    let aec_thread = spawn(CaptureKind::MicAec, aec_path);
    let loop_thread = spawn(CaptureKind::Loopback, loop_path);
    drop(tx); // only the threads hold senders now

    // UI loop: keep the latest meter per track and repaint a few times a second.
    let start = Instant::now();
    let mut mic_m = Meter { label: "mic", rms: 0.0, peak: 0.0 };
    let mut aec_m = Meter { label: "mic_aec", rms: 0.0, peak: 0.0 };
    let mut loop_m = Meter { label: "loopback", rms: 0.0, peak: 0.0 };
    while start.elapsed().as_secs() < secs {
        while let Ok(m) = rx.try_recv() {
            match m.label {
                "mic" => mic_m = m,
                "mic_aec" => aec_m = m,
                _ => loop_m = m,
            }
        }
        let remaining = secs.saturating_sub(start.elapsed().as_secs());
        print!(
            "\r{:>3}s mic[{}] aec[{}] loop[{}] ",
            remaining, bar(mic_m.rms, mic_m.peak), bar(aec_m.rms, aec_m.peak), bar(loop_m.rms, loop_m.peak)
        );
        std::io::stdout().flush().ok();
        thread::sleep(Duration::from_millis(250));
    }
    let elapsed = start.elapsed().as_secs_f64();

    running.store(false, Ordering::SeqCst);
    let mic_res = mic_thread.join().unwrap();
    let aec_res = aec_thread.join().unwrap();
    let loop_res = loop_thread.join().unwrap();
    println!("\n");

    report("mic", &mic_res, elapsed);
    report("mic_aec", &aec_res, elapsed);
    report("loopback", &loop_res, elapsed);

    println!("\nDone. Compare the WAVs: mic(raw) vs mic(AEC) should differ when the");
    println!("other side is talking through your speakers — AEC should remove that echo.");
    Ok(())
}

fn capture_thread(
    kind: CaptureKind,
    running: Arc<AtomicBool>,
    tx: mpsc::Sender<Meter>,
    wav_path: PathBuf,
) -> CaptureOutcome {
    let mut silent = true;
    let mut frames: u64 = 0;
    let mut note = String::new();
    let result = capture_inner(kind, &running, &tx, &wav_path, &mut silent, &mut frames, &mut note)
        .map_err(|e| e.to_string());
    CaptureOutcome { result, silent, frames, note }
}

fn capture_inner(
    kind: CaptureKind,
    running: &AtomicBool,
    tx: &mpsc::Sender<Meter>,
    wav_path: &PathBuf,
    silent: &mut bool,
    frames: &mut u64,
    note: &mut String,
) -> Res<()> {
    let label = kind.label();

    wasapi::initialize_mta().ok()?;
    let enumerator = DeviceEnumerator::new()?;
    // Mic / MicAec: default Console capture device (same physical mic).
    // Loopback: default Console RENDER device opened for capture — the wasapi
    // crate auto-sets the loopback flag for a Render device + Capture direction.
    let device = match kind {
        CaptureKind::Mic | CaptureKind::MicAec => {
            enumerator.get_default_device_for_role(&Direction::Capture, &Role::Console)?
        }
        CaptureKind::Loopback => enumerator.get_default_device_for_role(&Direction::Render, &Role::Console)?,
    };

    let mut client = device.get_iaudioclient()?;

    // For the AEC track, declare the stream as Communications BEFORE init so the
    // OS will apply the voice-capture DSP (incl. AEC) to it.
    if matches!(kind, CaptureKind::MicAec) {
        let props = AudioClientProperties::new().set_category(StreamCategory::Communications);
        client.set_properties(props)?;
    }

    // Ask WASAPI to deliver 16 kHz mono i16 directly (autoconvert handles
    // resample + channel mixdown in shared mode), so all tracks share a format.
    let fmt = WaveFormat::new(16, 16, &SampleType::Int, TARGET_RATE as usize, TARGET_CH as usize, None);
    let (_def, min_time) = client.get_device_period()?;
    let mode = StreamMode::EventsShared { autoconvert: true, buffer_duration_hns: min_time };
    client
        .initialize_client(&fmt, &Direction::Capture, &mode)
        .map_err(|e| format!("{label}: initialize_client failed (autoconvert to 16k mono may be unsupported on this endpoint): {e}"))?;

    // Enable AEC for the AEC track, referencing the default render endpoint as the
    // echo source. This is the non-lossy echo strategy from DESIGN.md §3.6.
    if matches!(kind, CaptureKind::MicAec) {
        match client.is_aec_supported() {
            Ok(true) => {
                let aec = client.get_aec_control()?;
                let render = enumerator.get_default_device_for_role(&Direction::Render, &Role::Console)?;
                aec.set_echo_cancellation_render_endpoint(Some(render.get_id()?))?;
                *note = format!("AEC supported & enabled (ref: {})", name_of(&render));
            }
            Ok(false) => *note = "AEC NOT supported on this mic endpoint".to_string(),
            Err(e) => *note = format!("AEC support check failed: {e}"),
        }
    }

    let h_event = client.set_get_eventhandle()?;
    let _buffer_frames = client.get_buffer_size()?;
    let capture = client.get_audiocaptureclient()?;

    let spec = hound::WavSpec {
        channels: TARGET_CH,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(wav_path, spec)?;

    let mut bytes: VecDeque<u8> = VecDeque::new();
    let mut win_sumsq: f64 = 0.0;
    let mut win_n: u64 = 0;
    let mut win_peak: f32 = 0.0;
    let mut last_emit = Instant::now();

    client.start_stream()?;
    while running.load(Ordering::SeqCst) {
        capture.read_from_device_to_deque(&mut bytes)?;
        // Drain complete 2-byte (i16, little-endian) samples.
        while bytes.len() >= 2 {
            let b0 = bytes.pop_front().unwrap();
            let b1 = bytes.pop_front().unwrap();
            let s = i16::from_le_bytes([b0, b1]);
            writer.write_sample(s)?;
            *frames += 1;
            let f = s as f32 / 32768.0;
            win_sumsq += (f * f) as f64;
            win_n += 1;
            win_peak = win_peak.max(f.abs());
            if f.abs() > 0.003 {
                *silent = false;
            }
        }
        // Emit a meter ~5x/sec.
        if last_emit.elapsed().as_millis() >= 200 && win_n > 0 {
            let rms = (win_sumsq / win_n as f64).sqrt() as f32;
            let _ = tx.send(Meter { label, rms, peak: win_peak });
            win_sumsq = 0.0;
            win_n = 0;
            win_peak = 0.0;
            last_emit = Instant::now();
        }
        // Event-driven wait with a short timeout so we notice the stop flag.
        let _ = h_event.wait_for_event(200);
    }
    client.stop_stream()?;
    writer.finalize()?;
    Ok(())
}

/// Render a small ASCII meter bar (cp932-safe, ASCII only). `#` = RMS level,
/// `|` = recent peak, on a ~60 dB scale so quiet speech is still visible.
fn bar(rms: f32, peak: f32) -> String {
    const W: usize = 16;
    let to_n = |v: f32| -> usize {
        let level = if v <= 0.0 { 0.0 } else { (20.0 * v.log10() + 60.0) / 60.0 };
        (level.clamp(0.0, 1.0) * W as f32).round() as usize
    };
    let n = to_n(rms);
    let peak_n = to_n(peak);
    let mut s = String::with_capacity(W);
    for i in 0..W {
        if i < n {
            s.push('#');
        } else if peak_n > 0 && i == peak_n.min(W - 1) {
            s.push('|');
        } else {
            s.push(' ');
        }
    }
    s
}

fn report(label: &str, outcome: &CaptureOutcome, elapsed: f64) {
    match &outcome.result {
        Ok(()) => {
            // Drift/dropout check: captured frames vs what 16 kHz * wall-clock
            // would predict. A large shortfall means dropouts or clock drift.
            let expected = (TARGET_RATE as f64 * elapsed).round() as i64;
            let actual = outcome.frames as i64;
            let pct = if expected > 0 { (actual - expected) as f64 / expected as f64 * 100.0 } else { 0.0 };
            let warn = if outcome.silent { "  WARNING: track ~silent (no audio captured)" } else { "" };
            println!(
                "[{label:<8}] OK  frames={actual}  expected~{expected}  drift={pct:+.2}%  ({:.2}s recorded){warn}",
                actual as f64 / TARGET_RATE as f64
            );
            if !outcome.note.is_empty() {
                println!("           {}", outcome.note);
            }
        }
        Err(e) => println!("[{label:<8}] FAILED: {e}"),
    }
}

fn print_devices() -> Res<()> {
    let enumerator = DeviceEnumerator::new()?;
    println!("=== Audio endpoints ===");
    for (dir_name, dir) in [("RENDER (speakers)", Direction::Render), ("CAPTURE (mics)", Direction::Capture)] {
        println!("\n{dir_name}:");
        let coll = enumerator.get_device_collection(&dir)?;
        let n = coll.get_nbr_devices()?;
        for i in 0..n {
            if let Ok(dev) = coll.get_device_at_index(i) {
                println!("  - {}", name_of(&dev));
            }
        }
        // Role defaults matter: Zoom often uses the COMMUNICATIONS default, which
        // can differ from the CONSOLE/MULTIMEDIA default used for media playback.
        for role in [Role::Console, Role::Multimedia, Role::Communications] {
            if let Ok(dev) = enumerator.get_default_device_for_role(&dir, &role) {
                println!("    default[{:<14}] = {}", role.to_string(), name_of(&dev));
            }
        }
    }
    Ok(())
}

fn name_of(dev: &Device) -> String {
    dev.get_friendlyname().unwrap_or_else(|_| "<unknown>".into())
}
