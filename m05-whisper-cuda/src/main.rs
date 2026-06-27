//! M0.5 — validate native whisper.cpp + CUDA on this machine.
//!
//! Usage:
//!   m05-whisper-cuda <model.bin> [audio.wav]
//!
//! Confirms the native engine runs on the GPU (not CPU), prints the transcript,
//! and reports the real-time factor so we can compare against the M0
//! faster-whisper baseline (RTF 0.06x, WER 0.0%).

use std::path::PathBuf;
use std::time::Instant;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn main() {
    if let Err(e) = run() {
        eprintln!("\nERROR: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .ok_or("usage: m05-whisper-cuda <model.bin> [audio.wav]")?;

    // Default to the M0 test clip so accuracy is directly comparable.
    let wav_path: PathBuf = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let root = env!("CARGO_MANIFEST_DIR");
            PathBuf::from(root).join("..").join("models").join("test.wav")
        });

    println!("model : {model_path}");
    println!("audio : {}", wav_path.display());

    let (samples, audio_secs) = load_wav_16k_mono(&wav_path)?;
    println!("clip  : {audio_secs:.1}s ({} samples)", samples.len());

    // --- load model on GPU ---
    let t0 = Instant::now();
    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())?;
    println!("loaded model in {:.1}s", t0.elapsed().as_secs_f32());
    // whisper.cpp prints "ggml_cuda_init" / "using CUDA" to stderr on a GPU build;
    // watch for it to confirm we are NOT on a CPU fallback.

    let mut state = ctx.create_state()?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // warm-up (first run pays kernel/JIT cost), then the measured run
    state.full(params.clone(), &samples)?;

    let t0 = Instant::now();
    state.full(params, &samples)?;
    let proc = t0.elapsed().as_secs_f32();

    // whisper-rs 0.16: full_n_segments() returns c_int (no Result); segment text
    // is read via get_segment(i) -> WhisperSegment::to_str_lossy().
    let n = state.full_n_segments();
    let mut text = String::new();
    for i in 0..n {
        if let Some(seg) = state.get_segment(i) {
            text.push_str(seg.to_str_lossy()?.trim());
            text.push(' ');
        }
    }

    let rtf = proc / audio_secs;
    println!("\nprocessing time : {proc:.2}s");
    println!(
        "real-time factor: {rtf:.2}x  ({} than real time)",
        if rtf < 1.0 { "FASTER" } else { "SLOWER" }
    );
    println!("\nTRANSCRIPT:\n{}", text.trim());
    println!("\n(compare to M0 ground truth in models/test.txt)");
    Ok(())
}

/// Load a 16 kHz mono WAV into f32 samples in [-1, 1]. The M0 test clip is
/// already 16 kHz/16-bit/mono, so this stays deliberately simple.
fn load_wav_16k_mono(path: &std::path::Path) -> Result<(Vec<f32>, f32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 {
        return Err(format!(
            "expected 16 kHz mono, got {} Hz / {} ch (regenerate with Make-TestWav.ps1)",
            spec.sample_rate, spec.channels
        )
        .into());
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };
    let secs = samples.len() as f32 / spec.sample_rate as f32;
    Ok((samples, secs))
}
