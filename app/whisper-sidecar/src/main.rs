//! Whisper-CUDA sidecar for Voice2Text.
//!
//! The main app is CPU-only (sherpa streaming) so its launcher stays small and
//! has no CUDA link. When the user wants the clean saved transcript, the app
//! spawns this process and streams each finalized utterance to it for accurate
//! re-transcription with Whisper large-v3 on the GPU.
//!
//! Protocol (binary on stdin, text on stdout; whisper logs go to stderr):
//!   startup  -> sidecar prints "READY\n" once the GPU model is loaded,
//!               or "ERROR <message>\n" and exits non-zero on failure.
//!   per job  -> app writes u32 LE sample-count, then that many f32 LE samples
//!               (16 kHz mono). Sidecar runs Whisper and prints exactly one
//!               line: the transcript (possibly empty) terminated by "\n".
//!   shutdown -> app closes stdin (EOF); sidecar exits.
//!
//! Usage: whisper-sidecar <model.bin> [language]

use std::io::{self, BufWriter, Read, Write};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn main() {
    let mut args = std::env::args().skip(1);
    let model = match args.next() {
        Some(m) => m,
        None => {
            // stdout is the control channel; emit a protocol ERROR line.
            let mut out = io::stdout();
            let _ = writeln!(out, "ERROR missing model path argument");
            let _ = out.flush();
            std::process::exit(2);
        }
    };
    let language = args.next().unwrap_or_else(|| "en".to_string());

    if let Err(code) = run(&model, &language) {
        std::process::exit(code);
    }
}

fn run(model: &str, language: &str) -> Result<(), i32> {
    let mut control = io::stdout();

    let ctx = match WhisperContext::new_with_params(model, WhisperContextParameters::default()) {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(control, "ERROR loading model: {e}");
            let _ = control.flush();
            return Err(3);
        }
    };
    let mut state = match ctx.create_state() {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(control, "ERROR creating state: {e}");
            let _ = control.flush();
            return Err(4);
        }
    };

    // Model is on the GPU and ready.
    let _ = writeln!(control, "READY");
    let _ = control.flush();

    let mut stdin = io::stdin().lock();
    let mut out = BufWriter::new(io::stdout().lock());

    loop {
        // Read the 4-byte sample count; clean EOF here means shutdown.
        let mut len_buf = [0u8; 4];
        match read_full(&mut stdin, &mut len_buf) {
            Ok(true) => {}
            Ok(false) => break, // EOF between jobs
            Err(_) => break,
        }
        let n = u32::from_le_bytes(len_buf) as usize;

        let mut bytes = vec![0u8; n * 4];
        if read_full(&mut stdin, &mut bytes).map(|ok| !ok).unwrap_or(true) {
            break; // truncated job -> give up
        }
        let samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let text = transcribe(&mut state, &samples, language);
        // Exactly one line per job, preserving request order.
        let _ = writeln!(out, "{text}");
        let _ = out.flush();
    }

    Ok(())
}

fn transcribe(state: &mut whisper_rs::WhisperState, samples: &[f32], language: &str) -> String {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(language));
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    if state.full(params, samples).is_err() {
        return String::new();
    }
    let mut text = String::new();
    for i in 0..state.full_n_segments() {
        if let Some(seg) = state.get_segment(i) {
            if let Ok(s) = seg.to_str_lossy() {
                text.push_str(s.trim());
                text.push(' ');
            }
        }
    }
    // Collapse to a single clean line (the protocol is line-delimited).
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Read exactly `buf.len()` bytes. Ok(true) = filled, Ok(false) = EOF (clean
/// shutdown or a truncated final record — either way the caller stops), Err =
/// I/O error.
fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut read = 0;
    while read < buf.len() {
        match r.read(&mut buf[read..]) {
            Ok(0) => return Ok(false),
            Ok(n) => read += n,
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}
