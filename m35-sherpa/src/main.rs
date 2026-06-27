//! M3.5 — validate streaming (online) ASR via sherpa-onnx for Google-style
//! low-latency partials.
//!
//! sherpa-rs only wraps the *offline* recognizers, so we drive the *online*
//! streaming recognizer through the raw `sherpa_rs_sys` FFI (the same API the
//! crate's keyword-spotter uses). Feeds a WAV in small chunks and prints the
//! incrementally-growing partial after each chunk + the finalized segment at each
//! endpoint — proving the streaming behavior and that it runs real-time on CPU.
//!
//! Usage:  m35-sherpa <model_dir> [wav]
//!   model_dir must contain encoder.onnx/decoder.onnx/joiner.onnx/tokens.txt
//!   (a sherpa-onnx streaming-zipformer model). Defaults wav to models/test.wav.

use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::time::Instant;

use sherpa_rs::sherpa_rs_sys as sys;

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(
        args.next()
            .ok_or("usage: m35-sherpa <model_dir> [wav]")?,
    );
    let wav_path = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models").join("test.wav")
    });

    // Locate the model files (names vary by release, so glob the dir).
    let encoder = find(&model_dir, "encoder")?;
    let decoder = find(&model_dir, "decoder")?;
    let joiner = find(&model_dir, "joiner")?;
    let tokens = model_dir.join("tokens.txt");
    println!("model dir: {}", model_dir.display());
    println!("  encoder: {}", encoder.file_name().unwrap().to_string_lossy());
    println!("  decoder: {}", decoder.file_name().unwrap().to_string_lossy());
    println!("  joiner : {}", joiner.file_name().unwrap().to_string_lossy());

    let (samples, sr) = load_wav(&wav_path)?;
    println!("audio  : {} ({:.1}s @ {} Hz)\n", wav_path.display(), samples.len() as f32 / sr as f32, sr);
    if sr != 16000 {
        return Err("model expects 16 kHz audio".into());
    }

    // Keep CStrings alive until after the recognizer is created.
    let c_encoder = CString::new(encoder.to_string_lossy().as_bytes())?;
    let c_decoder = CString::new(decoder.to_string_lossy().as_bytes())?;
    let c_joiner = CString::new(joiner.to_string_lossy().as_bytes())?;
    let c_tokens = CString::new(tokens.to_string_lossy().as_bytes())?;
    let c_provider = CString::new("cpu")?;
    let c_method = CString::new("greedy_search")?;

    let recognizer = unsafe {
        let model_config = sys::SherpaOnnxOnlineModelConfig {
            transducer: sys::SherpaOnnxOnlineTransducerModelConfig {
                encoder: c_encoder.as_ptr(),
                decoder: c_decoder.as_ptr(),
                joiner: c_joiner.as_ptr(),
            },
            tokens: c_tokens.as_ptr(),
            num_threads: 2,
            provider: c_provider.as_ptr(),
            debug: 0,
            // everything else zeroed (other model families unused)
            paraformer: std::mem::zeroed(),
            zipformer2_ctc: std::mem::zeroed(),
            model_type: std::ptr::null(),
            modeling_unit: std::ptr::null(),
            bpe_vocab: std::ptr::null(),
            tokens_buf: std::ptr::null(),
            tokens_buf_size: 0,
            nemo_ctc: std::mem::zeroed(),
        };
        let config = sys::SherpaOnnxOnlineRecognizerConfig {
            feat_config: sys::SherpaOnnxFeatureConfig { sample_rate: 16000, feature_dim: 80 },
            model_config,
            decoding_method: c_method.as_ptr(),
            max_active_paths: 4,
            // Endpoint detection → segments the stream into final utterances.
            enable_endpoint: 1,
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: 1.2,
            rule3_min_utterance_length: 20.0,
            hotwords_file: std::ptr::null(),
            hotwords_score: 0.0,
            ctc_fst_decoder_config: std::mem::zeroed(),
            rule_fsts: std::ptr::null(),
            rule_fars: std::ptr::null(),
            blank_penalty: 0.0,
            hotwords_buf: std::ptr::null(),
            hotwords_buf_size: 0,
            hr: std::mem::zeroed(),
        };
        let r = sys::SherpaOnnxCreateOnlineRecognizer(&config);
        if r.is_null() {
            return Err("SherpaOnnxCreateOnlineRecognizer failed (check model files)".into());
        }
        r
    };

    let stream = unsafe { sys::SherpaOnnxCreateOnlineStream(recognizer) };

    // Feed audio in 100 ms chunks, printing the partial each time it changes.
    const CHUNK: usize = 1600; // 0.1 s @ 16 kHz
    let t0 = Instant::now();
    let mut last = String::new();
    let mut seg = 0;
    let mut k = 0;
    println!("--- streaming (partials grow as audio is fed; FINAL at each endpoint) ---");
    while k < samples.len() {
        let end = (k + CHUNK).min(samples.len());
        unsafe {
            sys::SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, samples[k..end].as_ptr(), (end - k) as i32);
            while sys::SherpaOnnxIsOnlineStreamReady(recognizer, stream) != 0 {
                sys::SherpaOnnxDecodeOnlineStream(recognizer, stream);
            }
        }
        let pos = end as f32 / 16000.0;
        let text = result_text(recognizer, stream);
        if !text.is_empty() && text != last {
            println!("  [{pos:>4.1}s] {text}");
            last = text.clone();
        }
        if unsafe { sys::SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream) } != 0 {
            if !text.is_empty() {
                println!("  >>> FINAL #{seg}: {text}");
                seg += 1;
            }
            unsafe { sys::SherpaOnnxOnlineStreamReset(recognizer, stream) };
            last.clear();
        }
        k = end;
    }
    // Tail padding + finish to flush the last words.
    let tail = vec![0.0f32; 4800];
    unsafe {
        sys::SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, tail.as_ptr(), tail.len() as i32);
        sys::SherpaOnnxOnlineStreamInputFinished(stream);
        while sys::SherpaOnnxIsOnlineStreamReady(recognizer, stream) != 0 {
            sys::SherpaOnnxDecodeOnlineStream(recognizer, stream);
        }
    }
    let final_text = result_text(recognizer, stream);
    if !final_text.is_empty() {
        println!("  >>> FINAL #{seg}: {final_text}");
    }

    let proc = t0.elapsed().as_secs_f32();
    let audio = samples.len() as f32 / 16000.0;
    println!("\nprocessing {proc:.2}s for {audio:.1}s audio  ->  RTF {:.3} ({} than real time)",
        proc / audio, if proc < audio { "FASTER" } else { "SLOWER" });

    unsafe {
        sys::SherpaOnnxDestroyOnlineStream(stream);
        sys::SherpaOnnxDestroyOnlineRecognizer(recognizer);
    }
    Ok(())
}

fn result_text(rec: *const sys::SherpaOnnxOnlineRecognizer, stream: *const sys::SherpaOnnxOnlineStream) -> String {
    unsafe {
        let r = sys::SherpaOnnxGetOnlineStreamResult(rec, stream);
        let text = if r.is_null() || (*r).text.is_null() {
            String::new()
        } else {
            CStr::from_ptr((*r).text).to_string_lossy().into_owned()
        };
        if !r.is_null() {
            sys::SherpaOnnxDestroyOnlineRecognizerResult(r);
        }
        text
    }
}

fn find(dir: &std::path::Path, kind: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        let name = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
        if name.contains(kind) && name.ends_with(".onnx") && !name.contains("int8") {
            return Ok(p);
        }
    }
    // fall back to int8 if that's all there is
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        let name = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
        if name.contains(kind) && name.ends_with(".onnx") {
            return Ok(p);
        }
    }
    Err(format!("no {kind}*.onnx in {}", dir.display()).into())
}

fn load_wav(path: &std::path::Path) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => r.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
    };
    // downmix if needed
    let mono: Vec<f32> = if spec.channels > 1 {
        samples.chunks(spec.channels as usize).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect()
    } else {
        samples
    };
    Ok((mono, spec.sample_rate))
}
