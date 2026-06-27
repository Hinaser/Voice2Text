//! Thin safe-ish wrapper around the sherpa-onnx ONLINE (streaming) transducer
//! recognizer (validated in m35-sherpa). Encapsulates the raw FFI.

use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

use sherpa_rs::sherpa_rs_sys as sys;

pub struct Streamer {
    recognizer: *const sys::SherpaOnnxOnlineRecognizer,
    stream: *const sys::SherpaOnnxOnlineStream,
    // keep model-path CStrings alive for the recognizer's lifetime
    _keep: Vec<CString>,
}

impl Streamer {
    pub fn new(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let encoder = find(model_dir, "encoder")?;
        let decoder = find(model_dir, "decoder")?;
        let joiner = find(model_dir, "joiner")?;
        let tokens = model_dir.join("tokens.txt");

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
                enable_endpoint: 1,
                // snappier finals than the defaults (meeting-pace pauses)
                rule1_min_trailing_silence: 2.0,
                rule2_min_trailing_silence: 0.8,
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

        Ok(Self {
            recognizer,
            stream,
            _keep: vec![c_encoder, c_decoder, c_joiner, c_tokens, c_provider, c_method],
        })
    }

    /// Feed audio (16 kHz mono f32) and run the decoder on what's ready.
    pub fn accept(&mut self, samples: &[f32]) {
        unsafe {
            sys::SherpaOnnxOnlineStreamAcceptWaveform(self.stream, 16000, samples.as_ptr(), samples.len() as i32);
            while sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) != 0 {
                sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
            }
        }
    }

    /// Current (partial) hypothesis text.
    pub fn partial(&self) -> String {
        unsafe {
            let r = sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);
            let text = if r.is_null() || (*r).text.is_null() {
                String::new()
            } else {
                CStr::from_ptr((*r).text).to_string_lossy().into_owned()
            };
            if !r.is_null() {
                sys::SherpaOnnxDestroyOnlineRecognizerResult(r);
            }
            text.trim().to_string()
        }
    }

    pub fn is_endpoint(&self) -> bool {
        unsafe { sys::SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, self.stream) != 0 }
    }

    pub fn reset(&mut self) {
        unsafe { sys::SherpaOnnxOnlineStreamReset(self.recognizer, self.stream) };
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        unsafe {
            sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}

fn find(dir: &Path, kind: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let pick = |int8: bool| -> Option<PathBuf> {
        std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).find(|p| {
            let n = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
            n.contains(kind) && n.ends_with(".onnx") && (n.contains("int8") == int8)
        })
    };
    pick(false).or_else(|| pick(true)).ok_or_else(|| format!("no {kind}*.onnx in {}", dir.display()).into())
}
