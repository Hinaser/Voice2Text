//! Local-LLM sidecar for Voice2Text.
//!
//! Kept out of the (CPU-only) main app like the Whisper sidecar. Reads a meeting
//! transcript on stdin, summarizes it with Qwen2.5-3B-Instruct on the GPU, and
//! writes the summary to stdout, then exits. Model/llama.cpp logs go to stderr.
//!
//! Usage: llama-sidecar <model.gguf>

use std::io::{self, Read, Write};
use std::num::NonZeroU32;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

const N_CTX: u32 = 8192;
const MAX_GEN: usize = 600;
/// Cap transcript length so prompt + generation stay within the context window.
const MAX_INPUT_CHARS: usize = 16_000;

const SYSTEM_PROMPT: &str = "You are a meeting assistant for a non-native English speaker. \
Read the meeting transcript and reply with:\n\
1) A concise summary in 3-5 short sentences.\n\
2) A bulleted list of action items (who does what, and any deadline) if there are any; otherwise write \"Action items: none\".\n\
Use simple, clear English. Only use information present in the transcript.";

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = match args.next() {
        Some(m) => m,
        None => {
            eprintln!("usage: llama-sidecar <model.gguf>");
            std::process::exit(2);
        }
    };

    let mut transcript = String::new();
    if io::stdin().read_to_string(&mut transcript).is_err() {
        eprintln!("failed to read transcript from stdin");
        std::process::exit(3);
    }
    let transcript = transcript.trim();
    if transcript.is_empty() {
        // Nothing to summarize; emit empty and exit cleanly.
        return;
    }
    let transcript = if transcript.len() > MAX_INPUT_CHARS {
        // Keep the most recent portion (meeting ends matter most).
        &transcript[transcript.len() - MAX_INPUT_CHARS..]
    } else {
        transcript
    };

    if let Err(e) = run(&model_path, transcript) {
        eprintln!("llama-sidecar error: {e}");
        std::process::exit(1);
    }
}

fn run(model_path: &str, transcript: &str) -> Result<(), Box<dyn std::error::Error>> {
    let backend = LlamaBackend::init()?;

    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000); // offload all layers
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

    let prompt = build_prompt(&model, transcript);
    let tokens = model.str_to_token(&prompt, AddBos::Never)?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_CTX);
    let mut ctx = model.new_context(&backend, ctx_params)?;

    // Feed the prompt (only the last token needs logits).
    let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
    let last = tokens.len() as i32 - 1;
    for (i, tok) in tokens.iter().enumerate() {
        batch.add(*tok, i as i32, &[0], i as i32 == last)?;
    }
    ctx.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::temp(0.3),
        LlamaSampler::dist(0),
    ]);

    let mut out = io::stdout().lock();
    let mut n_cur = tokens.len() as i32;
    let mut idx = last;
    let mut decoder = String::new();
    for _ in 0..MAX_GEN {
        let token = sampler.sample(&ctx, idx);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        if let Ok(piece) = model.token_to_str(token, Special::Plaintext) {
            decoder.push_str(&piece);
            let _ = out.write_all(piece.as_bytes());
            let _ = out.flush();
        }
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
        idx = 0;
    }
    let _ = out.write_all(b"\n");
    let _ = out.flush();
    Ok(())
}

/// Build the chat prompt using the model's baked-in template, falling back to a
/// manual ChatML prompt (Qwen uses ChatML) if the template is missing.
fn build_prompt(model: &LlamaModel, transcript: &str) -> String {
    let user = format!("Transcript:\n{transcript}");
    let rendered = (|| {
        let tmpl = model.chat_template(None).ok()?;
        let sys = LlamaChatMessage::new("system".to_string(), SYSTEM_PROMPT.to_string()).ok()?;
        let usr = LlamaChatMessage::new("user".to_string(), user.clone()).ok()?;
        model.apply_chat_template(&tmpl, &[sys, usr], true).ok()
    })();
    rendered.unwrap_or_else(|| {
        format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        )
    })
}
