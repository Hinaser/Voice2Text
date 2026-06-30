//! Local-LLM sidecar for Voice2Text.
//!
//! Kept out of the (CPU-only) main app like the Whisper sidecar. Two modes:
//!   - one-shot **summary** (default): read a meeting transcript on stdin,
//!     summarize it with Qwen2.5-3B-Instruct on the GPU, write it to stdout, exit;
//!   - persistent **correction server** (`--serve`): load the model once, then
//!     loop correcting one speech-to-text line at a time using recent context, so
//!     live captions can be polished without paying the model-load cost per line.
//! Model/llama.cpp logs go to stderr.
//!
//! Usage: llama-sidecar <model.gguf> [--serve]

use std::io::{self, BufRead, Read, Write};
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

// --- Correction (serve) mode -------------------------------------------------

/// Smaller window for the correction server: a line plus a little context is
/// short, so a big KV cache would just be wasted per-request allocation.
const CORRECT_N_CTX: u32 = 2048;
/// Hard ceiling on generated tokens for one correction (a single utterance).
const CORRECT_MAX_GEN: usize = 200;
/// Field separator between the context blob and the target line.
const FIELD_SEP: char = '\u{1f}';
/// Record separator between individual context lines.
const REC_SEP: char = '\u{1e}';

const CORRECT_SYSTEM_PROMPT: &str = "You proofread live speech-to-text for a meeting attended by a \
non-native English speaker. You are given recent lines for context and one NEW line to fix. Correct \
ONLY clear speech-recognition mistakes in the NEW line — misheard or wrong words, homophones, wrong \
names or terms (use the context to disambiguate), and missing short function words. Keep the original \
meaning, wording, and language. Do NOT paraphrase, translate, summarize, answer questions, or add or \
remove information. If the NEW line already looks correct, repeat it unchanged. Output ONLY the \
corrected NEW line, with no quotes, labels, or extra text.";

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = match args.next() {
        Some(m) => m,
        None => {
            eprintln!("usage: llama-sidecar <model.gguf> [--serve]");
            std::process::exit(2);
        }
    };
    let serve = args.next().as_deref() == Some("--serve");

    let result = if serve {
        serve_corrections(&model_path)
    } else {
        run_summary(&model_path)
    };
    if let Err(e) = result {
        eprintln!("llama-sidecar error: {e}");
        std::process::exit(1);
    }
}

/// One-shot summary mode: transcript on stdin → summary on stdout.
fn run_summary(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut transcript = String::new();
    if io::stdin().read_to_string(&mut transcript).is_err() {
        return Err("failed to read transcript from stdin".into());
    }
    let transcript = transcript.trim();
    if transcript.is_empty() {
        return Ok(()); // nothing to summarize
    }
    let transcript = if transcript.len() > MAX_INPUT_CHARS {
        &transcript[transcript.len() - MAX_INPUT_CHARS..] // recent portion matters most
    } else {
        transcript
    };

    let backend = LlamaBackend::init()?;
    let model = load_model(&backend, model_path)?;

    let prompt = build_summary_prompt(&model, transcript);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_CTX);
    let mut ctx = model.new_context(&backend, ctx_params)?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::temp(0.3),
        LlamaSampler::dist(0),
    ]);

    let mut out = io::stdout().lock();
    generate(&model, &mut ctx, &mut sampler, &prompt, MAX_GEN, |piece| {
        let _ = out.write_all(piece.as_bytes());
        let _ = out.flush();
    })?;
    let _ = out.write_all(b"\n");
    let _ = out.flush();
    Ok(())
}

/// Persistent correction server: load once, then one request per stdin line
/// (`ctx1␞ctx2␞…␟target`) → one corrected line on stdout. Emits "READY" first.
fn serve_corrections(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let backend = LlamaBackend::init()?;
    let model = load_model(&backend, model_path)?;

    let mut out = io::stdout().lock();
    out.write_all(b"READY\n")?;
    out.flush()?;

    let stdin = io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break, // EOF / parent gone
            Ok(_) => {}
        }
        let req = line.trim_end_matches(['\n', '\r']);
        let (context, target) = parse_request(req);
        // Reply with exactly one line per request, even on failure (echo input),
        // so the client's request/response framing never desyncs.
        let corrected = correct_line(&backend, &model, &context, target)
            .unwrap_or_else(|_| target.to_string());
        let corrected = single_line(&corrected);
        out.write_all(corrected.as_bytes())?;
        out.write_all(b"\n")?;
        out.flush()?;
    }
    Ok(())
}

/// Split a request line into (context lines, target line).
fn parse_request(req: &str) -> (Vec<&str>, &str) {
    match req.split_once(FIELD_SEP) {
        Some((ctx_blob, target)) => {
            let context = ctx_blob.split(REC_SEP).filter(|s| !s.is_empty()).collect();
            (context, target)
        }
        None => (Vec::new(), req),
    }
}

/// Correct one line, using the recent context for disambiguation.
fn correct_line(
    backend: &LlamaBackend,
    model: &LlamaModel,
    context: &[&str],
    target: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    if target.trim().is_empty() {
        return Ok(target.to_string());
    }
    let prompt = build_correction_prompt(model, context, target);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(CORRECT_N_CTX))
        .with_n_batch(CORRECT_N_CTX);
    let mut ctx = model.new_context(backend, ctx_params)?;

    // Greedy decoding (top_k(1)) — corrections must be deterministic and avoid
    // creative rewrites; the client additionally rejects over-large edits.
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::top_k(1), LlamaSampler::dist(0)]);

    // Bound generation to roughly the input size so a runaway can't pad content.
    let target_tokens = model.str_to_token(target, AddBos::Never).map(|t| t.len()).unwrap_or(64);
    let max_gen = (target_tokens + 24).min(CORRECT_MAX_GEN);

    let mut acc = String::new();
    generate(model, &mut ctx, &mut sampler, &prompt, max_gen, |piece| acc.push_str(piece))?;
    Ok(acc)
}

/// Feed `prompt`, then sample up to `max_gen` tokens, handing each decoded piece
/// to `sink`. Shared by both modes. Stops at end-of-generation.
fn generate(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    sampler: &mut LlamaSampler,
    prompt: &str,
    max_gen: usize,
    mut sink: impl FnMut(&str),
) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = model.str_to_token(prompt, AddBos::Never)?;
    let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
    let last = tokens.len() as i32 - 1;
    for (i, tok) in tokens.iter().enumerate() {
        batch.add(*tok, i as i32, &[0], i as i32 == last)?;
    }
    ctx.decode(&mut batch)?;

    let mut n_cur = tokens.len() as i32;
    let mut idx = last;
    for _ in 0..max_gen {
        let token = sampler.sample(ctx, idx);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        if let Ok(piece) = model.token_to_str(token, Special::Plaintext) {
            sink(&piece);
        }
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
        idx = 0;
    }
    Ok(())
}

fn load_model(backend: &LlamaBackend, model_path: &str) -> Result<LlamaModel, Box<dyn std::error::Error>> {
    let params = LlamaModelParams::default().with_n_gpu_layers(1000); // offload all layers
    Ok(LlamaModel::load_from_file(backend, model_path, &params)?)
}

/// Collapse a model reply to a single clean line: drop everything after the
/// first newline, strip wrapping quotes and a stray "Corrected:" label.
fn single_line(s: &str) -> String {
    let mut t = s.trim();
    if let Some(rest) = t.strip_prefix("Corrected:") {
        t = rest.trim();
    }
    let t = t.lines().next().unwrap_or("").trim();
    let t = t.trim_matches(|c| c == '"' || c == '\'' || c == '`');
    t.trim().to_string()
}

/// Build the chat prompt with the model's template, falling back to ChatML.
fn build_chat_prompt(model: &LlamaModel, system: &str, user: &str) -> String {
    let rendered = (|| {
        let tmpl = model.chat_template(None).ok()?;
        let sys = LlamaChatMessage::new("system".to_string(), system.to_string()).ok()?;
        let usr = LlamaChatMessage::new("user".to_string(), user.to_string()).ok()?;
        model.apply_chat_template(&tmpl, &[sys, usr], true).ok()
    })();
    rendered.unwrap_or_else(|| {
        format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    })
}

fn build_summary_prompt(model: &LlamaModel, transcript: &str) -> String {
    build_chat_prompt(model, SYSTEM_PROMPT, &format!("Transcript:\n{transcript}"))
}

fn build_correction_prompt(model: &LlamaModel, context: &[&str], target: &str) -> String {
    let user = if context.is_empty() {
        format!("NEW line: {target}")
    } else {
        format!("Context:\n{}\n\nNEW line: {target}", context.join("\n"))
    };
    build_chat_prompt(model, CORRECT_SYSTEM_PROMPT, &user)
}
