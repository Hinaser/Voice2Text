//! Local-LLM meeting summary via the `llama-sidecar` process. Spawned per
//! request (one-shot): transcript on stdin → summary on stdout.

use std::io::Write;
use std::process::{Command, Stdio};

use crate::paths;

const SIDECAR_EXE: &str = "llama-sidecar.exe";
const MODEL_FILE: &str = "Qwen2.5-3B-Instruct-Q4_K_M.gguf";

/// Summarize `transcript` with the local LLM. Blocking — call off the UI thread.
pub fn run(models_override: &str, transcript: &str) -> Result<String, String> {
    let model = paths::models_dir(models_override).join(MODEL_FILE);
    if !model.exists() {
        return Err(format!("Summary model not found at {}", model.display()));
    }
    let exe = paths::sidecar_exe(SIDECAR_EXE);
    let mut cmd = Command::new(&exe);
    cmd.arg(&model)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    paths::hide_console(&mut cmd);
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to start summarizer ({}): {e}", exe.display()))?;
    {
        let mut stdin = child.stdin.take().ok_or("no stdin")?;
        stdin.write_all(transcript.as_bytes()).map_err(|e| e.to_string())?;
    } // stdin dropped → EOF, sidecar starts generating
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!("summarizer exited with {}", output.status));
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        Err("empty summary".into())
    } else {
        Ok(text)
    }
}
