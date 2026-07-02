//! Persistent LLM correction client.
//!
//! Drives the `llama-sidecar` in `--serve` mode (model loaded once) to polish
//! each finalized line using recent context — fixing the speech-recognition
//! errors that context makes obvious (homophones, names, dropped words). The
//! model is given the *Whisper* text (already accurate) so it only has to make
//! light fixes, not guess.
//!
//! Because an LLM can "correct" words that were actually right and silently
//! change meaning — worse than a visible ASR glitch for a non-native listener —
//! every correction passes [`accept_correction`], which rejects rewrites that
//! diverge too far from the original (a paraphrase/hallucination, not a fix).

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use crate::paths;

use super::events::Ui;

const SIDECAR_EXE: &str = "llama-sidecar.exe";
const DEFAULT_MODEL: &str = "Qwen2.5-3B-Instruct-Q4_K_M.gguf";
/// Request wire format mirrors the sidecar: `ctx1␞ctx2␟target`.
const FIELD_SEP: char = '\u{1f}';
const REC_SEP: char = '\u{1e}';
/// How many recent (corrected) lines to send as context.
const CONTEXT_LINES: usize = 4;
/// Reject a correction whose char edit distance exceeds this fraction of the
/// original length — a large change is almost certainly a paraphrase, not a fix.
const MAX_EDIT_FRAC: f32 = 0.4;
/// Also reject corrections that balloon the length (added/invented content).
const MAX_LEN_GROWTH: f32 = 1.6;
/// How long to wait for the model to load and print READY (cold disk reads of
/// a multi-GB GGUF can be slow).
const READY_TIMEOUT: Duration = Duration::from_secs(60);
/// How long to wait for one correction. Generation is bounded to roughly the
/// input length, so anything beyond this means the sidecar is wedged (GPU
/// fault, driver reset) — kill it rather than block the transcript pipeline
/// forever on a blocking read.
const CORRECT_TIMEOUT: Duration = Duration::from_secs(15);

pub struct Corrector {
    child: Child,
    stdin: ChildStdin,
    /// Replies arrive via a dedicated reader thread, so waits can time out —
    /// a blocking read here would let a hung sidecar freeze the caller (and
    /// with it every later transcript write) with no visible error.
    replies: Receiver<String>,
    /// Rolling window of recent accepted lines, oldest first.
    context: VecDeque<String>,
    ui: Ui,
    /// Latched on timeout/IO failure: the child is dead and every later
    /// `correct` is a cheap pass-through.
    dead: bool,
}

impl Corrector {
    /// Spawn the serve-mode sidecar and wait for its READY handshake. Returns
    /// `None` (after a status note) on any failure, so correction degrades to
    /// plain Whisper text.
    pub fn spawn(models_override: &str, model_file: &str, ui: &Ui) -> Option<Self> {
        let model = match resolve_model(models_override, model_file) {
            Some(m) => m,
            None => {
                ui.status("error", "LLM correction off: model not found");
                return None;
            }
        };
        let exe = paths::sidecar_exe(SIDECAR_EXE);
        let mut cmd = Command::new(&exe);
        cmd.arg(&model)
            .arg("--serve")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        paths::hide_console(&mut cmd);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                ui.status("error", format!("LLM correction off ({e})"));
                return None;
            }
        };
        let stdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;

        // Reader thread: forwards each stdout line; exits on EOF/error.
        let (tx, replies) = mpsc::channel::<String>();
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if tx.send(line.trim_end_matches(['\n', '\r']).to_string()).is_err() {
                            break;
                        }
                    }
                }
            }
        });

        let ready = matches!(replies.recv_timeout(READY_TIMEOUT), Ok(l) if l.trim() == "READY");
        if !ready {
            ui.status("error", "LLM correction unavailable");
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        Some(Self { child, stdin, replies, context: VecDeque::new(), ui: ui.clone(), dead: false })
    }

    /// Polish `text` with the recent context. Returns the accepted line (the
    /// correction if it passed the guard, otherwise `text` unchanged) and pushes
    /// it onto the context window for subsequent calls.
    pub fn correct(&mut self, text: &str) -> String {
        if self.dead {
            return text.to_string();
        }
        let accepted = match self.request(text) {
            Some(corrected) if accept_correction(text, &corrected) => corrected,
            _ => text.to_string(),
        };
        self.context.push_back(accepted.clone());
        while self.context.len() > CONTEXT_LINES {
            self.context.pop_front();
        }
        accepted
    }

    /// Send one request and wait (bounded) for the single-line reply. `None` on
    /// I/O error or timeout — in both cases the sidecar is killed and the
    /// corrector latched dead, since the request/reply framing is broken.
    fn request(&mut self, text: &str) -> Option<String> {
        let target = sanitize(text);
        if target.trim().is_empty() {
            return None;
        }
        let ctx: Vec<String> = self.context.iter().map(|l| sanitize(l)).collect();
        let mut req = ctx.join(&REC_SEP.to_string());
        req.push(FIELD_SEP);
        req.push_str(&target);
        req.push('\n');

        if self.stdin.write_all(req.as_bytes()).and_then(|_| self.stdin.flush()).is_err() {
            self.die("LLM correction stopped; continuing without it");
            return None;
        }
        match self.replies.recv_timeout(CORRECT_TIMEOUT) {
            Ok(line) => Some(line.trim().to_string()),
            Err(RecvTimeoutError::Timeout) => {
                self.die("LLM correction timed out; continuing without it");
                None
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.die("LLM correction stopped; continuing without it");
                None
            }
        }
    }

    /// Kill the sidecar and latch this corrector dead, surfacing why.
    fn die(&mut self, msg: &str) {
        if !self.dead {
            self.dead = true;
            let _ = self.child.kill();
            let _ = self.child.wait();
            self.ui.status("error", msg.to_string());
        }
    }
}

impl Drop for Corrector {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Resolve the correction model: an absolute path as-is, else a filename in the
/// models folder; empty falls back to the bundled Qwen default. `None` if absent.
fn resolve_model(models_override: &str, model_file: &str) -> Option<PathBuf> {
    let model_file = if model_file.trim().is_empty() { DEFAULT_MODEL } else { model_file.trim() };
    let candidate = Path::new(model_file);
    let model = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        paths::models_dir(models_override).join(model_file)
    };
    model.exists().then_some(model)
}

/// Strip control chars that would corrupt the line-based wire protocol.
fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect()
}

/// Whether a proposed correction is a plausible *fix* rather than a rewrite:
/// non-empty, not wildly longer, and within a bounded edit distance of the
/// original. Conservative on purpose — a missed fix is better than a confident
/// hallucination.
fn accept_correction(original: &str, corrected: &str) -> bool {
    let corrected = corrected.trim();
    if corrected.is_empty() {
        return false;
    }
    let orig = original.trim();
    if orig.is_empty() {
        return false;
    }
    let (olen, clen) = (orig.chars().count(), corrected.chars().count());
    if clen as f32 > olen as f32 * MAX_LEN_GROWTH + 8.0 {
        return false;
    }
    let dist = levenshtein(orig, corrected);
    (dist as f32) <= olen as f32 * MAX_EDIT_FRAC
}

/// Char-level Levenshtein distance (lines are short; O(n·m) is fine).
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn accepts_small_fixes() {
        // homophone / minor word fixes are within the edit budget
        assert!(accept_correction("their going to the meating", "they're going to the meeting"));
        assert!(accept_correction("lets discus the road map", "let's discuss the roadmap"));
        // unchanged is fine
        assert!(accept_correction("the budget is approved", "the budget is approved"));
    }

    #[test]
    fn rejects_paraphrase_and_hallucination() {
        // wholesale rewrite (meaning preserved but words changed too much)
        assert!(!accept_correction("ship it friday", "We have decided to release the product on Friday."));
        // empty / dropped content
        assert!(!accept_correction("the deadline is next week", ""));
        // invented extra content
        assert!(!accept_correction("ok", "ok, and also please remember to email the client about the invoice"));
    }
}
