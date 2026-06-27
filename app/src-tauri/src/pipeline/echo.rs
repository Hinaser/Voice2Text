//! Software echo suppression: when the attendees' audio plays out the speakers
//! it is re-captured by the mic and would be transcribed twice. Loopback is the
//! source of truth, so a mic line whose words are mostly covered by a recent
//! system line is dropped as echo.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// How long a system utterance stays eligible to explain a mic echo.
const WINDOW: Duration = Duration::from_secs(6);
/// Fraction of a mic line's words that must appear in a recent system line.
const THRESHOLD: f32 = 0.55;

pub struct EchoFilter {
    recent: VecDeque<(Instant, Vec<String>)>,
}

impl EchoFilter {
    pub fn new() -> Self {
        Self { recent: VecDeque::new() }
    }

    /// Record a finalized *system* line so future mic lines can be matched
    /// against it. Prunes entries older than the window.
    pub fn record_system(&mut self, now: Instant, text: &str) {
        while let Some((t, _)) = self.recent.front() {
            if now.duration_since(*t) > WINDOW {
                self.recent.pop_front();
            } else {
                break;
            }
        }
        self.recent.push_back((now, tokenize(text)));
    }

    /// True if `text` (a mic line) closely matches any recent system line.
    pub fn is_echo(&self, now: Instant, text: &str) -> bool {
        let toks = tokenize(text);
        if toks.is_empty() {
            return false;
        }
        self.recent
            .iter()
            .any(|(t, sys)| now.duration_since(*t) <= WINDOW && coverage(&toks, sys) >= THRESHOLD)
    }
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// Fraction of `mic` words covered by `sys` (multiset intersection / |mic|).
fn coverage(mic: &[String], sys: &[String]) -> f32 {
    if mic.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<&str, i32> = HashMap::new();
    for w in sys {
        *counts.entry(w.as_str()).or_insert(0) += 1;
    }
    let mut hit = 0;
    for w in mic {
        if let Some(c) = counts.get_mut(w.as_str()) {
            if *c > 0 {
                *c -= 1;
                hit += 1;
            }
        }
    }
    hit as f32 / mic.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        tokenize(s)
    }

    #[test]
    fn tokenize_splits_on_non_alphanumeric_and_lowercases() {
        assert_eq!(toks("Hello, World! 123"), vec!["hello", "world", "123"]);
    }

    #[test]
    fn coverage_is_fraction_of_mic_words_found() {
        assert_eq!(coverage(&toks("a b c d"), &toks("a b x y")), 0.5);
        assert_eq!(coverage(&toks("a b"), &toks("a b c")), 1.0);
        assert_eq!(coverage(&[], &toks("a b")), 0.0);
    }

    #[test]
    fn matching_mic_line_is_flagged_as_echo() {
        let mut f = EchoFilter::new();
        let now = Instant::now();
        f.record_system(now, "let us review the roadmap");
        assert!(f.is_echo(now, "let us review the roadmap"));
        // a genuine, different mic line is not echo
        assert!(!f.is_echo(now, "yeah I agree completely"));
    }

    #[test]
    fn stale_system_lines_do_not_match() {
        let mut f = EchoFilter::new();
        let past = Instant::now();
        f.record_system(past, "let us review the roadmap");
        let later = past + WINDOW + Duration::from_secs(1);
        assert!(!f.is_echo(later, "let us review the roadmap"));
    }
}
