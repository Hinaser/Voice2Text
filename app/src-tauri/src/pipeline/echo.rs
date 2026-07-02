//! Software echo suppression: when the attendees' audio plays out the speakers
//! it is re-captured by the mic and would be transcribed twice. Loopback is the
//! source of truth, so a mic line whose words are mostly covered by recent
//! system speech is dropped as echo.
//!
//! This is the *text* layer of echo defense (a signal-level AEC runs upstream on
//! the mic audio). It is built to survive the three ways naive line-vs-line
//! matching fails in practice:
//!   - **ordering race** — the mic's echoed utterance can finalize before the
//!     matching system line does, so we also fold the system's *live partial*
//!     into the reference, which is available well before the final;
//!   - **segmentation mismatch** — a mic echo can straddle two system
//!     utterances, so we match against the *union* of all recent system words
//!     rather than any single line;
//!   - **ASR drift** — the degraded mic re-capture transcribes slightly
//!     different words, so matching tolerates small per-word edit distance.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// How long system speech stays eligible to explain a mic echo.
const WINDOW: Duration = Duration::from_secs(6);
/// Fraction of a mic line's words that must be covered by recent system speech.
const THRESHOLD: f32 = 0.55;
/// Words this long or longer may match across an edit distance of 1 (tolerates
/// ASR drift like plurals/tense); shorter words must match exactly.
const FUZZY_MIN_LEN: usize = 4;
/// Mic lines shorter than this must be covered by a SINGLE system line. The
/// union-of-all-recent-words matching (needed for echoes that straddle two
/// system utterances) is statistically unsafe for short lines: common words
/// scattered across different lines ("i" here, "agree" there) would suppress
/// the user's brief genuine replies.
const UNION_MIN_WORDS: usize = 5;

pub struct EchoFilter {
    /// Finalized system lines within the window, oldest first.
    finals: VecDeque<(Instant, Vec<String>)>,
    /// The system track's in-flight partial, replaced as it grows. Folding this
    /// into the reference closes the race where a mic echo finalizes first.
    partial: Option<(Instant, Vec<String>)>,
}

impl EchoFilter {
    pub fn new() -> Self {
        Self { finals: VecDeque::new(), partial: None }
    }

    /// Record a finalized *system* line. Supersedes the live partial (the final
    /// is its authoritative form) and prunes anything past the window.
    pub fn record_final(&mut self, now: Instant, text: &str) {
        self.prune(now);
        self.finals.push_back((now, tokenize(text)));
        self.partial = None;
    }

    /// Record the system track's current partial so it can explain a mic echo
    /// that finalizes before the matching system line does. Replaces any prior
    /// partial (a partial only grows within one utterance).
    pub fn record_partial(&mut self, now: Instant, text: &str) {
        let toks = tokenize(text);
        if !toks.is_empty() {
            self.partial = Some((now, toks));
        }
    }

    /// True if `text` (a mic line) is mostly covered by recent system speech.
    pub fn is_echo(&mut self, now: Instant, text: &str) -> bool {
        let mic = tokenize(text);
        if mic.is_empty() {
            return false;
        }
        self.prune(now);

        // All in-window system lines: finals plus the live partial.
        let mut lines: Vec<&Vec<String>> = self.finals.iter().map(|(_, sys)| sys).collect();
        if let Some((t, sys)) = &self.partial {
            if now.duration_since(*t) <= WINDOW {
                lines.push(sys);
            }
        }
        if lines.is_empty() {
            return false;
        }

        // Short mic lines: require a single system line to explain them (see
        // UNION_MIN_WORDS). Longer echoes may straddle utterance boundaries,
        // so those are matched against the union of all recent words.
        if mic.len() < UNION_MIN_WORDS {
            return lines.iter().any(|sys| coverage_frac(&mic, sys) >= THRESHOLD);
        }
        let mut counts: HashMap<&str, i32> = HashMap::new();
        for sys in &lines {
            for w in sys.iter() {
                *counts.entry(w.as_str()).or_insert(0) += 1;
            }
        }
        covered(&mic, &mut counts) as f32 / mic.len() as f32 >= THRESHOLD
    }

    fn prune(&mut self, now: Instant) {
        while let Some((t, _)) = self.finals.front() {
            if now.duration_since(*t) > WINDOW {
                self.finals.pop_front();
            } else {
                break;
            }
        }
        if let Some((t, _)) = &self.partial {
            if now.duration_since(*t) > WINDOW {
                self.partial = None;
            }
        }
    }
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// Fraction of `mic` explained by one system line.
fn coverage_frac(mic: &[String], sys: &[String]) -> f32 {
    let mut counts: HashMap<&str, i32> = HashMap::new();
    for w in sys {
        *counts.entry(w.as_str()).or_insert(0) += 1;
    }
    covered(mic, &mut counts) as f32 / mic.len() as f32
}

/// Number of `mic` words explained by the system multiset `counts`, consuming a
/// match per word. Each mic word prefers an exact match, then a fuzzy one
/// (edit distance ≤ 1 for sufficiently long words) to absorb ASR drift.
fn covered(mic: &[String], counts: &mut HashMap<&str, i32>) -> usize {
    let mut hit = 0;
    for w in mic {
        if let Some(c) = counts.get_mut(w.as_str()) {
            if *c > 0 {
                *c -= 1;
                hit += 1;
                continue;
            }
        }
        // Fuzzy fallback: find any remaining system word close to this one.
        let key = counts
            .iter()
            .find(|(k, c)| **c > 0 && fuzzy_eq(w, k))
            .map(|(k, _)| *k);
        if let Some(k) = key {
            *counts.get_mut(k).unwrap() -= 1;
            hit += 1;
        }
    }
    hit
}

/// Two words count as the same if equal, or (when both are long enough) within
/// a single edit. Keeps short common words exact to avoid spurious matches.
fn fuzzy_eq(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if a.len() < FUZZY_MIN_LEN || b.len() < FUZZY_MIN_LEN {
        return false;
    }
    within_one_edit(a, b)
}

/// True if `a` and `b` differ by at most one insertion, deletion, or
/// substitution. Linear, allocation-free.
fn within_one_edit(a: &str, b: &str) -> bool {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let (la, lb) = (a.len(), b.len());
    if la.abs_diff(lb) > 1 {
        return false;
    }
    // Walk both, allowing one mismatch.
    let (mut i, mut j, mut edits) = (0usize, 0usize, 0u8);
    while i < la && j < lb {
        if a[i] == b[j] {
            i += 1;
            j += 1;
            continue;
        }
        if edits == 1 {
            return false;
        }
        edits += 1;
        match la.cmp(&lb) {
            std::cmp::Ordering::Greater => i += 1, // deletion from a
            std::cmp::Ordering::Less => j += 1,    // insertion into a
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            } // substitution
        }
    }
    // Any trailing leftover char counts as the single allowed edit.
    true
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
    fn within_one_edit_basics() {
        assert!(within_one_edit("roadmap", "roadmaps")); // insertion
        assert!(within_one_edit("review", "revies")); // substitution
        assert!(within_one_edit("review", "reviw")); // deletion
        assert!(!within_one_edit("review", "viewer")); // far apart
        assert!(!within_one_edit("kitten", "sitting")); // 3 edits
    }

    #[test]
    fn matching_mic_line_is_flagged_as_echo() {
        let mut f = EchoFilter::new();
        let now = Instant::now();
        f.record_final(now, "let us review the roadmap");
        assert!(f.is_echo(now, "let us review the roadmap"));
        // a genuine, different mic line is not echo
        assert!(!f.is_echo(now, "yeah I agree completely"));
    }

    #[test]
    fn echo_with_minor_asr_drift_still_matches() {
        let mut f = EchoFilter::new();
        let now = Instant::now();
        f.record_final(now, "let us review the roadmaps today");
        // mic re-capture drifts: plural dropped, one near-miss word
        assert!(f.is_echo(now, "let us review the roadmap todays"));
    }

    #[test]
    fn partial_closes_the_ordering_race() {
        let mut f = EchoFilter::new();
        let now = Instant::now();
        // System line hasn't finalized yet — only a partial exists — but the mic
        // echo already finalizes. It must still be caught.
        f.record_partial(now, "we should ship the release on friday");
        assert!(f.is_echo(now, "we should ship the release on friday"));
    }

    #[test]
    fn short_genuine_reply_is_not_suppressed_by_scattered_words() {
        let mut f = EchoFilter::new();
        let now = Instant::now();
        f.record_final(now, "i think we should ship");
        f.record_final(now, "do you agree with the plan");
        // "i" appears in the first line and "agree" in the second — pooled they
        // cover 2/3 of this genuine reply, but no single line explains it.
        assert!(!f.is_echo(now, "yeah i agree"));
        // A short line that IS a real echo of one system line is still caught.
        assert!(f.is_echo(now, "do you agree"));
    }

    #[test]
    fn echo_spanning_two_system_lines_matches_the_union() {
        let mut f = EchoFilter::new();
        let now = Instant::now();
        f.record_final(now, "the deadline is next week");
        f.record_final(now, "and the budget is approved");
        // mic captured one run-on that straddles both system utterances
        assert!(f.is_echo(now, "the deadline is next week and the budget"));
    }

    #[test]
    fn final_supersedes_partial() {
        let mut f = EchoFilter::new();
        let now = Instant::now();
        f.record_partial(now, "incomplete partial words here");
        f.record_final(now, "let us review the roadmap");
        // partial was cleared by the final; only the final's words count
        assert!(f.is_echo(now, "let us review the roadmap"));
        assert!(!f.is_echo(now, "incomplete partial words here"));
    }

    #[test]
    fn stale_system_speech_does_not_match() {
        let mut f = EchoFilter::new();
        let past = Instant::now();
        f.record_final(past, "let us review the roadmap");
        f.record_partial(past, "some other live words");
        let later = past + WINDOW + Duration::from_secs(1);
        assert!(!f.is_echo(later, "let us review the roadmap"));
        assert!(!f.is_echo(later, "some other live words"));
    }
}
