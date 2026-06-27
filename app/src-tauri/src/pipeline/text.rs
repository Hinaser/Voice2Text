//! Text post-processing for ASR output: punctuation, full-width→ASCII
//! normalization, and sentence/`I` casing. Pure functions — unit-tested below.

use sherpa_rs::punctuate::Punctuation;

/// Turn raw upper-case ASR text into a polished line: optionally add punctuation
/// (lower-casing first, as the model expects), normalize CJK punctuation to
/// ASCII, then truecase.
pub fn finalize(caps_text: &str, punct: Option<&mut Punctuation>) -> String {
    let with_punct = match punct {
        Some(p) => p.add_punctuation(&caps_text.to_lowercase()),
        None => caps_text.to_string(),
    };
    truecase(&normalize_punct(&with_punct))
}

/// Map the zh-en punctuation model's full-width marks to ASCII and tidy spacing.
fn normalize_punct(s: &str) -> String {
    let mut t = s
        .replace('，', ", ").replace('、', ", ").replace('。', ". ")
        .replace('？', "? ").replace('！', "! ").replace('：', ": ").replace('；', "; ");
    t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    for p in [",", ".", "?", "!", ":", ";"] {
        t = t.replace(&format!(" {p}"), p);
    }
    t.trim().to_string()
}

/// Capitalize sentence starts and the standalone pronoun "I".
fn truecase(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut sentence_start = true;
    for c in lower.chars() {
        if sentence_start && c.is_alphabetic() {
            out.extend(c.to_uppercase());
            sentence_start = false;
        } else {
            out.push(c);
        }
        if c == '.' || c == '!' || c == '?' {
            sentence_start = true;
        }
    }
    out.split(' ')
        .map(|w| if w == "i" || w.starts_with("i'") { format!("I{}", &w[1..]) } else { w.to_string() })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Lightweight casing for live partials (no punctuation model in the hot path).
pub fn truecase_partial(s: &str) -> String {
    truecase(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_full_width_punct_and_spacing() {
        assert_eq!(normalize_punct("hello ， world 。"), "hello, world.");
        assert_eq!(normalize_punct("yes ！ no ？"), "yes! no?");
    }

    #[test]
    fn collapses_repeated_whitespace() {
        assert_eq!(normalize_punct("a   b\tc"), "a b c");
    }

    #[test]
    fn truecase_capitalizes_sentences() {
        assert_eq!(truecase("hello world. how are you?"), "Hello world. How are you?");
    }

    #[test]
    fn truecase_fixes_standalone_i() {
        assert_eq!(truecase("then i said i'm fine"), "Then I said I'm fine");
        // "i" inside a word must not be touched
        assert_eq!(truecase("the list is big"), "The list is big");
    }

    #[test]
    fn finalize_without_punct_model_still_cases() {
        assert_eq!(finalize("HELLO WORLD", None), "Hello world");
    }
}
