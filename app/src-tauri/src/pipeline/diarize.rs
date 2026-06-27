//! Speaker diarization: label each finalized utterance "Speaker N" using
//! CAM++ embeddings + cosine matching (sherpa-rs). Wraps the FFI handles.

use std::path::Path;

use sherpa_rs::embedding_manager::EmbeddingManager;
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};

use super::RATE;

/// Cosine distance threshold for treating an embedding as a known speaker.
const SPEAKER_THRESHOLD: f32 = 0.5;

pub struct Diarizer {
    extractor: Option<EmbeddingExtractor>,
    manager: EmbeddingManager,
    next_id: usize,
    last: String,
}

impl Diarizer {
    pub fn new(model: &Path) -> Self {
        let extractor = if model.exists() {
            EmbeddingExtractor::new(ExtractorConfig {
                model: model.to_string_lossy().into_owned(),
                provider: None,
                num_threads: Some(1),
                debug: false,
            })
            .ok()
        } else {
            None
        };
        let dim = extractor.as_ref().map(|e| e.embedding_size as i32).unwrap_or(512);
        Self { extractor, manager: EmbeddingManager::new(dim), next_id: 1, last: "Speaker 1".into() }
    }

    /// Best-effort speaker label for an utterance's audio. Falls back to the
    /// previous label for very short clips or on error.
    pub fn label(&mut self, audio: &[f32]) -> String {
        let Some(ex) = self.extractor.as_mut() else { return "Speaker 1".into() };
        if audio.len() < RATE / 2 {
            return self.last.clone();
        }
        match ex.compute_speaker_embedding(audio.to_vec(), RATE as u32) {
            Ok(mut emb) => {
                let name = self.manager.search(&emb, SPEAKER_THRESHOLD).unwrap_or_else(|| {
                    let n = format!("Speaker {}", self.next_id);
                    self.next_id += 1;
                    let _ = self.manager.add(n.clone(), &mut emb);
                    n
                });
                self.last = name.clone();
                name
            }
            Err(_) => self.last.clone(),
        }
    }
}
