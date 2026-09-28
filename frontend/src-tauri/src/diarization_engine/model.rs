//! ONNX speaker-embedding model wrapper.
//!
//! Loads a single-input/single-output NeMo-style ONNX speaker-embedding network
//! (TitaNet / SpeakerNet) and turns 16kHz mono audio into a fixed-length
//! voiceprint vector. The single input, named `audio_signal` on the bundled
//! models, expects channels-first log-mel features of shape `[1, 64, time]`
//! (see `fbank.rs`). The input/output tensor names are still read from the
//! model itself rather than hardcoded, since different releases may use
//! different node names.

use super::fbank::{compute_fbank, NUM_MEL_BINS};
use ndarray::Array3;
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum DiarizationModelError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("Audio segment too short to extract a voiceprint (need at least 25ms of audio)")]
    SegmentTooShort,
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
    #[error("ONNX Runtime unavailable: {0}")]
    RuntimeUnavailable(String),
}

pub struct SpeakerEmbeddingModel {
    session: Session,
    input_name: String,
    output_name: String,
}

impl SpeakerEmbeddingModel {
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, DiarizationModelError> {
        crate::ensure_onnx_runtime_available()
            .map_err(|error| DiarizationModelError::RuntimeUnavailable(error.to_string()))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .commit_from_file(model_path.as_ref())?;

        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .ok_or_else(|| DiarizationModelError::OutputNotFound("model has no inputs".into()))?;
        let output_name = session
            .outputs
            .first()
            .map(|o| o.name.clone())
            .ok_or_else(|| DiarizationModelError::OutputNotFound("model has no outputs".into()))?;

        if session.inputs.len() > 1 {
            let extra: Vec<&str> = session.inputs[1..].iter().map(|i| i.name.as_str()).collect();
            log::warn!(
                "Speaker embedding model has {} inputs beyond '{}' ({:?}) that this loader does not supply; \
                 inference may fail if the model requires them",
                extra.len(),
                input_name,
                extra
            );
        }

        log::info!(
            "Loaded speaker embedding model from {}: input='{}', output='{}'",
            model_path.as_ref().display(),
            input_name,
            output_name
        );

        Ok(Self {
            session,
            input_name,
            output_name,
        })
    }

    /// Extracts a single L2-normalized embedding vector from 16kHz mono audio samples.
    pub fn embed(&mut self, samples_16k_mono: &[f32]) -> Result<Vec<f32>, DiarizationModelError> {
        let (features, num_frames) = compute_fbank(samples_16k_mono);
        if num_frames == 0 {
            return Err(DiarizationModelError::SegmentTooShort);
        }

        // Channels-first: [batch=1, mel_bins=64, time=num_frames], matching `audio_signal`.
        let input = Array3::from_shape_vec((1, NUM_MEL_BINS, num_frames), features)
            .expect("fbank feature buffer matches (NUM_MEL_BINS, num_frames) shape");

        let outputs = self.session.run(inputs![
            self.input_name.as_str() => TensorRef::from_array_view(input.view())?,
        ])?;

        let output = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| DiarizationModelError::OutputNotFound(self.output_name.clone()))?
            .try_extract_array::<f32>()?;

        let mut embedding: Vec<f32> = output.iter().copied().collect();

        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for v in embedding.iter_mut() {
                *v /= norm;
            }
        }

        Ok(embedding)
    }
}

/// Cosine similarity between two L2-normalized embeddings, in `[-1.0, 1.0]`.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_similarity_one() {
        let a = vec![0.6, 0.8];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn mismatched_lengths_return_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
