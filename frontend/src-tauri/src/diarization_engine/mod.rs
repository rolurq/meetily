//! Speaker diarization: identifies "who spoke when" within a recorded meeting.
//!
//! # Module structure
//! - `fbank`: Kaldi-style log-mel filterbank feature extraction feeding the embedding model.
//! - `model`: ONNX speaker-embedding session wrapper.
//! - `engine`: model catalog/download/lifecycle, clustering, and speaker-profile matching.
//! - `wav`: tiny WAV writer for speaker sample clips.
//! - `commands`: Tauri command interface for frontend integration.

pub mod commands;
pub mod engine;
pub mod fbank;
pub mod model;
pub mod wav;

pub use engine::{DiarizationEngine, DownloadProgress, ModelInfo, ModelStatus};
