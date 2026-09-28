//! Diarization engine: model catalog/download/lifecycle, plus the clustering
//! and speaker-matching logic that turns per-segment embeddings into speaker
//! assignments against persisted [`SpeakerProfile`] voiceprints.

use super::model::{cosine_similarity, DiarizationModelError, SpeakerEmbeddingModel};
use crate::database::models::SpeakerProfile;
use crate::database::repositories::speaker_profile::{blob_to_embedding, SpeakerProfilesRepository};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

/// A cluster is considered the same speaker as an existing profile above this similarity.
pub const SPEAKER_MATCH_THRESHOLD: f32 = 0.72;
/// Two segments are considered the same local speaker above this similarity.
const CLUSTER_THRESHOLD: f32 = 0.75;
/// Segments shorter than this contribute too little signal for a reliable voiceprint.
const MIN_SEGMENT_SECONDS: f64 = 0.3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Missing,
    Downloading { progress: u8 },
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_mb: u32,
    pub purpose: String,
    pub description: String,
    pub status: ModelStatus,
}

struct ModelSpec {
    name: &'static str,
    filename: &'static str,
    size_mb: u32,
    exact_bytes: u64,
    purpose: &'static str,
    description: &'static str,
    download_url: &'static str,
}

const MODEL_CATALOG: &[ModelSpec] = &[
    ModelSpec {
        name: "speaker-id-fast",
        filename: "nemo_en_speakerverification_speakernet.onnx",
        size_mb: 22,
        exact_bytes: 23_411_863,
        purpose: "Fast",
        description: "Compact, low-latency voiceprint model. Best when diarizing on modest hardware or when meetings are processed right after recording.",
        download_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx",
    },
    ModelSpec {
        name: "speaker-id-accurate",
        filename: "nemo_en_titanet_large.onnx",
        size_mb: 97,
        exact_bytes: 101_405_493,
        purpose: "Accurate",
        description: "Larger voiceprint model with better separation between similar-sounding speakers. Recommended when getting speaker identity right matters more than processing speed.",
        download_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx",
    },
];

fn find_spec(model_name: &str) -> Option<&'static ModelSpec> {
    MODEL_CATALOG.iter().find(|s| s.name == model_name)
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    pub downloaded_mb: f64,
    pub total_mb: f64,
    pub speed_mbps: f64,
    pub percent: u8,
}

pub struct DiarizationEngine {
    models_dir: PathBuf,
    current_model: Arc<RwLock<Option<SpeakerEmbeddingModel>>>,
    current_model_name: Arc<RwLock<Option<String>>>,
    active_downloads: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl DiarizationEngine {
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let models_dir = match models_dir {
            Some(dir) => dir.join("diarization"),
            None => std::env::current_dir()?.join("models").join("diarization"),
        };
        if !models_dir.exists() {
            std::fs::create_dir_all(&models_dir)?;
        }
        Ok(Self {
            models_dir,
            current_model: Arc::new(RwLock::new(None)),
            current_model_name: Arc::new(RwLock::new(None)),
            active_downloads: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub async fn get_models_directory(&self) -> PathBuf {
        self.models_dir.clone()
    }

    pub fn discover_models(&self) -> Vec<ModelInfo> {
        MODEL_CATALOG
            .iter()
            .map(|spec| {
                let path = self.models_dir.join(spec.filename);
                let status = match std::fs::metadata(&path) {
                    Ok(meta) if meta.len() == spec.exact_bytes => ModelStatus::Available,
                    Ok(_) => ModelStatus::Error("File size mismatch; please retry the download".into()),
                    Err(_) => ModelStatus::Missing,
                };
                ModelInfo {
                    name: spec.name.to_string(),
                    path,
                    size_mb: spec.size_mb,
                    purpose: spec.purpose.to_string(),
                    description: spec.description.to_string(),
                    status,
                }
            })
            .collect()
    }

    pub async fn get_current_model(&self) -> Option<String> {
        self.current_model_name.read().await.clone()
    }

    pub async fn is_model_loaded(&self) -> bool {
        self.current_model.read().await.is_some()
    }

    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        if self.current_model_name.read().await.as_deref() == Some(model_name) {
            return Ok(());
        }
        let spec = find_spec(model_name).ok_or_else(|| anyhow!("Unknown diarization model: {}", model_name))?;
        let path = self.models_dir.join(spec.filename);
        let model_path = path.clone();
        let model = tokio::task::spawn_blocking(move || SpeakerEmbeddingModel::load(&model_path))
            .await
            .map_err(|e| anyhow!("Model load task failed: {}", e))?
            .map_err(|e: DiarizationModelError| anyhow!("Failed to load model: {}", e))?;

        *self.current_model.write().await = Some(model);
        *self.current_model_name.write().await = Some(model_name.to_string());
        log::info!("Loaded diarization model: {}", model_name);
        Ok(())
    }

    pub async fn unload_model(&self) {
        self.current_model.write().await.take();
        self.current_model_name.write().await.take();
    }

    pub async fn delete_model(&self, model_name: &str) -> Result<()> {
        let spec = find_spec(model_name).ok_or_else(|| anyhow!("Unknown diarization model: {}", model_name))?;
        let path = self.models_dir.join(spec.filename);
        if self.current_model_name.read().await.as_deref() == Some(model_name) {
            self.unload_model().await;
        }
        if path.exists() {
            tokio::fs::remove_file(&path).await?;
        }
        Ok(())
    }

    pub async fn cancel_download(&self, model_name: &str) {
        if let Some(token) = self.active_downloads.lock().await.get(model_name) {
            token.cancel();
        }
    }

    /// Downloads a catalogued model with progress reporting and cancellation support.
    pub async fn download_model(
        &self,
        model_name: &str,
        progress_callback: impl Fn(DownloadProgress) + Send + 'static,
    ) -> Result<()> {
        let spec = find_spec(model_name).ok_or_else(|| anyhow!("Unknown diarization model: {}", model_name))?;
        let final_path = self.models_dir.join(spec.filename);
        let temp_path = self.models_dir.join(format!("{}.part", spec.filename));

        let cancellation = CancellationToken::new();
        self.active_downloads
            .lock()
            .await
            .insert(model_name.to_string(), cancellation.clone());

        let result = self
            .download_to_file(spec, &temp_path, &cancellation, progress_callback)
            .await;

        self.active_downloads.lock().await.remove(model_name);

        match result {
            Ok(()) => {
                tokio::fs::rename(&temp_path, &final_path).await?;
                Ok(())
            }
            Err(e) => {
                let _ = tokio::fs::remove_file(&temp_path).await;
                Err(e)
            }
        }
    }

    async fn download_to_file(
        &self,
        spec: &ModelSpec,
        temp_path: &std::path::Path,
        cancellation: &CancellationToken,
        progress_callback: impl Fn(DownloadProgress) + Send + 'static,
    ) -> Result<()> {
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;

        let client = reqwest::Client::builder()
            .tcp_nodelay(true)
            .timeout(std::time::Duration::from_secs(3600))
            .build()?;

        let response = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(anyhow!("Download cancelled")),
            response = client.get(spec.download_url).send() => response?,
        };

        if !response.status().is_success() {
            return Err(anyhow!("Download failed with status {}", response.status()));
        }

        let total_bytes = response.content_length().unwrap_or(spec.exact_bytes);
        let mut file = tokio::fs::File::create(temp_path).await?;
        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;
        let start = std::time::Instant::now();
        let mut last_report = std::time::Instant::now();

        while let Some(chunk) = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(anyhow!("Download cancelled")),
            chunk = stream.next() => chunk,
        } {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if last_report.elapsed() >= std::time::Duration::from_millis(500) || downloaded == total_bytes {
                let elapsed = start.elapsed().as_secs_f64().max(0.001);
                let speed_mbps = downloaded as f64 / (1024.0 * 1024.0) / elapsed;
                let percent = ((downloaded as f64 / total_bytes.max(1) as f64) * 100.0).min(100.0) as u8;
                progress_callback(DownloadProgress {
                    downloaded_mb: downloaded as f64 / (1024.0 * 1024.0),
                    total_mb: total_bytes as f64 / (1024.0 * 1024.0),
                    speed_mbps,
                    percent,
                });
                last_report = std::time::Instant::now();
            }
        }
        file.flush().await?;
        Ok(())
    }

    /// Extracts an embedding for one audio segment using the currently loaded model.
    pub async fn embed_segment(&self, samples_16k_mono: &[f32]) -> Result<Vec<f32>> {
        let mut guard = self.current_model.write().await;
        let model = guard.as_mut().ok_or_else(|| anyhow!("No diarization model loaded"))?;
        model
            .embed(samples_16k_mono)
            .map_err(|e| anyhow!("Failed to extract speaker embedding: {}", e))
    }

    pub async fn get_current_model_name_or_unknown(&self) -> String {
        self.current_model_name
            .read()
            .await
            .clone()
            .unwrap_or_else(|| "unknown".to_string())
    }
}

/// One transcript segment to diarize, with its recording-relative time bounds (seconds).
pub struct SegmentInput {
    pub transcript_id: String,
    pub audio_start_time: f64,
    pub audio_end_time: f64,
}

/// The speaker assigned to one segment, with the confidence of that match.
#[derive(Debug, Clone, Serialize)]
pub struct SpeakerAssignment {
    pub transcript_id: String,
    pub speaker_id: String,
    pub speaker_name: String,
    pub is_named: bool,
    pub confidence: f32,
    /// Representative audio slice for this speaker (start/end within the recording),
    /// used by the caller to cut a sample clip for the identification UI.
    pub sample_start: f64,
    pub sample_end: f64,
}

struct LocalCluster {
    centroid: Vec<f32>,
    member_indices: Vec<usize>,
}

/// Runs greedy online clustering over segment embeddings, then matches each
/// resulting cluster against persisted speaker profiles (creating a new
/// "Unknown Speaker #N" profile when no existing voiceprint matches closely
/// enough), and returns a per-segment speaker assignment.
pub async fn diarize_segments(
    pool: &SqlitePool,
    engine: &DiarizationEngine,
    full_audio_16k_mono: &[f32],
    segments: &[SegmentInput],
) -> Result<Vec<SpeakerAssignment>> {
    const SAMPLE_RATE: f64 = 16000.0;
    let model_name = engine.get_current_model_name_or_unknown().await;

    // 1. Extract an embedding per segment (skipping ones too short to be reliable).
    let mut embeddings: Vec<(usize, Vec<f32>)> = Vec::new();
    for (idx, segment) in segments.iter().enumerate() {
        let duration = segment.audio_end_time - segment.audio_start_time;
        if duration < MIN_SEGMENT_SECONDS {
            continue;
        }
        let start_sample = (segment.audio_start_time * SAMPLE_RATE).max(0.0) as usize;
        let end_sample = ((segment.audio_end_time * SAMPLE_RATE) as usize).min(full_audio_16k_mono.len());
        if start_sample >= end_sample {
            continue;
        }
        match engine
            .embed_segment(&full_audio_16k_mono[start_sample..end_sample])
            .await
        {
            Ok(embedding) => embeddings.push((idx, embedding)),
            Err(e) => log::warn!("Skipping segment {} for diarization: {}", segment.transcript_id, e),
        }
    }

    // 2. Greedy online clustering by cosine similarity into local (per-meeting) speakers.
    let mut clusters: Vec<LocalCluster> = Vec::new();
    let mut segment_cluster: HashMap<usize, usize> = HashMap::new();
    for (idx, embedding) in &embeddings {
        let mut best: Option<(usize, f32)> = None;
        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            let sim = cosine_similarity(&cluster.centroid, embedding);
            if best.map(|(_, best_sim)| sim > best_sim).unwrap_or(true) {
                best = Some((cluster_idx, sim));
            }
        }
        match best {
            Some((cluster_idx, sim)) if sim >= CLUSTER_THRESHOLD => {
                let cluster = &mut clusters[cluster_idx];
                cluster.member_indices.push(*idx);
                let n = cluster.member_indices.len() as f32;
                for (c, e) in cluster.centroid.iter_mut().zip(embedding.iter()) {
                    *c = (*c * (n - 1.0) + e) / n;
                }
                segment_cluster.insert(*idx, cluster_idx);
            }
            _ => {
                segment_cluster.insert(*idx, clusters.len());
                clusters.push(LocalCluster {
                    centroid: embedding.clone(),
                    member_indices: vec![*idx],
                });
            }
        }
    }

    // 3. Match each local cluster against persisted speaker profiles.
    let existing_profiles = SpeakerProfilesRepository::list_profiles(pool).await?;
    let mut assignments = Vec::with_capacity(embeddings.len());

    for cluster in &clusters {
        let best_match = existing_profiles
            .iter()
            .map(|profile| (profile, cosine_similarity(&blob_to_embedding(&profile.embedding), &cluster.centroid)))
            .filter(|(_, sim)| sim.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let (profile, confidence): (SpeakerProfile, f32) = match best_match {
            Some((profile, sim)) if sim >= SPEAKER_MATCH_THRESHOLD => {
                SpeakerProfilesRepository::update_embedding(pool, &profile.id, &cluster.centroid).await?;
                (profile.clone(), sim)
            }
            _ => {
                let name = SpeakerProfilesRepository::next_unknown_speaker_name(pool).await?;
                let profile = SpeakerProfilesRepository::create_profile(
                    pool,
                    &name,
                    false,
                    &cluster.centroid,
                    &model_name,
                    None,
                )
                .await?;
                (profile, 1.0)
            }
        };

        let first_member = cluster.member_indices.first().copied();
        let (sample_start, sample_end) = first_member
            .and_then(|idx| segments.get(idx))
            .map(|s| (s.audio_start_time, s.audio_end_time))
            .unwrap_or((0.0, 0.0));

        for &member_idx in &cluster.member_indices {
            assignments.push(SpeakerAssignment {
                transcript_id: segments[member_idx].transcript_id.clone(),
                speaker_id: profile.id.clone(),
                speaker_name: profile.name.clone(),
                is_named: profile.is_named,
                confidence,
                sample_start,
                sample_end,
            });
        }
    }

    Ok(assignments)
}
