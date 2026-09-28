use super::engine::{diarize_segments, DiarizationEngine, DownloadProgress, ModelInfo, ModelStatus, SegmentInput};
use super::wav::write_wav_16k_mono;
use crate::database::repositories::meeting::MeetingsRepository;
use crate::database::repositories::speaker_profile::SpeakerProfilesRepository;
use crate::database::repositories::transcript::TranscriptsRepository;
use crate::state::AppState;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{command, AppHandle, Emitter, Manager, Runtime};

pub static DIARIZATION_ENGINE: Mutex<Option<Arc<DiarizationEngine>>> = Mutex::new(None);
static MODELS_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

pub fn set_models_directory<R: Runtime>(app: &AppHandle<R>) {
    let app_data_dir = app.path().app_data_dir().expect("Failed to get app data dir");
    let models_dir = app_data_dir.join("models");
    if !models_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&models_dir) {
            log::error!("Failed to create models directory: {}", e);
            return;
        }
    }
    *MODELS_DIR.lock().unwrap() = Some(models_dir);
}

fn get_models_directory() -> Option<PathBuf> {
    MODELS_DIR.lock().unwrap().clone()
}

fn get_engine() -> Option<Arc<DiarizationEngine>> {
    DIARIZATION_ENGINE.lock().unwrap().as_ref().cloned()
}

#[command]
pub async fn diarization_init() -> Result<(), String> {
    let mut guard = DIARIZATION_ENGINE.lock().unwrap();
    if guard.is_some() {
        return Ok(());
    }
    let engine = DiarizationEngine::new_with_models_dir(get_models_directory())
        .map_err(|e| format!("Failed to initialize diarization engine: {}", e))?;
    *guard = Some(Arc::new(engine));
    Ok(())
}

#[command]
pub async fn diarization_get_available_models() -> Result<Vec<ModelInfo>, String> {
    let engine = get_engine().ok_or_else(|| "Diarization engine not initialized".to_string())?;
    Ok(engine.discover_models())
}

#[command]
pub async fn diarization_get_current_model() -> Result<Option<String>, String> {
    let engine = get_engine().ok_or_else(|| "Diarization engine not initialized".to_string())?;
    Ok(engine.get_current_model().await)
}

#[command]
pub async fn diarization_is_model_loaded() -> Result<bool, String> {
    let engine = get_engine().ok_or_else(|| "Diarization engine not initialized".to_string())?;
    Ok(engine.is_model_loaded().await)
}

#[command]
pub async fn diarization_load_model<R: Runtime>(app_handle: AppHandle<R>, model_name: String) -> Result<(), String> {
    let engine = get_engine().ok_or_else(|| "Diarization engine not initialized".to_string())?;
    let result = engine.load_model(&model_name).await.map_err(|e| e.to_string());
    if result.is_ok() {
        let _ = app_handle.emit("diarization-model-loading-completed", serde_json::json!({ "modelName": model_name }));
    }
    result
}

#[command]
pub async fn diarization_download_model<R: Runtime>(app_handle: AppHandle<R>, model_name: String) -> Result<(), String> {
    let engine = get_engine().ok_or_else(|| "Diarization engine not initialized".to_string())?;

    let app_handle_clone = app_handle.clone();
    let model_name_clone = model_name.clone();
    let progress_callback = move |progress: DownloadProgress| {
        let _ = app_handle_clone.emit(
            "diarization-model-download-progress",
            serde_json::json!({
                "modelName": model_name_clone,
                "progress": progress.percent,
                "downloaded_mb": progress.downloaded_mb,
                "total_mb": progress.total_mb,
                "speed_mbps": progress.speed_mbps,
                "status": if progress.percent == 100 { "completed" } else { "downloading" }
            }),
        );
    };

    match engine.download_model(&model_name, progress_callback).await {
        Ok(()) => {
            let _ = app_handle.emit(
                "diarization-model-download-complete",
                serde_json::json!({ "modelName": model_name }),
            );
            Ok(())
        }
        Err(e) => {
            let _ = app_handle.emit(
                "diarization-model-download-error",
                serde_json::json!({ "modelName": model_name, "error": e.to_string() }),
            );
            Err(format!("Failed to download diarization model: {}", e))
        }
    }
}

#[command]
pub async fn diarization_cancel_download(model_name: String) -> Result<(), String> {
    let engine = get_engine().ok_or_else(|| "Diarization engine not initialized".to_string())?;
    engine.cancel_download(&model_name).await;
    Ok(())
}

#[command]
pub async fn diarization_delete_model(model_name: String) -> Result<(), String> {
    let engine = get_engine().ok_or_else(|| "Diarization engine not initialized".to_string())?;
    engine.delete_model(&model_name).await.map_err(|e| e.to_string())
}

#[command]
pub async fn open_diarization_models_folder() -> Result<(), String> {
    let models_dir = get_models_directory()
        .ok_or_else(|| "Diarization models directory not initialized".to_string())?
        .join("diarization");
    if !models_dir.exists() {
        std::fs::create_dir_all(&models_dir).map_err(|e| format!("Failed to create directory: {}", e))?;
    }
    let folder_path = models_dir.to_string_lossy().to_string();

    #[cfg(target_os = "windows")]
    std::process::Command::new("explorer").arg(&folder_path).spawn().map_err(|e| e.to_string())?;
    #[cfg(target_os = "macos")]
    std::process::Command::new("open").arg(&folder_path).spawn().map_err(|e| e.to_string())?;
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open").arg(&folder_path).spawn().map_err(|e| e.to_string())?;

    Ok(())
}

/// Ensures a diarization model is loaded, auto-loading the first available one if none is.
async fn ensure_model_ready(engine: &DiarizationEngine) -> Result<(), String> {
    if engine.is_model_loaded().await {
        return Ok(());
    }
    let models = engine.discover_models();
    let available = models
        .iter()
        .find(|m| matches!(m.status, ModelStatus::Available))
        .ok_or_else(|| "No diarization model downloaded. Download one from Settings to enable speaker identification.".to_string())?;
    engine.load_model(&available.name).await.map_err(|e| e.to_string())
}

#[derive(Debug, Serialize)]
pub struct SpeakerProfileDto {
    pub id: String,
    pub name: String,
    pub is_named: bool,
    pub sample_count: i64,
    pub sample_audio_path: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl From<crate::database::models::SpeakerProfile> for SpeakerProfileDto {
    fn from(p: crate::database::models::SpeakerProfile) -> Self {
        Self {
            id: p.id,
            name: p.name,
            is_named: p.is_named,
            sample_count: p.sample_count,
            sample_audio_path: p.sample_audio_path,
            created_at: p.created_at.0.to_rfc3339(),
            updated_at: p.updated_at.0.to_rfc3339(),
        }
    }
}

/// Reads a speaker sample clip's raw bytes so the frontend can play it back, without
/// exposing a general-purpose file-read endpoint. Only paths inside the recordings
/// folder (where diarization writes its sample clips) are allowed.
#[command]
pub async fn diarization_read_sample_audio(path: String) -> Result<Vec<u8>, String> {
    let requested = std::path::PathBuf::from(&path);
    let recordings_root = crate::audio::recording_preferences::get_default_recordings_folder();

    let canonical_requested = requested.canonicalize().map_err(|e| format!("Sample clip not found: {}", e))?;
    let canonical_root = recordings_root
        .canonicalize()
        .map_err(|e| format!("Recordings folder not found: {}", e))?;
    if !canonical_requested.starts_with(&canonical_root) {
        return Err("Sample clip is outside the recordings folder".to_string());
    }

    tokio::fs::read(&canonical_requested)
        .await
        .map_err(|e| format!("Failed to read sample clip: {}", e))
}

#[command]
pub async fn diarization_list_speaker_profiles(state: tauri::State<'_, AppState>) -> Result<Vec<SpeakerProfileDto>, String> {
    let pool = state.db_manager.pool();
    SpeakerProfilesRepository::list_profiles(pool)
        .await
        .map(|profiles| profiles.into_iter().map(SpeakerProfileDto::from).collect())
        .map_err(|e| e.to_string())
}

#[command]
pub async fn diarization_rename_speaker(
    state: tauri::State<'_, AppState>,
    speaker_id: String,
    new_name: String,
) -> Result<(), String> {
    if new_name.trim().is_empty() {
        return Err("Speaker name cannot be empty".to_string());
    }
    let pool = state.db_manager.pool();
    SpeakerProfilesRepository::rename_profile(pool, &speaker_id, new_name.trim())
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[command]
pub async fn diarization_delete_speaker_profile(state: tauri::State<'_, AppState>, speaker_id: String) -> Result<(), String> {
    let pool = state.db_manager.pool();
    SpeakerProfilesRepository::delete_profile(pool, &speaker_id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Reassigns a single transcript segment to a different (or brand-new) speaker profile,
/// then folds that segment's voiceprint into the target profile so future diarization
/// benefits from the correction. Used when the user fixes a mis-identified speaker.
#[command]
pub async fn diarization_reassign_segment<R: Runtime>(
    app_handle: AppHandle<R>,
    state: tauri::State<'_, AppState>,
    meeting_id: String,
    transcript_id: String,
    target_speaker_id: Option<String>,
    new_speaker_name: Option<String>,
) -> Result<SpeakerProfileDto, String> {
    let engine = get_engine().ok_or_else(|| "Diarization engine not initialized".to_string())?;
    let pool = state.db_manager.pool();

    let meeting = MeetingsRepository::get_meeting_metadata(pool, &meeting_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Meeting not found".to_string())?;
    let folder_path = meeting.folder_path.ok_or_else(|| "Meeting has no recorded audio".to_string())?;

    let transcripts = MeetingsRepository::get_all_transcripts(pool, &meeting_id)
        .await
        .map_err(|e| e.to_string())?;
    let segment = transcripts
        .iter()
        .find(|t| t.id == transcript_id)
        .ok_or_else(|| "Transcript segment not found".to_string())?;
    let (Some(start), Some(end)) = (segment.audio_start_time, segment.audio_end_time) else {
        return Err("Segment has no audio timing information".to_string());
    };

    let audio_path = std::path::PathBuf::from(&folder_path).join("audio.mp4");
    let decoded = crate::audio::decoder::decode_audio_file(&audio_path).map_err(|e| e.to_string())?;
    let full_audio = decoded.to_whisper_format();
    let start_sample = ((start * 16000.0) as usize).min(full_audio.len());
    let end_sample = ((end * 16000.0) as usize).min(full_audio.len());
    if start_sample >= end_sample {
        return Err("Segment audio range is empty".to_string());
    }
    let embedding = engine
        .embed_segment(&full_audio[start_sample..end_sample])
        .await
        .map_err(|e| e.to_string())?;

    let profile = match (target_speaker_id, new_speaker_name) {
        (Some(target_id), _) => {
            SpeakerProfilesRepository::update_embedding(pool, &target_id, &embedding)
                .await
                .map_err(|e| e.to_string())?;
            SpeakerProfilesRepository::get_profile(pool, &target_id)
                .await
                .map_err(|e| e.to_string())?
                .ok_or_else(|| "Speaker profile not found".to_string())?
        }
        (None, Some(name)) if !name.trim().is_empty() => {
            let model_name = engine.get_current_model_name_or_unknown().await;
            SpeakerProfilesRepository::create_profile(pool, name.trim(), true, &embedding, &model_name, None)
                .await
                .map_err(|e| e.to_string())?
        }
        _ => return Err("Either target_speaker_id or new_speaker_name must be provided".to_string()),
    };

    TranscriptsRepository::set_segment_speaker(pool, &transcript_id, &profile.id, 1.0)
        .await
        .map_err(|e| e.to_string())?;

    let _ = app_handle.emit(
        "diarization-segment-updated",
        serde_json::json!({ "meetingId": meeting_id, "transcriptId": transcript_id, "speakerId": profile.id }),
    );

    Ok(profile.into())
}

/// Reassigns every segment in a meeting currently attributed to `from_speaker_id` to a
/// different (or brand-new) speaker profile, and folds that speaker's sample clip into
/// the target profile's voiceprint. Used from the post-diarization "name these speakers"
/// panel when a whole detected speaker was matched to the wrong person.
#[command]
pub async fn diarization_reassign_meeting_speaker(
    state: tauri::State<'_, AppState>,
    meeting_id: String,
    from_speaker_id: String,
    target_speaker_id: Option<String>,
    new_speaker_name: Option<String>,
) -> Result<SpeakerProfileDto, String> {
    let engine = get_engine().ok_or_else(|| "Diarization engine not initialized".to_string())?;
    let pool = state.db_manager.pool();

    let from_profile = SpeakerProfilesRepository::get_profile(pool, &from_speaker_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Speaker profile not found".to_string())?;

    let embedding = if let Some(sample_path) = &from_profile.sample_audio_path {
        let decoded = crate::audio::decoder::decode_audio_file(std::path::Path::new(sample_path))
            .map_err(|e| e.to_string())?;
        let samples = decoded.to_whisper_format();
        Some(engine.embed_segment(&samples).await.map_err(|e| e.to_string())?)
    } else {
        None
    };

    let profile = match (target_speaker_id, new_speaker_name) {
        (Some(target_id), _) => {
            if let Some(embedding) = &embedding {
                SpeakerProfilesRepository::update_embedding(pool, &target_id, embedding)
                    .await
                    .map_err(|e| e.to_string())?;
            }
            SpeakerProfilesRepository::get_profile(pool, &target_id)
                .await
                .map_err(|e| e.to_string())?
                .ok_or_else(|| "Speaker profile not found".to_string())?
        }
        (None, Some(name)) if !name.trim().is_empty() => {
            let model_name = engine.get_current_model_name_or_unknown().await;
            let profile_embedding = embedding.unwrap_or_else(|| blob_to_embedding_public(&from_profile));
            SpeakerProfilesRepository::create_profile(pool, name.trim(), true, &profile_embedding, &model_name, None)
                .await
                .map_err(|e| e.to_string())?
        }
        _ => return Err("Either target_speaker_id or new_speaker_name must be provided".to_string()),
    };

    TranscriptsRepository::reassign_speaker(pool, &meeting_id, &from_speaker_id, &profile.id, 1.0)
        .await
        .map_err(|e| e.to_string())?;

    Ok(profile.into())
}

fn blob_to_embedding_public(profile: &crate::database::models::SpeakerProfile) -> Vec<f32> {
    crate::database::repositories::speaker_profile::blob_to_embedding(&profile.embedding)
}

#[derive(Debug, Serialize, Clone)]
pub struct DiarizedSpeaker {
    pub speaker_id: String,
    pub name: String,
    pub is_named: bool,
    pub confidence: f32,
    pub sample_audio_path: Option<String>,
}

/// Diarizes a completed meeting recording: extracts a voiceprint for every
/// transcript segment, clusters them into distinct speakers, matches those
/// speakers against saved profiles (creating new "Unknown Speaker #N"
/// profiles for voices never heard before), and persists the result.
#[command]
pub async fn diarize_meeting<R: Runtime>(
    app_handle: AppHandle<R>,
    state: tauri::State<'_, AppState>,
    meeting_id: String,
) -> Result<Vec<DiarizedSpeaker>, String> {
    let engine = get_engine().ok_or_else(|| "Diarization engine not initialized".to_string())?;
    ensure_model_ready(&engine).await?;

    let pool = state.db_manager.pool().clone();

    let meeting = MeetingsRepository::get_meeting_metadata(&pool, &meeting_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Meeting not found".to_string())?;
    let folder_path = meeting
        .folder_path
        .ok_or_else(|| "Meeting has no recorded audio to diarize".to_string())?;
    let audio_path = std::path::PathBuf::from(&folder_path).join("audio.mp4");
    if !audio_path.exists() {
        return Err(format!("Recording audio not found at {}", audio_path.display()));
    }

    let transcripts = MeetingsRepository::get_all_transcripts(&pool, &meeting_id)
        .await
        .map_err(|e| e.to_string())?;

    let segments: Vec<SegmentInput> = transcripts
        .iter()
        .filter_map(|t| match (t.audio_start_time, t.audio_end_time) {
            (Some(start), Some(end)) if end > start => Some(SegmentInput {
                transcript_id: t.id.clone(),
                audio_start_time: start,
                audio_end_time: end,
            }),
            _ => None,
        })
        .collect();

    if segments.is_empty() {
        return Ok(Vec::new());
    }

    let audio_path_clone = audio_path.clone();
    let decoded = tokio::task::spawn_blocking(move || crate::audio::decoder::decode_audio_file(&audio_path_clone))
        .await
        .map_err(|e| format!("Audio decode task failed: {}", e))?
        .map_err(|e| format!("Failed to decode recording audio: {}", e))?;
    let full_audio = decoded.to_whisper_format();

    let assignments = diarize_segments(&pool, &engine, &full_audio, &segments)
        .await
        .map_err(|e| e.to_string())?;

    for assignment in &assignments {
        TranscriptsRepository::set_segment_speaker(&pool, &assignment.transcript_id, &assignment.speaker_id, assignment.confidence as f64)
            .await
            .map_err(|e| e.to_string())?;
    }

    // Cut one short sample clip per distinct speaker for the identification UI.
    let speakers_dir = std::path::PathBuf::from(&folder_path).join("speakers");
    if !speakers_dir.exists() {
        std::fs::create_dir_all(&speakers_dir).map_err(|e| e.to_string())?;
    }

    let mut seen = std::collections::HashSet::new();
    let mut speakers = Vec::new();
    for assignment in &assignments {
        if !seen.insert(assignment.speaker_id.clone()) {
            continue;
        }
        const MAX_SAMPLE_SECONDS: f64 = 6.0;
        let start_sample = ((assignment.sample_start * 16000.0) as usize).min(full_audio.len());
        let end_sample = (((assignment.sample_start + MAX_SAMPLE_SECONDS).min(assignment.sample_end.max(assignment.sample_start + 1.0)) * 16000.0) as usize)
            .min(full_audio.len());

        let sample_audio_path = if end_sample > start_sample {
            let clip_path = speakers_dir.join(format!("{}.wav", assignment.speaker_id));
            match write_wav_16k_mono(&clip_path, &full_audio[start_sample..end_sample]) {
                Ok(()) => {
                    let path_str = clip_path.to_string_lossy().to_string();
                    let _ = SpeakerProfilesRepository::set_sample_audio_path(&pool, &assignment.speaker_id, &path_str).await;
                    Some(path_str)
                }
                Err(e) => {
                    log::warn!("Failed to write speaker sample clip: {}", e);
                    None
                }
            }
        } else {
            None
        };

        speakers.push(DiarizedSpeaker {
            speaker_id: assignment.speaker_id.clone(),
            name: assignment.speaker_name.clone(),
            is_named: assignment.is_named,
            confidence: assignment.confidence,
            sample_audio_path,
        });
    }

    let _ = app_handle.emit(
        "diarization-complete",
        serde_json::json!({ "meetingId": meeting_id, "speakers": speakers }),
    );

    Ok(speakers)
}

/// Fire-and-forget diarization kicked off right after a meeting's transcript is saved.
/// Failures are logged rather than surfaced, since diarization is a best-effort
/// enrichment step that must never block saving the meeting itself.
pub fn spawn_diarize_meeting<R: Runtime>(app_handle: AppHandle<R>, meeting_id: String) {
    tokio::spawn(async move {
        let state = app_handle.state::<AppState>();
        if let Err(e) = diarize_meeting(app_handle.clone(), state, meeting_id.clone()).await {
            log::warn!("Diarization skipped for meeting {}: {}", meeting_id, e);
        }
    });
}
