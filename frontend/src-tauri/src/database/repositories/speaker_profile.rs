use crate::database::models::SpeakerProfile;
use chrono::Utc;
use sqlx::{Error as SqlxError, SqlitePool};
use tracing::info;
use uuid::Uuid;

pub struct SpeakerProfilesRepository;

/// Serializes an embedding vector to little-endian f32 bytes for BLOB storage.
pub fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Deserializes a BLOB column back into an embedding vector.
pub fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

impl SpeakerProfilesRepository {
    /// Allocates the next "Unknown Speaker #N" label, guaranteed unique within this database.
    pub async fn next_unknown_speaker_name(pool: &SqlitePool) -> Result<String, SqlxError> {
        let mut transaction = pool.begin().await?;
        let (next_number,): (i64,) =
            sqlx::query_as("SELECT next_number FROM unknown_speaker_counter WHERE id = 1")
                .fetch_one(&mut *transaction)
                .await?;
        sqlx::query("UPDATE unknown_speaker_counter SET next_number = next_number + 1 WHERE id = 1")
            .execute(&mut *transaction)
            .await?;
        transaction.commit().await?;
        Ok(format!("Unknown Speaker #{}", next_number))
    }

    /// Creates a new speaker profile (named, or an auto-generated "Unknown Speaker #N").
    pub async fn create_profile(
        pool: &SqlitePool,
        name: &str,
        is_named: bool,
        embedding: &[f32],
        embedding_model: &str,
        sample_audio_path: Option<&str>,
    ) -> Result<SpeakerProfile, SqlxError> {
        let id = format!("speaker-{}", Uuid::new_v4());
        let now = Utc::now();
        let embedding_blob = embedding_to_blob(embedding);

        sqlx::query(
            "INSERT INTO speaker_profiles
                (id, name, is_named, embedding, embedding_dim, embedding_model, sample_count, sample_audio_path, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
        )
        .bind(&id)
        .bind(name)
        .bind(is_named)
        .bind(&embedding_blob)
        .bind(embedding.len() as i64)
        .bind(embedding_model)
        .bind(sample_audio_path)
        .bind(now)
        .bind(now)
        .execute(pool)
        .await?;

        info!("Created speaker profile '{}' ({})", name, id);

        Ok(SpeakerProfile {
            id,
            name: name.to_string(),
            is_named,
            embedding: embedding_blob,
            embedding_dim: embedding.len() as i64,
            embedding_model: embedding_model.to_string(),
            sample_count: 1,
            sample_audio_path: sample_audio_path.map(|s| s.to_string()),
            created_at: now.into(),
            updated_at: now.into(),
        })
    }

    pub async fn list_profiles(pool: &SqlitePool) -> Result<Vec<SpeakerProfile>, SqlxError> {
        sqlx::query_as::<_, SpeakerProfile>(
            "SELECT * FROM speaker_profiles ORDER BY is_named DESC, name ASC",
        )
        .fetch_all(pool)
        .await
    }

    pub async fn get_profile(
        pool: &SqlitePool,
        id: &str,
    ) -> Result<Option<SpeakerProfile>, SqlxError> {
        sqlx::query_as::<_, SpeakerProfile>("SELECT * FROM speaker_profiles WHERE id = ?")
            .bind(id)
            .fetch_optional(pool)
            .await
    }

    /// Renames a profile, marking it as user-identified (`is_named = true`).
    pub async fn rename_profile(
        pool: &SqlitePool,
        id: &str,
        new_name: &str,
    ) -> Result<bool, SqlxError> {
        let result = sqlx::query(
            "UPDATE speaker_profiles SET name = ?, is_named = 1, updated_at = ? WHERE id = ?",
        )
        .bind(new_name)
        .bind(Utc::now())
        .bind(id)
        .execute(pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn delete_profile(pool: &SqlitePool, id: &str) -> Result<bool, SqlxError> {
        let mut transaction = pool.begin().await?;
        sqlx::query("UPDATE transcripts SET speaker_id = NULL, speaker_confidence = NULL WHERE speaker_id = ?")
            .bind(id)
            .execute(&mut *transaction)
            .await?;
        let result = sqlx::query("DELETE FROM speaker_profiles WHERE id = ?")
            .bind(id)
            .execute(&mut *transaction)
            .await?;
        transaction.commit().await?;
        Ok(result.rows_affected() > 0)
    }

    /// Folds a newly observed embedding into a profile's stored voiceprint using a running
    /// average weighted by how many samples have contributed to it so far, then re-normalizes.
    /// Called both when diarization confirms an existing match and when the user manually
    /// corrects a speaker assignment (so the profile keeps improving over time).
    pub async fn update_embedding(
        pool: &SqlitePool,
        id: &str,
        new_embedding: &[f32],
    ) -> Result<(), SqlxError> {
        let profile = Self::get_profile(pool, id).await?;
        let Some(profile) = profile else {
            return Ok(());
        };
        let existing = blob_to_embedding(&profile.embedding);
        let n = profile.sample_count.max(1) as f32;

        let merged: Vec<f32> = if existing.len() == new_embedding.len() {
            existing
                .iter()
                .zip(new_embedding.iter())
                .map(|(a, b)| (a * n + b) / (n + 1.0))
                .collect()
        } else {
            new_embedding.to_vec()
        };
        let norm = merged.iter().map(|v| v * v).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 1e-6 {
            merged.iter().map(|v| v / norm).collect()
        } else {
            merged
        };

        sqlx::query(
            "UPDATE speaker_profiles SET embedding = ?, sample_count = sample_count + 1, updated_at = ? WHERE id = ?",
        )
        .bind(embedding_to_blob(&normalized))
        .bind(Utc::now())
        .bind(id)
        .execute(pool)
        .await?;
        Ok(())
    }

    pub async fn set_sample_audio_path(
        pool: &SqlitePool,
        id: &str,
        sample_audio_path: &str,
    ) -> Result<(), SqlxError> {
        sqlx::query("UPDATE speaker_profiles SET sample_audio_path = ? WHERE id = ?")
            .bind(sample_audio_path)
            .bind(id)
            .execute(pool)
            .await?;
        Ok(())
    }
}
