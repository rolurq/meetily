-- Speaker diarization support.
--
-- speaker_profiles holds every speaker voiceprint the app knows about,
-- including auto-created "Unknown Speaker #N" profiles created the first
-- time a new voice is detected. is_named distinguishes profiles the user
-- has actually assigned a real name to from auto-generated placeholders.
CREATE TABLE IF NOT EXISTS speaker_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    is_named INTEGER NOT NULL DEFAULT 0,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 1,
    sample_audio_path TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Monotonic counter used to name new profiles "Unknown Speaker #N"
-- without ever reusing a number within this database.
CREATE TABLE IF NOT EXISTS unknown_speaker_counter (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    next_number INTEGER NOT NULL DEFAULT 1
);
INSERT OR IGNORE INTO unknown_speaker_counter (id, next_number) VALUES (1, 1);

-- Per-segment diarization result: which profile spoke this segment, and
-- how confident the match against that profile's stored voiceprint was.
ALTER TABLE transcripts ADD COLUMN speaker_id TEXT REFERENCES speaker_profiles(id);
ALTER TABLE transcripts ADD COLUMN speaker_confidence REAL;

CREATE INDEX IF NOT EXISTS idx_transcripts_speaker_id ON transcripts(speaker_id);
