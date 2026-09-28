//! Minimal mono 16-bit PCM WAV writer, used to cut short speaker sample clips
//! for the "name this speaker" UI. Avoids pulling in a dedicated audio-writer
//! crate for what is just a handful of header bytes plus the sample data.

use std::io::Write;
use std::path::Path;

pub fn write_wav_16k_mono(path: &Path, samples: &[f32]) -> std::io::Result<()> {
    const SAMPLE_RATE: u32 = 16000;
    const BITS_PER_SAMPLE: u16 = 16;
    const NUM_CHANNELS: u16 = 1;

    let byte_rate = SAMPLE_RATE * NUM_CHANNELS as u32 * (BITS_PER_SAMPLE as u32 / 8);
    let block_align = NUM_CHANNELS * (BITS_PER_SAMPLE / 8);
    let data_size = (samples.len() * 2) as u32;

    let mut file = std::fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_size).to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // PCM fmt chunk size
    file.write_all(&1u16.to_le_bytes())?; // PCM format
    file.write_all(&NUM_CHANNELS.to_le_bytes())?;
    file.write_all(&SAMPLE_RATE.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let value = (clamped * i16::MAX as f32) as i16;
        file.write_all(&value.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_valid_header_and_correct_data_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.wav");
        let samples = vec![0.0f32; 1600]; // 0.1s
        write_wav_16k_mono(&path, &samples).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 44 + samples.len() * 2);
    }
}
