//! Kaldi-style log-mel filterbank feature extraction.
//!
//! Speaker-embedding models (WeSpeaker/NeMo, as distributed for ONNX inference)
//! are trained on 80-dimensional log-mel filterbank features computed with the
//! same conventions as Kaldi's `compute-fbank-feats`: 25ms/10ms framing, a
//! Povey window, pre-emphasis, and per-utterance mean normalization. This
//! module reproduces that pipeline so embeddings extracted here line up with
//! what the models expect.

use realfft::RealFftPlanner;
use std::f32::consts::PI;

const SAMPLE_RATE: f32 = 16000.0;
const FRAME_LENGTH_MS: f32 = 25.0;
const FRAME_SHIFT_MS: f32 = 10.0;
const NUM_MEL_BINS: usize = 80;
const LOW_FREQ: f32 = 20.0;
const PREEMPHASIS_COEFF: f32 = 0.97;
const LOG_FLOOR: f32 = 1e-10;

fn frame_length_samples() -> usize {
    (SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0).round() as usize
}

fn frame_shift_samples() -> usize {
    (SAMPLE_RATE * FRAME_SHIFT_MS / 1000.0).round() as usize
}

fn fft_size(frame_length: usize) -> usize {
    frame_length.next_power_of_two()
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Builds the triangular mel filterbank matrix: `num_mel_bins` rows, each with
/// weights over the `fft_bins` power-spectrum bins (0..=fft_size/2).
fn build_mel_filterbank(fft_size: usize, num_mel_bins: usize) -> Vec<Vec<(usize, f32)>> {
    let high_freq = SAMPLE_RATE / 2.0;
    let mel_low = hz_to_mel(LOW_FREQ);
    let mel_high = hz_to_mel(high_freq);
    let mel_step = (mel_high - mel_low) / (num_mel_bins as f32 + 1.0);

    let num_fft_bins = fft_size / 2 + 1;
    let fft_bin_freq = |bin: usize| -> f32 { bin as f32 * SAMPLE_RATE / fft_size as f32 };

    let mut filters = Vec::with_capacity(num_mel_bins);
    for m in 0..num_mel_bins {
        let mel_left = mel_low + mel_step * m as f32;
        let mel_center = mel_low + mel_step * (m as f32 + 1.0);
        let mel_right = mel_low + mel_step * (m as f32 + 2.0);

        let mut weights = Vec::new();
        for bin in 0..num_fft_bins {
            let mel = hz_to_mel(fft_bin_freq(bin));
            let weight = if mel > mel_left && mel < mel_right {
                if mel <= mel_center {
                    (mel - mel_left) / (mel_center - mel_left)
                } else {
                    (mel_right - mel) / (mel_right - mel_center)
                }
            } else {
                0.0
            };
            if weight > 0.0 {
                weights.push((bin, weight));
            }
        }
        filters.push(weights);
    }
    filters
}

fn povey_window(frame_length: usize) -> Vec<f32> {
    let n = frame_length as f32;
    (0..frame_length)
        .map(|i| {
            let x = 0.5 - 0.5 * (2.0 * PI * i as f32 / (n - 1.0)).cos();
            x.powf(0.85)
        })
        .collect()
}

/// Extracts mean-normalized 80-dim log-mel filterbank features from 16kHz mono audio.
///
/// Returns a `[num_frames][NUM_MEL_BINS]` matrix flattened row-major, along with
/// the frame count, ready to feed to an ONNX speaker-embedding session as a
/// `[1, num_frames, NUM_MEL_BINS]` tensor.
pub fn compute_fbank(samples: &[f32]) -> (Vec<f32>, usize) {
    let frame_length = frame_length_samples();
    let frame_shift = frame_shift_samples();
    let fft_len = fft_size(frame_length);

    if samples.len() < frame_length {
        return (Vec::new(), 0);
    }
    let num_frames = 1 + (samples.len() - frame_length) / frame_shift;
    if num_frames == 0 {
        return (Vec::new(), 0);
    }

    let window = povey_window(frame_length);
    let mel_filters = build_mel_filterbank(fft_len, NUM_MEL_BINS);

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let mut scratch = fft.make_scratch_vec();
    let mut spectrum = fft.make_output_vec();

    let mut features = vec![0.0f32; num_frames * NUM_MEL_BINS];

    for frame_idx in 0..num_frames {
        let start = frame_idx * frame_shift;
        let mut frame: Vec<f32> = samples[start..start + frame_length].to_vec();

        // Remove DC offset.
        let mean = frame.iter().sum::<f32>() / frame.len() as f32;
        for s in frame.iter_mut() {
            *s -= mean;
        }

        // Pre-emphasis (using the frame's own leading sample as history for the first tap).
        let mut preemphasized = vec![0.0f32; frame_length];
        preemphasized[0] = frame[0] - PREEMPHASIS_COEFF * frame[0];
        for i in 1..frame_length {
            preemphasized[i] = frame[i] - PREEMPHASIS_COEFF * frame[i - 1];
        }

        // Povey window.
        for i in 0..frame_length {
            preemphasized[i] *= window[i];
        }

        // Zero-pad to FFT size and run the real FFT.
        let mut fft_input = fft.make_input_vec();
        fft_input[..frame_length].copy_from_slice(&preemphasized);
        fft.process_with_scratch(&mut fft_input, &mut spectrum, &mut scratch)
            .expect("fbank FFT should never fail on a correctly sized buffer");

        // Mel filterbank on the power spectrum, then log with a numerical floor.
        for (mel_idx, filter) in mel_filters.iter().enumerate() {
            let mut energy = 0.0f32;
            for &(bin, weight) in filter {
                let power = spectrum[bin].norm_sqr();
                energy += power * weight;
            }
            features[frame_idx * NUM_MEL_BINS + mel_idx] = energy.max(LOG_FLOOR).ln();
        }
    }

    // Per-utterance cepstral mean normalization (subtract the per-bin mean across time),
    // matching the input normalization WeSpeaker/NeMo embedding models were trained with.
    let mut means = [0.0f32; NUM_MEL_BINS];
    for frame_idx in 0..num_frames {
        for bin in 0..NUM_MEL_BINS {
            means[bin] += features[frame_idx * NUM_MEL_BINS + bin];
        }
    }
    for m in means.iter_mut() {
        *m /= num_frames as f32;
    }
    for frame_idx in 0..num_frames {
        for bin in 0..NUM_MEL_BINS {
            features[frame_idx * NUM_MEL_BINS + bin] -= means[bin];
        }
    }

    (features, num_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hz_mel_roundtrip_is_monotonic() {
        assert!(hz_to_mel(1000.0) > hz_to_mel(500.0));
        assert!(hz_to_mel(8000.0) > hz_to_mel(1000.0));
    }

    #[test]
    fn silence_produces_finite_features() {
        let samples = vec![0.0f32; 16000]; // 1 second of silence
        let (features, num_frames) = compute_fbank(&samples);
        assert!(num_frames > 0);
        assert_eq!(features.len(), num_frames * NUM_MEL_BINS);
        assert!(features.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn short_audio_yields_no_frames() {
        let samples = vec![0.0f32; 10];
        let (features, num_frames) = compute_fbank(&samples);
        assert_eq!(num_frames, 0);
        assert!(features.is_empty());
    }

    #[test]
    fn frame_count_matches_kaldi_snip_edges_formula() {
        let samples = vec![0.1f32; 16000]; // 1s @ 16kHz
        let (_features, num_frames) = compute_fbank(&samples);
        let frame_length = frame_length_samples();
        let frame_shift = frame_shift_samples();
        let expected = 1 + (samples.len() - frame_length) / frame_shift;
        assert_eq!(num_frames, expected);
    }
}
