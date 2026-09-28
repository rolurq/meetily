//! NeMo-style log-mel filterbank feature extraction.
//!
//! The bundled speaker-embedding models (NeMo TitaNet / SpeakerNet, exported to ONNX)
//! take a single `audio_signal` input of shape `[batch, 64, time]` — 64 mel channels,
//! channels-first, matching NeMo's `AudioToMelSpectrogramPreprocessor` defaults:
//! a centered (reflect-padded) STFT with a Hann window, a Slaney mel scale with
//! area-normalized filters, and per-feature (per mel-bin) mean/variance normalization.
//! This is a different convention from Kaldi/WeSpeaker fbank (which uses a Povey
//! window, non-centered framing, and a different mel scale) — this module intentionally
//! reproduces NeMo's pipeline, not Kaldi's, so features line up with what these models
//! were trained on.

use realfft::RealFftPlanner;
use std::f32::consts::PI;

const SAMPLE_RATE: f32 = 16000.0;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400; // 25ms @ 16kHz
const HOP_LENGTH: usize = 160; // 10ms @ 16kHz
pub const NUM_MEL_BINS: usize = 64;
const PREEMPHASIS_COEFF: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8; // 2^-24, matches NeMo's log_zero_guard_value
const NORMALIZE_EPS: f32 = 1e-5; // matches NeMo's per-feature normalization CONSTANT

/// Slaney-scale Hz -> mel, matching librosa's default (htk=False) and NeMo's mel filterbank.
fn hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP; // 15.0
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        let logstep = (6.4f32).ln() / 27.0;
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    }
}

/// Slaney-scale mel -> Hz (inverse of [`hz_to_mel`]).
fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP; // 15.0
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        let logstep = (6.4f32).ln() / 27.0;
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    }
}

/// Builds librosa/NeMo-style mel filters: `num_mel_bins` rows of sparse (fft_bin, weight)
/// pairs over the `n_fft/2 + 1` power-spectrum bins, with Slaney area normalization.
fn build_mel_filterbank(num_mel_bins: usize) -> Vec<Vec<(usize, f32)>> {
    let num_fft_bins = N_FFT / 2 + 1;
    let fft_bin_freq = |bin: usize| -> f32 { bin as f32 * SAMPLE_RATE / N_FFT as f32 };

    // num_mel_bins + 2 edge points, evenly spaced in mel space, converted back to Hz.
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(SAMPLE_RATE / 2.0);
    let mel_points: Vec<f32> = (0..=num_mel_bins + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (num_mel_bins + 1) as f32))
        .collect();

    let mut filters = Vec::with_capacity(num_mel_bins);
    for m in 0..num_mel_bins {
        let (left, center, right) = (mel_points[m], mel_points[m + 1], mel_points[m + 2]);
        let lower_diff = center - left;
        let upper_diff = right - center;
        let slaney_norm = 2.0 / (right - left);

        let mut weights = Vec::new();
        for bin in 0..num_fft_bins {
            let f = fft_bin_freq(bin);
            let lower = (f - left) / lower_diff;
            let upper = (right - f) / upper_diff;
            let weight = lower.min(upper).max(0.0) * slaney_norm;
            if weight > 0.0 {
                weights.push((bin, weight));
            }
        }
        filters.push(weights);
    }
    filters
}

fn hann_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (len - 1) as f32).cos())
        .collect()
}

/// Reflect-pads `samples` by `pad` on each side, matching `torch.stft(..., pad_mode="reflect")`.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(samples[i.min(n.saturating_sub(1))]);
    }
    out.extend_from_slice(samples);
    for i in 0..pad {
        let idx = n.saturating_sub(2).saturating_sub(i);
        out.push(samples[idx.min(n.saturating_sub(1))]);
    }
    out
}

/// Extracts 64-dim, per-feature-normalized log-mel features from 16kHz mono audio,
/// following NeMo's `AudioToMelSpectrogramPreprocessor` defaults.
///
/// Returns a `[NUM_MEL_BINS][num_frames]` matrix flattened row-major (channels-first,
/// matching the `audio_signal` input layout these models expect), along with the frame count.
pub fn compute_fbank(samples: &[f32]) -> (Vec<f32>, usize) {
    if samples.is_empty() {
        return (Vec::new(), 0);
    }

    // Pre-emphasis over the whole signal (NeMo applies this once, not per-frame).
    let mut preemphasized = vec![0.0f32; samples.len()];
    preemphasized[0] = samples[0];
    for i in 1..samples.len() {
        preemphasized[i] = samples[i] - PREEMPHASIS_COEFF * samples[i - 1];
    }

    // Center (reflect-pad) so framing matches torch.stft(center=True).
    let pad = N_FFT / 2;
    let padded = reflect_pad(&preemphasized, pad);

    let num_frames = 1 + samples.len() / HOP_LENGTH;
    if num_frames == 0 {
        return (Vec::new(), 0);
    }

    let window = hann_window(WIN_LENGTH);
    // The Hann window (length WIN_LENGTH) is centered within the N_FFT-sized analysis buffer.
    let window_offset = (N_FFT - WIN_LENGTH) / 2;
    let mel_filters = build_mel_filterbank(NUM_MEL_BINS);

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut scratch = fft.make_scratch_vec();
    let mut spectrum = fft.make_output_vec();

    // Channels-first: features[mel_bin * num_frames + frame_idx].
    let mut features = vec![0.0f32; NUM_MEL_BINS * num_frames];

    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_LENGTH;
        let mut fft_input = fft.make_input_vec();
        if start + N_FFT <= padded.len() {
            for i in 0..WIN_LENGTH {
                fft_input[window_offset + i] = padded[start + window_offset + i] * window[i];
            }
        } else {
            // Last frame may run past the (already center-padded) buffer; zero-pad the tail.
            for i in 0..WIN_LENGTH {
                let sample_idx = start + window_offset + i;
                if sample_idx < padded.len() {
                    fft_input[window_offset + i] = padded[sample_idx] * window[i];
                }
            }
        }

        fft.process_with_scratch(&mut fft_input, &mut spectrum, &mut scratch)
            .expect("fbank FFT should never fail on a correctly sized buffer");

        for (mel_idx, filter) in mel_filters.iter().enumerate() {
            let mut energy = 0.0f32;
            for &(bin, weight) in filter {
                energy += spectrum[bin].norm_sqr() * weight;
            }
            features[mel_idx * num_frames + frame_idx] = (energy + LOG_ZERO_GUARD).ln();
        }
    }

    // Per-feature normalization: for each mel bin, subtract its mean and divide by its
    // std across time, matching NeMo's normalize="per_feature".
    for mel_idx in 0..NUM_MEL_BINS {
        let row = &mut features[mel_idx * num_frames..(mel_idx + 1) * num_frames];
        let mean = row.iter().sum::<f32>() / num_frames as f32;
        let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / num_frames as f32;
        let std = variance.sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) / (std + NORMALIZE_EPS);
        }
    }

    (features, num_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hz_mel_roundtrip_is_close() {
        for hz in [100.0f32, 500.0, 1000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(hz);
            let back = mel_to_hz(mel);
            assert!((back - hz).abs() < 0.05, "hz={hz} mel={mel} back={back}");
        }
    }

    #[test]
    fn hz_to_mel_is_monotonic() {
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
    fn empty_audio_yields_no_frames() {
        let (features, num_frames) = compute_fbank(&[]);
        assert_eq!(num_frames, 0);
        assert!(features.is_empty());
    }

    #[test]
    fn frame_count_matches_centered_stft_formula() {
        let samples = vec![0.1f32; 16000]; // 1s @ 16kHz
        let (_features, num_frames) = compute_fbank(&samples);
        assert_eq!(num_frames, 1 + samples.len() / HOP_LENGTH);
    }

    #[test]
    fn each_mel_channel_is_normalized() {
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 16000.0).sin() * 0.3)
            .collect();
        let (features, num_frames) = compute_fbank(&samples);
        for mel_idx in 0..NUM_MEL_BINS {
            let row = &features[mel_idx * num_frames..(mel_idx + 1) * num_frames];
            let mean = row.iter().sum::<f32>() / num_frames as f32;
            assert!(mean.abs() < 1e-3, "mel {mel_idx} mean {mean} not ~0");
        }
    }
}
