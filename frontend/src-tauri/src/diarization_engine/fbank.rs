//! Log-mel filterbank feature extraction for the bundled NeMo speaker-embedding models.
//!
//! Verified against the actual reference implementation these `.onnx` exports ship
//! with — `k2-fsa/sherpa-onnx`'s `SpeakerEmbeddingExtractorNeMoImpl` (which builds
//! features via `kaldi-native-fbank`'s `OnlineStream` with `snip_edges=true`,
//! `is_librosa=true`, `remove_dc_offset=false`) — rather than assumed from NeMo's
//! Python preprocessor in isolation. Concretely, per `kaldi-native-fbank`:
//! - Kaldi-style snip-edge framing (frame `f` starts at `f * frame_shift`; no
//!   centering/reflect-padding of the signal).
//! - Pre-emphasis applied per frame (using the frame's own first sample as history),
//!   then a periodic Hann window (`a = 2*pi/N`, not `2*pi/(N-1)`).
//! - The windowed frame is left-aligned and zero-padded up to the FFT size (400 -> 512).
//! - A Slaney-scale, area-normalized mel filterbank (librosa's default), 64 bins.
//! - Power spectrum, natural log with NeMo's `log_zero_guard_value` epsilon.
//! - Per-feature (per mel-bin) mean/variance normalization across time
//!   (`NormalizePerFeature` in `speaker-embedding-extractor-nemo-impl.h`).
//! Output is channels-first (`[mel_bin][frame]`) since the model's `audio_signal`
//! input is `[batch, 64, time]` (`Transpose12` is applied before the ONNX call in
//! the reference implementation; we just build it in that layout directly).

use realfft::RealFftPlanner;
use std::f32::consts::PI;

const SAMPLE_RATE: f32 = 16000.0;
const N_FFT: usize = 512; // next power of two >= WIN_LENGTH
const WIN_LENGTH: usize = 400; // 25ms @ 16kHz
const HOP_LENGTH: usize = 160; // 10ms @ 16kHz
pub const NUM_MEL_BINS: usize = 64;
const PREEMPHASIS_COEFF: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8; // 2^-24, matches NeMo's log_zero_guard_value
const NORMALIZE_EPS: f32 = 1e-5; // matches NormalizePerFeature's epsilon

/// Slaney-scale Hz -> mel (`MelScaleSlaney` in kaldi-native-fbank's mel-computations.h).
fn hz_to_mel(hz: f32) -> f32 {
    if hz <= 1000.0 {
        hz * 3.0 / 200.0
    } else {
        15.0 + 14.545_078_5 * (hz / 1000.0).ln()
    }
}

/// Slaney-scale mel -> Hz (`InverseMelScaleSlaney`).
fn mel_to_hz(mel: f32) -> f32 {
    if mel <= 15.0 {
        200.0 / 3.0 * mel
    } else {
        1000.0 * ((mel - 15.0) * 0.068_751_78).exp()
    }
}

/// Builds Slaney-scale, area-normalized mel filters (librosa/`is_librosa=true` convention):
/// `num_mel_bins` rows of sparse (fft_bin, weight) pairs over the `n_fft/2 + 1` power bins.
fn build_mel_filterbank(num_mel_bins: usize) -> Vec<Vec<(usize, f32)>> {
    let num_fft_bins = N_FFT / 2 + 1;
    let fft_bin_width = SAMPLE_RATE / N_FFT as f32;

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(SAMPLE_RATE / 2.0);
    let mel_delta = (mel_high - mel_low) / (num_mel_bins as f32 + 1.0);

    let mut filters = Vec::with_capacity(num_mel_bins);
    for m in 0..num_mel_bins {
        let left_hz = mel_to_hz(mel_low + m as f32 * mel_delta);
        let center_hz = mel_to_hz(mel_low + (m as f32 + 1.0) * mel_delta);
        let right_hz = mel_to_hz(mel_low + (m as f32 + 2.0) * mel_delta);
        let slaney_norm = 2.0 / (right_hz - left_hz);

        let mut weights = Vec::new();
        for bin in 0..num_fft_bins {
            let hz = fft_bin_width * bin as f32;
            if hz > left_hz && hz < right_hz {
                let weight = if hz <= center_hz {
                    (hz - left_hz) / (center_hz - left_hz)
                } else {
                    (right_hz - hz) / (right_hz - center_hz)
                } * slaney_norm;
                weights.push((bin, weight));
            }
        }
        filters.push(weights);
    }
    filters
}

/// Periodic Hann window (`a = 2*pi/N`), matching kaldi-native-fbank's "hann" window type
/// (this is torch/numpy's periodic convention, distinct from a symmetric Hann window).
fn hann_window(len: usize) -> Vec<f32> {
    let a = 2.0 * PI / len as f32;
    (0..len).map(|i| 0.5 - 0.5 * (a * i as f32).cos()).collect()
}

/// Extracts 64-dim, per-feature-normalized log-mel features from 16kHz mono audio.
///
/// Returns a `[NUM_MEL_BINS][num_frames]` matrix flattened row-major (channels-first,
/// matching the `audio_signal` input layout these models expect), along with the frame count.
pub fn compute_fbank(samples: &[f32]) -> (Vec<f32>, usize) {
    if samples.len() < WIN_LENGTH {
        return (Vec::new(), 0);
    }
    // Kaldi snip-edges: frames must fit entirely within the waveform.
    let num_frames = 1 + (samples.len() - WIN_LENGTH) / HOP_LENGTH;

    let window = hann_window(WIN_LENGTH);
    let mel_filters = build_mel_filterbank(NUM_MEL_BINS);

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut scratch = fft.make_scratch_vec();
    let mut spectrum = fft.make_output_vec();

    // Channels-first: features[mel_bin * num_frames + frame_idx].
    let mut features = vec![0.0f32; NUM_MEL_BINS * num_frames];

    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_LENGTH;
        let frame = &samples[start..start + WIN_LENGTH];

        // Pre-emphasis per frame (frame's own first sample stands in for prior history).
        let mut processed = vec![0.0f32; WIN_LENGTH];
        processed[0] = frame[0] - PREEMPHASIS_COEFF * frame[0];
        for i in 1..WIN_LENGTH {
            processed[i] = frame[i] - PREEMPHASIS_COEFF * frame[i - 1];
        }

        // Window, then left-aligned zero-pad up to the FFT size.
        let mut fft_input = fft.make_input_vec();
        for i in 0..WIN_LENGTH {
            fft_input[i] = processed[i] * window[i];
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
    // (population) std across time, matching NormalizePerFeature.
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
        let expected = 1 + (samples.len() - WIN_LENGTH) / HOP_LENGTH;
        assert_eq!(num_frames, expected);
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
