"""
Pitch Pipeline - Real-Time Grammar Validation Integration

Integrates with existing tonic_sa_detection.py to provide end-to-end pipeline:
Audio → Sa Detection → pYIN Pitch Tracking → Swara Quantization → Grammar Validation

Supports both real-time streaming (23ms per frame) and offline file analysis.
"""

import numpy as np
import librosa
import time
from typing import List, Dict, Optional, Generator, Tuple
from dataclasses import dataclass
from collections import deque

# Import from existing tonic detection system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tonic_sa_detection import TonicSaDetector

# Import grammar validation components
from .swara_quantizer import SwaraQuantizer, SwaraResult
from .grammar_validator import RagaGrammarValidator, ValidationEvent
from .raga_database import get_raga_info

@dataclass
class FrameResult:
    """Result for a single audio frame"""
    timestamp_ms: float
    frequency_hz: float
    voiced: bool
    voicing_prob: float
    swara_result: Optional[SwaraResult]
    validation_event: Optional[ValidationEvent]

class RealTimeGrammarPipeline:
    """
    Real-time audio analysis pipeline for raga grammar validation.
    
    Pipeline stages:
    1. Audio framing (2048 samples, hop=512, ~23ms at 22050 Hz)
    2. Tonic Sa detection (Carnatic HPS ensemble)
    3. pYIN pitch tracking per frame
    4. Swara quantization using Sa reference
    5. Grammar validation via FSM
    """
    
    def __init__(
        self,
        raga_name: str,
        sr: int = 22050,
        frame_length: int = 2048,
        hop_length: int = 512,
        pyin_frame_length: int = 1024,
        pyin_hop_length: int = 256,
        min_frame_rms: float = 0.01,
    ):
        """
        Initialize pipeline for specific raga.
        
        Args:
            raga_name: Raga to validate against
            sr: Sample rate (Hz)
            
        Raises:
            ValueError: If raga not found
        """
        self.raga_name = raga_name
        self.sr = sr
        
        # Verify raga exists
        if not get_raga_info(raga_name):
            raise ValueError(f"Raga '{raga_name}' not found in database")
        
        # Audio processing parameters
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.pyin_frame_length = pyin_frame_length
        self.pyin_hop_length = pyin_hop_length
        self.min_frame_rms = min_frame_rms

        # Real-time pitch stabilization state
        # Keep this intentionally lightweight for fast note transitions.
        self._f0_history = deque(maxlen=3)
        self._swara_history = deque(maxlen=3)
        self._stable_swara: Optional[str] = None
        self._candidate_swara: Optional[str] = None
        self._candidate_count: int = 0
        self._last_frequency_hz: Optional[float] = None
        self._last_timestamp_ms: Optional[float] = None
        self._prev_instant_slope: float = 0.0
        self._last_output_swara: Optional[str] = None
        self._last_output_change_ms: Optional[float] = None
        self._unvoiced_ms_accum: float = 0.0
        self._drift_counter: int = 0
        
        # Initialize components
        self.tonic_detector = TonicSaDetector(sr=sr)
        self.swara_quantizer: Optional[SwaraQuantizer] = None
        self.grammar_validator: Optional[RagaGrammarValidator] = None
        self.sa_frequency: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        """True when tonic is detected and processing components are initialized."""
        return self.sa_frequency is not None and self.swara_quantizer is not None and self.grammar_validator is not None

    def initialize_with_sa(self, sa_frequency: float):
        """
        Initialize quantizer and grammar validator with a detected Sa frequency.
        
        **This is CRITICAL**: Once Sa is locked, all pitch detection is anchored to it.
        All 12 Carnatic swaras are calculated as:
        - Swara_Hz = Sa_Hz × Carnatic_Ratio
        
        This tonic-ratio basis means the system is immune to frequency drift,
        microphone issues, or room acoustics—only the RELATIVE pitch matters.
        
        Args:
            sa_frequency: The tonic Sa frequency in Hz (must be > 0)
            
        Raises:
            ValueError: If sa_frequency is invalid
        """
        if not sa_frequency or sa_frequency <= 0:
            raise ValueError(f"Invalid Sa frequency: {sa_frequency}")
        if not np.isfinite(sa_frequency):
            raise ValueError(f"Sa frequency not finite: {sa_frequency}")
            
        self.sa_frequency = sa_frequency
        self.swara_quantizer = SwaraQuantizer(self.sa_frequency)
        self.grammar_validator = RagaGrammarValidator(self.raga_name)
        
        print(f"✓ Sa locked to {self.sa_frequency:.2f} Hz")
        print(f"✓ All swaras calculated from this Sa")
        print(f"✓ Quantization range: {self.sa_frequency:.2f}–{2*self.sa_frequency:.2f} Hz")
        
    def detect_tonic_from_file(self, audio_path: str) -> float:
        """
        Detect Sa frequency from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detected Sa frequency in Hz
            
        Raises:
            RuntimeError: If Sa detection fails
        """
        print(f"Detecting tonic Sa from {audio_path}...")
        
        result = self.tonic_detector.ensemble_detection(audio_path, verbose=True)
        
        if 'sa_frequency' not in result:
            raise RuntimeError(f"Sa detection failed: {result.get('error', 'Unknown error')}")
        
        self.sa_frequency = result['sa_frequency']

        # Initialize quantizer and validator with detected Sa
        self.initialize_with_sa(self.sa_frequency)
        
        print(f"Detected Sa: {self.sa_frequency:.2f} Hz")
        print(f"Initialized for raga: {self.raga_name}")
        
        return self.sa_frequency
    
    def detect_tonic_from_audio(self, y: np.ndarray) -> float:
        """
        Detect Sa frequency from audio array.
        
        Args:
            y: Audio samples
            
        Returns:
            Detected Sa frequency in Hz
        """
        # Save to temporary file for tonic detection
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            try:
                sf.write(tmp.name, y, self.sr)
                return self.detect_tonic_from_file(tmp.name)
            finally:
                os.unlink(tmp.name)  # Clean up temp file

    def _apply_swara_hysteresis(self, swara_result: SwaraResult) -> SwaraResult:
        """
        Apply temporal hysteresis to avoid jitter between neighboring swaras.

        Switch rules:
        - Same as stable note: accept immediately.
        - Very high confidence: switch immediately.
        - Otherwise require 2 consecutive frames before switching.
        """
        if self.swara_quantizer is None:
            return swara_result

        # Initialize stable state
        if self._stable_swara is None:
            self._stable_swara = swara_result.swara
            self._candidate_swara = None
            self._candidate_count = 0
            return swara_result

        # No change needed
        if swara_result.swara == self._stable_swara:
            self._candidate_swara = None
            self._candidate_count = 0
            return swara_result

        # Strong evidence: allow immediate switch
        if swara_result.confidence >= 0.82:
            self._stable_swara = swara_result.swara
            self._candidate_swara = None
            self._candidate_count = 0
            return swara_result

        # Accumulate candidate evidence for switching
        if self._candidate_swara == swara_result.swara:
            self._candidate_count += 1
        else:
            self._candidate_swara = swara_result.swara
            self._candidate_count = 1

        # Switch only after repeated evidence
        if self._candidate_count >= 2 and swara_result.confidence >= 0.58:
            self._stable_swara = swara_result.swara
            self._candidate_swara = None
            self._candidate_count = 0
            return swara_result

        # Hold previous stable swara for this frame
        held = self.swara_quantizer.to_swara_tonic_band(
            self.swara_quantizer.get_swara_frequency(self._stable_swara, swara_result.octave)
        )
        if held is not None:
            held.confidence = max(held.confidence * 0.85, swara_result.confidence * 0.70)
            return held

        return swara_result

    def _apply_frequency_continuity(self, frequency_hz: float, timestamp_ms: float) -> float:
        """
        Apply continuity constraint to limit abrupt octave-like jumps.

        Dynamic-programming-like local continuity: clamp frame-to-frame jump
        in semitones while keeping latency low.
        """
        if frequency_hz <= 0:
            return frequency_hz

        if self._last_frequency_hz is None or self._last_timestamp_ms is None:
            return frequency_hz

        dt_sec = max((timestamp_ms - self._last_timestamp_ms) / 1000.0, 1e-3)

        # At ~10ms hops this is ~7 semitone/frame. Scales with dt.
        max_jump_per_10ms = 7.0
        max_jump_semitones = max_jump_per_10ms * (dt_sec / 0.01)

        st = 12.0 * np.log2((frequency_hz + 1e-9) / (self._last_frequency_hz + 1e-9))
        if abs(st) <= max_jump_semitones:
            clamped = frequency_hz
        else:
            ratio = 2 ** (max_jump_semitones / 12.0)
            upper = self._last_frequency_hz * ratio
            lower = self._last_frequency_hz / ratio
            clamped = float(np.clip(frequency_hz, lower, upper))

        return clamped

    def _commit_frequency_state(self, frequency_hz: float, timestamp_ms: float) -> None:
        """Commit frequency state after final per-frame decision."""
        if frequency_hz > 0:
            self._last_frequency_hz = float(frequency_hz)
            self._last_timestamp_ms = float(timestamp_ms)

    def _soft_reset_tracking_state(self) -> None:
        """Reset transient tracking state without touching tonic Sa lock."""
        self._f0_history.clear()
        self._swara_history.clear()
        self._stable_swara = None
        self._candidate_swara = None
        self._candidate_count = 0
        self._last_frequency_hz = None
        self._last_timestamp_ms = None
        self._prev_instant_slope = 0.0
        self._last_output_swara = None
        self._last_output_change_ms = None
        self._drift_counter = 0

    def _update_unvoiced_and_maybe_reset(self, timestamp_ms: float, voiced: bool) -> None:
        """
        Track unvoiced duration and reset stale continuity/hysteresis after pauses.
        """
        if self._last_timestamp_ms is None:
            dt_ms = 10.0
        else:
            dt_ms = max(1.0, float(timestamp_ms - self._last_timestamp_ms))

        if voiced:
            self._unvoiced_ms_accum = 0.0
            return

        self._unvoiced_ms_accum += dt_ms
        if self._unvoiced_ms_accum >= 120.0:
            self._soft_reset_tracking_state()
            self._unvoiced_ms_accum = 0.0

    def _apply_min_duration_hold(
        self,
        swara_result: SwaraResult,
        timestamp_ms: float,
        slope_st_per_sec: float,
    ) -> SwaraResult:
        """
        Enforce minimum stable-note duration (C_min = 60 ms).

        If a short, low-slope deviation appears briefly, keep previous stable swara.
        """
        if self.swara_quantizer is None:
            return swara_result

        if self._last_output_swara is None:
            self._last_output_swara = swara_result.swara
            self._last_output_change_ms = float(timestamp_ms)
            return swara_result

        if swara_result.swara == self._last_output_swara:
            return swara_result

        elapsed_ms = float(timestamp_ms - (self._last_output_change_ms or timestamp_ms))
        short_deviation = elapsed_ms < 60.0
        low_motion = slope_st_per_sec < 15.0
        not_very_strong = swara_result.confidence < 0.86

        if short_deviation and low_motion and not_very_strong:
            held = self.swara_quantizer.to_swara_tonic_band(
                self.swara_quantizer.get_swara_frequency(self._last_output_swara, swara_result.octave)
            )
            if held is not None:
                held.confidence = max(held.confidence * 0.85, swara_result.confidence * 0.7)
                return held

        # Accept actual change
        self._last_output_swara = swara_result.swara
        self._last_output_change_ms = float(timestamp_ms)
        return swara_result

    def _local_slope_semitones_per_sec(self, frequency_hz: float, timestamp_ms: float) -> float:
        """Compute local pitch slope for stable-note region detection."""
        if frequency_hz <= 0 or self._last_frequency_hz is None or self._last_timestamp_ms is None:
            return 0.0
        dt_sec = max((timestamp_ms - self._last_timestamp_ms) / 1000.0, 1e-3)
        st = 12.0 * np.log2((frequency_hz + 1e-9) / (self._last_frequency_hz + 1e-9))
        return float(abs(st / dt_sec))

    def _is_stable_bidir_anchor(self, frequency_hz: float, timestamp_ms: float, threshold: float = 15.0) -> Tuple[bool, float]:
        """
        Low-latency 3-point anchoring approximation for stability.

        Uses current instantaneous slope and previous instantaneous slope.
        This is causal and introduces no extra buffering while approximating
        backward/forward local-neighbor checks.
        """
        slope_now = self._local_slope_semitones_per_sec(frequency_hz, timestamp_ms)
        stable = (slope_now < threshold) or (self._prev_instant_slope < threshold)
        self._prev_instant_slope = slope_now
        return stable, slope_now

    def _band_energy(self, spectrum: np.ndarray, freqs: np.ndarray, center_hz: float, bw_cents: float) -> float:
        """Energy in a cents-width band around a center frequency."""
        if center_hz <= 0:
            return 0.0
        ratio = 2 ** (bw_cents / 1200.0)
        f_lo = center_hz / ratio
        f_hi = center_hz * ratio
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(mask):
            return 0.0
        return float(np.sum(spectrum[mask]))

    def _estimate_swara_from_spectrum(self, audio_frame: np.ndarray) -> Optional[Tuple[SwaraResult, float, float]]:
        """
        Estimate swara directly from tonic-anchored harmonic template matching.

        Returns:
            (SwaraResult, estimated_frequency_hz, voicing_prob) or None
        """
        if self.swara_quantizer is None or self.sa_frequency is None:
            return None

        n = len(audio_frame)
        if n == 0:
            return None

        n_fft = max(8192, int(2 ** np.ceil(np.log2(max(1, n)))))
        windowed = audio_frame.astype(np.float64, copy=False) * np.hanning(n)
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

        band_mask = (freqs >= max(40.0, 0.5 * self.sa_frequency)) & (freqs <= min(self.sr / 2.0, 4.0 * self.sa_frequency))
        if not np.any(band_mask):
            return None

        total_band_energy = float(np.sum(spectrum[band_mask])) + 1e-9

        harmonic_numbers = [1, 2, 3, 4]
        # Favor fundamental strongly to reduce complementary/harmonic confusion
        # (e.g., Ga/Ni where one note's harmonic can overlap another's).
        harmonic_weights = [2.2, 0.55, 0.30, 0.20]

        scores: List[Tuple[str, float, float]] = []
        for swara, ratio in self.swara_quantizer.CARNATIC_RATIOS.items():
            f0 = self.sa_frequency * ratio
            score = 0.0
            f0_energy = 0.0
            upper_harm_energy = 0.0
            for h, w in zip(harmonic_numbers, harmonic_weights):
                fh = f0 * h
                if fh >= (self.sr / 2.0):
                    continue
                e = self._band_energy(spectrum, freqs, fh, bw_cents=40.0)
                score += w * e
                if h == 1:
                    f0_energy = e
                else:
                    upper_harm_energy += e

            # Penalize candidates supported mostly by harmonics but with weak
            # fundamental energy (common source of Ga<->Ni swaps).
            fundamental_ratio = f0_energy / (0.35 * upper_harm_energy + 1e-9)
            score *= float(np.clip(fundamental_ratio, 0.35, 1.30))
            scores.append((swara, f0, float(score)))

        if not scores:
            return None

        scores.sort(key=lambda x: x[2], reverse=True)
        best_swara, best_f0, best_score = scores[0]
        second_score = scores[1][2] if len(scores) > 1 else 0.0

        # Peak near predicted fundamental for cents deviation.
        f0_band_ratio = 2 ** (120.0 / 1200.0)
        f0_lo = best_f0 / f0_band_ratio
        f0_hi = best_f0 * f0_band_ratio
        f0_mask = (freqs >= f0_lo) & (freqs <= f0_hi)
        if np.any(f0_mask):
            local_freqs = freqs[f0_mask]
            local_spec = spectrum[f0_mask]
            peak_idx = int(np.argmax(local_spec))
            estimated_f = float(local_freqs[peak_idx])
        else:
            estimated_f = float(best_f0)

        swara_result = self.swara_quantizer.to_swara_tonic_band(estimated_f)
        if swara_result is None:
            # Fallback to ideal swara anchor when fundamental peak is slightly off.
            swara_result = self.swara_quantizer.to_swara_tonic_band(best_f0)
            estimated_f = float(best_f0)
        if swara_result is None:
            return None

        dominance = (best_score - second_score) / (best_score + 1e-9)
        energy_ratio = best_score / total_band_energy
        voicing_prob = float(np.clip(0.5 * dominance + 1.5 * energy_ratio, 0.0, 1.0))

        # Blend spectral-template confidence with cents-based swara confidence.
        swara_result.confidence = float(np.clip(0.5 * swara_result.confidence + 0.5 * voicing_prob, 0.0, 1.0))

        return swara_result, estimated_f, voicing_prob

    def _estimate_swara_from_acpt(self, audio_frame: np.ndarray) -> Optional[Tuple[SwaraResult, float, float]]:
        """
        Low-latency ACPT (autocorrelation-based) swara estimation.

        Uses a causal trailing frame, normalized autocorrelation, tonic-constrained
        lag search, and parabolic peak refinement for stable monophonic tracking.
        """
        if self.swara_quantizer is None or self.sa_frequency is None:
            return None

        x = audio_frame.astype(np.float64, copy=False)
        n = len(x)
        if n < 128:
            return None

        # DC removal + mild pre-emphasis to suppress low-frequency drift
        x = x - np.mean(x)
        x = np.append(x[0], x[1:] - 0.97 * x[:-1])

        # Apply analysis window
        x = x * np.hanning(n)

        # Tonic-focused frequency range for lag search
        fmin = max(70.0, self.sa_frequency * 0.9)
        fmax = min(700.0, self.sa_frequency * 2.1)
        lag_min = int(max(1, np.floor(self.sr / fmax)))
        lag_max = int(min(n - 2, np.ceil(self.sr / fmin)))
        if lag_max <= lag_min:
            return None

        # Raw autocorrelation (positive lags)
        acf = np.correlate(x, x, mode='full')[n - 1:]
        if len(acf) <= lag_max:
            return None

        # Normalized autocorrelation over lag range (robust to amplitude changes)
        energy = np.cumsum(x * x)

        def segment_energy(start: int, end: int) -> float:
            if start > end:
                return 0.0
            return float(energy[end] - (energy[start - 1] if start > 0 else 0.0))

        lags = np.arange(lag_min, lag_max + 1, dtype=np.int32)
        nacf = np.zeros_like(lags, dtype=np.float64)

        for i, lag in enumerate(lags):
            e1 = segment_energy(0, n - lag - 1)
            e2 = segment_energy(lag, n - 1)
            denom = np.sqrt(max(e1 * e2, 1e-12))
            nacf[i] = acf[lag] / denom

        # Reject unvoiced / weakly periodic frames
        peak_i = int(np.argmax(nacf))
        peak_score = float(nacf[peak_i])
        if peak_score < 0.22:
            return None

        # Parabolic refinement around the best lag
        best_lag = float(lags[peak_i])
        if 0 < peak_i < (len(nacf) - 1):
            y0, y1, y2 = nacf[peak_i - 1], nacf[peak_i], nacf[peak_i + 1]
            denom = (y0 - 2.0 * y1 + y2)
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
                best_lag += delta

        if best_lag <= 0:
            return None

        estimated_f = float(self.sr / best_lag)
        swara_result = self.swara_quantizer.to_swara_tonic_band(estimated_f)
        if swara_result is None:
            return None

        # ACPT confidence combines peak strength and peak dominance
        baseline = float(np.median(nacf))
        dominance = max(0.0, peak_score - baseline)
        voicing_prob = float(np.clip((peak_score - 0.15) / 0.75, 0.0, 1.0))
        acpt_conf = float(np.clip(0.6 * voicing_prob + 0.4 * np.clip(dominance / 0.5, 0.0, 1.0), 0.0, 1.0))
        swara_result.confidence = float(np.clip(0.65 * swara_result.confidence + 0.35 * acpt_conf, 0.0, 1.0))

        return swara_result, estimated_f, voicing_prob

    def _estimate_swara_from_yin(self, audio_frame: np.ndarray) -> Optional[Tuple[SwaraResult, float, float]]:
        """Estimate swara using YIN pitch tracking in tonic-focused range."""
        if self.swara_quantizer is None or self.sa_frequency is None:
            return None

        fmin = max(70.0, self.sa_frequency * 0.9)
        fmax = min(700.0, self.sa_frequency * 2.1)

        yin_f0 = librosa.yin(
            audio_frame,
            fmin=fmin,
            fmax=fmax,
            sr=self.sr,
            frame_length=self.pyin_frame_length,
            hop_length=self.pyin_hop_length,
        )

        if len(yin_f0) == 0:
            return None

        yin_valid = yin_f0[~np.isnan(yin_f0)]
        if len(yin_valid) == 0:
            return None

        frequency = float(np.median(yin_valid))
        voicing_prob = float(min(1.0, (len(yin_valid) / max(1, len(yin_f0)))))

        self._f0_history.append(frequency)
        stabilized_frequency = float(np.median(np.array(self._f0_history, dtype=np.float64)))

        swara_result = self.swara_quantizer.to_swara_tonic_band(stabilized_frequency)
        if swara_result is None:
            return None

        swara_result.confidence = float(np.clip(0.55 * swara_result.confidence + 0.45 * voicing_prob, 0.0, 1.0))
        return swara_result, stabilized_frequency, voicing_prob

    def _refine_frequency_near_swara(
        self,
        audio_frame: np.ndarray,
        swara_result: SwaraResult,
        current_frequency: float,
    ) -> float:
        """
        Refine frequency estimate by searching local FFT peak around the selected swara.

        This reduces cents deviation jitter while preserving real pitch movement.
        """
        if self.swara_quantizer is None or current_frequency <= 0:
            return current_frequency

        n = len(audio_frame)
        if n < 128:
            return current_frequency

        ideal_f = self.swara_quantizer.get_swara_frequency(swara_result.swara, swara_result.octave)
        n_fft = max(8192, int(2 ** np.ceil(np.log2(max(1, n)))))
        x = audio_frame.astype(np.float64, copy=False) * np.hanning(n)
        spec = np.abs(np.fft.rfft(x, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

        # Search around ideal swara and current estimate (intersection-like guard)
        ratio_ideal = 2 ** (110.0 / 1200.0)
        ratio_curr = 2 ** (80.0 / 1200.0)
        f_lo = max(40.0, min(ideal_f / ratio_ideal, current_frequency / ratio_curr))
        f_hi = min(self.sr / 2.0, max(ideal_f * ratio_ideal, current_frequency * ratio_curr))
        if f_hi <= f_lo:
            return current_frequency

        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(mask):
            return current_frequency

        local_freqs = freqs[mask]
        local_spec = spec[mask]
        idx = int(np.argmax(local_spec))
        refined_f = float(local_freqs[idx])

        # Small parabolic interpolation for sub-bin precision
        if 0 < idx < (len(local_spec) - 1):
            y0, y1, y2 = local_spec[idx - 1], local_spec[idx], local_spec[idx + 1]
            denom = (y0 - 2.0 * y1 + y2)
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
                bin_hz = float(local_freqs[1] - local_freqs[0]) if len(local_freqs) > 1 else 0.0
                refined_f += delta * bin_hz

        return refined_f
    
    def analyze_frame(self, audio_frame: np.ndarray, 
                     timestamp_ms: float) -> Optional[FrameResult]:
        """
        Analyze single audio frame for pitch and grammar validation.
        
        **ALGORITHM:**
        1. Filter out low-energy frames (RMS gate)
        2. Extract pitch using YIN in tonic-focused frequency range
        3. Apply temporal stabilization (median filter + EMA)
        4. Quantize stabilized pitch to Carnatic swara using tonic ratios
        5. Validate quantized swara against raga grammar
        
        This ensures clean, stable swara detection with no frequency confusion.
        
        Args:
            audio_frame: Audio samples for this frame
            timestamp_ms: Timestamp of frame start
            
        Returns:
            FrameResult or None if analysis fails
        """
        if self.swara_quantizer is None or self.grammar_validator is None:
            return None
        
        try:
            # ===== STAGE 1: RMS GATE - Filter silent/low-energy frames =====
            frame_rms = float(np.sqrt(np.mean(np.square(audio_frame))))
            if frame_rms < self.min_frame_rms:
                self._update_unvoiced_and_maybe_reset(timestamp_ms, voiced=False)
                return FrameResult(
                    timestamp_ms=timestamp_ms,
                    frequency_hz=0,
                    voiced=False,
                    voicing_prob=0,
                    swara_result=None,
                    validation_event=None
                )

            # ===== STAGE 2: HYBRID TONIC-FOCUSED SWARA ESTIMATION =====
            if not self.sa_frequency:
                return None

            acpt_estimate = self._estimate_swara_from_acpt(audio_frame)
            spectral_estimate = self._estimate_swara_from_spectrum(audio_frame)
            yin_estimate = None
            # Keep YIN as expensive fallback only when both low-latency paths fail.
            if acpt_estimate is None and spectral_estimate is None:
                yin_estimate = self._estimate_swara_from_yin(audio_frame)

            if acpt_estimate is None and spectral_estimate is None and yin_estimate is None:
                # Bridge short glottal closures/unvoiced gaps by holding last stable pitch.
                if (
                    self._last_frequency_hz is not None
                    and self._last_timestamp_ms is not None
                    and (timestamp_ms - self._last_timestamp_ms) <= 70.0
                ):
                    held_result = self.swara_quantizer.to_swara_tonic_band(self._last_frequency_hz)
                    if held_result is not None:
                        validation_event = self.grammar_validator.validate_swara(held_result, timestamp_ms)
                        validation_event.frequency_hz = self._last_frequency_hz
                        return FrameResult(
                            timestamp_ms=timestamp_ms,
                            frequency_hz=float(self._last_frequency_hz),
                            voiced=True,
                            voicing_prob=0.18,
                            swara_result=held_result,
                            validation_event=validation_event,
                        )
                self._update_unvoiced_and_maybe_reset(timestamp_ms, voiced=False)
                return FrameResult(
                    timestamp_ms=timestamp_ms,
                    frequency_hz=0,
                    voiced=False,
                    voicing_prob=0,
                    swara_result=None,
                    validation_event=None
                )

            if acpt_estimate is not None and spectral_estimate is not None:
                sw_acpt, f_acpt, v_acpt = acpt_estimate
                sw_spec, f_spec, v_spec = spectral_estimate

                last_swara = self._swara_history[-1] if len(self._swara_history) > 0 else None

                if sw_acpt.swara == sw_spec.swara:
                    swara_result = sw_acpt
                    swara_result.confidence = float(np.clip(0.65 * sw_acpt.confidence + 0.35 * sw_spec.confidence, 0.0, 1.0))
                    stabilized_frequency = float((0.65 * f_acpt) + (0.35 * f_spec))
                    voicing_prob = float(np.clip(0.65 * v_acpt + 0.35 * v_spec, 0.0, 1.0))
                else:
                    acpt_score = sw_acpt.confidence + (0.15 if last_swara == sw_acpt.swara else 0.0)
                    spec_score = sw_spec.confidence + (0.10 if last_swara == sw_spec.swara else 0.0)

                    if abs(acpt_score - spec_score) < 0.10:
                        pick_acpt = abs(sw_acpt.cents_deviation) <= abs(sw_spec.cents_deviation)
                    else:
                        pick_acpt = acpt_score > spec_score

                    if pick_acpt:
                        swara_result, stabilized_frequency, voicing_prob = sw_acpt, f_acpt, v_acpt
                    else:
                        swara_result, stabilized_frequency, voicing_prob = sw_spec, f_spec, v_spec
            elif acpt_estimate is not None:
                swara_result, stabilized_frequency, voicing_prob = acpt_estimate
            elif spectral_estimate is not None:
                swara_result, stabilized_frequency, voicing_prob = spectral_estimate
            else:
                swara_result, stabilized_frequency, voicing_prob = yin_estimate

            # ===== STAGE 3A: CONTINUITY CONSTRAINT =====
            stabilized_frequency = self._apply_frequency_continuity(stabilized_frequency, timestamp_ms)

            # ===== STAGE 3B: LOCAL FREQUENCY REFINEMENT =====
            refined_frequency = self._refine_frequency_near_swara(
                audio_frame, swara_result, stabilized_frequency
            )
            refined_result = self.swara_quantizer.to_swara_tonic_band(refined_frequency)
            if refined_result is not None and refined_result.swara == swara_result.swara:
                refined_result.confidence = max(refined_result.confidence, swara_result.confidence)
                swara_result = refined_result
                stabilized_frequency = refined_frequency

            # ===== STAGE 3C: INSTANTANEOUS STABILITY ANCHOR =====
            is_stable_region, slope_st_per_sec = self._is_stable_bidir_anchor(
                stabilized_frequency,
                timestamp_ms,
            )

            # Direct grid quantization for stable regions (strict boundaries).
            if is_stable_region and abs(swara_result.cents_deviation) <= 32.0:
                grid_result = self.swara_quantizer.to_swara_tonic_band(
                    self.swara_quantizer.get_swara_frequency(swara_result.swara, swara_result.octave)
                )
                if grid_result is not None:
                    # Keep vocal naturalness: do not force correction for >32 cents.
                    grid_result.confidence = max(grid_result.confidence * 0.92, swara_result.confidence)
                    swara_result = grid_result
                    stabilized_frequency = self.swara_quantizer.get_swara_frequency(
                        swara_result.swara,
                        swara_result.octave,
                    )

            # Update continuity state only after all per-frame refinements.
            self._commit_frequency_state(stabilized_frequency, timestamp_ms)

            # ===== STAGE 4: LABEL STABILIZATION =====
            # Suppress isolated one-frame label flips by short majority vote.
            self._swara_history.append(swara_result.swara)
            if len(self._swara_history) >= 2:
                counts: Dict[str, int] = {}
                for s in self._swara_history:
                    counts[s] = counts.get(s, 0) + 1
                dominant_swara, dominant_count = max(counts.items(), key=lambda kv: kv[1])
                if dominant_count >= 2 and dominant_swara != swara_result.swara and swara_result.confidence < 0.55:
                    corrected = self.swara_quantizer.to_swara_tonic_band(
                        self.swara_quantizer.get_swara_frequency(dominant_swara, swara_result.octave)
                    )
                    if corrected is not None:
                        corrected.confidence = max(corrected.confidence, swara_result.confidence * 0.85)
                        swara_result = corrected

            # ===== STAGE 4B: TEMPORAL HYSTERESIS =====
            swara_result = self._apply_swara_hysteresis(swara_result)

            # ===== STAGE 4C: MIN-DURATION HOLD FOR BRIEF DEVIATIONS =====
            swara_result = self._apply_min_duration_hold(
                swara_result, timestamp_ms, slope_st_per_sec
            )

            # Drift guard: if repeated low-confidence or high-deviation frames appear,
            # clear stale continuity state to prevent "wrong-note lock".
            if (swara_result.confidence < 0.30) or (abs(swara_result.cents_deviation) > 45.0):
                self._drift_counter += 1
            else:
                self._drift_counter = 0

            if self._drift_counter >= 4:
                self._soft_reset_tracking_state()
                # Keep current frame result; reset affects following frames.

            self._update_unvoiced_and_maybe_reset(timestamp_ms, voiced=True)
            
            # ===== STAGE 5: GRAMMAR VALIDATION =====
            validation_event = self.grammar_validator.validate_swara(
                swara_result, timestamp_ms)
            validation_event.frequency_hz = stabilized_frequency
            
            return FrameResult(
                timestamp_ms=timestamp_ms,
                frequency_hz=stabilized_frequency,
                voiced=True,
                voicing_prob=voicing_prob,
                swara_result=swara_result,
                validation_event=validation_event
            )
            
        except Exception as e:
            print(f"Frame analysis error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_file_streaming(self, audio_path: str, 
                              min_confidence: float = 0.3) -> Generator[FrameResult, None, None]:
        """
        Stream-analyze an audio file frame by frame.
        
        Args:
            audio_path: Path to audio file
            min_confidence: Minimum voicing confidence to process
            
        Yields:
            FrameResult objects for each valid frame
        """
        # Detect tonic first
        self.detect_tonic_from_file(audio_path)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        
        # Process frame by frame
        for frame_start in range(0, len(y) - self.frame_length, self.hop_length):
            frame_end = frame_start + self.frame_length
            audio_frame = y[frame_start:frame_end]
            
            timestamp_ms = (frame_start / sr) * 1000
            
            result = self.analyze_frame(audio_frame, timestamp_ms)
            if result and result.voicing_prob >= min_confidence:
                yield result
    
    def analyze_file(self, audio_path: str, verbose: bool = True) -> Dict:
        """
        Complete offline analysis of an audio file.
        
        Args:
            audio_path: Path to audio file
            verbose: Print progress information
            
        Returns:
            Complete analysis report with summary statistics
        """
        if verbose:
            print(f"\nAnalyzing {audio_path} for raga {self.raga_name}")
            print("=" * 60)
        
        start_time = time.time()
        
        # Collect all frame results
        frame_results = list(self.analyze_file_streaming(audio_path))
        
        analysis_time = time.time() - start_time
        
        if not frame_results:
            return {
                'success': False,
                'error': 'No valid frames detected',
                'analysis_time': analysis_time
            }
        
        # Extract validation events
        validation_events = [r.validation_event for r in frame_results 
                           if r.validation_event is not None]
        
        # Compile summary
        total_frames = len(frame_results)
        voiced_frames = len(validation_events)
        error_events = [e for e in validation_events if e.error_type is not None]
        
        # Get summary from grammar validator
        grammar_summary = self.grammar_validator.get_validation_summary()
        
        # Audio duration
        duration_sec = frame_results[-1].timestamp_ms / 1000 if frame_results else 0
        
        report = {
            'success': True,
            'audio_file': audio_path,
            'raga': self.raga_name,
            'detected_sa_hz': self.sa_frequency,
            'duration_sec': duration_sec,
            'analysis_time_sec': analysis_time,
            'total_frames': total_frames,
            'voiced_frames': voiced_frames,
            'total_errors': len(error_events),
            'error_rate': len(error_events) / max(voiced_frames, 1),
            'grammar_summary': grammar_summary,
            'error_events': error_events,
            'frame_results': frame_results  # Full detail for debugging
        }
        
        if verbose:
            print(f"\nAnalysis Summary:")
            print(f"  Duration: {duration_sec:.1f}s | Frames: {voiced_frames}/{total_frames}")
            print(f"  Sa frequency: {self.sa_frequency:.2f} Hz")
            print(f"  Errors detected: {len(error_events)} ({report['error_rate']:.1%})")
            print(f"  Analysis time: {analysis_time:.2f}s")
            
            if error_events:
                print(f"\nError Breakdown:")
                error_types = {}
                for event in error_events:
                    et = event.error_type.value if event.error_type else 'unknown'
                    error_types[et] = error_types.get(et, 0) + 1
                
                for error_type, count in error_types.items():
                    print(f"  {error_type}: {count}")
        
        return report
    
    def reset_analysis(self):
        """Reset all analysis state for new file"""
        if self.grammar_validator:
            self.grammar_validator.reset()
        self.sa_frequency = None
        self.swara_quantizer = None
        self._f0_history.clear()
        self._swara_history.clear()
        self._stable_swara = None
        self._candidate_swara = None
        self._candidate_count = 0
        self._last_frequency_hz = None
        self._last_timestamp_ms = None
        self._prev_instant_slope = 0.0
        self._last_output_swara = None
        self._last_output_change_ms = None
        self._unvoiced_ms_accum = 0.0
        self._drift_counter = 0

class BatchAnalyzer:
    """Analyze multiple audio files for comparative validation studies"""
    
    def __init__(self, raga_name: str):
        self.raga_name = raga_name
        self.pipeline = RealTimeGrammarPipeline(raga_name)
    
    def analyze_directory(self, dir_path: str, file_pattern: str = "*.wav") -> List[Dict]:
        """
        Analyze all matching files in a directory.
        
        Args:
            dir_path: Directory containing audio files
            file_pattern: Glob pattern for file matching
            
        Returns:
            List of analysis reports
        """
        import glob
        
        audio_files = glob.glob(os.path.join(dir_path, file_pattern))
        results = []
        
        for audio_file in audio_files:
            print(f"\nProcessing: {os.path.basename(audio_file)}")
            try:
                self.pipeline.reset_analysis()
                result = self.pipeline.analyze_file(audio_file, verbose=False)
                result['filename'] = os.path.basename(audio_file)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results.append({
                    'filename': os.path.basename(audio_file),
                    'success': False,
                    'error': str(e)
                })
        
        return results