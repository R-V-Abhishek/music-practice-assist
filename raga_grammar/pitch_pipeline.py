"""
Pitch Pipeline - Real-Time Grammar Validation Integration

Integrates with existing tonic_sa_detection.py to provide end-to-end pipeline:
Audio → Sa Detection → pYIN Pitch Tracking → Swara Quantization → Grammar Validation

Supports both real-time streaming (23ms per frame) and offline file analysis.
"""

import numpy as np
import librosa
import time
import math
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

class ShruthiHMM:
    """GC-SHMM forward prediction for gap filling."""
    def __init__(self, quantizer: SwaraQuantizer):
        self.quantizer = quantizer
        self.num_states = 22
        self.alpha = np.ones(self.num_states) / self.num_states
        
    def reset(self):
        self.alpha = np.ones(self.num_states) / self.num_states
        
    def step_forward(self, emission_probs: Optional[np.ndarray] = None):
        new_alpha = np.zeros(self.num_states)
        for j in range(self.num_states):
            sum_prob = 0.0
            for i in range(self.num_states):
                cent_i = self.quantizer.SHRUTI_CENTS_22[i]
                cent_j = self.quantizer.SHRUTI_CENTS_22[j]
                
                dist1 = abs(cent_j - cent_i)
                dist2 = abs(cent_j + 1200.0 - cent_i)
                dist3 = abs(cent_j - 1200.0 - cent_i)
                min_dist = min(dist1, dist2, dist3)
                
                # grammar constraints: 200 cents max leap in gap prediction
                if min_dist <= 200.0:
                    trans_prob = np.exp(- (0.1 / 100.0) * min_dist)
                else:
                    trans_prob = 0.0
                
                sum_prob += self.alpha[i] * trans_prob
            
            new_alpha[j] = sum_prob
        
        if emission_probs is not None:
            new_alpha *= emission_probs
            
        s = np.sum(new_alpha)
        if s > 0:
            self.alpha = new_alpha / s
        else:
            self.alpha = np.ones(self.num_states) / self.num_states
            
    def get_best_state(self) -> int:
        return int(np.argmax(self.alpha))


class RealTimeGrammarPipeline:
    """
    Real-time audio analysis pipeline for raga grammar validation.
    
    Pipeline stages:
    1. Audio framing (2048 samples, hop=1024, ~46ms at 22050 Hz)
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
        hop_length: int = 1024,
        pyin_frame_length: int = 1024,
        pyin_hop_length: int = 256,
        min_frame_rms: float = 0.008,
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

        # Responsiveness-first tuning: react quickly to swara changes
        self._hysteresis_fast_switch_conf = 0.60      # was 0.50
        self._hysteresis_candidate_min_count = 1      
        self._hysteresis_candidate_switch_conf = 0.35 
        self._min_duration_hold_ms = 18.0             
        self._min_duration_low_motion_st_per_sec = 6.0  
        self._min_duration_not_strong_conf = 0.60     
        self._majority_hold_conf_max = 0.25           
        self._hysteresis_log2_margin = 25.0 / 1200.0  # was 15.0 / 1200.0
        
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
        self._f0_history = deque(maxlen=2)    
        self._swara_history = deque(maxlen=2)  
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
        self._octave_error_count: int = 0
        self._octave_error_accept_after: int = 4

        # MPM (McLeod Pitch Method) tuning
        self._mpm_rms_threshold: float = 0.008
        self._mpm_key_threshold: float = 0.93
        self._mpm_min_hz: float = 50.0
        self._mpm_max_hz: float = 800.0  # Indian classical: female taara ~1108Hz, 800Hz is safe ceiling
        self._mpm_pitch_history: deque[float] = deque(maxlen=3)

        # Adaptive 2-state Kalman smoothing [pitch_hz, velocity_hz_per_frame]
        self._kalman_state: Optional[np.ndarray] = None
        self._kalman_P: Optional[np.ndarray] = None
        self._kalman_q_base: float = 2.0    
        self._kalman_r_base: float = 20.0   
        
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
        self.hmm = ShruthiHMM(self.swara_quantizer)

        qmin_hz, qmax_hz = self.swara_quantizer.get_supported_frequency_range()
        
        print(f"✓ Sa locked to {self.sa_frequency:.2f} Hz")
        print(f"✓ All swaras calculated from this Sa")
        print(f"✓ Quantization range: {qmin_hz:.2f}–{qmax_hz:.2f} Hz")
        
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

        fd, tmp_path = tempfile.mkstemp(prefix='sa_detect_', suffix='.wav')
        os.close(fd)

        try:
            sf.write(tmp_path, y, self.sr)
            return self.detect_tonic_from_file(tmp_path)
        finally:
            # Temp input is cached by path in tonic detector; evict this key so
            # repeated live sessions don't retain many stale temp entries.
            self.tonic_detector.evict_audio_cache(tmp_path)

            # Windows can hold short-lived read locks after decoder access.
            # Retry unlink with small backoff to avoid noisy cleanup failures.
            for attempt in range(12):
                try:
                    os.unlink(tmp_path)
                    break
                except FileNotFoundError:
                    break
                except (PermissionError, OSError):
                    if attempt == 11:
                        # Best-effort cleanup; avoid failing tonic detection path
                        # because of transient OS file lock behavior.
                        pass
                    else:
                        time.sleep(0.05 * (attempt + 1))

    def _apply_swara_hysteresis(self, swara_result: SwaraResult) -> SwaraResult:
        """
        Frequency-displacement based note switching.

        Switches to a new swara based on how far the raw pitch has moved away
        from the current stable note's center in cents. This ensures that
        sustained-vowel glides trigger note changes as soon as the pitch is
        clearly in the new note's territory, regardless of per-frame confidence.
        """
        if self.swara_quantizer is None:
            return swara_result

        # Initialize stable state on first frame
        if self._stable_swara is None:
            self._stable_swara = swara_result.swara
            self._candidate_swara = None
            self._candidate_count = 0
            return swara_result

        # No change needed — pitch is still on the stable swara
        if swara_result.swara == self._stable_swara:
            self._candidate_swara = None
            self._candidate_count = 0
            return swara_result

        # ── PRIMARY GATE: Frequency-displacement based switching ──────────────
        # Compute how far the ACPT raw frequency has moved from the OLD stable
        # note center in cents. If >= 50 cents away, the pitch has crossed the
        # midpoint between the two swaras — switch immediately, no conf needed.
        stable_freq = self.swara_quantizer.get_swara_frequency(self._stable_swara, swara_result.octave)
        if stable_freq > 0:
            new_note_freq = self.swara_quantizer.get_swara_frequency(swara_result.swara, swara_result.octave)
            # Re-derive raw freq: it's the new note's center offset by how many
            # cents the quantizer reported the deviation FROM the new note.
            cents_dev = swara_result.cents_deviation if swara_result.cents_deviation is not None else 0.0
            raw_freq = new_note_freq * (2 ** (cents_dev / 1200.0)) if new_note_freq > 0 else 0.0
            if raw_freq > 0:
                displacement_from_old = abs(1200.0 * math.log2(raw_freq / stable_freq))

                # Hysteresis margin in log-frequency space: switch only when the
                # new label is at least 25 cents closer than currently locked swara.
                old_log = math.log2(stable_freq)
                new_log = math.log2(new_note_freq) if new_note_freq > 0 else old_log
                cur_log = math.log2(raw_freq)
                old_dist = abs(cur_log - old_log)
                new_dist = abs(cur_log - new_log)
                significantly_closer = (old_dist - new_dist) >= self._hysteresis_log2_margin

                if displacement_from_old >= 50.0:
                    if not significantly_closer and swara_result.confidence < self._hysteresis_fast_switch_conf:
                        held = self.swara_quantizer.to_swara_tonic_band(
                            self.swara_quantizer.get_swara_frequency(self._stable_swara, swara_result.octave)
                        )
                        if held is not None:
                            held.confidence = max(held.confidence * 0.85, swara_result.confidence * 0.70)
                            return held
                    self._stable_swara = swara_result.swara
                    self._candidate_swara = None
                    self._candidate_count = 0
                    return swara_result

        # ── SECONDARY GATE: High-confidence instant switch ────────────────────
        if swara_result.confidence >= self._hysteresis_fast_switch_conf:
            self._stable_swara = swara_result.swara
            self._candidate_swara = None
            self._candidate_count = 0
            return swara_result

        # Accumulate candidate evidence for the new note
        if self._candidate_swara == swara_result.swara:
            self._candidate_count += 1
        else:
            self._candidate_swara = swara_result.swara
            self._candidate_count = 1

        if (
            self._candidate_count >= self._hysteresis_candidate_min_count
            and swara_result.confidence >= self._hysteresis_candidate_switch_conf
        ):
            self._stable_swara = swara_result.swara
            self._candidate_swara = None
            self._candidate_count = 0
            return swara_result

        # Hold previous stable swara — pitch is ambiguous midway
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

        Corrects sudden near-octave flips when they happen in a short time window.
        """
        if (
            frequency_hz <= 0
            or self._last_frequency_hz is None
            or self._last_timestamp_ms is None
        ):
            return frequency_hz

        dt_ms = float(timestamp_ms - self._last_timestamp_ms)
        if dt_ms <= 0 or dt_ms > 120.0:
            return frequency_hz

        jump_cents = 1200.0 * math.log2((frequency_hz + 1e-9) / (self._last_frequency_hz + 1e-9))
        abs_jump = abs(jump_cents)

        # Near-octave flips (roughly 2x / 0.5x) are corrected directly.
        if 900.0 <= abs_jump <= 1500.0:
            corrected = frequency_hz / 2.0 if jump_cents > 0 else frequency_hz * 2.0
            return corrected

        # Extreme outlier jumps are rejected.
        if abs_jump > 1800.0:
            return float(self._last_frequency_hz)

        return frequency_hz

    def _apply_kalman_smoothing(self, frequency_hz: float, confidence: float) -> float:
        """Apply adaptive 2-state Kalman smoothing to pitch track."""
        if frequency_hz <= 0:
            self._kalman_state = None
            self._kalman_P = None
            return frequency_hz

        if self._kalman_state is None or self._kalman_P is None:
            self._kalman_state = np.array([float(frequency_hz), 0.0], dtype=np.float64)
            self._kalman_P = np.array([[20.0, 0.0], [0.0, 40.0]], dtype=np.float64)
            return float(frequency_hz)

        # Near-constant velocity model
        F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Gamaka-adaptive process noise scaling
        pred_pitch = float(self._kalman_state[0])
        q_scale = 3.0 if abs(float(frequency_hz) - pred_pitch) > 8.0 else 1.0
        q = self._kalman_q_base * q_scale
        Q = np.array([[q, 0.0], [0.0, q * 0.25]], dtype=np.float64)

        # Confidence-aware measurement noise
        r = self._kalman_r_base / max(0.1, float(confidence))
        R = np.array([[r]], dtype=np.float64)

        # Predict
        x_pred = F @ self._kalman_state
        P_pred = F @ self._kalman_P @ F.T + Q

        # Update
        z = np.array([[float(frequency_hz)]], dtype=np.float64)
        y = z - (H @ x_pred).reshape(1, 1)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self._kalman_state = x_pred + (K @ y).reshape(2)
        I = np.eye(2, dtype=np.float64)
        self._kalman_P = (I - K @ H) @ P_pred
        return float(self._kalman_state[0])

    def _apply_octave_jump_guard(self, raw_frequency_hz: float) -> float:
        """
        Reject suspicious octave-like jumps unless sustained for several frames.
        """
        if raw_frequency_hz <= 0:
            self._octave_error_count = 0
            return raw_frequency_hz

        if self._kalman_state is None:
            self._octave_error_count = 0
            return raw_frequency_hz

        ref_pitch = float(self._kalman_state[0])
        if ref_pitch <= 0:
            self._octave_error_count = 0
            return raw_frequency_hz

        jump_log2 = abs(math.log2((raw_frequency_hz + 1e-9) / (ref_pitch + 1e-9)))
        if jump_log2 > 0.6:
            self._octave_error_count += 1
            if self._octave_error_count > self._octave_error_accept_after:
                # Accept sustained jump and reset filter around new pitch.
                self._kalman_state = np.array([float(raw_frequency_hz), 0.0], dtype=np.float64)
                self._kalman_P = np.array([[20.0, 0.0], [0.0, 40.0]], dtype=np.float64)
                self._octave_error_count = 0
                return raw_frequency_hz
            # Temporary spike: keep previous stabilized reference.
            return ref_pitch

        self._octave_error_count = 0
        return raw_frequency_hz

    def _commit_frequency_state(self, frequency_hz: float, timestamp_ms: float) -> None:
        """Commit frequency state after final per-frame decision."""
        if frequency_hz > 0:
            self._last_frequency_hz = float(frequency_hz)
            self._last_timestamp_ms = float(timestamp_ms)

    def _soft_reset_tracking_state(self) -> None:
        """Reset transient tracking state without touching tonic Sa lock."""
        self._f0_history.clear()
        self._mpm_pitch_history.clear()
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
        self._kalman_state = None
        self._kalman_P = None
        self._octave_error_count = 0
        if hasattr(self, 'hmm'):
            self.hmm.reset()

        if self.grammar_validator is not None:
            self.grammar_validator.reset_phrase_buffer()

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
        if self._unvoiced_ms_accum >= 300.0:
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
        short_deviation = elapsed_ms < self._min_duration_hold_ms
        low_motion = slope_st_per_sec < self._min_duration_low_motion_st_per_sec
        not_very_strong = swara_result.confidence < self._min_duration_not_strong_conf

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

        band_mask = (freqs >= max(30.0, 0.25 * self.sa_frequency)) & (freqs <= min(self.sr / 2.0, 4.0 * self.sa_frequency))
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

        # Tonic-anchored range from anumaandra Sa to ati taara Sa.
        qmin_hz, qmax_hz = self.swara_quantizer.get_supported_frequency_range()
        fmin = max(60.0, qmin_hz * 0.95)
        fmax = min(1200.0, qmax_hz * 1.05)
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

    def _estimate_swara_from_mpm(self, audio_frame: np.ndarray) -> Optional[Tuple[SwaraResult, float, float]]:
        """
        Estimate swara using McLeod Pitch Method (MPM) with NSDF.

        Steps:
        1) RMS gate
        2) NSDF over lag range
        3) Skip initial positive lobe
        4) Collect positive local maxima
        5) First-above-key-threshold selection
        6) Parabolic lag refinement
        7) Hz conversion + sanity guard
        8) Short median smoothing (3 frames)
        """
        if self.swara_quantizer is None or self.sa_frequency is None:
            return None

        x = audio_frame.astype(np.float64, copy=False)
        n = len(x)
        if n < 256:
            return None

        # Step A: RMS gate
        rms = float(np.sqrt(np.mean(x * x)))
        if rms < self._mpm_rms_threshold:
            return None

        # Use configured frame length (2048 in offline path, smaller in low-latency path).
        size = int(min(n, self.frame_length))
        if size < 256:
            return None

        x = x[:size]
        max_lag = size // 2
        if max_lag < 3:
            return None

        # Step B: NSDF computation
        nsdf = np.zeros(max_lag, dtype=np.float64)
        for tau in range(max_lag):
            x1 = x[: size - tau]
            x2 = x[tau:size]
            acf_tau = float(np.dot(x1, x2))
            m_tau = float(np.dot(x1, x1) + np.dot(x2, x2))
            nsdf[tau] = (2.0 * acf_tau / m_tau) if m_tau > 0.0 else 0.0

        # Step C: ignore initial positive lobe
        i = 1
        while i < (max_lag - 1) and nsdf[i] > 0.0:
            i += 1

        # Step D: peak collection
        peaks: List[Tuple[int, float]] = []
        for k in range(max(i + 1, 1), max_lag - 1):
            v = float(nsdf[k])
            if v > 0.0 and v > nsdf[k - 1] and v >= nsdf[k + 1]:
                peaks.append((k, v))

        if not peaks:
            return None

        # Step E: key-threshold selection
        max_peak = max(p[1] for p in peaks)
        threshold = self._mpm_key_threshold * max_peak
        selected_idx, selected_val = peaks[-1]
        for p_idx, p_val in peaks:
            if p_val >= threshold:
                selected_idx, selected_val = p_idx, p_val
                break

        # Step F: parabolic interpolation
        t0 = float(selected_idx)
        if 0 < selected_idx < (max_lag - 1):
            x1 = float(nsdf[selected_idx - 1])
            x2 = float(nsdf[selected_idx])
            x3 = float(nsdf[selected_idx + 1])
            a = (x1 + x3 - 2.0 * x2) / 2.0
            b = (x3 - x1) / 2.0
            if abs(a) > 1e-12:
                t0 = t0 - (b / (2.0 * a))

        if t0 <= 0.0:
            return None

        # Step G/H: lag -> frequency + bounds
        pitch_hz = float(self.sr / t0)
        if pitch_hz < self._mpm_min_hz or pitch_hz > self._mpm_max_hz:
            return None

        # Additional tonic-focused support range guard
        qmin_hz, qmax_hz = self.swara_quantizer.get_supported_frequency_range()
        if pitch_hz < (qmin_hz * 0.92) or pitch_hz > (qmax_hz * 1.08):
            return None

        # Step I: short median post-filter
        self._mpm_pitch_history.append(pitch_hz)
        if len(self._mpm_pitch_history) == self._mpm_pitch_history.maxlen:
            stabilized_frequency = float(np.median(np.array(self._mpm_pitch_history, dtype=np.float64)))
        else:
            stabilized_frequency = pitch_hz

        swara_result = self.swara_quantizer.to_swara_tonic_band(stabilized_frequency)
        if swara_result is None:
            return None

        voicing_prob = float(np.clip(selected_val, 0.0, 1.0))
        swara_result.confidence = float(
            np.clip(0.6 * swara_result.confidence + 0.4 * voicing_prob, 0.0, 1.0)
        )
        return swara_result, stabilized_frequency, voicing_prob

    def _estimate_swara_from_yin(self, audio_frame: np.ndarray) -> Optional[Tuple[SwaraResult, float, float]]:
        """Estimate swara using YIN pitch tracking in tonic-focused range."""
        if self.swara_quantizer is None or self.sa_frequency is None:
            return None

        qmin_hz, qmax_hz = self.swara_quantizer.get_supported_frequency_range()
        fmin = max(60.0, qmin_hz * 0.95)
        fmax = min(1200.0, qmax_hz * 1.05)

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
                if self.grammar_validator is not None:
                    self.grammar_validator.tick_direction(timestamp_ms)
                return FrameResult(
                    timestamp_ms=timestamp_ms,
                    frequency_hz=0,
                    voiced=False,
                    voicing_prob=0,
                    swara_result=None,
                    validation_event=None
                )

            # ===== STAGE 2: LOW-LATENCY HZ ESTIMATION =====
            if not self.sa_frequency:
                return None

            mpm_estimate = self._estimate_swara_from_mpm(audio_frame)
            acpt_estimate = self._estimate_swara_from_acpt(audio_frame)

            raw_hz = None
            if mpm_estimate is not None and acpt_estimate is not None:
                mpm_hz, mpm_conf = mpm_estimate[1], mpm_estimate[2]
                acpt_hz, acpt_conf = acpt_estimate[1], acpt_estimate[2]
                # Reject blend if estimators disagree by >0.5 octave (spike indicator)
                if abs(math.log2((mpm_hz + 1e-9) / (acpt_hz + 1e-9))) > 0.5:
                    # Trust the lower estimate — vocal fundamentals are lower than harmonics
                    raw_hz = mpm_hz if mpm_hz < acpt_hz else acpt_hz
                else:
                    total_conf = mpm_conf + acpt_conf
                    if total_conf > 0:
                        raw_hz = (mpm_hz * mpm_conf + acpt_hz * acpt_conf) / total_conf
            elif acpt_estimate is not None:
                raw_hz = acpt_estimate[1]
            elif mpm_estimate is not None:
                raw_hz = mpm_estimate[1]

            if raw_hz is not None:
                # ===== STAGE 2C: FST CORRECTION =====
                fst_result = self.swara_quantizer.fst_correct(raw_hz, getattr(self, '_last_shruti_idx', None))
                if fst_result is not None:
                    swara_result, best_shruti_idx = fst_result
                    self._last_shruti_idx = best_shruti_idx
                    stabilized_frequency = raw_hz
                    voicing_prob = swara_result.confidence
                    
                    # Update HMM forward pass with actual observation
                    emission = np.zeros(22)
                    raw_cents = 1200.0 * np.log2(raw_hz / self.sa_frequency)
                    while raw_cents < 0: raw_cents += 1200.0
                    while raw_cents >= 1200.0: raw_cents -= 1200.0
                    for i, c in enumerate(self.swara_quantizer.SHRUTI_CENTS_22):
                        dev1 = raw_cents - c
                        dev2 = (raw_cents - 1200.0) - c
                        dev3 = (raw_cents + 1200.0) - c
                        dev = min([dev1, dev2, dev3], key=abs)
                        emission[i] = np.exp(-0.5 * (dev / 25.0)**2)
                    self.hmm.step_forward(emission)
                else:
                    raw_hz = None

            if raw_hz is None:
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
                        self.grammar_validator.tick_direction(timestamp_ms)
                        return FrameResult(
                            timestamp_ms=timestamp_ms,
                            frequency_hz=float(self._last_frequency_hz),
                            voiced=True,
                            voicing_prob=0.18,
                            swara_result=held_result,
                            validation_event=validation_event,
                        )
                        
                # ---------------- NEW GC-SHMM GAP FILL ---------------- #
                # If we've been tracking, and now we hit a >70ms gap, use GC-SHMM
                # Skip first 150ms of silence to suppress phantom end-notes
                if self._last_frequency_hz is not None and 150.0 <= getattr(self, '_unvoiced_ms_accum', 0) < 300.0:
                    self.hmm.step_forward(None) # uniform emission
                    best_s = self.hmm.get_best_state()
                    swara_name = self.swara_quantizer.SHRUTI_TO_SWARA[best_s]
                    
                    # Estimate octave based on previous frame
                    last_octave = 0
                    if getattr(self, '_last_frequency_hz', None):
                        oct_est = int(np.round(np.log2(self._last_frequency_hz / self.sa_frequency)))
                        last_octave = max(self.swara_quantizer.MIN_OCTAVE_OFFSET, min(self.swara_quantizer.MAX_OCTAVE_OFFSET, oct_est))
                        
                    predicted_result = SwaraResult(
                        swara=swara_name,
                        octave=last_octave,
                        cents_deviation=0.0,
                        confidence=0.15 # Low confidence
                    )
                    
                    freq_hz = self.swara_quantizer.sa_frequency * (2 ** (self.swara_quantizer.SHRUTI_CENTS_22[best_s] / 1200.0)) * (2 ** last_octave)
                    
                    validation_event = self.grammar_validator.validate_swara(predicted_result, timestamp_ms)
                    validation_event.frequency_hz = freq_hz
                    self.grammar_validator.tick_direction(timestamp_ms)
                    
                    # Do not update self._last_frequency_hz from HMM to avoid diverging forever
                    self._update_unvoiced_and_maybe_reset(timestamp_ms, voiced=False)
                    return FrameResult(
                        timestamp_ms=timestamp_ms,
                        frequency_hz=freq_hz,
                        voiced=True, # Mark as voiced so UI plots it
                        voicing_prob=0.15,
                        swara_result=predicted_result,
                        validation_event=validation_event
                    )
                # ------------------------------------------------------ #
                
                self._update_unvoiced_and_maybe_reset(timestamp_ms, voiced=False)
                return FrameResult(
                    timestamp_ms=timestamp_ms,
                    frequency_hz=0,
                    voiced=False,
                    voicing_prob=0,
                    swara_result=None,
                    validation_event=None
                )

            # ===== STAGE 3A: CONTINUITY CONSTRAINT =====
            stabilized_frequency = self._apply_frequency_continuity(stabilized_frequency, timestamp_ms)
            stabilized_frequency = self._apply_octave_jump_guard(stabilized_frequency)
            
            # Kalman on cents for smoother pitch
            raw_cents = 1200.0 * np.log2(stabilized_frequency / self.sa_frequency)
            smooth_cents = self._apply_kalman_smoothing(raw_cents, confidence=voicing_prob)
            stabilized_frequency = self.sa_frequency * (2 ** (smooth_cents / 1200.0))

            # ===== STAGE 3B: LOCAL FREQUENCY REFINEMENT =====
            # Disabled to allow continuous frequency gliding ("aas" and "oos") 
            # as requested by the user, rather than snapping to peaks.
            # refined_frequency = self._refine_frequency_near_swara(
            #     audio_frame, swara_result, stabilized_frequency
            # )
            # refined_result = self.swara_quantizer.to_swara_tonic_band(refined_frequency)
            # if refined_result is not None and refined_result.swara == swara_result.swara:
            #     refined_result.confidence = max(refined_result.confidence, swara_result.confidence)
            #     swara_result = refined_result
            #     stabilized_frequency = refined_frequency

            # ===== STAGE 3C: INSTANTANEOUS STABILITY ANCHOR =====
            # Disabled hysteresis to allow fully responsive raw note changes
            # swara_result = self._apply_swara_hysteresis(swara_result)
            # stable, slope = self._is_stable_bidir_anchor(stabilized_frequency, timestamp_ms)
            # swara_result = self._apply_min_duration_hold(swara_result, timestamp_ms, slope)
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
                if (
                    dominant_count >= 2
                    and dominant_swara != swara_result.swara
                    and swara_result.confidence < self._majority_hold_conf_max
                ):
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
            
            self.grammar_validator.tick_direction(timestamp_ms)
            
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
        self._mpm_pitch_history.clear()
        self._kalman_state = None
        self._kalman_P = None
        self._octave_error_count = 0

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