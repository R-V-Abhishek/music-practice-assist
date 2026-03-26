"""
Live audio processor for raga practice.

Flow:
1. Collect short pre-roll for tonic Sa detection
2. Lock tonic
3. Process streaming chunks frame-by-frame
4. Quantize swaras and validate raga grammar in real time
5. Emit debounced forbidden-note alerts for UI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import librosa
import numpy as np

from .feedback_generator import FeedbackGenerator
from .grammar_validator import ErrorType
from .pitch_pipeline import FrameResult, RealTimeGrammarPipeline
from .swara_quantizer import get_swara_ordinal


@dataclass
class LiveProcessorConfig:
    target_sr: int = 22050
    bootstrap_seconds: float = 1.5
    min_voicing_prob: float = 0.25
    forbidden_trigger_frames: int = 3
    forbidden_clear_frames: int = 3
    frame_length: int = 768
    hop_length: int = 128
    pyin_frame_length: int = 512
    pyin_hop_length: int = 128
    min_frame_rms: float = 0.02
    min_swara_confidence: float = 0.25


class LiveAudioProcessor:
    """Stateful live audio processor for one student session."""

    def __init__(self, raga_name: str, config: Optional[LiveProcessorConfig] = None):
        self.raga_name = raga_name
        self.config = config or LiveProcessorConfig()
        self.pipeline = RealTimeGrammarPipeline(
            raga_name=raga_name,
            sr=self.config.target_sr,
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length,
            pyin_frame_length=self.config.pyin_frame_length,
            pyin_hop_length=self.config.pyin_hop_length,
            min_frame_rms=self.config.min_frame_rms,
        )
        self.feedback = FeedbackGenerator()

        self._audio_buffer = np.array([], dtype=np.float32)
        self._bootstrap_buffer = np.array([], dtype=np.float32)
        self._processed_samples = 0

        self._tonic_locked = False
        self._tonic_hz: Optional[float] = None
        self._input_sr: Optional[int] = None

        self._consecutive_forbidden = 0
        self._consecutive_clean = 0
        self._forbidden_alert_active = False
        # Time-based forbidden note debounce (in ms)
        self._forbidden_start_ms: Optional[float] = None
        self._clean_start_ms: Optional[float] = None
        # Trigger alert after 300ms of continuous forbidden note
        self._forbidden_trigger_ms: float = 300.0
        # Clear alert after 200ms of consecutive clean frames
        self._forbidden_clear_ms: float = 200.0

    def _reconfigure_for_input_sr(self, input_sr: int) -> None:
        """Bind pipeline to mic sample rate and preserve time-window durations."""
        if input_sr <= 0 or input_sr == self.config.target_sr:
            return

        ratio = float(input_sr) / float(self.config.target_sr)

        self.config.target_sr = int(input_sr)
        self.config.frame_length = max(256, int(round(self.config.frame_length * ratio)))
        self.config.hop_length = max(64, int(round(self.config.hop_length * ratio)))
        self.config.pyin_frame_length = max(256, int(round(self.config.pyin_frame_length * ratio)))
        self.config.pyin_hop_length = max(64, int(round(self.config.pyin_hop_length * ratio)))

        self.pipeline = RealTimeGrammarPipeline(
            raga_name=self.raga_name,
            sr=self.config.target_sr,
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length,
            pyin_frame_length=self.config.pyin_frame_length,
            pyin_hop_length=self.config.pyin_hop_length,
            min_frame_rms=self.config.min_frame_rms,
        )

        self._audio_buffer = np.array([], dtype=np.float32)
        self._bootstrap_buffer = np.array([], dtype=np.float32)
        self._processed_samples = 0

    @property
    def stage(self) -> str:
        if not self._tonic_locked:
            return "bootstrapping"
        return "running"

    @property
    def tonic_hz(self) -> Optional[float]:
        return self._tonic_hz

    def _resample_if_needed(self, chunk: np.ndarray, chunk_sr: int) -> np.ndarray:
        if chunk_sr == self.config.target_sr:
            return chunk.astype(np.float32, copy=False)

        resampled = librosa.resample(
            y=chunk.astype(np.float32, copy=False),
            orig_sr=chunk_sr,
            target_sr=self.config.target_sr,
            res_type="soxr_hq",
        )
        return resampled.astype(np.float32, copy=False)

    def _try_lock_tonic(self) -> Optional[Dict[str, Any]]:
        if self._tonic_locked:
            return None

        bootstrap_needed = int(self.config.bootstrap_seconds * self.config.target_sr)
        if len(self._bootstrap_buffer) < bootstrap_needed:
            return None

        bootstrap_audio = self._bootstrap_buffer[:bootstrap_needed]
        sa = self.pipeline.detect_tonic_from_audio(bootstrap_audio)
        self._tonic_hz = sa
        self._tonic_locked = True

        return {
            "type": "session_status",
            "stage": "locked",
            "tonic_hz": round(sa, 3),
        }

    def _frame_to_event(self, frame: FrameResult) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "type": "frame_event",
            "timestamp_ms": round(frame.timestamp_ms, 2),
            "frequency_hz": round(frame.frequency_hz, 3),
            "voiced": frame.voiced,
            "voicing_prob": round(frame.voicing_prob, 3),
        }

        if frame.swara_result is not None:
            event["swara"] = frame.swara_result.swara
            event["swara_index"] = get_swara_ordinal(frame.swara_result.swara)
            event["octave"] = frame.swara_result.octave
            event["cents_deviation"] = round(frame.swara_result.cents_deviation, 2)
            event["swara_confidence"] = round(frame.swara_result.confidence, 3)

        if frame.validation_event is not None:
            event["direction"] = frame.validation_event.direction.value
            event["error_type"] = (
                frame.validation_event.error_type.value
                if frame.validation_event.error_type is not None
                else None
            )
            event["description"] = frame.validation_event.description

        return event

    def _build_alert_events(self, frame: FrameResult) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        validation = frame.validation_event
        ts = frame.timestamp_ms

        if validation is None or validation.error_type != ErrorType.FORBIDDEN_NOTE:
            # Reset forbidden timer; start/continue clean timer
            self._forbidden_start_ms = None
            if self._clean_start_ms is None:
                self._clean_start_ms = ts

            if (
                self._forbidden_alert_active
                and self._clean_start_ms is not None
                and (ts - self._clean_start_ms) >= self._forbidden_clear_ms
            ):
                self._forbidden_alert_active = False
                self._clean_start_ms = None
                alerts.append(
                    {
                        "type": "alert_event",
                        "alert": "forbidden_note",
                        "state": "cleared",
                        "timestamp_ms": round(ts, 2),
                    }
                )
            return alerts

        # Forbidden note frame: reset clean timer; start/continue forbidden timer
        self._clean_start_ms = None
        if self._forbidden_start_ms is None:
            self._forbidden_start_ms = ts

        if (
            not self._forbidden_alert_active
            and (ts - self._forbidden_start_ms) >= self._forbidden_trigger_ms
        ):
            self._forbidden_alert_active = True
            feedback = self.feedback.generate_feedback(validation, self.raga_name)
            alerts.append(
                {
                    "type": "alert_event",
                    "alert": "forbidden_note",
                    "state": "active",
                    "timestamp_ms": round(ts, 2),
                    "swara": validation.swara,
                    "direction": validation.direction.value,
                    "message": feedback.get("message", validation.description),
                    "suggestion": feedback.get("suggestion", ""),
                    "confidence": round(validation.confidence, 3),
                }
            )

        return alerts

    def process_audio_chunk(self, audio_chunk: np.ndarray, chunk_sr: int) -> Dict[str, Any]:
        """
        Process one incoming PCM chunk and return UI events.

        Args:
            audio_chunk: Float32 mono PCM in range [-1, 1]
            chunk_sr: Input chunk sample rate
        """
        if audio_chunk.ndim != 1:
            audio_chunk = np.mean(audio_chunk, axis=1)

        if self._input_sr is None:
            self._input_sr = int(chunk_sr)
            self._reconfigure_for_input_sr(self._input_sr)

        chunk = self._resample_if_needed(audio_chunk, chunk_sr)
        if len(chunk) == 0:
            return {"stage": self.stage, "events": []}

        self._bootstrap_buffer = np.concatenate([self._bootstrap_buffer, chunk])
        self._audio_buffer = np.concatenate([self._audio_buffer, chunk])

        events: List[Dict[str, Any]] = []

        status_event = self._try_lock_tonic()
        if status_event is not None:
            events.append(status_event)

        if not self._tonic_locked:
            return {
                "stage": self.stage,
                "tonic_hz": self._tonic_hz,
                "events": events,
            }

        frame_len = self.pipeline.frame_length
        hop = self.pipeline.hop_length

        while len(self._audio_buffer) >= frame_len:
            frame = self._audio_buffer[:frame_len]
            ts_ms = (self._processed_samples / self.config.target_sr) * 1000.0
            result = self.pipeline.analyze_frame(frame, ts_ms)

            # Pass any frame that has a valid tracked frequency (>0 Hz)
            if result is not None and result.frequency_hz > 0:
                events.append(self._frame_to_event(result))
                events.extend(self._build_alert_events(result))

            self._audio_buffer = self._audio_buffer[hop:]
            self._processed_samples += hop

        return {
            "stage": self.stage,
            "tonic_hz": self._tonic_hz,
            "events": events,
        }

    def get_session_summary(self) -> Dict[str, Any]:
        if self.pipeline.grammar_validator is None:
            return {
                "raga": self.raga_name,
                "stage": self.stage,
                "tonic_hz": self._tonic_hz,
                "message": "No validated frames yet",
            }

        summary = self.pipeline.grammar_validator.get_validation_summary()
        feedback_summary = self.feedback.generate_session_summary(
            self.pipeline.grammar_validator.events,
            self.raga_name,
        )

        return {
            "raga": self.raga_name,
            "stage": self.stage,
            "tonic_hz": self._tonic_hz,
            "validation": summary,
            "feedback": feedback_summary,
        }
