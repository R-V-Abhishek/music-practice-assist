import numpy as np

from raga_grammar.grammar_validator import Direction, ErrorType, ValidationEvent
from raga_grammar.live_audio_processor import LiveAudioProcessor, LiveProcessorConfig
from raga_grammar.pitch_pipeline import FrameResult
from raga_grammar.swara_quantizer import SwaraResult


class _StubGrammarValidator:
    def __init__(self):
        self.events = []

    def get_validation_summary(self):
        return {
            "raga": "Kāṁbhōji",
            "total_swaras": len(self.events),
            "total_errors": len([e for e in self.events if e.error_type is not None]),
            "error_rate": 0.0,
            "error_breakdown": {},
            "duration_ms": 0,
        }


class _StubPipeline:
    def __init__(self, errors):
        self.frame_length = 8
        self.hop_length = 4
        self._errors = errors
        self._idx = 0
        self.grammar_validator = _StubGrammarValidator()

    def detect_tonic_from_audio(self, _audio):
        return 260.0

    def analyze_frame(self, _audio_frame, timestamp_ms):
        err = self._errors[self._idx] if self._idx < len(self._errors) else None
        self._idx += 1

        sw = SwaraResult(swara="Ni2", octave=0, cents_deviation=0.0, confidence=1.0)
        event = ValidationEvent(
            timestamp_ms=timestamp_ms,
            swara="Ni2",
            octave=0,
            frequency_hz=487.5,
            cents_deviation=0.0,
            error_type=err,
            direction=Direction.ASCENDING,
            description="test",
            confidence=1.0,
        )
        self.grammar_validator.events.append(event)
        return FrameResult(
            timestamp_ms=timestamp_ms,
            frequency_hz=487.5,
            voiced=True,
            voicing_prob=1.0,
            swara_result=sw,
            validation_event=event,
        )


def test_tonic_bootstrap_locks_after_preroll():
    cfg = LiveProcessorConfig(target_sr=22050, bootstrap_seconds=0.01)
    p = LiveAudioProcessor("Kāṁbhōji", config=cfg)
    p.pipeline = _StubPipeline(errors=[])

    chunk = np.zeros(int(0.02 * cfg.target_sr), dtype=np.float32)
    out = p.process_audio_chunk(chunk, cfg.target_sr)

    assert p.tonic_hz == 260.0
    assert out["stage"] == "running"
    assert any(e.get("type") == "session_status" and e.get("stage") == "locked" for e in out["events"])


def test_forbidden_note_alert_debounce_and_clear():
    cfg = LiveProcessorConfig(
        target_sr=22050,
        bootstrap_seconds=0.0,
        forbidden_trigger_frames=3,
        forbidden_clear_frames=2,
    )
    p = LiveAudioProcessor("Kāṁbhōji", config=cfg)

    # 3 forbidden -> trigger, then 2 clean -> clear
    errors = [
        ErrorType.FORBIDDEN_NOTE,
        ErrorType.FORBIDDEN_NOTE,
        ErrorType.FORBIDDEN_NOTE,
        None,
        None,
    ]
    p.pipeline = _StubPipeline(errors=errors)
    p._tonic_locked = True
    p._tonic_hz = 260.0

    # Enough samples for 5 frames with frame=8 hop=4 needs 24 samples
    chunk = np.zeros(24, dtype=np.float32)
    out = p.process_audio_chunk(chunk, cfg.target_sr)

    alerts = [e for e in out["events"] if e.get("type") == "alert_event"]
    assert any(a.get("state") == "active" for a in alerts)
    assert any(a.get("state") == "cleared" for a in alerts)
