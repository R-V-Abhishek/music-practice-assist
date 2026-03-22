"""UI feedback adapter utilities for live dashboard payloads."""

from __future__ import annotations

from typing import Any, Dict


def compact_frame_label(frame_event: Dict[str, Any]) -> str:
    swara = frame_event.get("swara", "-")
    freq = frame_event.get("frequency_hz", 0)
    t = frame_event.get("timestamp_ms", 0)
    return f"[{int(t)} ms] {swara} @ {freq} Hz"


def alert_text(alert_event: Dict[str, Any]) -> str:
    if alert_event.get("state") == "cleared":
        return "Forbidden note alert cleared"

    msg = alert_event.get("message", "Forbidden note detected")
    suggestion = alert_event.get("suggestion")
    if suggestion:
        return f"{msg} | Suggestion: {suggestion}"
    return msg
