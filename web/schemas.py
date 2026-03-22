"""Schema helpers for websocket payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StartSessionRequest:
    raga_name: str
    bootstrap_seconds: float = 3.0


@dataclass
class AudioChunkMessage:
    type: str
    sample_rate: int
    audio_b64: str
    chunk_id: Optional[int] = None
