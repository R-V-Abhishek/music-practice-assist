"""In-memory session manager for live raga dashboard."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from raga_grammar.live_audio_processor import LiveAudioProcessor, LiveProcessorConfig


@dataclass
class LiveSession:
    session_id: str
    raga_name: str
    processor: LiveAudioProcessor


class SessionManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, LiveSession] = {}

    def create_session(self, raga_name: str, bootstrap_seconds: float = 1.5) -> LiveSession:
        session_id = str(uuid.uuid4())
        config = LiveProcessorConfig(bootstrap_seconds=bootstrap_seconds)
        session = LiveSession(
            session_id=session_id,
            raga_name=raga_name,
            processor=LiveAudioProcessor(raga_name=raga_name, config=config),
        )

        with self._lock:
            self._sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> Optional[LiveSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> Optional[dict]:
        with self._lock:
            session = self._sessions.pop(session_id, None)

        if not session:
            return None

        return session.processor.get_session_summary()
