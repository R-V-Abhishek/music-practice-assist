from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from raga_grammar.raga_database import get_raga_info, list_available_ragas
from web.session_manager import SessionManager


class StartSessionBody(BaseModel):
    raga_name: str
    bootstrap_seconds: float = 1.5


app = FastAPI(title="Live Raga Practice")
session_manager = SessionManager()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/ragas")
def ragas() -> list[str]:
    return sorted(list_available_ragas())


@app.post("/session/start")
def start_session(body: StartSessionBody) -> Dict[str, Any]:
    if get_raga_info(body.raga_name) is None:
        raise HTTPException(status_code=404, detail=f"Raga not found: {body.raga_name}")

    session = session_manager.create_session(
        raga_name=body.raga_name,
        bootstrap_seconds=body.bootstrap_seconds,
    )
    return {
        "session_id": session.session_id,
        "raga": session.raga_name,
        "stage": session.processor.stage,
    }


@app.post("/session/{session_id}/stop")
def stop_session(session_id: str) -> Dict[str, Any]:
    summary = session_manager.close_session(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary


@app.websocket("/ws/live/{session_id}")
async def live_ws(websocket: WebSocket, session_id: str):
    session = session_manager.get_session(session_id)
    if session is None:
        await websocket.close(code=1008, reason="Unknown session")
        return

    await websocket.accept()
    print(f"WebSocket connected for session {session_id}")

    try:
        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type")

            if msg_type == "stop":
                await websocket.send_json({"type": "session_status", "stage": "stopping"})
                break

            if msg_type != "audio_chunk":
                await websocket.send_json({"type": "error", "message": "Unsupported message type"})
                continue

            audio_b64 = payload.get("audio_b64")
            sample_rate = int(payload.get("sample_rate", 0))

            if not audio_b64 or sample_rate <= 0:
                print(f"Invalid audio params: b64={bool(audio_b64)}, sr={sample_rate}")
                await websocket.send_json({"type": "error", "message": "Missing audio_b64/sample_rate"})
                continue

            try:
                raw = base64.b64decode(audio_b64)
                audio = np.frombuffer(raw, dtype=np.float32)
                print(f"Received audio chunk: {len(audio)} samples @ {sample_rate} Hz")
            except Exception as e:
                print(f"Audio decode error: {e}")
                await websocket.send_json({"type": "error", "message": f"Invalid PCM payload: {str(e)}"})
                continue

            result = session.processor.process_audio_chunk(audio_chunk=audio, chunk_sr=sample_rate)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
        return
