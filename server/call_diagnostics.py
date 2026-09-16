"""Bounded, content-free call diagnostics. Never accepts audio, text or feature arrays."""
from datetime import datetime, timezone
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Request, HTTPException
from .service_access import origin_allowed
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PATH = Path(__file__).resolve().parents[1] / '.runtime' / 'call-events.jsonl'
logger = logging.getLogger('eric.call_events')
logger.setLevel(logging.INFO)
logger.propagate = False
router = APIRouter()


def configure():
    if logger.handlers:
        return
    PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(PATH, maxBytes=2_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    if os.environ.get('MOSHI_BACKEND') == 'torch' or os.environ.get('VOICE_REQUIRE_AUTH')=='1':
        # Modal retains container output after scale-down; local disk does not.
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream)


def emit(event, **fields):
    configure()
    logger.info(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}, allow_nan=False))


class ClientDiagnostic(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    event: Literal['client_sample','client_end','socket_close','audio_state','microphone_mute','microphone_unmute','page_visibility']
    event_id: str = Field(pattern=r'^[a-f0-9-]{36}$')
    client_id: str = Field(pattern=r'^[a-f0-9-]{36}$')
    call_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{8}$')
    reason: Literal['user_stop','playback_backlog','transport_backlog','page_closed','microphone_ended','connection_error','session_limit','server_error','audio_unavailable'] | None = None
    elapsed_ms: float = Field(ge=0,le=1e12)
    input_frames: int = Field(ge=0,le=1e9)
    output_frames: int = Field(ge=0,le=1e9)
    last_server_frame: int = Field(ge=0,le=1e9)
    frame_gap_ms: float | None = Field(default=None,ge=0,le=1e12)
    capture_gap_ms: float | None = Field(default=None,ge=0,le=1e12)
    playback_buffer_s: float = Field(ge=0,le=1e6)
    socket_buffer_bytes: int = Field(ge=0,le=1e12)
    socket_state: int = Field(ge=0,le=3)
    audio_state: Literal['running','suspended','interrupted','closed','unknown']
    microphone_state: Literal['live','ended','unknown']
    microphone_muted: bool
    hidden: bool
    close_code: int | None = Field(default=None,ge=0,le=65535)
    close_clean: bool | None = None


@router.post('/api/call-diagnostics')
async def receive(request: Request):
    if not origin_allowed(request.headers):
        raise HTTPException(403)
    body = await request.body()
    if len(body)>4096:
        raise HTTPException(413)
    try:
        record=ClientDiagnostic.model_validate_json(body)
    except ValidationError:
        raise HTTPException(422, 'Invalid diagnostic fields')
    data=record.model_dump(exclude_none=True)
    emit(data.pop('event'), **data)
    return {'saved':True}
