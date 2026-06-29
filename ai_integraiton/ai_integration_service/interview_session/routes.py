"""FastAPI router for interview session endpoints.
Register in app.py:
    from interview_session.routes import router as interview_router
    app.include_router(interview_router)
"""
import os, json, logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .tts_service import TTSService
from .question_manager import QuestionManager
from .integrity_analyzer import IntegrityAnalyzer

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/interview", tags=["Interview Session"])

_tts = TTSService(
    model_dir=os.getenv("TTS_MODEL_DIR", "./models/tts"),
    cache_dir=os.getenv("TTS_CACHE_DIR", "./tts_cache"),
)
_qm = QuestionManager(_tts)
VIDEO_DIR = os.getenv("INTERVIEW_VIDEO_DIR", "./uploads/interviews")
os.makedirs(VIDEO_DIR, exist_ok=True)


# ═══════════════════════════════════════════════
#   REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════

class QuestionInput(BaseModel):
    id: Optional[str] = Field(None, description="Unique question ID (optional)")
    text: str = Field(..., description="Question text to be spoken by TTS")
    language: str = Field("tr", description="Language code: 'tr' or 'en'")
    order_index: int = Field(0, description="Display order (0-based)")
    max_duration: int = Field(120, description="Max answer duration in seconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "q1",
                    "text": "Kendinizi kısaca tanıtır mısınız?",
                    "language": "tr",
                    "order_index": 0,
                    "max_duration": 120
                }
            ]
        }
    }


class PrepareRequest(BaseModel):
    session_id: str = Field(..., description="Interview session UUID")
    questions: List[QuestionInput] = Field(..., description="List of interview questions")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "test-session-1",
                    "questions": [
                        {
                            "id": "q1",
                            "text": "Kendinizi kısaca tanıtır mısınız?",
                            "language": "tr",
                            "order_index": 0,
                            "max_duration": 120
                        },
                        {
                            "id": "q2",
                            "text": "Daha önceki iş deneyimlerinizden bahseder misiniz?",
                            "language": "tr",
                            "order_index": 1,
                            "max_duration": 120
                        },
                        {
                            "id": "q3",
                            "text": "Why are you interested in this position?",
                            "language": "en",
                            "order_index": 2,
                            "max_duration": 120
                        }
                    ]
                }
            ]
        }
    }


class PreparedQuestion(BaseModel):
    id: Optional[str]
    order_index: int
    text: str
    language: str
    max_duration: int
    audio_url: Optional[str] = Field(None, description="TTS audio file URL")
    audio_key: Optional[str] = Field(None, description="Cache key for the generated WAV")


class PrepareResponse(BaseModel):
    session_id: Optional[str]
    questions: List[PreparedQuestion]


class IntegrityRaw(BaseModel):
    tab_switches_count: int = 0
    tab_switches_total_seconds: float = 0.0
    focus_losses_count: int = 0
    copy_paste_attempts: int = 0
    face_absence_seconds: float = 0.0
    multi_face_frames: int = 0


class IntegrityResult(BaseModel):
    raw: IntegrityRaw
    score: str = Field(..., description="CLEAN | MINOR_CONCERNS | SIGNIFICANT_ANOMALIES")
    summary: str


class SubmitResponse(BaseModel):
    session_id: str
    video_path: Optional[str]
    integrity: IntegrityResult
    timestamps: List[Dict[str, Any]]
    status: str


# ═══════════════════════════════════════════════
#   ENDPOINTS
# ═══════════════════════════════════════════════

@router.post(
    "/prepare",
    response_model=PrepareResponse,
    summary="Pre-cache TTS audio for interview questions",
    description=(
        "Accepts a list of interview questions and synthesizes TTS audio for each "
        "using Piper (Turkish + English voices). Returns cache URLs that the frontend "
        "can fetch during the interview. Audio is cached by content hash, so identical "
        "questions across sessions are synthesized only once."
    ),
)
async def prepare(payload: PrepareRequest):
    questions = [q.model_dump() for q in payload.questions]
    if not questions:
        raise HTTPException(status_code=400, detail="no questions provided")
    prepared = _qm.prepare(questions)
    return {"session_id": payload.session_id, "questions": prepared}


@router.get(
    "/tts/{key}",
    summary="Serve cached TTS audio file",
    description="Returns the WAV file for a previously cached TTS synthesis. "
                "The key is obtained from /prepare response.",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "WAV audio file",
        },
        404: {"description": "Cache key not found"},
    },
)
async def tts_serve(key: str):
    p = _tts.cache_path(key)
    if not p.exists():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(str(p), media_type="audio/wav", filename=f"{key}.wav")


@router.post(
    "/submit",
    response_model=SubmitResponse,
    summary="Submit recorded interview video + metadata",
    description=(
        "Multipart upload containing the interview video (WebM) and metadata JSON. "
        "Metadata must include timestamps per question and integrity signals collected "
        "by the frontend (tab switches, focus losses, copy-paste attempts, etc.). "
        "Returns aggregated integrity analysis; actual MCP pipeline hand-off is TODO."
    ),
)
async def submit(
    session_id: str = Form(
        ...,
        description="Interview session UUID",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    ),
    metadata: str = Form(
        ...,
        description=(
            'JSON string with keys: timestamps (list), integrity_signals (dict), '
            'face_absence_seconds (float), multi_face_frames (int)'
        ),
        examples=[
            '{"timestamps":[{"q_index":0,"q_start":0.0,"a_start":5.2,"a_end":45.1}],'
            '"integrity_signals":{"tab_switches":[{"at":10.5,"duration":2.3}],'
            '"focus_losses":[],"copy_paste_attempts":0},'
            '"face_absence_seconds":3.5,"multi_face_frames":0}'
        ],
    ),
    video: Optional[UploadFile] = File(
        None, description="Interview video recording (WebM format)"
    ),
):
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid metadata JSON: {e}")

    video_path = None
    if video is not None:
        video_path = os.path.join(VIDEO_DIR, f"{session_id}.webm")
        with open(video_path, "wb") as f:
            f.write(await video.read())
        log.info(f"Video saved: {video_path}")
        
        from models.minio_client import MinioClientWrapper
        minio_client = MinioClientWrapper()
        minio_url = minio_client.upload_file(video_path, f"interviews/{session_id}.webm")
        if minio_url:
            video_path = minio_url

    integrity = IntegrityAnalyzer.analyze(
        meta.get("integrity_signals", {}),
        face_absence_seconds=float(meta.get("face_absence_seconds", 0.0)),
        multi_face_frames=int(meta.get("multi_face_frames", 0)),
    )

    # Persist the integrity result so the analyze pipeline can look it up by session_id
    # when the Spring Boot analyze request arrives (which carries the same session_id as interview_id).
    try:
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from models.result_store import ResultStore
        _store = ResultStore()
        existing = _store.get(session_id) or {}
        existing.update({
            "session_id": session_id,
            "integrity_result": integrity,
            "timestamps": meta.get("timestamps", []),
            "video_path": video_path,
        })
        _store.save(session_id, existing)
        log.info(f"[submit] Integrity result stored for session={session_id}, score={integrity.get('score')}")
    except Exception as e:
        log.warning(f"[submit] Could not persist integrity result: {e}")

    return JSONResponse({
        "session_id": session_id,
        "video_path": video_path,
        "integrity": integrity,
        "timestamps": meta.get("timestamps", []),
        "status": "received",
    })