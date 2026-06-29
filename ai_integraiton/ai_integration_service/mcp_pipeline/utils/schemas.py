"""
Pydantic Data Schemas - Standard data formats across all streams.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Stream 1: Resume ──

class ResumeAnalysis(BaseModel):
    full_name: str = ""
    contact_information: dict[str, str] = {}
    education: list[str] = []
    work_experience: list[str] = []
    technical_skills: list[str] = []
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_text: str = ""


# ── Stream 2: Speech Recognition ──

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str

class SpeechTranscript(BaseModel):
    full_text: str = ""
    detected_language: str = ""
    segments: list[TranscriptSegment] = []
    word_count: int = 0
    content_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ── Stream 3: Speech Emotion ──

class SpeechEmotionSegment(BaseModel):
    start_time: float
    end_time: float
    emotion: str            # "positive", "neutral", "negative"
    confidence: float = 0.0

class SpeechEmotionResult(BaseModel):
    segments: list[SpeechEmotionSegment] = []
    dominant_emotion: str = "neutral"
    emotion_distribution: dict[str, float] = {}
    positivity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    duration_seconds: float = 0.0


# ── Stream 4: Facial Emotion ──

class FacialEmotionRecord(BaseModel):
    frame: int
    time_sec: float
    time_min: float
    face_id: int
    emotion: str
    confidence: float
    raw_emotion: str
    raw_confidence: float

class FacialEmotionMinuteSummary(BaseModel):
    minute: int
    time_range: str
    dominant_emotion: str
    count: int
    total_detections: int
    percentage: float
    all_emotions: dict[str, int] = {}

class FacialEmotionResult(BaseModel):
    records: list[FacialEmotionRecord] = []
    minute_summary: list[FacialEmotionMinuteSummary] = []
    dominant_emotion: str = "neutral"
    emotion_distribution: dict[str, float] = {}
    positivity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    total_frames_analyzed: int = 0
    faces_detected: int = 0


# ── Fusion: Combined Result ──

class StreamScore(BaseModel):
    stream_name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0, le=1.0)
    details: dict = {}

class CandidateEvaluation(BaseModel):
    candidate_name: str = ""
    evaluation_date: str = Field(default_factory=lambda: datetime.now().isoformat())

    resume: Optional[ResumeAnalysis] = None
    transcript: Optional[SpeechTranscript] = None
    speech_emotion: Optional[SpeechEmotionResult] = None
    facial_emotion: Optional[FacialEmotionResult] = None

    stream_scores: list[StreamScore] = []
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)

    recommendation: str = ""
    strengths: list[str] = []
    concerns: list[str] = []
    summary: str = ""

    processing_time_seconds: float = 0.0
    pipeline_version: str = "1.0.0"
