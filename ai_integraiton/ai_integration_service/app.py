"""
AI Integration Service
======================
REST API bridge between the Spring Boot backend and the MCP AI Pipeline.

Architecture:
    Spring Boot Backend  <--REST/Kafka-->  AI Integration Service  <--MCP-->  AI Pipeline
                                                    |
                                                    ├── /analyze           (trigger full pipeline)
                                                    ├── /results/{id}      (aggregated scores)
                                                    ├── /results/{id}/xai  (SHAP/LIME explanations)
                                                    └── /results/{id}/feedback (NLG candidate report)

This service:
  1. Receives interview completion events (REST or Kafka)
  2. Orchestrates the MCP pipeline (CV + Video → 4 AI streams)
  3. Stores results in MongoDB-compatible JSON
  4. Generates XAI explanations (feature contribution analysis)
  5. Produces NLG feedback reports for both HR and Candidates
"""

import os
import sys
import json
import uuid
import time
import logging
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mcp_pipeline"))

from xai.explainer import XAIExplainer
from reports.nlg_generator import NLGReportGenerator
from models.result_store import ResultStore
from models.minio_client import MinioClientWrapper

# ══════════════════════════════════════════════
#   LOGGING
# ══════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai_integration_service")

# ══════════════════════════════════════════════
#   FASTAPI APPLICATION
# ══════════════════════════════════════════════

app = FastAPI(
    title="AI Integration Service",
    description="Bridge between Spring Boot HR Backend and MCP AI Pipeline with XAI support",
    version="1.0.0",
    docs_url="/api/v1/ai/docs",
    openapi_url="/api/v1/ai/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════
#   GLOBAL INSTANCES
# ══════════════════════════════════════════════

result_store = ResultStore()
xai_explainer = XAIExplainer()
nlg_generator = NLGReportGenerator()

# ══════════════════════════════════════════════
#   REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════

class QuestionItem(BaseModel):
    """A single interview question, used for VIDEO interviews where questions are preset."""
    id: Optional[str] = None
    order_index: int = 0
    text: str
    language: Optional[str] = "tr"
    max_duration: Optional[int] = None


class AnalyzeRequest(BaseModel):
    """Request payload from Spring Boot when interview is COMPLETED."""
    application_id: str = Field(..., description="UUID of the Application entity")
    candidate_id: str = Field(..., description="UUID of the Candidate entity")
    candidate_name: str = Field(default="", description="Candidate full name")
    video_url: str = Field(..., description="S3/MinIO URL or local path of the interview video")
    resume_url: Optional[str] = Field(None, description="S3/MinIO URL or local path of the resume PDF")
    resume_parsed_data: Optional[dict] = Field(None, description="Pre-parsed CV data cached at upload time; skips parse_resume() in the pipeline")
    job_title: str = Field(default="", description="Job position title")
    job_requirements: str = Field(default="", description="Job requirements text for CV scoring")
    integrity_metadata: Optional[dict] = Field(None, description="Integrity signals collected during the interview")
    session_type: Optional[str] = Field(None, description="e.g. LIVE_WEBRTC for live recording vs async one-way")

    interview_type: Optional[str] = Field(
        None,
        description="VIDEO (preset questions + timestamps) or LIVE (HR speaks questions in real time)",
    )
    questions: Optional[List[QuestionItem]] = Field(
        None,
        description="Preset interview questions (VIDEO only). Empty/None for LIVE.",
    )
    interview_session_id: Optional[str] = Field(
        None,
        description="Interview UUID used to look up timestamps stored by /interview/submit (VIDEO only)",
    )
    previous_video_report: Optional[str] = Field(
        None,
        description="HR report text from the prior AI_ONE_WAY (video) analysis. "
                    "When present for LIVE interviews, CV analysis is skipped and the "
                    "NLG report is written comparatively. When absent, full 4-stream runs.",
    )

class AnalyzeResponse(BaseModel):
    """Immediate response when analysis is triggered."""
    analysis_id: str
    application_id: str
    status: str = "PROCESSING"
    message: str = "AI analysis pipeline started"

class AnalysisResult(BaseModel):
    """Full analysis result stored in MongoDB."""
    id: str
    application_id: str
    candidate_id: str
    candidate_name: str
    status: str  # PROCESSING, COMPLETED, FAILED
    created_at: str
    completed_at: Optional[str] = None
    processing_time_seconds: Optional[float] = None

    # Stream results
    cv_analysis_result: Optional[dict] = None
    interview_transcript: Optional[str] = None
    speech_emotion_result: Optional[dict] = None
    facial_emotion_result: Optional[dict] = None

    # Aggregated scores
    final_score: Optional[float] = None
    recommendation: Optional[str] = None
    stream_scores: Optional[list] = None

    # XAI data
    xai_explanations: Optional[dict] = None

    # NLG reports
    hr_report: Optional[str] = None
    candidate_feedback: Optional[str] = None

    # Per-question Q&A analysis
    per_question_analysis: Optional[list] = None
    dialogue_transcript: Optional[str] = None
    interview_type: Optional[str] = None
    question_pairing_method: Optional[str] = None

class DecisionRequest(BaseModel):
    """HR decision payload."""
    decision: str = Field(..., description="ACCEPTED or REJECTED")
    hr_notes: Optional[str] = None

class FeedbackResponse(BaseModel):
    """Candidate-facing XAI feedback."""
    application_id: str
    candidate_name: str
    decision: Optional[str] = None
    feedback_report: str
    improvement_suggestions: list[str] = []
    strength_areas: list[str] = []

# ══════════════════════════════════════════════
#   BACKGROUND PIPELINE EXECUTION
# ══════════════════════════════════════════════

def _download_file(url: str, dest_path: str) -> str:
    """Download a file from an HTTP/HTTPS URL to dest_path. Returns dest_path."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading {url} → {dest_path}")

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} when downloading file from URL")
        data = response.read()

    with open(dest_path, "wb") as f:
        f.write(data)

    size = os.path.getsize(dest_path)
    if size < 1024:
        raise RuntimeError(
            f"Downloaded file is only {size} bytes — likely an error response from storage (e.g. expired presigned URL)"
        )

    logger.info(f"Downloaded {size:,} bytes → {dest_path}")
    return dest_path


def _resolve_path(url_or_path: str, dest_dir: str, filename: str) -> str:
    """Return a local file path: download if HTTP URL, otherwise use as-is."""
    if url_or_path and (url_or_path.startswith("http://") or url_or_path.startswith("https://")):
        return _download_file(url_or_path, os.path.join(dest_dir, filename))
    return url_or_path


def _run_pipeline_background(analysis_id: str, request: AnalyzeRequest):
    """
    Background task that runs the full MCP pipeline.
    Called asynchronously after the /analyze endpoint returns.
    """
    temp_dir = f"./pipeline_output/{analysis_id}/downloads"
    try:
        logger.info(
            f"[{analysis_id}] Starting AI pipeline for application={request.application_id} "
            f"session_type={request.session_type!r} interview_type={request.interview_type!r}"
        )

        start_time = time.time()

        # Resolve HTTP URLs to local file paths (MinIO presigned URLs, etc.)
        video_path = _resolve_path(request.video_url, temp_dir, "interview_video.webm")
        resume_path = _resolve_path(request.resume_url, temp_dir, "resume.pdf") if request.resume_url else None

        # Import pipeline (lazy to avoid startup overhead)
        from mcp_pipeline.agent.orchestrator import run_pipeline_direct, load_config

        config = load_config(str(PROJECT_ROOT / "mcp_pipeline" / "config.yaml"))
        config.setdefault("general", {})["output_dir"] = f"./pipeline_output/{analysis_id}"

        # Resolve interview type. Map legacy session_type values when interview_type missing.
        interview_type = (request.interview_type or "").upper()
        if not interview_type:
            interview_type = "LIVE" if (request.session_type or "").upper() == "LIVE_WEBRTC" else "VIDEO"

        # For VIDEO, look up timestamps stored by /interview/submit (keyed by session_id == interview_id)
        timestamps = None
        if interview_type == "VIDEO" and request.interview_session_id:
            session_data = result_store.get(request.interview_session_id)
            if session_data:
                timestamps = session_data.get("timestamps") or []
                logger.info(
                    f"[{analysis_id}] Loaded {len(timestamps)} question timestamps for "
                    f"session={request.interview_session_id}"
                )

        # Fallback: timestamps travel via integrity_metadata when frontend doesn't call /interview/submit directly
        if not timestamps and request.integrity_metadata:
            fallback_ts = request.integrity_metadata.get("timestamps") or []
            if fallback_ts:
                timestamps = fallback_ts
                logger.info(
                    f"[{analysis_id}] Using {len(timestamps)} timestamps from integrity_metadata (fallback)"
                )

        questions_payload = None
        if request.questions:
            questions_payload = [q.model_dump() for q in request.questions]

        # For LIVE interviews with a prior video report, CV was already assessed —
        # skip Stream 1 by clearing resume paths so the orchestrator omits it.
        has_prior_video = bool(request.previous_video_report)
        if interview_type == "LIVE" and has_prior_video:
            resume_path = None

        # Run the 4-stream pipeline (skip parse_resume if cached data is available)
        evaluation = run_pipeline_direct(
            video_path=video_path,
            resume_path=resume_path,
            job_requirements=request.job_requirements,
            config=config,
            resume_parsed_data=None if (interview_type == "LIVE" and has_prior_video) else request.resume_parsed_data,
            interview_type=interview_type,
            questions=questions_payload,
            timestamps=timestamps,
            has_prior_video=has_prior_video,
        )

        processing_time = round(time.time() - start_time, 2)

        # Load raw stream results
        raw_path = f"./pipeline_output/{analysis_id}/raw_stream_results.json"
        raw_results = {}
        if os.path.exists(raw_path):
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_results = json.load(f)
            
            # Upload raw file to MinIO for backup/S3 migration readiness
            minio_client = MinioClientWrapper()
            minio_client.upload_file(raw_path, f"pipeline_output/{analysis_id}/raw_stream_results.json")

        # Generate XAI explanations
        logger.info(f"[{analysis_id}] Generating XAI explanations...")
        xai_data = xai_explainer.explain(
            stream_scores=[s.model_dump() for s in evaluation.stream_scores],
            raw_results=raw_results,
            final_score=evaluation.final_score,
            recommendation=evaluation.recommendation
        )

        qa_data = raw_results.get("qa", {}) or {}

        # Determine interview language from HR-configured question settings (majority vote)
        interview_language = "tr"
        if request.questions:
            from collections import Counter
            lang_counter = Counter((q.language or "tr").lower() for q in request.questions)
            interview_language = lang_counter.most_common(1)[0][0]
        logger.info(f"[{analysis_id}] Interview language resolved: {interview_language}")

        # Build CV grounding summary to prevent NLG hallucination.
        # parse_resume() may use different field names; try all known variants.
        resume_data = raw_results.get("resume") or {}

        def _list_field(*keys):
            """Return the first non-empty list found under any of the given keys."""
            for k in keys:
                v = resume_data.get(k)
                if v:
                    return v if isinstance(v, list) else [v]
            return []

        def _item_str(e, title_key="title", company_key="company"):
            if isinstance(e, str):
                return e
            if isinstance(e, dict):
                t = e.get(title_key) or e.get("role") or e.get("position") or "?"
                c = e.get(company_key) or e.get("organization") or e.get("institution") or ""
                return f"{t} at {c}" if c else t
            return str(e)

        cv_summary_parts = []

        skills = _list_field("skills", "technical_skills")
        if skills:
            skill_strs = [
                s if isinstance(s, str) else s.get("name", str(s))
                for s in skills[:15]
            ]
            cv_summary_parts.append("Skills: " + ", ".join(skill_strs))

        experiences = _list_field("experiences", "work_experiences", "work_experience")
        if experiences:
            exp_strs = [_item_str(e) for e in experiences[:4]]
            cv_summary_parts.append("Work Experience: " + "; ".join(exp_strs))

        education = _list_field("education", "educations")
        if education:
            edu_strs = [_item_str(e, title_key="degree") for e in education[:2]]
            cv_summary_parts.append("Education: " + "; ".join(edu_strs))

        # Job-relevance strengths/gaps assessed by score_resume() — tied to job description
        if resume_data.get("strengths"):
            cv_summary_parts.append(
                "CV strengths vs. job: " + "; ".join(str(s) for s in resume_data["strengths"][:4])
            )
        if resume_data.get("gaps"):
            cv_summary_parts.append(
                "CV gaps vs. job: " + "; ".join(str(g) for g in resume_data["gaps"][:3])
            )

        # Raw CV text as fallback context (first 1000 chars)
        raw_cv = (resume_data.get("raw_text") or "")[:1000]
        if raw_cv and not cv_summary_parts:
            cv_summary_parts.append("CV excerpt:\n" + raw_cv)

        cv_summary = "\n".join(cv_summary_parts) or None

        full_transcript = raw_results.get("transcript", {}).get("full_text", "") or ""
        transcript_excerpt = full_transcript[:2000] or None

        # Build emotion analysis summary for NLG context
        speech_emo = raw_results.get("speech_emotion") or {}
        facial_emo = raw_results.get("facial_emotion") or {}
        emotion_parts = []
        if speech_emo.get("dominant_emotion"):
            emotion_parts.append(
                f"Speech emotion: dominant={speech_emo['dominant_emotion']}, "
                f"positivity={speech_emo.get('positivity_score', 0):.2f}"
            )
        if facial_emo.get("dominant_emotion"):
            emotion_parts.append(
                f"Facial emotion: dominant={facial_emo['dominant_emotion']}, "
                f"positivity={facial_emo.get('positivity_score', 0):.2f}"
            )
        emotion_summary = "\n".join(emotion_parts) or None

        # Generate NLG reports
        logger.info(f"[{analysis_id}] Generating LLM-authored NLG reports...")
        # Integrity is computed below — generate it first so it can be passed to HR report.
        from interview_session.integrity_analyzer import IntegrityAnalyzer
        fer_result = raw_results.get("facial_emotion") or {}
        meta = request.integrity_metadata or {}
        face_absence_seconds = float(
            fer_result.get("face_absence_seconds", meta.get("face_absence_seconds", 0.0))
        )
        multi_face_frames = int(
            fer_result.get("multi_face_frames", meta.get("multi_face_frames", 0))
        )
        integrity_result = IntegrityAnalyzer.analyze(
            frontend_signals=meta.get("integrity_signals", {}),
            face_absence_seconds=face_absence_seconds,
            multi_face_frames=multi_face_frames,
        )

        hr_report = nlg_generator.generate_hr_report_llm(
            candidate_name=request.candidate_name or evaluation.candidate_name,
            job_title=request.job_title,
            evaluation=evaluation.model_dump(),
            xai_data=xai_data,
            qa_data=qa_data,
            integrity=integrity_result,
            interview_language=interview_language,
            cv_summary=cv_summary,
            transcript_excerpt=transcript_excerpt,
            emotion_summary=emotion_summary,
            job_requirements=request.job_requirements,
            interview_type=interview_type,
            previous_video_report=request.previous_video_report,
        )

        candidate_feedback = nlg_generator.generate_candidate_feedback_llm(
            candidate_name=request.candidate_name or evaluation.candidate_name,
            job_title=request.job_title,
            evaluation=evaluation.model_dump(),
            xai_data=xai_data,
            qa_data=qa_data,
            interview_language=interview_language,
            cv_summary=cv_summary,
            transcript_excerpt=transcript_excerpt,
            emotion_summary=emotion_summary,
            job_requirements=request.job_requirements,
            interview_type=interview_type,
            previous_video_report=request.previous_video_report,
        )

        # Store complete result
        result = {
            "id": analysis_id,
            "application_id": request.application_id,
            "candidate_id": request.candidate_id,
            "candidate_name": request.candidate_name or evaluation.candidate_name,
            "status": "COMPLETED",
            "created_at": result_store.get(analysis_id, {}).get("created_at", datetime.now().isoformat()),
            "completed_at": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "cv_analysis_result": raw_results.get("resume"),
            "interview_transcript": raw_results.get("transcript", {}).get("full_text"),
            "speech_emotion_result": raw_results.get("speech_emotion"),
            "facial_emotion_result": raw_results.get("facial_emotion"),
            "final_score": evaluation.final_score,
            "recommendation": evaluation.recommendation,
            "stream_scores": [s.model_dump() for s in evaluation.stream_scores],
            "xai_explanations": xai_data,
            "hr_report": hr_report,
            "candidate_feedback": candidate_feedback,
            "integrity_result": integrity_result,
            "per_question_analysis": qa_data.get("per_question_analysis", []),
            "dialogue_transcript": qa_data.get("dialogue_transcript", ""),
            "interview_type": interview_type,
            "question_pairing_method": qa_data.get("question_pairing_method"),
        }

        result_store.save(analysis_id, result)
        logger.info(f"[{analysis_id}] Pipeline completed. Score={evaluation.final_score:.2f}, "
                     f"Recommendation={evaluation.recommendation}, Time={processing_time:.1f}s")

    except Exception as e:
        logger.error(f"[{analysis_id}] Pipeline failed: {e}", exc_info=True)
        result_store.update_status(analysis_id, "FAILED", error=str(e))


# ══════════════════════════════════════════════
#   API ENDPOINTS
# ══════════════════════════════════════════════

class ParseResumeRequest(BaseModel):
    resume_url: str = Field(..., description="Presigned URL or local path to a resume PDF")

@app.post("/api/v1/ai/resume/parse")
async def parse_resume_endpoint(request: ParseResumeRequest):
    """
    Parse a resume PDF and return structured data.
    Called once at CV upload time so the result can be cached and reused.
    """
    from mcp_pipeline.mcp_servers.resume_server import parse_resume
    from mcp_pipeline.agent.orchestrator import load_config

    config = load_config(str(PROJECT_ROOT / "mcp_pipeline" / "config.yaml"))
    ollama_cfg = config.get("ollama", {})

    with tempfile.TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, "resume.pdf")
        urllib.request.urlretrieve(request.resume_url, local_path)
        parsed = parse_resume(
            pdf_path=local_path,
            gemini_api_key=config.get("resume", {}).get("gemini_api_key", ""),
            model_name=ollama_cfg.get("model", "gemma4:e4b"),
        )

    logger.info(f"Resume parsed via standalone endpoint: full_name={parsed.get('full_name')}")
    return parsed


@app.post("/api/v1/ai/analyze", response_model=AnalyzeResponse)
async def trigger_analysis(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Trigger AI analysis pipeline for a completed interview.
    Called by Spring Boot when Interview status changes to COMPLETED.

    This endpoint returns immediately; the pipeline runs in background.
    Poll /results/{id} or wait for Kafka callback.
    """
    analysis_id = str(uuid.uuid4())

    # Create initial record
    result_store.save(analysis_id, {
        "id": analysis_id,
        "application_id": request.application_id,
        "candidate_id": request.candidate_id,
        "candidate_name": request.candidate_name,
        "status": "PROCESSING",
        "created_at": datetime.now().isoformat(),
    })

    # Start pipeline in background
    background_tasks.add_task(_run_pipeline_background, analysis_id, request)

    return AnalyzeResponse(
        analysis_id=analysis_id,
        application_id=request.application_id,
        status="PROCESSING",
        message="AI analysis pipeline started. Poll /results/{analysis_id} for status."
    )


@app.get("/api/v1/ai/results/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """
    Get aggregated analysis results for an application.
    Maps to: GET /api/v1/results/{candidateId}
    """
    result = result_store.get(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return result


@app.get("/api/v1/ai/results/{analysis_id}/xai")
async def get_xai_explanations(analysis_id: str):
    """
    Get XAI explanations (feature contributions, SHAP-like analysis).
    Maps to: GET /api/v1/results/{candidateId}/xai
    """
    result = result_store.get(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    if result.get("status") != "COMPLETED":
        raise HTTPException(status_code=202, detail="Analysis still processing")

    return {
        "analysis_id": analysis_id,
        "xai_explanations": result.get("xai_explanations", {}),
        "stream_scores": result.get("stream_scores", []),
        "final_score": result.get("final_score"),
        "recommendation": result.get("recommendation"),
    }


@app.get("/api/v1/ai/results/{analysis_id}/feedback")
async def get_candidate_feedback(analysis_id: str):
    """
    Get NLG-generated candidate feedback report.
    Maps to: GET /api/v1/results/{candidateId}/feedback
    Shown to candidate AFTER HR decision.
    """
    result = result_store.get(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    if result.get("status") != "COMPLETED":
        raise HTTPException(status_code=202, detail="Analysis still processing")

    return FeedbackResponse(
        application_id=result.get("application_id", ""),
        candidate_name=result.get("candidate_name", ""),
        decision=result.get("decision"),
        feedback_report=result.get("candidate_feedback", "Feedback not yet generated."),
        improvement_suggestions=result.get("xai_explanations", {}).get("improvement_suggestions", []),
        strength_areas=result.get("xai_explanations", {}).get("strength_areas", []),
    )


@app.get("/api/v1/ai/results/{analysis_id}/hr-report")
async def get_hr_report(analysis_id: str):
    """
    Get detailed HR analysis report with XAI-backed insights.
    Maps to: GET /api/v1/hr/applications/{applicationId}/analysis
    """
    result = result_store.get(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")

    return {
        "analysis_id": analysis_id,
        "candidate_name": result.get("candidate_name"),
        "status": result.get("status"),
        "final_score": result.get("final_score"),
        "recommendation": result.get("recommendation"),
        "stream_scores": result.get("stream_scores"),
        "hr_report": result.get("hr_report"),
        "xai_explanations": result.get("xai_explanations"),
        "cv_analysis_result": result.get("cv_analysis_result"),
        "processing_time_seconds": result.get("processing_time_seconds"),
    }


@app.post("/api/v1/ai/results/{analysis_id}/decision")
async def record_hr_decision(analysis_id: str, decision: DecisionRequest):
    """
    Record HR decision (ACCEPTED/REJECTED).
    If REJECTED, regenerate candidate feedback with improvement focus.
    Maps to: PATCH /api/v1/hr/applications/{id}/decision
    """
    result = result_store.get(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")

    result["decision"] = decision.decision
    result["hr_notes"] = decision.hr_notes
    result["decision_at"] = datetime.now().isoformat()

    # If REJECTED → regenerate candidate feedback with improvement suggestions
    if decision.decision == "REJECTED":
        logger.info(f"[{analysis_id}] Generating rejection feedback with XAI insights...")
        result["candidate_feedback"] = nlg_generator.generate_rejection_feedback(
            candidate_name=result.get("candidate_name", ""),
            job_title="",
            evaluation=result,
            xai_data=result.get("xai_explanations", {})
        )

    result_store.save(analysis_id, result)

    return {"status": "ok", "decision": decision.decision, "analysis_id": analysis_id}


class PdfReportRequest(BaseModel):
    type: str = Field(default="hr", description='"hr" or "candidate"')
    candidate_name: str = Field(default="")
    job_title: str = Field(default="")
    refresh: bool = Field(default=False, description="Force regeneration even if cached")


@app.post("/api/v1/ai/results/{analysis_id}/report/pdf")
async def generate_report_pdf(analysis_id: str, request: PdfReportRequest):
    """
    Generate (or return cached) a PDF report for the given analysis.
    Uploads to MinIO under reports/{analysis_id}/{type}-report.pdf and
    returns a presigned URL (7-day expiry).
    """
    from reports.pdf_generator import generate_hr_pdf, generate_candidate_pdf

    report_type = (request.type or "hr").lower().strip()
    if report_type not in ("hr", "candidate"):
        raise HTTPException(status_code=400, detail='type must be "hr" or "candidate"')

    object_name = f"reports/{analysis_id}/{report_type}-report.pdf"
    minio = MinioClientWrapper()

    # Return cached PDF if it already exists (unless refresh is requested)
    if not request.refresh and minio.object_exists(object_name):
        url = minio.get_presigned_url(object_name)
        logger.info(f"[{analysis_id}] Returning cached {report_type} PDF: {object_name}")
        return {"pdf_url": url, "cached": True}

    result = result_store.get(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    if result.get("status") != "COMPLETED":
        raise HTTPException(status_code=202, detail="Analysis still processing")

    candidate_info = {
        "name": request.candidate_name or result.get("candidate_name", ""),
        "job_title": request.job_title,
    }

    try:
        if report_type == "hr":
            pdf_bytes = generate_hr_pdf(result, candidate_info)
        else:
            pdf_bytes = generate_candidate_pdf(result, candidate_info)
    except Exception as e:
        logger.error(f"[{analysis_id}] PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    minio.upload_bytes(pdf_bytes, object_name, "application/pdf")
    url = minio.get_presigned_url(object_name)
    logger.info(f"[{analysis_id}] Generated and uploaded {report_type} PDF ({len(pdf_bytes):,} bytes)")
    return {"pdf_url": url, "cached": False}


@app.get("/api/v1/ai/health")
async def health_check():
    """Health check endpoint for service discovery."""
    return {
        "status": "healthy",
        "service": "ai-integration-service",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

from interview_session.routes import router as interview_router
app.include_router(interview_router)

# ══════════════════════════════════════════════
#   STARTUP
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
