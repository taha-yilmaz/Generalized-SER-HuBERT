"""
Q&A Pipeline Orchestrator
=========================
Branches between VIDEO (preset questions + timestamps) and LIVE (LLM-discovered
Q&A from a stereo dialogue) and produces:
  - per_question_analysis: list of per-Q LLM evaluations
  - dialogue_transcript:   "HR: ... / Candidate: ..." formatted text
  - dialogue_turns:        structured list of turns
  - question_pairing_method: "preset_timestamps" | "llm_dialogue" | "llm_mono"
  - avg_quality_score:     mean quality score across questions (for fusion)
"""

from __future__ import annotations

import logging
import time
from typing import List, Dict, Any, Optional, Tuple

log = logging.getLogger(__name__)


def run_qa_analysis(
    interview_type: str,
    transcript_segments: List[Dict[str, Any]],
    questions: Optional[List[Dict[str, Any]]],
    timestamps: Optional[List[Dict[str, Any]]],
    job_context: str = "",
    candidate_resume_summary: Optional[str] = None,
    dual_channel_transcripts: Optional[Tuple[List[Dict], List[Dict]]] = None,
) -> Dict[str, Any]:
    """Run the per-question analysis pipeline.

    Args:
        interview_type: "VIDEO" or "LIVE" (LIVE_MONO falls under LIVE).
        transcript_segments: mono Whisper segments (used for VIDEO and LIVE_MONO).
        questions: preset question list for VIDEO; ignored for LIVE.
        timestamps: q_index/q_start/a_start/a_end array for VIDEO; ignored for LIVE.
        job_context: job requirements text (forwarded to per-Q LLM call).
        candidate_resume_summary: optional resume summary for richer LLM context.
        dual_channel_transcripts: (hr_segments, candidate_segments) for stereo LIVE.

    Returns:
        dict with per_question_analysis, dialogue_transcript, dialogue_turns,
        question_pairing_method, avg_quality_score.
    """
    from mcp_servers import qa_analysis_server as qa

    pipeline_start = time.time()
    interview_type = (interview_type or "VIDEO").upper()

    # ──────────────────────────────────────────────────────────
    #   Step 1: derive Q&A pairs and dialogue turns
    # ──────────────────────────────────────────────────────────
    pairs: List[Dict[str, Any]] = []
    dialogue_turns: List[Dict[str, Any]] = []
    pairing_method = "preset_timestamps"

    if interview_type == "VIDEO":
        pairs = qa.pair_qa_from_timestamps(
            transcript_segments or [],
            questions or [],
            timestamps or [],
        )
        # Synthesize dialogue turns from the preset Q + answer text
        for p in pairs:
            if p.get("question_text"):
                dialogue_turns.append({
                    "speaker": "HR",
                    "start": p.get("q_start", 0),
                    "end": p.get("a_start", 0),
                    "text": p["question_text"],
                })
            if p.get("answer_text"):
                dialogue_turns.append({
                    "speaker": "Candidate",
                    "start": p.get("a_start", 0),
                    "end": p.get("a_end", 0),
                    "text": p["answer_text"],
                })
        pairing_method = "preset_timestamps"

    else:  # LIVE
        if dual_channel_transcripts:
            hr_segs, cand_segs = dual_channel_transcripts
            pairs, dialogue_turns = qa.pair_qa_from_dialogue(
                hr_segs or [], cand_segs or [], job_context=job_context,
            )
            pairing_method = "llm_dialogue"
        else:
            # LIVE_MONO fallback — content-based diarization
            log.info("[qa] LIVE without stereo channels — running content-based diarization")
            dialogue_turns = qa.diarize_mono_dialogue(
                transcript_segments or [], job_context=job_context,
            )
            hr_turns = [t for t in dialogue_turns if t["speaker"] == "HR"]
            cand_turns = [t for t in dialogue_turns if t["speaker"] == "Candidate"]
            pairs, _ = qa.pair_qa_from_dialogue(
                [{"start": t["start"], "end": t["end"], "text": t["text"]} for t in hr_turns],
                [{"start": t["start"], "end": t["end"], "text": t["text"]} for t in cand_turns],
                job_context=job_context,
            )
            pairing_method = "llm_mono"

    # ──────────────────────────────────────────────────────────
    #   Step 2: per-question LLM analysis
    # ──────────────────────────────────────────────────────────
    per_question_analysis: List[Dict[str, Any]] = []
    quality_scores: List[float] = []

    for i, pair in enumerate(pairs):
        t0 = time.time()
        analysis = qa.analyze_question(
            question_text=pair.get("question_text", ""),
            answer_text=pair.get("answer_text", ""),
            job_context=job_context,
            candidate_resume_summary=candidate_resume_summary,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        log.info(f"[qa] question {i+1}/{len(pairs)} analyzed in {elapsed_ms}ms")

        entry = dict(pair)
        entry.update(analysis)
        per_question_analysis.append(entry)
        quality_scores.append(analysis.get("quality_score", 0.0))

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    dialogue_transcript = qa.render_dialogue_transcript(dialogue_turns)

    log.info(
        f"[qa] pipeline done in {time.time()-pipeline_start:.1f}s, "
        f"{len(per_question_analysis)} pairs, avg_quality={avg_quality:.2f}, "
        f"pairing={pairing_method}"
    )

    return {
        "per_question_analysis": per_question_analysis,
        "dialogue_transcript": dialogue_transcript,
        "dialogue_turns": dialogue_turns,
        "question_pairing_method": pairing_method,
        "avg_quality_score": avg_quality,
    }
