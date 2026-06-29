"""
Q&A Analysis Server
===================
Per-question interview analysis using gemma4:e4b (Ollama).

Two modes:
  - VIDEO: deterministic Q-to-segment pairing using the timestamps captured by
    /api/v1/interview/submit (q_index, q_start, a_start, a_end).
  - LIVE:  HR speaks questions in real time, no preset list. Stereo recording
    has HR on the left channel and Candidate on the right. Each channel is
    transcribed separately (speaker label = channel) and gemma4:e4b is asked
    to identify the substantive HR-question / Candidate-answer pairs.

For every Q&A pair, gemma4:e4b produces a per-question analysis:
  {relevance_score, quality_score, commentary, strengths[], weaknesses[]}
"""

from __future__ import annotations

import logging
import sys
import os
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm.ollama_client import generate_json, generate

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
#   VIDEO PATH — preset questions + submit timestamps
# ════════════════════════════════════════════════════════════

def pair_qa_from_timestamps(
    transcript_segments: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    timestamps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Slice Whisper segments per (q_start..a_end) window and pair each with the
    matching preset question.

    Args:
        transcript_segments: [{start, end, text}, ...]
        questions: [{id, order_index, text, ...}, ...]
        timestamps: [{q_index, q_start, a_start, a_end}, ...]

    Returns:
        [{question_id, order_index, question_text, answer_text,
          q_start, a_start, a_end}, ...]
    """
    # Sort by order_index so array position (0-based) matches the frontend's q_index (currentIdx)
    sorted_qs = sorted(questions or [], key=lambda q: int(q.get("order_index", 0)))
    by_idx = {i: q for i, q in enumerate(sorted_qs)}
    pairs: List[Dict[str, Any]] = []

    for ts in timestamps or []:
        try:
            q_idx = int(ts.get("q_index"))
        except (TypeError, ValueError):
            continue

        question = by_idx.get(q_idx)
        if not question:
            log.warning(f"[qa] No question text for q_index={q_idx}, skipping")
            continue

        a_start = float(ts.get("a_start", 0))
        a_end = float(ts.get("a_end", a_start))
        q_start = float(ts.get("q_start", a_start))

        # Collect segments that overlap with the answer window [a_start, a_end).
        # Overlap-based (not start-only) so a long Whisper segment whose start
        # falls before a_start but whose end reaches into this window is still
        # captured — prevents missing words when the previous fallback path was
        # skipped because a later segment was already found for this question.
        answer_parts: List[str] = []
        for seg in transcript_segments or []:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", seg_start))
            if seg_start >= a_end or seg_end <= a_start:
                continue
            text = (seg.get("text") or "").strip()
            if text:
                answer_parts.append(text)

        answer_text = " ".join(answer_parts).strip()

        pairs.append({
            "question_id": question.get("id"),
            "order_index": q_idx,
            "question_text": question.get("text", ""),
            "answer_text": answer_text,
            "q_start": q_start,
            "a_start": a_start,
            "a_end": a_end,
        })

    pairs.sort(key=lambda p: p.get("order_index", 0))
    return pairs


# ════════════════════════════════════════════════════════════
#   LIVE PATH — stereo dialogue, LLM-discovered Q&A
# ════════════════════════════════════════════════════════════

def _interleave_channel_segments(
    hr_segments: List[Dict[str, Any]],
    candidate_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge two channel transcripts into a single time-ordered turn list."""
    turns: List[Dict[str, Any]] = []
    for seg in hr_segments or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        turns.append({
            "speaker": "HR",
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
            "text": text,
        })
    for seg in candidate_segments or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        turns.append({
            "speaker": "Candidate",
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
            "text": text,
        })
    turns.sort(key=lambda t: t["start"])
    return _merge_consecutive_same_speaker(turns)


def _merge_consecutive_same_speaker(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent turns from the same speaker (Whisper segmentation noise)."""
    if not turns:
        return []
    merged: List[Dict[str, Any]] = [dict(turns[0])]
    for t in turns[1:]:
        last = merged[-1]
        # Same speaker AND turn boundary <= 1.2s apart → fuse
        if t["speaker"] == last["speaker"] and t["start"] - last["end"] <= 1.2:
            last["end"] = t["end"]
            last["text"] = (last["text"] + " " + t["text"]).strip()
        else:
            merged.append(dict(t))
    return merged


def render_dialogue_transcript(turns: List[Dict[str, Any]]) -> str:
    """Render `[{speaker, start, text}, ...]` as 'HR: ...\\nCandidate: ...'."""
    lines: List[str] = []
    for t in turns or []:
        speaker = t.get("speaker", "Speaker")
        text = (t.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def pair_qa_from_dialogue(
    hr_segments: List[Dict[str, Any]],
    candidate_segments: List[Dict[str, Any]],
    job_context: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Discover substantive Q&A pairs from a two-speaker dialogue using gemma4:e4b.

    Returns (qa_pairs, dialogue_turns).
    """
    turns = _interleave_channel_segments(hr_segments, candidate_segments)
    if not turns:
        return [], []

    dialogue_text = render_dialogue_transcript(turns)

    prompt = f"""You are reviewing a job interview transcript with two speakers (HR and Candidate).
Identify each substantive question that the HR speaker asked and pair it with the
candidate's answer. Skip greetings, filler, "okay", "next question", small talk.

Job context: {job_context[:600] if job_context else "(none provided)"}

Transcript:
{dialogue_text[:6000]}

Return ONLY this JSON (no markdown, no commentary):
{{"pairs": [
  {{"order_index": 0,
    "question_text": "<the HR question, verbatim or near-verbatim>",
    "answer_text":   "<the candidate's full answer, joined>"
  }}
]}}

Rules:
- Each pair represents one substantive HR question. If HR asked multiple in a row
  before the candidate replied, combine them into one question_text.
- Order pairs in chronological order, starting from order_index 0.
- If a candidate answer spans multiple turns, concatenate them with spaces.
- If you cannot find any substantive questions, return {{"pairs": []}}."""

    try:
        result = generate_json(prompt, temperature=0.1, max_tokens=2048)
    except Exception as e:
        log.warning(f"[qa] LLM dialogue pairing failed: {e}")
        result = {}

    raw_pairs = result.get("pairs") if isinstance(result, dict) else None
    if not isinstance(raw_pairs, list):
        log.warning("[qa] LLM returned no pairs; falling back to heuristic")
        raw_pairs = _heuristic_pair_dialogue(turns)

    pairs: List[Dict[str, Any]] = []
    for i, p in enumerate(raw_pairs):
        if not isinstance(p, dict):
            continue
        q_text = (p.get("question_text") or "").strip()
        a_text = (p.get("answer_text") or "").strip()
        if not q_text or not a_text:
            continue
        pairs.append({
            "question_id": None,
            "order_index": int(p.get("order_index", i)),
            "question_text": q_text,
            "answer_text": a_text,
        })

    pairs.sort(key=lambda p: p.get("order_index", 0))
    return pairs, turns


def _heuristic_pair_dialogue(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback when LLM fails: greedy alternation. HR turn → next Candidate turn."""
    pairs: List[Dict[str, Any]] = []
    pending_q: Optional[str] = None
    order = 0
    for t in turns:
        if t["speaker"] == "HR":
            pending_q = (pending_q + " " + t["text"]).strip() if pending_q else t["text"]
        else:  # Candidate
            if pending_q:
                pairs.append({
                    "order_index": order,
                    "question_text": pending_q,
                    "answer_text": t["text"],
                })
                order += 1
                pending_q = None
    return pairs


# ════════════════════════════════════════════════════════════
#   PER-QUESTION LLM ANALYSIS
# ════════════════════════════════════════════════════════════

def analyze_question(
    question_text: str,
    answer_text: str,
    job_context: str = "",
    candidate_resume_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Score and explain a single Q&A pair using gemma4:e4b. JSON output."""
    if not answer_text or len(answer_text.strip()) < 3:
        return {
            "relevance_score": 0.0,
            "quality_score": 0.0,
            "commentary": "Aday bu soruya anlamlı bir cevap vermedi.",
            "strengths": [],
            "weaknesses": ["Cevap çok kısa veya eksik."],
            "low_confidence": True,
        }

    resume_block = f"\nCandidate background:\n{candidate_resume_summary[:600]}" if candidate_resume_summary else ""

    prompt = f"""You are an experienced HR interviewer evaluating a single interview question and answer.

Job context:
{job_context[:800] if job_context else "(general)"}{resume_block}

Question:
{question_text[:1000]}

Candidate's answer:
{answer_text[:2500]}

Return ONLY this JSON (no markdown, no commentary outside JSON):
{{"relevance_score": 0.0,
  "quality_score":   0.0,
  "commentary":      "2-3 sentence professional evaluation in the same language as the answer",
  "strengths":       ["..."],
  "weaknesses":      ["..."]}}

Scoring:
- relevance_score [0.0-1.0]: how directly the answer addresses the question.
- quality_score   [0.0-1.0]: depth, clarity, structure, and use of concrete examples.
- commentary: balanced, professional, no exact numeric scores in the prose.
- strengths/weaknesses: 1-3 short bullets each. Empty list is acceptable."""

    try:
        result = generate_json(prompt, temperature=0.2, max_tokens=768)
    except Exception as e:
        log.warning(f"[qa] LLM analyze_question failed: {e}")
        result = {}

    if not isinstance(result, dict) or "relevance_score" not in result:
        return {
            "relevance_score": 0.4,
            "quality_score": 0.4,
            "commentary": "Otomatik değerlendirme tamamlanamadı; cevap insan tarafından gözden geçirilmeli.",
            "strengths": [],
            "weaknesses": [],
            "low_confidence": True,
        }

    try:
        relevance = float(result.get("relevance_score", 0.5))
    except (TypeError, ValueError):
        relevance = 0.5
    try:
        quality = float(result.get("quality_score", 0.5))
    except (TypeError, ValueError):
        quality = 0.5

    commentary = (result.get("commentary") or "").strip()
    strengths = result.get("strengths") if isinstance(result.get("strengths"), list) else []
    weaknesses = result.get("weaknesses") if isinstance(result.get("weaknesses"), list) else []

    return {
        "relevance_score": max(0.0, min(1.0, relevance)),
        "quality_score": max(0.0, min(1.0, quality)),
        "commentary": commentary or "(no commentary returned)",
        "strengths": [str(s).strip() for s in strengths if str(s).strip()],
        "weaknesses": [str(w).strip() for w in weaknesses if str(w).strip()],
    }


# ════════════════════════════════════════════════════════════
#   LIVE_MONO fallback — content-based diarization
# ════════════════════════════════════════════════════════════

def diarize_mono_dialogue(
    transcript_segments: List[Dict[str, Any]],
    job_context: str = "",
) -> List[Dict[str, Any]]:
    """When stereo channels are not available, ask gemma4:e4b to label each Whisper
    segment as HR or Candidate based on content cues."""
    if not transcript_segments:
        return []

    numbered = "\n".join(
        f"[{i}] ({s.get('start',0):.1f}s) {(s.get('text') or '').strip()}"
        for i, s in enumerate(transcript_segments) if (s.get('text') or '').strip()
    )

    prompt = f"""Below is a job interview transcript. Each line is a Whisper segment
prefixed by its index. Label every segment as either "HR" (interviewer) or
"Candidate" (interviewee).

Job context: {job_context[:400] if job_context else "(general)"}

Segments:
{numbered[:5500]}

Return ONLY this JSON:
{{"labels": [{{"index": 0, "speaker": "HR"}}, {{"index": 1, "speaker": "Candidate"}}]}}"""

    try:
        result = generate_json(prompt, temperature=0.1, max_tokens=1024)
    except Exception as e:
        log.warning(f"[qa] mono diarization LLM failed: {e}")
        return []

    labels = result.get("labels") if isinstance(result, dict) else None
    if not isinstance(labels, list):
        return []

    label_by_idx = {int(it.get("index", -1)): it.get("speaker", "Candidate")
                    for it in labels if isinstance(it, dict)}

    turns: List[Dict[str, Any]] = []
    for i, seg in enumerate(transcript_segments):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = label_by_idx.get(i, "Candidate")
        turns.append({
            "speaker": "HR" if speaker == "HR" else "Candidate",
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
            "text": text,
        })
    return _merge_consecutive_same_speaker(turns)
