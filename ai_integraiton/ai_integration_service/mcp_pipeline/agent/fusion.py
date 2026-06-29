"""
Fusion Module - Weighted Aggregation & Scoring
Combines outputs from 4 AI streams into an HR Decision Panel score.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.schemas import CandidateEvaluation, StreamScore

DEFAULT_WEIGHTS = {
    "resume_score": 0.35,
    "transcript_score": 0.25,
    "speech_emotion": 0.20,
    "facial_emotion": 0.20,
}

# Used for LIVE interviews that follow a prior video interview (CV already assessed).
LIVE_WEIGHTS = {
    "resume_score":     0.00,
    "transcript_score": 0.40,
    "speech_emotion":   0.30,
    "facial_emotion":   0.30,
}


def get_weights(interview_type: str, has_prior_video: bool = False) -> dict:
    """Return the correct weight set for the interview type.

    LIVE_WEIGHTS are only used when the interview is LIVE *and* a prior video
    report exists (meaning CV was already assessed). Otherwise DEFAULT_WEIGHTS
    are used so the full 4-stream pipeline runs normally.
    """
    if (interview_type or "").upper() == "LIVE" and has_prior_video:
        return LIVE_WEIGHTS
    return DEFAULT_WEIGHTS


def compute_integrated_score(
    resume_data: dict | None = None,
    transcript_data: dict | None = None,
    speech_emotion_data: dict | None = None,
    facial_emotion_data: dict | None = None,
    weights: dict | None = None,
    candidate_name: str = "Unknown"
) -> CandidateEvaluation:
    """Compute weighted final score from all stream outputs."""

    if weights is None:
        weights = DEFAULT_WEIGHTS

    stream_scores = []
    strengths = []
    concerns = []

    # -- Stream 1: Resume --
    resume_score = 0.0
    if resume_data and not resume_data.get("error"):
        resume_score = resume_data.get("relevance_score", 0.5)
        exp_count = len(resume_data.get("work_experience", []))
        skill_count = len(resume_data.get("technical_skills", []))

        if resume_score >= 0.7:
            strengths.append(f"Strong resume-to-job alignment ({resume_score:.0%})")
        if exp_count >= 3:
            strengths.append(f"Extensive work experience ({exp_count} positions)")
        if resume_score < 0.4:
            concerns.append(f"Low resume relevance ({resume_score:.0%})")
        if skill_count < 3:
            concerns.append("Limited technical skill set")

    stream_scores.append(StreamScore(
        stream_name="Resume Analysis",
        score=resume_score,
        weight=weights["resume_score"],
        weighted_score=resume_score * weights["resume_score"],
        details={"relevance": resume_score}
    ))

    # -- Stream 2: Transcript --
    transcript_score = 0.0
    if transcript_data:
        transcript_score = transcript_data.get("content_score", 0.0)
        word_count = transcript_data.get("word_count", 0)

        if transcript_score >= 0.7:
            strengths.append("High-quality interview responses with strong content")
        if word_count > 100:
            strengths.append("Detailed and comprehensive verbal responses")
        if transcript_score < 0.4:
            concerns.append("Interview responses lacked depth or relevance")
        if word_count < 30:
            concerns.append("Notably brief verbal responses")

    stream_scores.append(StreamScore(
        stream_name="Speech Transcript",
        score=transcript_score,
        weight=weights["transcript_score"],
        weighted_score=transcript_score * weights["transcript_score"],
        details={"content_quality": transcript_score}
    ))

    # -- Stream 3: Speech Emotion --
    speech_pos = 0.5
    if speech_emotion_data:
        speech_pos = speech_emotion_data.get("positivity_score", 0.5)
        dominant = speech_emotion_data.get("dominant_emotion", "neutral")

        if dominant == "positive":
            strengths.append("Positive and energetic vocal tone throughout")
        elif dominant == "negative":
            concerns.append("Noticeable negativity detected in vocal tone")

        neg_ratio = speech_emotion_data.get("emotion_distribution", {}).get("negative", 0)
        if neg_ratio > 0.4:
            concerns.append(f"Negative vocal tone in {neg_ratio*100:.0f}% of interview")

    stream_scores.append(StreamScore(
        stream_name="Speech Emotion",
        score=speech_pos,
        weight=weights["speech_emotion"],
        weighted_score=speech_pos * weights["speech_emotion"],
        details={"positivity": speech_pos,
                 "dominant": speech_emotion_data.get("dominant_emotion", "N/A") if speech_emotion_data else "N/A"}
    ))

    # -- Stream 4: Facial Emotion --
    facial_pos = 0.5
    if facial_emotion_data:
        facial_pos = facial_emotion_data.get("positivity_score", 0.5)
        dominant = facial_emotion_data.get("dominant_emotion", "neutral")
        dist = facial_emotion_data.get("emotion_distribution", {})

        happy_r = dist.get("happy", 0)
        if happy_r > 0.3:
            strengths.append(f"Predominantly positive facial expressions ({happy_r*100:.0f}% happy)")
        if dominant in ["angry", "disgust", "fear"]:
            concerns.append(f"Dominant facial expression: {dominant}")
        if dist.get("neutral", 0) > 0.7:
            concerns.append("Predominantly neutral expressions — low engagement")

    stream_scores.append(StreamScore(
        stream_name="Facial Emotion",
        score=facial_pos,
        weight=weights["facial_emotion"],
        weighted_score=facial_pos * weights["facial_emotion"],
        details={"positivity": facial_pos,
                 "dominant": facial_emotion_data.get("dominant_emotion", "N/A") if facial_emotion_data else "N/A"}
    ))

    # -- Final Score --
    final_score = sum(s.weighted_score for s in stream_scores)

    if final_score >= 0.75:
        recommendation = "Strong Hire"
    elif final_score >= 0.60:
        recommendation = "Hire"
    elif final_score >= 0.45:
        recommendation = "Maybe"
    else:
        recommendation = "No Hire"

    summary = _generate_summary(candidate_name, final_score, recommendation,
                                stream_scores, strengths, concerns)

    return CandidateEvaluation(
        candidate_name=candidate_name,
        stream_scores=stream_scores,
        final_score=round(final_score, 3),
        recommendation=recommendation,
        strengths=strengths[:5],
        concerns=concerns[:5],
        summary=summary
    )


def _generate_summary(name, score, rec, stream_scores, strengths, concerns):
    lines = [
        f"Candidate Evaluation Summary: {name}",
        "=" * 50,
        f"Final Score: {score:.2f}/1.00 | Recommendation: {rec}",
        "",
        "Stream Scores:",
    ]
    for ss in stream_scores:
        bar = "█" * int(ss.score * 20) + "░" * (20 - int(ss.score * 20))
        lines.append(f"  {ss.stream_name:20s} [{bar}] {ss.score:.2f} (x{ss.weight:.2f} = {ss.weighted_score:.2f})")

    if strengths:
        lines.append("\nStrengths:")
        for s in strengths[:5]:
            lines.append(f"  + {s}")
    if concerns:
        lines.append("\nAreas of Concern:")
        for c in concerns[:5]:
            lines.append(f"  - {c}")

    return "\n".join(lines)
