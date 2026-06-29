"""
NLG Report Generator
====================
Generates natural language reports for HR specialists and candidates
using XAI explanation data from the pipeline.

Two report types:
  1. HR Report: Detailed technical analysis with scores, XAI insights, charts
  2. Candidate Feedback: Constructive, professional feedback with improvement areas

The candidate feedback follows XAI principles:
  - Explains WHAT was analyzed (not just a score)
  - Shows HOW each area contributed to the evaluation
  - Provides actionable improvement suggestions (especially for rejections)
"""

import logging
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("nlg_generator")


def _build_xai_header(
    candidate_name: str,
    job_title: str,
    evaluation: dict,
    xai_data: dict,
) -> str:
    """Build the deterministic structured XAI block that always appears at the top of the HR report."""
    final_score = evaluation.get("final_score", 0)
    recommendation = evaluation.get("recommendation", "N/A")
    stream_scores = evaluation.get("stream_scores", [])
    contributions = xai_data.get("feature_contributions", [])
    counterfactuals = xai_data.get("counterfactual_analysis", [])
    confidence = xai_data.get("confidence_level", 0)

    lines = [
        "=" * 60,
        f"  HR ANALYSIS REPORT — CONFIDENTIAL",
        "=" * 60,
        f"  Candidate : {candidate_name}",
        f"  Position  : {job_title or 'N/A'}",
        f"  Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  AI Conf.  : {confidence:.0%}",
        "=" * 60,
        "",
        f"FINAL SCORE: {final_score:.2f} / 1.00",
        f"RECOMMENDATION: {recommendation}",
        "",
        "─" * 40,
        "STREAM BREAKDOWN",
        "─" * 40,
    ]

    for ss in stream_scores:
        bar = "█" * int(ss["score"] * 20) + "░" * (20 - int(ss["score"] * 20))
        lines.append(
            f"  {ss['stream_name']:20s} [{bar}] {ss['score']:.2f} "
            f"(weight: {ss['weight']:.0%}, contribution: {ss['weighted_score']:.3f})"
        )

    if contributions:
        lines.extend(["", "─" * 40, "XAI FEATURE CONTRIBUTIONS", "─" * 40])
        for c in contributions:
            lines.append(
                f"  {c['stream']:20s} → {c['contribution_percentage']:.1f}% of decision "
                f"[{c['impact']} impact, {c['impact_magnitude']}]"
            )

    if counterfactuals:
        lines.extend(["", "─" * 40, "COUNTERFACTUAL ANALYSIS (What-if scenarios)", "─" * 40])
        for cf in counterfactuals:
            lines.append(f"  • {cf['description']}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def _format_qa_block_for_prompt(per_question: list, max_len: int = 4000) -> str:
    """Render the per-question analysis into a compact text block for the LLM prompt."""
    if not per_question:
        return "(No per-question analysis available.)"
    lines = []
    for i, q in enumerate(per_question, start=1):
        question = (q.get("question_text") or "").strip()
        answer = (q.get("answer_text") or "").strip()
        commentary = (q.get("commentary") or "").strip()
        relevance = q.get("relevance_score", 0.0)
        quality = q.get("quality_score", 0.0)
        strengths = ", ".join(q.get("strengths") or [])
        weaknesses = ", ".join(q.get("weaknesses") or [])
        lines.append(
            f"[Q{i}] relevance={relevance:.2f} quality={quality:.2f}\n"
            f"  Question: {question}\n"
            f"  Answer:   {answer[:400]}{'...' if len(answer) > 400 else ''}\n"
            f"  Commentary: {commentary}\n"
            f"  Strengths:  {strengths or '(none)'}\n"
            f"  Weaknesses: {weaknesses or '(none)'}"
        )
    block = "\n\n".join(lines)
    return block[:max_len]


class NLGReportGenerator:
    """Generates NLG reports for HR and candidates."""

    def generate_hr_report(
        self,
        candidate_name: str,
        job_title: str,
        evaluation: dict,
        xai_data: dict,
        raw_results: dict
    ) -> str:
        """
        Generate detailed HR analysis report.
        Includes: scores, XAI feature contributions, stream breakdowns, recommendation reasoning.
        """
        final_score = evaluation.get("final_score", 0)
        recommendation = evaluation.get("recommendation", "N/A")
        stream_scores = evaluation.get("stream_scores", [])
        reasoning = xai_data.get("decision_reasoning", "")
        contributions = xai_data.get("feature_contributions", [])
        counterfactuals = xai_data.get("counterfactual_analysis", [])
        confidence = xai_data.get("confidence_level", 0)

        lines = [
            "=" * 60,
            f"  HR ANALYSIS REPORT — CONFIDENTIAL",
            "=" * 60,
            f"  Candidate : {candidate_name}",
            f"  Position  : {job_title or 'N/A'}",
            f"  Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"  AI Conf.  : {confidence:.0%}",
            "=" * 60,
            "",
            f"FINAL SCORE: {final_score:.2f} / 1.00",
            f"RECOMMENDATION: {recommendation}",
            "",
            "─" * 40,
            "STREAM BREAKDOWN",
            "─" * 40,
        ]

        for ss in stream_scores:
            bar = "█" * int(ss["score"] * 20) + "░" * (20 - int(ss["score"] * 20))
            lines.append(
                f"  {ss['stream_name']:20s} [{bar}] {ss['score']:.2f} "
                f"(weight: {ss['weight']:.0%}, contribution: {ss['weighted_score']:.3f})"
            )

        lines.extend(["", "─" * 40, "XAI FEATURE CONTRIBUTIONS", "─" * 40])
        for c in contributions:
            lines.append(
                f"  {c['stream']:20s} → {c['contribution_percentage']:.1f}% of decision "
                f"[{c['impact']} impact, {c['impact_magnitude']}]"
            )

        lines.extend(["", "─" * 40, "DECISION REASONING", "─" * 40, f"  {reasoning}"])

        # Counterfactual analysis
        lines.extend(["", "─" * 40, "COUNTERFACTUAL ANALYSIS (What-if scenarios)", "─" * 40])
        for cf in counterfactuals:
            lines.append(f"  • {cf['description']}")

        # Strengths & concerns
        strengths = evaluation.get("strengths", [])
        concerns = evaluation.get("concerns", [])

        if strengths:
            lines.extend(["", "─" * 40, "STRENGTHS", "─" * 40])
            for s in strengths:
                lines.append(f"  ✓ {s}")
        if concerns:
            lines.extend(["", "─" * 40, "AREAS OF CONCERN", "─" * 40])
            for c in concerns:
                lines.append(f"  ✗ {c}")

        # Transcript excerpt
        transcript = raw_results.get("transcript", {}).get("full_text", "")
        if transcript:
            excerpt = transcript[:500] + ("..." if len(transcript) > 500 else "")
            lines.extend(["", "─" * 40, "INTERVIEW TRANSCRIPT (excerpt)", "─" * 40, f"  {excerpt}"])

        lines.extend(["", "=" * 60, "  END OF REPORT", "=" * 60])

        report = "\n".join(lines)
        logger.info(f"HR report generated for {candidate_name}: {len(report)} chars")
        return report

    def generate_candidate_feedback(
        self,
        candidate_name: str,
        job_title: str,
        evaluation: dict,
        xai_data: dict
    ) -> str:
        """
        Generate candidate-facing feedback report.
        Professional, constructive, and XAI-backed.
        Does NOT reveal exact scores — focuses on qualitative insights.
        """
        strengths = xai_data.get("strength_areas", [])
        improvements = xai_data.get("improvement_suggestions", [])
        stream_scores = evaluation.get("stream_scores", [])

        lines = [
            f"Dear {candidate_name},",
            "",
            f"Thank you for your interest in the {job_title or 'position'} and for taking the time "
            f"to complete the interview process. Below is a summary of your evaluation based on "
            f"multiple assessment dimensions.",
            "",
            "── ASSESSMENT OVERVIEW ──",
            "",
        ]

        # Qualitative level per stream (no exact scores)
        level_map = {
            (0.0, 0.3): "Needs Improvement",
            (0.3, 0.5): "Developing",
            (0.5, 0.7): "Competent",
            (0.7, 0.85): "Strong",
            (0.85, 1.01): "Excellent",
        }

        for ss in stream_scores:
            score = ss["score"]
            level = "N/A"
            for (lo, hi), lbl in level_map.items():
                if lo <= score < hi:
                    level = lbl
                    break
            lines.append(f"  • {ss['stream_name']}: {level}")

        if strengths:
            lines.extend(["", "── YOUR STRENGTHS ──", ""])
            for s in strengths:
                lines.append(f"  ✓ {s}")

        if improvements:
            lines.extend(["", "── AREAS FOR GROWTH ──", ""])
            for imp in improvements:
                lines.append(f"  → {imp}")

        lines.extend([
            "",
            "── ABOUT THIS ASSESSMENT ──",
            "",
            "This evaluation was conducted using multiple AI-powered analysis streams:",
            "  1. Resume & Skills Analysis — alignment with job requirements",
            "  2. Interview Response Quality — depth and relevance of your answers",
            "  3. Speech Tone Analysis — vocal confidence and engagement patterns",
            "  4. Facial Expression Analysis — non-verbal communication signals",
            "",
            "Each dimension contributes to the overall assessment with transparent, "
            "explainable weighting. This feedback is generated to help you understand "
            "which areas were strongest and where there is room for development.",
            "",
            "We wish you the best in your career journey.",
            "",
            "Best regards,",
            "AI-Recruiter Assessment System"
        ])

        report = "\n".join(lines)
        logger.info(f"Candidate feedback generated for {candidate_name}: {len(report)} chars")
        return report

    # ════════════════════════════════════════════════════════════
    #   LLM-AUTHORED REPORTS (gemma4:e4b)
    # ════════════════════════════════════════════════════════════

    def generate_hr_report_llm(
        self,
        candidate_name: str,
        job_title: str,
        evaluation: dict,
        xai_data: dict,
        qa_data: dict,
        integrity: dict | None = None,
        interview_language: str = "tr",
        cv_summary: str | None = None,
        transcript_excerpt: str | None = None,
        emotion_summary: str | None = None,
        job_requirements: str | None = None,
        interview_type: str = "VIDEO",
        previous_video_report: str | None = None,
    ) -> str:
        """LLM-authored formal HR evaluation report (Markdown).

        Includes overall summary, per-question breakdown, strengths/concerns,
        and integrity flags. Falls back to template-based report on LLM failure.
        Always prepends the deterministic structured XAI block.
        """
        xai_header = _build_xai_header(candidate_name, job_title, evaluation, xai_data)

        try:
            from llm.ollama_client import generate as llm_generate
        except Exception as e:
            logger.warning(f"Ollama client import failed: {e} — falling back to template")
            fallback = self.generate_hr_report(
                candidate_name=candidate_name, job_title=job_title,
                evaluation=evaluation, xai_data=xai_data,
                raw_results={"transcript": {"full_text": qa_data.get("dialogue_transcript", "")}},
            )
            return xai_header + "\n\n" + fallback

        # LIVE interview with a prior video report → comparative report, no CV re-analysis
        if (interview_type or "").upper() == "LIVE" and previous_video_report:
            return self._generate_live_hr_report_llm(
                candidate_name=candidate_name,
                job_title=job_title,
                evaluation=evaluation,
                qa_data=qa_data,
                integrity=integrity,
                interview_language=interview_language,
                transcript_excerpt=transcript_excerpt,
                emotion_summary=emotion_summary,
                previous_video_report=previous_video_report,
                xai_header=xai_header,
                llm_generate=llm_generate,
            )

        per_question = qa_data.get("per_question_analysis", []) or []
        qa_block = _format_qa_block_for_prompt(per_question)
        final_score = evaluation.get("final_score", 0.0)
        recommendation = evaluation.get("recommendation", "N/A")
        stream_scores = evaluation.get("stream_scores", []) or []
        stream_summary = "\n".join(
            f"  - {s.get('stream_name')}: {s.get('score', 0):.2f} (weight {s.get('weight', 0):.0%})"
            for s in stream_scores
        ) or "  (no stream scores)"
        integrity_summary = ""
        if integrity:
            integrity_summary = (
                f"\nIntegrity status: {integrity.get('score', 'UNKNOWN')} — "
                f"{integrity.get('summary', '')}"
            )

        lang_instruction = (
            "Write entirely in English. Do not use any Turkish."
            if interview_language == "en"
            else "Write entirely in Turkish. Do not use any English."
        )

        system_prompt = (
            "You are an experienced HR analyst writing a confidential evaluation report. "
            "Use a formal, professional tone. Be specific and reference per-question evidence "
            "and the provided CV/transcript data. "
            "Output Markdown only. Do NOT invent facts not present in the data."
        )
        prompt = f"""Write a confidential HR evaluation report for the following candidate.

Candidate: {candidate_name or '(unknown)'}
Position:  {job_title or '(unspecified)'}
Date:      {datetime.now().strftime('%Y-%m-%d %H:%M')}

Final score: {final_score:.2f} / 1.00
Recommendation: {recommendation}
Stream breakdown:
{stream_summary}{integrity_summary}

Job requirements:
{job_requirements[:800] if job_requirements else "(not provided)"}

Candidate CV summary (only cite skills/experience listed here):
{cv_summary or "(not available)"}

Emotion analysis results:
{emotion_summary or "(not available)"}

Interview transcript excerpt (what the candidate actually said):
{transcript_excerpt or "(no transcript available)"}

Per-question analysis:
{qa_block}

Produce a Markdown report with this structure:
## Executive Summary
- 2-3 sentences. Include score and recommendation.

## Question-by-Question Analysis
- For each question (numbered Q1, Q2 ...): re-state the question briefly, then 2-3 sentences
  evaluating the candidate's answer, citing strengths/weaknesses where relevant.

## Strengths
- 2-5 concise bullets. Cite specific skills or experiences from the CV that align with the
  job requirements. If emotion results are positive, include a bullet about communication style.

## Areas of Concern
- 2-5 concise bullets tied to evidence. Include observations from the emotion analysis
  (e.g. vocal tone, facial expression patterns) if they show notable patterns.

## Integrity & Risk Signals
- 1-3 bullets. If integrity is CLEAN, state "No integrity concerns observed."

## Final Recommendation
- 2-3 sentence justification of the recommendation.

{lang_instruction}
IMPORTANT: Only reference skills, experiences, and statements explicitly present in the
CV summary or transcript above. Do not invent, assume, or imply information not in the data.
Do not include any prose outside this Markdown structure."""

        try:
            text = llm_generate(prompt, system=system_prompt, temperature=0.3, max_tokens=2048)
            if text and len(text.strip()) > 100:
                logger.info(f"HR report (LLM) generated for {candidate_name}: {len(text)} chars")
                return xai_header + "\n\n" + text.strip()
            logger.warning("LLM HR report empty/short — falling back to template")
        except Exception as e:
            logger.warning(f"LLM HR report failed: {e} — falling back to template")

        fallback = self.generate_hr_report(
            candidate_name=candidate_name, job_title=job_title,
            evaluation=evaluation, xai_data=xai_data,
            raw_results={"transcript": {"full_text": qa_data.get("dialogue_transcript", "")}},
        )
        return xai_header + "\n\n" + fallback

    def _generate_live_hr_report_llm(
        self,
        candidate_name: str,
        job_title: str,
        evaluation: dict,
        qa_data: dict,
        integrity: dict | None,
        interview_language: str,
        transcript_excerpt: str | None,
        emotion_summary: str | None,
        previous_video_report: str,
        xai_header: str,
        llm_generate,
    ) -> str:
        """HR report for a LIVE interview that follows a prior video interview.

        CV is NOT re-analyzed here. The prior video report is used as comparative
        context only — the LLM must not repeat its findings verbatim.
        """
        per_question = qa_data.get("per_question_analysis", []) or []
        qa_block = _format_qa_block_for_prompt(per_question)
        final_score = evaluation.get("final_score", 0.0)
        recommendation = evaluation.get("recommendation", "N/A")
        stream_scores = evaluation.get("stream_scores", []) or []
        stream_summary = "\n".join(
            f"  - {s.get('stream_name')}: {s.get('score', 0):.2f} (weight {s.get('weight', 0):.0%})"
            for s in stream_scores
            if s.get("weight", 0) > 0
        ) or "  (no stream scores)"
        integrity_summary = ""
        if integrity:
            integrity_summary = (
                f"\nIntegrity status: {integrity.get('score', 'UNKNOWN')} — "
                f"{integrity.get('summary', '')}"
            )
        lang_instruction = (
            "Write entirely in English. Do not use any Turkish."
            if interview_language == "en"
            else "Write entirely in Turkish. Do not use any English."
        )
        prior_context = (
            "Prior video interview report (reference for comparison ONLY — "
            "do NOT repeat its findings verbatim):\n"
            + previous_video_report[:1500]
        )

        system_prompt = (
            "You are an experienced HR analyst writing a confidential evaluation report "
            "for a LIVE interview that follows a prior one-way AI video interview. "
            "The candidate's CV has already been assessed in the video stage — "
            "do NOT re-analyze or re-state CV content here. "
            "Focus exclusively on what happened in this live session: dialogue quality, "
            "spontaneity, behavioral consistency, and communication under real-time conditions. "
            "Where the prior video report is provided, write comparatively "
            "(e.g. 'In the video stage the candidate scored X; in this live session ...'). "
            "Use a formal, professional tone. Output Markdown only. Do NOT invent facts."
        )
        prompt = f"""Write a confidential LIVE INTERVIEW HR evaluation report.

Candidate: {candidate_name or '(unknown)'}
Position:  {job_title or '(unspecified)'}
Date:      {datetime.now().strftime('%Y-%m-%d %H:%M')}

Live session final score: {final_score:.2f} / 1.00
Recommendation: {recommendation}

Live session stream breakdown (CV not re-assessed in this stage):
{stream_summary}{integrity_summary}

Emotion signals (live session):
{emotion_summary or "(not available)"}

Live dialogue transcript excerpt:
{transcript_excerpt or "(no transcript available)"}

Per-question live dialogue analysis:
{qa_block}

{prior_context}

Produce a Markdown report with this structure:
## Executive Summary
- 2-3 sentences. State this is the live follow-up interview. Include live score and
  recommendation. If prior video report is available, compare overall trajectory
  (improved / consistent / declined).

## Dialogue Quality & Spontaneous Response Analysis
- For each Q&A pair (Q1, Q2 ...): 2-3 sentences evaluating depth, spontaneity,
  and whether the answer held up under real-time pressure.

## Behavioral Consistency
- 2-4 bullets comparing behavioral signals (vocal tone, facial expression, response
  depth) between this live session and the prior video interview.
  If no prior report: assess internal consistency within this session.

## Communication Under Real Conditions
- 2-3 bullets on active listening, composure, and follow-up handling during
  the unscripted HR dialogue.

## Areas of Concern
- 2-4 bullets tied to live-session evidence only. Include emotion observations.

## Integrity & Risk Signals
- 1-3 bullets. If integrity is CLEAN, state "No integrity concerns observed."

## Final Recommendation
- 2-3 sentences. Cross-reference with video stage outcome where available.

{lang_instruction}
IMPORTANT: Do NOT re-analyze or re-state CV content. Only reference what the candidate
actually said or demonstrated during this live session. Do not invent facts."""

        try:
            text = llm_generate(prompt, system=system_prompt, temperature=0.3, max_tokens=2048)
            if text and len(text.strip()) > 100:
                logger.info(f"LIVE HR report (LLM) generated for {candidate_name}: {len(text)} chars")
                return xai_header + "\n\n" + text.strip()
            logger.warning("LIVE LLM HR report empty/short — falling back to template")
        except Exception as e:
            logger.warning(f"LIVE LLM HR report failed: {e} — falling back to template")

        fallback = self.generate_hr_report(
            candidate_name=candidate_name, job_title=job_title,
            evaluation=evaluation, xai_data={},
            raw_results={"transcript": {"full_text": qa_data.get("dialogue_transcript", "")}},
        )
        return xai_header + "\n\n" + fallback

    def generate_candidate_feedback_llm(
        self,
        candidate_name: str,
        job_title: str,
        evaluation: dict,
        xai_data: dict,
        qa_data: dict,
        interview_language: str = "tr",
        cv_summary: str | None = None,
        transcript_excerpt: str | None = None,
        emotion_summary: str | None = None,
        job_requirements: str | None = None,
        interview_type: str = "VIDEO",
        previous_video_report: str | None = None,
    ) -> str:
        """LLM-authored constructive candidate feedback (Markdown).

        Strengths-first, no exact numeric scores, suggests growth areas. Falls
        back to template on LLM failure.
        """
        try:
            from llm.ollama_client import generate as llm_generate
        except Exception as e:
            logger.warning(f"Ollama client import failed: {e} — falling back to template")
            return self.generate_candidate_feedback(
                candidate_name=candidate_name, job_title=job_title,
                evaluation=evaluation, xai_data=xai_data,
            )

        # LIVE with prior video → comparative candidate feedback, no CV re-analysis
        if (interview_type or "").upper() == "LIVE" and previous_video_report:
            return self._generate_live_candidate_feedback_llm(
                candidate_name=candidate_name,
                job_title=job_title,
                evaluation=evaluation,
                qa_data=qa_data,
                interview_language=interview_language,
                transcript_excerpt=transcript_excerpt,
                emotion_summary=emotion_summary,
                previous_video_report=previous_video_report,
                llm_generate=llm_generate,
            )

        per_question = qa_data.get("per_question_analysis", []) or []
        qa_block = _format_qa_block_for_prompt(per_question)

        # Language-dependent section headers and instruction
        if interview_language == "en":
            sections = (
                "## Overall Impression",
                "## Your Strengths",
                "## Question-by-Question Feedback",
                "## Areas for Growth",
                "## Closing",
            )
            lang_instruction = "Write entirely in English. Do not use any Turkish."
        else:
            sections = (
                "## Genel İzlenim",
                "## Güçlü Yönleriniz",
                "## Soru Bazlı Geri Bildirim",
                "## Geliştirilmesi Önerilen Alanlar",
                "## Kapanış",
            )
            lang_instruction = "Write entirely in Turkish. Do not use any English."

        h_impression, h_strengths, h_qa, h_growth, h_closing = sections

        system_prompt = (
            "You are an empathetic career coach writing constructive feedback for an interview "
            "candidate. Use a warm, encouraging, professional tone. Never reveal exact numeric "
            "scores. Be specific about strengths and offer actionable growth suggestions based "
            "only on the provided CV and transcript data. "
            "Output Markdown only."
        )
        prompt = f"""Write constructive interview feedback for the following candidate.

Candidate: {candidate_name or '(candidate)'}
Position:  {job_title or 'the position you applied for'}

Job requirements (use this to frame strengths and growth areas):
{job_requirements[:600] if job_requirements else "(not provided)"}

Candidate CV summary (ground your feedback in this — only cite what is listed here):
{cv_summary or "(not available)"}

Emotion analysis results (use to comment on communication style, tone, and presence):
{emotion_summary or "(not available)"}

Interview transcript excerpt (what the candidate actually said):
{transcript_excerpt or "(no transcript available)"}

Per-question analysis (HR-internal — do NOT cite scores in your output):
{qa_block}

Produce Markdown with this structure:
{h_impression}
- 2-3 sentences, warm and balanced. Address the candidate by name once.

{h_strengths}
- 3-5 specific bullets. Include:
  * At least one bullet citing a concrete skill or experience from the CV relevant to the job.
  * At least one bullet referencing the candidate's communication style based on emotion results
    (e.g. vocal presence, facial engagement) — keep it encouraging and constructive.
  * Any notable points from the interview responses.

{h_qa}
- For each question (Q1, Q2 ...): one short paragraph that:
  - acknowledges what worked
  - suggests one concrete growth opportunity (no exact scores).

{h_growth}
- 2-4 actionable bullets. Include at least one bullet that connects a CV skill gap to the job
  requirements and suggests how to close it. Include one tip on emotional presence if relevant.

{h_closing}
- 2 sentences of encouragement.

{lang_instruction}
IMPORTANT: Only reference skills, experiences, and statements explicitly present in the
CV summary or transcript above. Do not invent, assume, or imply information not in the data.
Do not include any text outside this Markdown structure."""

        try:
            text = llm_generate(prompt, system=system_prompt, temperature=0.4, max_tokens=2048)
            if text and len(text.strip()) > 100:
                logger.info(f"Candidate feedback (LLM) generated for {candidate_name}: {len(text)} chars")
                return text.strip()
            logger.warning("LLM candidate feedback empty/short — falling back to template")
        except Exception as e:
            logger.warning(f"LLM candidate feedback failed: {e} — falling back to template")

        return self.generate_candidate_feedback(
            candidate_name=candidate_name, job_title=job_title,
            evaluation=evaluation, xai_data=xai_data,
        )

    def _generate_live_candidate_feedback_llm(
        self,
        candidate_name: str,
        job_title: str,
        evaluation: dict,
        qa_data: dict,
        interview_language: str,
        transcript_excerpt: str | None,
        emotion_summary: str | None,
        previous_video_report: str,
        llm_generate,
    ) -> str:
        """Candidate-facing feedback for a LIVE interview with a prior video stage.

        Does not re-analyze CV. Writes comparatively where prior report is present.
        """
        per_question = qa_data.get("per_question_analysis", []) or []
        qa_block = _format_qa_block_for_prompt(per_question)

        if interview_language == "en":
            sections = (
                "## Overall Impression",
                "## Your Strengths in This Live Interview",
                "## Question-by-Question Feedback",
                "## Areas for Growth",
                "## Closing",
            )
            lang_instruction = "Write entirely in English. Do not use any Turkish."
        else:
            sections = (
                "## Genel İzlenim",
                "## Bu Canlı Görüşmedeki Güçlü Yönleriniz",
                "## Soru Bazlı Geri Bildirim",
                "## Geliştirilmesi Önerilen Alanlar",
                "## Kapanış",
            )
            lang_instruction = "Write entirely in Turkish. Do not use any English."

        h_impression, h_strengths, h_qa, h_growth, h_closing = sections
        prior_context = (
            "Your prior video interview report (for context — do NOT copy it, "
            "only reference it where relevant):\n"
            + previous_video_report[:1200]
        )

        system_prompt = (
            "You are an empathetic career coach writing constructive feedback for a candidate "
            "who has just completed a LIVE interview following a prior video interview. "
            "Do NOT re-evaluate or mention the candidate's CV — that was covered in the video stage. "
            "Focus exclusively on how the candidate communicated and performed in this live session. "
            "Where the prior video report is provided, you may briefly compare "
            "(e.g. 'Compared to your video interview, your responses showed ...'). "
            "Never reveal exact numeric scores. Be warm, specific, and encouraging. "
            "Output Markdown only."
        )
        prompt = f"""Write constructive live interview feedback for the following candidate.

Candidate: {candidate_name or '(candidate)'}
Position:  {job_title or 'the position you applied for'}

Emotion analysis results (comment on communication style, tone, presence):
{emotion_summary or "(not available)"}

Live dialogue transcript excerpt:
{transcript_excerpt or "(no transcript available)"}

Per-question analysis (do NOT reveal numeric scores in your output):
{qa_block}

{prior_context}

Produce Markdown with this structure:
{h_impression}
- 2-3 sentences, warm and balanced. Note this is the live follow-up interview.
  Briefly compare to video stage if prior report is available.

{h_strengths}
- 3-5 specific bullets from this live session only:
  * Communication style (vocal presence, composure, clarity) based on emotion results.
  * Response quality and depth during the live dialogue.
  * Any notable strengths from the Q&A pairs.

{h_qa}
- For each question (Q1, Q2 ...): one short paragraph that acknowledges what worked
  and suggests one concrete growth opportunity.

{h_growth}
- 2-4 actionable bullets focused on live communication skills:
  spontaneity, listening, follow-up handling, composure under pressure.

{h_closing}
- 2 sentences of encouragement referencing the overall journey (video + live).

{lang_instruction}
IMPORTANT: Do NOT reference CV content. Only reference what the candidate said or
demonstrated in this live session. Do not invent facts."""

        try:
            text = llm_generate(prompt, system=system_prompt, temperature=0.4, max_tokens=2048)
            if text and len(text.strip()) > 100:
                logger.info(f"LIVE candidate feedback (LLM) generated for {candidate_name}: {len(text)} chars")
                return text.strip()
            logger.warning("LIVE LLM candidate feedback empty/short — falling back to template")
        except Exception as e:
            logger.warning(f"LIVE LLM candidate feedback failed: {e} — falling back to template")

        return self.generate_candidate_feedback(
            candidate_name=candidate_name, job_title=job_title,
            evaluation=evaluation, xai_data={},
        )

    def generate_rejection_feedback(
        self,
        candidate_name: str,
        job_title: str,
        evaluation: dict,
        xai_data: dict
    ) -> str:
        """
        Generate constructive rejection feedback.
        Focuses heavily on improvement areas while acknowledging strengths.
        XAI-backed: uses counterfactual analysis to show what could improve.
        """
        strengths = xai_data.get("strength_areas", [])
        improvements = xai_data.get("improvement_suggestions", [])
        counterfactuals = xai_data.get("counterfactual_analysis", [])

        lines = [
            f"Dear {candidate_name},",
            "",
            f"Thank you for your interest in the {job_title or 'position'} and for investing "
            f"your time in the interview process. After careful evaluation, we have decided "
            f"to move forward with other candidates at this time.",
            "",
            "We want to provide you with constructive feedback to support your professional development.",
            "",
        ]

        if strengths:
            lines.extend(["── WHAT WENT WELL ──", ""])
            for s in strengths:
                lines.append(f"  ✓ {s}")
            lines.append("")

        if improvements:
            lines.extend(["── DEVELOPMENT OPPORTUNITIES ──", ""])
            for imp in improvements:
                lines.append(f"  → {imp}")
            lines.append("")

        # Counterfactual insights (simplified for candidate)
        top_lift = None
        if counterfactuals:
            top_lift = max(counterfactuals, key=lambda x: x.get("max_possible_lift", 0))

        if top_lift and top_lift["max_possible_lift"] > 0.05:
            stream = top_lift["stream"]
            lines.extend([
                "── KEY INSIGHT ──",
                "",
                f"  Our analysis suggests that strengthening your performance in the "
                f"'{stream}' area could have the most significant positive impact on "
                f"future evaluations.",
                ""
            ])

        lines.extend([
            "── NEXT STEPS ──",
            "",
            "  • Review the development areas mentioned above",
            "  • Consider practicing with mock interviews to improve verbal and non-verbal communication",
            "  • Keep your resume updated with relevant skills and accomplishments",
            "  • You are welcome to re-apply for future positions",
            "",
            "We value the effort you put into this process and encourage you to continue growing.",
            "",
            "Best regards,",
            "AI-Recruiter Assessment System"
        ])

        report = "\n".join(lines)
        logger.info(f"Rejection feedback generated for {candidate_name}")
        return report
