"""
XAI Explainer Module
====================
Generates explainable AI outputs for each analysis stream.

Approach: Feature Contribution Analysis (SHAP-like)
- Calculates how much each stream contributed to the final decision
- Identifies which specific features (emotions, CV skills, speech tone) drove the score
- Produces human-readable explanations for both HR and candidates

This module does NOT require actual SHAP/LIME computation on neural networks.
Instead, it uses the weighted fusion scores and raw stream outputs to produce
interpretable explanations — a practical XAI approach for production systems.
"""

import logging
from typing import Optional

logger = logging.getLogger("xai_explainer")


class XAIExplainer:
    """
    Generates XAI (Explainable AI) data from pipeline results.

    Produces:
      1. Feature contributions (which stream drove the score)
      2. Per-stream explanations (what specific signals were detected)
      3. Counterfactual insights ("score would be X if Y were different")
      4. Strength areas & improvement suggestions for candidate feedback
    """

    # Emotion valence mapping (same as fusion module)
    FACIAL_VALENCE = {
        'happy': 1.0, 'surprise': 0.7, 'neutral': 0.5,
        'sad': 0.2, 'fear': 0.15, 'angry': 0.1, 'disgust': 0.05
    }

    SPEECH_VALENCE = {
        'positive': 1.0, 'neutral': 0.5, 'negative': 0.1
    }

    def explain(
        self,
        stream_scores: list[dict],
        raw_results: dict,
        final_score: float,
        recommendation: str
    ) -> dict:
        """
        Generate comprehensive XAI explanations.

        Args:
            stream_scores: List of StreamScore dicts from fusion
            raw_results: Raw output from each AI stream
            final_score: Weighted final score (0-1)
            recommendation: "Strong Hire", "Hire", "Maybe", "No Hire"

        Returns:
            XAI explanation dict with feature_contributions, per_stream, counterfactuals, etc.
        """
        explanations = {
            "feature_contributions": self._compute_feature_contributions(stream_scores, final_score),
            "per_stream_explanations": self._explain_streams(stream_scores, raw_results),
            "counterfactual_analysis": self._compute_counterfactuals(stream_scores, final_score),
            "decision_reasoning": self._generate_decision_reasoning(stream_scores, final_score, recommendation),
            "strength_areas": self._identify_strengths(stream_scores, raw_results),
            "improvement_suggestions": self._identify_improvements(stream_scores, raw_results),
            "confidence_level": self._compute_confidence(stream_scores),
        }

        logger.info(f"XAI explanations generated: {len(explanations['feature_contributions'])} contributions, "
                     f"confidence={explanations['confidence_level']:.2f}")

        return explanations

    def _compute_feature_contributions(self, stream_scores: list[dict], final_score: float) -> list[dict]:
        """
        Calculate how much each stream contributed to the final score.
        Similar to SHAP feature importance values.
        """
        contributions = []
        for ss in stream_scores:
            contribution_pct = (ss["weighted_score"] / final_score * 100) if final_score > 0 else 0
            contributions.append({
                "stream": ss["stream_name"],
                "raw_score": round(ss["score"], 3),
                "weight": round(ss["weight"], 2),
                "weighted_score": round(ss["weighted_score"], 3),
                "contribution_percentage": round(contribution_pct, 1),
                "impact": "positive" if ss["score"] >= 0.5 else "negative",
                "impact_magnitude": "high" if abs(ss["score"] - 0.5) > 0.25 else
                                   "medium" if abs(ss["score"] - 0.5) > 0.1 else "low",
            })

        # Sort by contribution
        contributions.sort(key=lambda x: x["weighted_score"], reverse=True)
        return contributions

    def _explain_streams(self, stream_scores: list[dict], raw_results: dict) -> dict:
        """Generate human-readable explanation for each stream."""
        explanations = {}

        for ss in stream_scores:
            name = ss["stream_name"]
            score = ss["score"]

            if name == "Resume Analysis":
                explanations[name] = self._explain_resume(score, raw_results.get("resume", {}))
            elif name == "Speech Transcript":
                explanations[name] = self._explain_transcript(score, raw_results.get("transcript", {}))
            elif name == "Speech Emotion":
                explanations[name] = self._explain_speech_emotion(score, raw_results.get("speech_emotion", {}))
            elif name == "Facial Emotion":
                explanations[name] = self._explain_facial_emotion(score, raw_results.get("facial_emotion", {}))

        return explanations

    def _explain_resume(self, score: float, data: dict) -> dict:
        exp = {"score": score, "signals": []}
        if data:
            skills = data.get("technical_skills", [])
            exp_count = len(data.get("work_experience", []))
            edu_count = len(data.get("education", []))

            if exp_count > 0:
                exp["signals"].append(f"Detected {exp_count} work experience entries")
            if skills:
                exp["signals"].append(f"Found {len(skills)} technical skills: {', '.join(skills[:5])}")
            if edu_count > 0:
                exp["signals"].append(f"Found {edu_count} education entries")
            if score >= 0.7:
                exp["signals"].append("Resume shows strong alignment with job requirements")
            elif score < 0.4:
                exp["signals"].append("Resume shows limited alignment with job requirements")
        return exp

    def _explain_transcript(self, score: float, data: dict) -> dict:
        exp = {"score": score, "signals": []}
        if data:
            word_count = data.get("word_count", 0)
            lang = data.get("detected_language", "unknown")
            exp["signals"].append(f"Transcribed {word_count} words in {lang}")
            if word_count > 200:
                exp["signals"].append("Candidate provided detailed and thorough responses")
            elif word_count < 50:
                exp["signals"].append("Responses were notably brief — limited verbal engagement")
        return exp

    def _explain_speech_emotion(self, score: float, data: dict) -> dict:
        exp = {"score": score, "signals": []}
        if data:
            dist = data.get("emotion_distribution", {})
            dominant = data.get("dominant_emotion", "neutral")
            exp["signals"].append(f"Dominant speech emotion: {dominant}")
            for emo, ratio in sorted(dist.items(), key=lambda x: -x[1]):
                if ratio > 0.1:
                    exp["signals"].append(f"  {emo}: {ratio:.0%} of interview duration")
        return exp

    def _explain_facial_emotion(self, score: float, data: dict) -> dict:
        exp = {"score": score, "signals": []}
        if data:
            dist = data.get("emotion_distribution", {})
            dominant = data.get("dominant_emotion", "neutral")
            faces = data.get("faces_detected", 0)
            exp["signals"].append(f"Dominant facial expression: {dominant} ({faces} face detections)")
            for emo, ratio in sorted(dist.items(), key=lambda x: -x[1]):
                if ratio > 0.05:
                    exp["signals"].append(f"  {emo}: {ratio:.0%}")
        return exp

    def _compute_counterfactuals(self, stream_scores: list[dict], final_score: float) -> list[dict]:
        """
        Counterfactual analysis: "If stream X had scored Y, the final score would be Z"
        Helps HR understand which factors could change the outcome.
        """
        counterfactuals = []
        for ss in stream_scores:
            # What if this stream scored perfectly (1.0)?
            hypothetical_perfect = final_score - ss["weighted_score"] + ss["weight"] * 1.0
            # What if this stream scored worst (0.0)?
            hypothetical_worst = final_score - ss["weighted_score"]

            counterfactuals.append({
                "stream": ss["stream_name"],
                "current_score": round(ss["score"], 3),
                "if_perfect": round(hypothetical_perfect, 3),
                "if_absent": round(hypothetical_worst, 3),
                "max_possible_lift": round(hypothetical_perfect - final_score, 3),
                "description": (
                    f"If {ss['stream_name']} scored 1.0, final score would be "
                    f"{hypothetical_perfect:.2f} (currently {final_score:.2f})"
                )
            })

        return counterfactuals

    def _generate_decision_reasoning(self, stream_scores, final_score, recommendation) -> str:
        """Generate a natural-language reasoning chain for the decision."""
        top_stream = max(stream_scores, key=lambda x: x["weighted_score"])
        bottom_stream = min(stream_scores, key=lambda x: x["weighted_score"])

        reasoning = (
            f"The candidate received an overall score of {final_score:.2f}/1.00, "
            f"leading to a '{recommendation}' recommendation. "
            f"The strongest signal came from {top_stream['stream_name']} "
            f"(score: {top_stream['score']:.2f}, contributing {top_stream['weighted_score']:.2f}). "
        )

        if bottom_stream["score"] < 0.4:
            reasoning += (
                f"The weakest area was {bottom_stream['stream_name']} "
                f"(score: {bottom_stream['score']:.2f}), which lowered the overall assessment."
            )

        return reasoning

    def _identify_strengths(self, stream_scores, raw_results) -> list[str]:
        """Identify candidate's strong areas based on stream analysis."""
        strengths = []
        for ss in stream_scores:
            if ss["score"] >= 0.7:
                strengths.append(f"Strong performance in {ss['stream_name']} ({ss['score']:.0%})")

        # Specific signals
        facial = raw_results.get("facial_emotion", {})
        if facial.get("emotion_distribution", {}).get("happy", 0) > 0.3:
            strengths.append("Positive and engaging facial expressions throughout the interview")

        speech = raw_results.get("speech_emotion", {})
        if speech.get("dominant_emotion") == "positive":
            strengths.append("Confident and positive vocal tone")

        transcript = raw_results.get("transcript", {})
        if transcript.get("word_count", 0) > 150:
            strengths.append("Articulate and detailed verbal responses")

        return strengths

    def _identify_improvements(self, stream_scores, raw_results) -> list[str]:
        """Identify areas for improvement (used in candidate rejection feedback)."""
        improvements = []
        for ss in stream_scores:
            if ss["score"] < 0.4:
                if ss["stream_name"] == "Resume Analysis":
                    improvements.append("Consider highlighting more relevant technical skills and experience on your resume")
                elif ss["stream_name"] == "Speech Transcript":
                    improvements.append("Try to provide more detailed responses during interviews to better showcase your knowledge")
                elif ss["stream_name"] == "Speech Emotion":
                    improvements.append("Practice speaking with more confidence and enthusiasm during interviews")
                elif ss["stream_name"] == "Facial Emotion":
                    improvements.append("Maintaining positive facial expressions can help convey engagement and enthusiasm")

        transcript = raw_results.get("transcript", {})
        if transcript.get("word_count", 0) < 50:
            improvements.append("Expanding your answers with specific examples and details would strengthen your interview performance")

        return improvements

    def _compute_confidence(self, stream_scores) -> float:
        """
        Compute confidence level of the overall assessment.
        Higher when streams agree, lower when they diverge.
        """
        scores = [ss["score"] for ss in stream_scores]
        if not scores:
            return 0.0

        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # Low variance = high agreement = high confidence
        # Variance of 0 → confidence 1.0, variance of 0.25 → confidence ~0.5
        confidence = max(0.0, 1.0 - variance * 4)
        return round(confidence, 3)
