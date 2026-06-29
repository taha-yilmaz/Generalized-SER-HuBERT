"""
MCP Server - Stream 2: Speech-to-Text (Whisper)
================================================
Adapted from the project's original implementation.
Model: Whisper Large (OpenAI)
Output: full_text + segments (start, end, text)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SpeechRecognitionServer")

# Global model cache
_whisper_model = None
_current_model_size = None


def get_whisper_model(model_size: str = "large", device: str = "auto"):
    """Load faster-whisper model (4x faster than openai-whisper)."""
    global _whisper_model, _current_model_size

    if _whisper_model is not None and _current_model_size == model_size:
        return _whisper_model

    from faster_whisper import WhisperModel
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # compute_type: GPU'da float16, CPU'da int8 (çok hızlı)
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"📦 faster-whisper '{model_size}' yükleniyor ({device}, {compute_type})...")
    _whisper_model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    _current_model_size = model_size
    print("✅ faster-whisper model ready!")

    return _whisper_model


@mcp.tool()
def transcribe_audio(
    audio_path: str,
    model_size: str = "large",
    language: str = "null",
    device: str = "auto"
) -> dict:
    """Transcribe audio using faster-whisper."""
    if not os.path.exists(audio_path):
        return {"error": f"File not found: {audio_path}"}

    model = get_whisper_model(model_size, device)

    print(f"🎙️ Starting transcription: {audio_path}")

    segments_gen, info = model.transcribe(
        audio_path,
        language=language if language else None,
        beam_size=5,
        vad_filter=True,  # sessiz kısımları atla, daha da hızlı
    )

    # segments_gen lazy generator — tükete tüket
    segments_list = []
    full_text_parts = []
    for seg in segments_gen:
        segments_list.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())

    full_text = " ".join(full_text_parts)

    output = {
        "full_text": full_text,
        "detected_language": info.language,
        "segments": segments_list,
        "word_count": len(full_text.split()),
    }

    print(f"✅ Transcription completed: {output['word_count']} words "
          f"({info.language}, {info.duration:.1f}s audio)")
    return output


@mcp.tool()
def score_transcript(transcript_text: str, gemini_api_key: str = "", job_context: str = "") -> dict:
    """
    Scores interview transcript content quality.

    Args:
        transcript_text: Transkript metni
        gemini_api_key: DEPRECATED — unused
        job_context: Job context

    Returns:
        {"content_score": float, "key_topics": list, "summary": str, "word_count": int}
    """
    word_count = len(transcript_text.split())

    # Basit heuristik skor (LLM fail ederse fallback)
    if word_count < 20:
        score = 0.2
    elif word_count < 50:
        score = 0.4
    elif word_count < 200:
        score = 0.7
    else:
        score = 0.85

    result = {"content_score": score, "word_count": word_count, "key_topics": []}

    # LLM-based advanced scoring
    try:
        from llm.ollama_client import generate_json

        prompt = f"""Evaluate this interview response for content quality.

Transcript:
{transcript_text[:3000]}

Job context:
{job_context[:1000] if job_context else "General interview"}

Return ONLY this JSON (no markdown, no explanations):
{{"content_score": 0.0, "key_topics": ["topic"], "summary": "brief evaluation"}}

Guidelines:
- content_score: 0.0 to 1.0 float representing depth, clarity, and relevance
- key_topics: list of main subjects the candidate discussed
- summary: one sentence evaluation"""

        llm_result = generate_json(prompt, temperature=0.1, max_tokens=512)
        if llm_result and "content_score" in llm_result:
            # LLM değerlerini merge et (word_count'u koru)
            try:
                llm_result["content_score"] = float(llm_result["content_score"])
            except (TypeError, ValueError):
                llm_result["content_score"] = score  # heuristik skora fallback
            if not isinstance(llm_result.get("key_topics"), list):
                llm_result["key_topics"] = []
            result.update(llm_result)
    except Exception as e:
        print(f"⚠️ LLM scoring failed: {e}")

    return result


if __name__ == "__main__":
    print("🚀 Speech Recognition MCP Server starting...")
    mcp.run(transport="stdio")
