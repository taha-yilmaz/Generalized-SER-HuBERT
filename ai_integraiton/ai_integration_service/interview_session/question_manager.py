"""Pre-caches TTS for all interview questions and exposes URLs."""
from .tts_service import TTSService

class QuestionManager:
    def __init__(self, tts: TTSService, base_url: str = "/api/v1/interview/tts"):
        self.tts = tts; self.base_url = base_url

    def prepare(self, questions: list) -> list:
        """questions: [{id, text, language, order_index, max_duration?}]"""
        out = []
        for q in questions:
            lang = q.get("language", "tr")
            key = self.tts.synthesize(q["text"], lang)
            out.append({
                "id": q.get("id"),
                "order_index": q.get("order_index", 0),
                "text": q["text"],
                "language": lang,
                "max_duration": q.get("max_duration", 120),
                "audio_url": f"{self.base_url}/{key}" if key else None,
                "audio_key": key,
            })
        return sorted(out, key=lambda x: x["order_index"])
