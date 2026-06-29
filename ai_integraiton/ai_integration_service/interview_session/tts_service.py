"""Piper TTS wrapper with content-hash caching. Handles multiple Piper API versions."""
import os, hashlib, logging, wave
from pathlib import Path
log = logging.getLogger(__name__)

# Global model cache
_model_cache = {}


class TTSService:
    def __init__(self, model_dir="./models/tts", cache_dir="./tts_cache",
                 tr_model="tr_TR-dfki-medium", en_model="en_US-lessac-medium"):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models = {"tr": tr_model, "en": en_model}

    def _key(self, text: str, lang: str) -> str:
        return hashlib.sha256(f"{lang}:{text}".encode("utf-8")).hexdigest()[:32]

    def cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.wav"

    def _get_voice(self, lang: str):
        if lang in _model_cache:
            return _model_cache[lang]

        from piper import PiperVoice
        model_name = self.models[lang]
        model_path = self.model_dir / f"{model_name}.onnx"
        config_path = self.model_dir / f"{model_name}.onnx.json"

        if not model_path.exists():
            log.error(f"Piper model not found: {model_path}")
            return None

        log.info(f"Loading Piper voice: {model_name}")
        voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        _model_cache[lang] = voice
        log.info(f"Piper voice ready: {model_name}")
        return voice

    def synthesize(self, text: str, lang: str = "tr") -> str:
        """Generate WAV for text. Returns cache key."""
        lang = "tr" if lang.lower().startswith("tr") else "en"
        key = self._key(text, lang)
        out = self.cache_path(key)

        if out.exists() and out.stat().st_size > 0:
            return key

        voice = self._get_voice(lang)
        if voice is None:
            return ""

        try:
            self._write_wav(voice, text, out)
            log.info(f"TTS generated: {out.name} ({out.stat().st_size} bytes)")
            
            # Upload to MinIO
            from models.minio_client import MinioClientWrapper
            minio_client = MinioClientWrapper()
            minio_url = minio_client.upload_file(str(out), f"tts_audio/{key}.wav")
            if minio_url:
                log.info(f"TTS uploaded to MinIO: {minio_url}")
                
        except Exception as e:
            log.error(f"Piper synthesis error: {e}", exc_info=True)
            if out.exists():
                out.unlink()
            return ""

        return key

    def _write_wav(self, voice, text: str, out_path: Path):
        """Write WAV handling multiple Piper API versions."""
        # Voice config'inden sample rate al (Piper config.json'da var)
        sample_rate = getattr(voice.config, "sample_rate", 22050)

        # API v1: synthesize() generator döndürür
        # API v2: synthesize(text, wav_file) direkt yazar
        # API v3: synthesize_stream_raw() raw bytes generator döndürür

        # Önce generator approach'u dene
        try:
            audio_bytes = bytearray()

            # synthesize_stream_raw varsa onu kullan (en güvenilir)
            if hasattr(voice, "synthesize_stream_raw"):
                for chunk in voice.synthesize_stream_raw(text):
                    audio_bytes.extend(chunk)
            else:
                # synthesize() generator testi
                result = voice.synthesize(text)
                if result is not None and hasattr(result, "__iter__"):
                    for chunk in result:
                        # AudioChunk nesnesi olabilir veya bytes
                        if hasattr(chunk, "audio_int16_bytes"):
                            audio_bytes.extend(chunk.audio_int16_bytes)
                        elif hasattr(chunk, "audio"):
                            audio_bytes.extend(chunk.audio)
                        elif isinstance(chunk, (bytes, bytearray)):
                            audio_bytes.extend(chunk)
                        else:
                            audio_bytes.extend(bytes(chunk))

            if not audio_bytes:
                raise RuntimeError("Piper returned no audio data")

            # WAV header manuel olarak yaz
            with wave.open(str(out_path), "wb") as wav_file:
                wav_file.setnchannels(1)          # mono
                wav_file.setsampwidth(2)          # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(bytes(audio_bytes))

        except TypeError:
            # Eski API: synthesize(text, wav_file) — wav_file parametre alıyor
            with wave.open(str(out_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                voice.synthesize(text, wav_file)

    def get_wav_bytes(self, key: str) -> bytes:
        p = self.cache_path(key)
        return p.read_bytes() if p.exists() else b""