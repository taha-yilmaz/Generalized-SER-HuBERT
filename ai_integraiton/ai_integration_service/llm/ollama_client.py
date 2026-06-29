"""Ollama (Gemma 3 4B) client. Drop-in replacement for previous Gemini calls."""
import os, json, logging, requests
log = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model, self.url = model, url

    def generate(self, prompt: str, system: str = None, temperature: float = 0.3,
                 max_tokens: int = 1024, json_mode: bool = False) -> str:
        payload = {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system: payload["system"] = system
        if json_mode: payload["format"] = "json"
        try:
            r = requests.post(self.url, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            log.error(f"Ollama call failed: {e}")
            return ""

    def generate_json(self, prompt: str, system: str = None, **kw) -> dict:
        txt = self.generate(prompt, system=system, json_mode=True, **kw)
        try: return json.loads(txt) if txt else {}
        except json.JSONDecodeError:
            log.warning(f"Ollama returned invalid JSON: {txt[:200]}")
            return {}

_default = OllamaClient()
def generate(prompt, **kw): return _default.generate(prompt, **kw)
def generate_json(prompt, **kw): return _default.generate_json(prompt, **kw)
