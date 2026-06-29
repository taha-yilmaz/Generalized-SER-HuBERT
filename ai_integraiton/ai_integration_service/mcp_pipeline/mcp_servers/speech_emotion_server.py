"""
MCP Server - Stream 3: Speech Emotion Analysis (HuBERT)
========================================================
Adapted from the project's original implementation.
Model: Custom-trained model — architecture preserved.
Etiketler: 0=negative, 1=neutral, 2=positive
Output: Segment-level + dominant emotion + positivity score
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SpeechEmotionServer")

# Global model cache
_ser_model = None
_ser_extractor = None
_ser_device = None


def get_ser_model(model_dir: str, device: str = "auto"):
    """
    
    model_dir: HuBERT_SER directory (config.json + model.safetensors + preprocessor_config.json)
    """
    global _ser_model, _ser_extractor, _ser_device

    if _ser_model is not None:
        return _ser_model, _ser_extractor, _ser_device

    import torch
    from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification

    # Device selection logic (mps > cuda > cpu)
    if device == "auto":
        if torch.backends.mps.is_available():
            _ser_device = torch.device("mps")
        elif torch.cuda.is_available():
            _ser_device = torch.device("cuda")
        else:
            _ser_device = torch.device("cpu")
    else:
        _ser_device = torch.device(device)

    print(f"📦 HuBERT SER model loading: {model_dir} ({_ser_device})")

    # 
    _ser_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    _ser_model = HubertForSequenceClassification.from_pretrained(model_dir)
    _ser_model.to(_ser_device)
    _ser_model.eval()

    print("✅ HuBERT SER model ready! (Custom-trained model — architecture preserved.")
    return _ser_model, _ser_extractor, _ser_device


def predict_chunk(audio_chunk: np.ndarray, model, extractor, device) -> tuple:
    """
    
    adapted to work on numpy arrays.

    Returns:
        (label, confidence)  → ("positive", 0.87)
    """
    import torch
    import torch.nn.functional as F

    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    inputs = extractor(
        audio_chunk,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred_id = np.argmax(scores)
    confidence = float(scores[pred_id])

    return id2label[pred_id], confidence, {id2label[i]: float(scores[i]) for i in range(3)}


@mcp.tool()
def analyze_speech_emotion(
    audio_path: str,
    model_dir: str = "./models/HuBERT_SER",
    chunk_duration: float = 5.0,
    device: str = "auto"
) -> dict:
    """
    Audio filendaki ses duygusunu chunk bazlı analyzes.
    

    Args:
        audio_path: WAV audio file path (16kHz)
        model_dir: HuBERT_SER model directory path
        chunk_duration: Audio chunk duration (seconds)
        device: "auto", "cuda", "mps", "cpu"

    Returns:
        SpeechEmotionResult: segments + dominant + distribution + positivity_score
    """
    import librosa

    model, extractor, dev = get_ser_model(model_dir, device)

    # Load audio at 16kHz
    print(f"🎵 Starting speech emotion analysis: {audio_path}")
    speech, sr = librosa.load(audio_path, sr=16000)

    total_duration = len(speech) / sr
    chunk_samples = int(chunk_duration * sr)

    segments = []
    all_emotions = []

    # Chunk-based analysis
    for start_idx in range(0, len(speech), chunk_samples):
        end_idx = min(start_idx + chunk_samples, len(speech))
        chunk = speech[start_idx:end_idx]

        # Skip very short chunks (< 0.5 seconds)
        if len(chunk) < sr * 0.5:
            continue

        start_sec = start_idx / sr
        end_sec = end_idx / sr

        label, confidence, probs = predict_chunk(chunk, model, extractor, dev)

        segments.append({
            "start_time": round(start_sec, 2),
            "end_time": round(end_sec, 2),
            "emotion": label,
            "confidence": round(confidence, 3)
        })
        all_emotions.append(label)

    # Overall distribution
    total = len(all_emotions) if all_emotions else 1
    distribution = {
        "positive": all_emotions.count("positive") / total,
        "neutral": all_emotions.count("neutral") / total,
        "negative": all_emotions.count("negative") / total,
    }
    dominant = max(distribution, key=distribution.get) if all_emotions else "neutral"

    # Positivity score
    positivity = (
        distribution["positive"] * 1.0 +
        distribution["neutral"] * 0.5 +
        distribution["negative"] * 0.1
    )

    result = {
        "segments": segments,
        "dominant_emotion": dominant,
        "emotion_distribution": {k: round(v, 3) for k, v in distribution.items()},
        "positivity_score": round(positivity, 3),
        "duration_seconds": round(total_duration, 2)
    }

    print(f"✅ Speech emotion analysis completed: Dominant={dominant}, Positivity={positivity:.2f}")
    return result


if __name__ == "__main__":
    print("🚀 Speech Emotion MCP Server starting...")
    mcp.run(transport="stdio")
