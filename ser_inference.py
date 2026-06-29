"""
Speech Emotion Recognition (SER) Inference Script
=================================================
This script performs Speech Emotion Recognition using a custom fine-tuned 
HuBERT-Large model. It analyzes audio in chunks, extracting features and 
classifying them into three emotion classes: Negative, Neutral, Positive.

The model expects audio input at a 16kHz sampling rate.
"""

import os
import argparse
import numpy as np
import torch
import librosa
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification

# Dictionary mapping model output indices to emotion labels
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

class SERPredictor:
    def __init__(self, model_dir: str, device: str = "auto"):
        """
        Initialize the HuBERT SER Predictor.

        Args:
            model_dir (str): Path to the directory containing the model weights, 
                             config.json, and preprocessor_config.json.
            device (str): Device to run the model on ('auto', 'cuda', 'mps', 'cpu').
        """
        self.model_dir = model_dir
        self.device = self._select_device(device)
        self.extractor = None
        self.model = None
        self._load_model()

    def _select_device(self, device_str: str) -> torch.device:
        """Selects the best available device for computation."""
        if device_str == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    def _load_model(self):
        """Loads the Wav2Vec2 feature extractor and HuBERT model."""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory '{self.model_dir}' not found. "
                "Please download the weights and place them in the correct directory."
            )
        
        print(f"📦 Loading HuBERT SER model from '{self.model_dir}' onto {self.device}...")
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_dir)
        self.model = HubertForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")

    def predict_chunk(self, audio_chunk: np.ndarray) -> tuple:
        """
        Predict emotion for a single chunk of audio.

        Args:
            audio_chunk (np.ndarray): 1D numpy array containing the audio signal (16kHz).

        Returns:
            tuple: (predicted_label, confidence, probabilities_dict)
        """
        inputs = self.extractor(
            audio_chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_id = int(np.argmax(scores))
        confidence = float(scores[pred_id])

        probs_dict = {ID2LABEL[i]: float(scores[i]) for i in range(3)}
        return ID2LABEL[pred_id], confidence, probs_dict

    def analyze_audio(self, audio_path: str, chunk_duration: float = 5.0) -> dict:
        """
        Perform chunk-based emotion analysis on an entire audio file.

        Args:
            audio_path (str): Path to the audio file (.wav, .mp3, etc.).
            chunk_duration (float): Duration of each chunk in seconds.

        Returns:
            dict: Analysis results containing segments, dominant emotion, 
                  emotion distribution, and a positivity score.
        """
        print(f"🎵 Analyzing audio file: {audio_path}")
        
        # Load audio and resample to 16kHz
        speech, sr = librosa.load(audio_path, sr=16000)
        total_duration = len(speech) / sr
        chunk_samples = int(chunk_duration * sr)

        segments = []
        all_emotions = []

        # Process audio in chunks
        for start_idx in range(0, len(speech), chunk_samples):
            end_idx = min(start_idx + chunk_samples, len(speech))
            chunk = speech[start_idx:end_idx]

            # Skip chunks shorter than 0.5 seconds to avoid noisy predictions
            if len(chunk) < sr * 0.5:
                continue

            start_sec = start_idx / sr
            end_sec = end_idx / sr

            label, confidence, _ = self.predict_chunk(chunk)

            segments.append({
                "start_time": round(start_sec, 2),
                "end_time": round(end_sec, 2),
                "emotion": label,
                "confidence": round(confidence, 3)
            })
            all_emotions.append(label)

        # Calculate overall distribution
        total = len(all_emotions) if all_emotions else 1
        distribution = {
            "positive": all_emotions.count("positive") / total,
            "neutral": all_emotions.count("neutral") / total,
            "negative": all_emotions.count("negative") / total,
        }
        
        # Determine dominant emotion
        dominant = max(distribution, key=distribution.get) if all_emotions else "neutral"

        # Calculate positivity score (custom metric)
        # Weighted sum: Positive=1.0, Neutral=0.5, Negative=0.1
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

        return result

def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition using HuBERT")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--model_dir", type=str, default="./models/HuBERT_SER", help="Directory containing model weights")
    parser.add_argument("--chunk_duration", type=float, default=5.0, help="Chunk duration in seconds")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Computation device")
    
    args = parser.parse_args()

    try:
        predictor = SERPredictor(model_dir=args.model_dir, device=args.device)
        result = predictor.analyze_audio(args.audio, chunk_duration=args.chunk_duration)
        
        print("\n=== Analysis Results ===")
        print(f"File: {args.audio} ({result['duration_seconds']}s)")
        print(f"Dominant Emotion: {result['dominant_emotion'].upper()}")
        print(f"Positivity Score: {result['positivity_score']:.2f} / 1.00")
        print("\nEmotion Distribution:")
        for em, val in result["emotion_distribution"].items():
            print(f"  - {em.capitalize()}: {val*100:.1f}%")
            
        print("\nSegments:")
        for seg in result["segments"]:
            print(f"  [{seg['start_time']:05.2f}s - {seg['end_time']:05.2f}s] {seg['emotion'].capitalize()} (Conf: {seg['confidence']:.2f})")
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
