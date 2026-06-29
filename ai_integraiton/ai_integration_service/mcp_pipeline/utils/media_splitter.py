"""
Media Splitter - Video to Audio (16 kHz WAV) splitter.
"""

import subprocess
import os
from pathlib import Path
import cv2


def convert_webm_to_mp4(webm_path: str) -> str:
    """Convert a MediaRecorder WebM (VP8/VP9) to H.264 MP4 for reliable OpenCV frame reading."""
    mp4_path = str(Path(webm_path).with_suffix(".mp4"))
    print(f"🔄 Converting WebM → MP4 for reliable frame extraction...")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path,
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
         "-c:a", "copy", mp4_path],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0 or not Path(mp4_path).exists():
        print(f"⚠️ WebM→MP4 conversion failed, using original file: {result.stderr[-200:]}")
        return webm_path
    return mp4_path


def split_video(video_path: str, output_dir: str) -> dict:
    """Extract audio from video and return video metadata."""
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "audio_16khz.wav")

    # Convert WebM to MP4 before anything else so OpenCV reads all frames correctly
    if video_path.lower().endswith(".webm"):
        video_path = convert_webm_to_mp4(video_path)

    # Extract 16kHz mono WAV via ffmpeg
    print("🔊 Extracting audio (16 kHz, mono)...")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             audio_path],
            capture_output=True, text=True, timeout=300
        )
        if not os.path.exists(audio_path):
            audio_path = None
            print("⚠️ No audio track found in video!")
    except FileNotFoundError:
        print("❌ ffmpeg not found! Please install: https://ffmpeg.org/download.html")
        audio_path = None
    except subprocess.TimeoutExpired:
        print("❌ Audio extraction timed out!")
        audio_path = None

    # Video metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30  # guard against bogus WebM FPS metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"✅ Media split completed: {width}x{height} @ {fps:.1f} FPS, {duration_sec:.1f}s")

    return {
        "audio_path": audio_path,
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "width": width,
        "height": height
    }


def probe_audio_channels(media_path: str) -> int:
    """Return the number of audio channels in the given media file (0 if no audio)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels", "-of", "csv=p=0", media_path],
            capture_output=True, text=True, timeout=30,
        )
        out = (result.stdout or "").strip()
        return int(out) if out else 0
    except Exception as e:
        print(f"⚠️ ffprobe failed: {e}")
        return 0


def split_stereo_channels(media_path: str, output_dir: str) -> dict:
    """Split a stereo media file into two mono 16 kHz WAVs (left=HR, right=Candidate).

    Returns {"left_path": ..., "right_path": ..., "is_stereo": bool}.
    If the input has < 2 audio channels, returns {"is_stereo": False, "left_path": None,
    "right_path": None} so the caller can fall back to mono diarization.
    """
    os.makedirs(output_dir, exist_ok=True)
    channels = probe_audio_channels(media_path)
    if channels < 2:
        print(f"ℹ️ Media has {channels} audio channel(s); skipping stereo split")
        return {"left_path": None, "right_path": None, "is_stereo": False}

    left_path = os.path.join(output_dir, "audio_left_16khz.wav")
    right_path = os.path.join(output_dir, "audio_right_16khz.wav")

    # Use pan filter to extract each channel reliably (works even when -map_channel
    # behaves differently across ffmpeg versions and codecs).
    print("🔊 Splitting stereo audio into HR(left) + Candidate(right) channels...")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", media_path,
             "-af", "pan=mono|c0=c0",
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             left_path],
            capture_output=True, text=True, timeout=300, check=False,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", media_path,
             "-af", "pan=mono|c0=c1",
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             right_path],
            capture_output=True, text=True, timeout=300, check=False,
        )
    except FileNotFoundError:
        print("❌ ffmpeg not found while splitting stereo channels.")
        return {"left_path": None, "right_path": None, "is_stereo": False}
    except subprocess.TimeoutExpired:
        print("❌ Stereo split timed out.")
        return {"left_path": None, "right_path": None, "is_stereo": False}

    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        print("⚠️ Stereo channel WAV files were not produced.")
        return {"left_path": None, "right_path": None, "is_stereo": False}

    return {"left_path": left_path, "right_path": right_path, "is_stereo": True}
