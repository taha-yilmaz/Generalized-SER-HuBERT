"""
Orchestrator Agent
==================
Main control flow:
  INPUT DATA → Media Splitter → 4 Stream → FUSION → RESULT

Two modes:
  A) direct - Imports server code directly (development / local)
  B) mcp    - Runs servers via subprocess + MCP Client (production)

Outputs (under pipeline_output/):
  - candidate_evaluation.json  (final scores)
  - raw_stream_results.json    (raw output from each stream)
  - hr_report.txt              (readable HR report)
  - interview_transcript.txt   (interview transcript)
  - analyzed_video.mp4         (processed video: bounding box + emotion + FPS)
  - facial_emotions.csv        (frame-level emotion log)
  - facial_emotions_summary.csv (minute-level summary)
"""

import sys, os, json, time, shutil, asyncio, yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.fusion import compute_integrated_score, get_weights
from agent.qa_pipeline import run_qa_analysis
from utils.media_splitter import split_video, split_stereo_channels


def load_config(config_path: str = "config.yaml") -> dict:
    config_file = Path(__file__).parent.parent / config_path
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    print(f"⚠️ Config not found: {config_file}, using defaults.")
    return {}


# ╔══════════════════════════════════════════════════════════╗
# ║   MOD A: DIRECT CALL (Development / Local)               ║
# ╚══════════════════════════════════════════════════════════╝

def run_pipeline_direct(
    video_path: str,
    resume_path: str | None = None,
    job_requirements: str = "",
    config: dict | None = None,
    resume_parsed_data: dict | None = None,
    interview_type: str = "VIDEO",
    questions: list | None = None,
    timestamps: list | None = None,
    has_prior_video: bool = False,
) -> dict:
    """
    Run pipeline by directly importing server code.
    For development and debugging.
    """
    if config is None:
        config = load_config()

    start_time = time.time()
    output_dir = config.get("general", {}).get("output_dir", "./pipeline_output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  🎭 HR Multimodal Pipeline - Direct Mode")
    print("=" * 60)

    results = {}

    # ─── STEP 1: Media Split ───
    print("\n📦 Step 1: Media Splitting...")
    media_dir = os.path.join(output_dir, "media")
    media_info = split_video(video_path, media_dir)
    audio_path = media_info.get("audio_path")

    # For LIVE interviews, also try to split the stereo channels (HR=left, Candidate=right)
    interview_type_norm = (interview_type or "VIDEO").upper()
    stereo_info = {"is_stereo": False, "left_path": None, "right_path": None}
    if interview_type_norm == "LIVE":
        stereo_info = split_stereo_channels(video_path, media_dir)
        if not stereo_info.get("is_stereo"):
            print("ℹ️ LIVE recording is mono — falling back to LLM content-based diarization")

    # ─── STEP 2: Run Streams ───

    # --- Stream 1: Resume ---
    if resume_parsed_data or resume_path:
        print("\n📄 Stream 1: Resume Analysis...")
        from mcp_servers.resume_server import parse_resume, score_resume

        gemini_key = config.get("resume", {}).get("gemini_api_key", "")
        gemini_model = config.get("resume", {}).get("gemini_model", "gemini-2.5-flash")

        if resume_parsed_data:
            # Reuse cached parse result — skip the slow LLM extraction call
            print("   ✅ Using pre-parsed CV data (skipping parse_resume)")
            resume_data = dict(resume_parsed_data)
        else:
            resume_data = parse_resume(
                pdf_path=resume_path,
                gemini_api_key=gemini_key,
                model_name=gemini_model
            )

        if job_requirements and not resume_data.get("error"):
            score_data = score_resume(
                resume_data=resume_data,
                job_requirements=job_requirements,
                gemini_api_key=gemini_key
            )
            resume_data["relevance_score"] = score_data.get("relevance_score", 0.5)
            resume_data["strengths"] = score_data.get("strengths", [])
            resume_data["gaps"] = score_data.get("gaps", [])

        results["resume"] = resume_data

    # --- Stream 2: Speech Recognition ---
    if audio_path:
        print("\n🎙️ Stream 2: Speech Recognition (Whisper)...")
        from mcp_servers.speech_recognition_server import transcribe_audio, score_transcript

        stt_config = config.get("speech_recognition", {})
        transcript_data = transcribe_audio(
            audio_path=audio_path,
            model_size=stt_config.get("model_size", "large"),
            language=stt_config.get("language", "tr"),
        )

        if not transcript_data.get("error"):
            # Content scoring
            score_data = score_transcript(
                transcript_text=transcript_data.get("full_text", ""),
                gemini_api_key=config.get("resume", {}).get("gemini_api_key", ""),
                job_context=job_requirements
            )
            transcript_data["content_score"] = score_data.get("content_score", 0.5)

        results["transcript"] = transcript_data

        # ── Q&A Analysis (per-question LLM evaluation) ──
        print("\n💬 Q&A: Per-question analysis (gemma4:e4b)...")
        dual_channel = None
        if interview_type_norm == "LIVE" and stereo_info.get("is_stereo"):
            print("   🎧 Transcribing HR channel (left)...")
            hr_transcript = transcribe_audio(
                audio_path=stereo_info["left_path"],
                model_size=stt_config.get("model_size", "large"),
                language=stt_config.get("language", "tr"),
            )
            print("   🎧 Transcribing Candidate channel (right)...")
            cand_transcript = transcribe_audio(
                audio_path=stereo_info["right_path"],
                model_size=stt_config.get("model_size", "large"),
                language=stt_config.get("language", "tr"),
            )
            dual_channel = (
                hr_transcript.get("segments", []) if not hr_transcript.get("error") else [],
                cand_transcript.get("segments", []) if not cand_transcript.get("error") else [],
            )

        resume_summary = None
        if results.get("resume"):
            resume_summary = (results["resume"].get("raw_text") or "")[:1500] or None

        try:
            qa_result = run_qa_analysis(
                interview_type=interview_type_norm,
                transcript_segments=transcript_data.get("segments", []),
                questions=questions,
                timestamps=timestamps,
                job_context=job_requirements,
                candidate_resume_summary=resume_summary,
                dual_channel_transcripts=dual_channel,
            )
            results["qa"] = qa_result
            print(f"   ✅ Q&A analysis: {len(qa_result.get('per_question_analysis', []))} pairs, "
                  f"avg quality={qa_result.get('avg_quality_score', 0):.2f}")
        except Exception as e:
            print(f"   ⚠️ Q&A analysis failed: {e}")
            results["qa"] = {
                "per_question_analysis": [],
                "dialogue_transcript": "",
                "dialogue_turns": [],
                "question_pairing_method": None,
                "avg_quality_score": 0.0,
            }

    # --- Stream 3: Speech Emotion ---
    if audio_path:
        print("\n🎵 Stream 3: Speech Emotion (HuBERT)...")
        from mcp_servers.speech_emotion_server import analyze_speech_emotion

        ser_config = config.get("speech_emotion", {})
        speech_emo_data = analyze_speech_emotion(
            audio_path=audio_path,
            model_dir=ser_config.get("model_dir", "./models/HuBERT_SER"),
            chunk_duration=ser_config.get("chunk_duration_sec", 5.0),
        )
        results["speech_emotion"] = speech_emo_data

    # --- Stream 4: Facial Emotion (+ Processed Video Output) ---
    print("\n🎭 Stream 4: Facial Emotion Analysis...")
    from mcp_servers.facial_emotion_server import analyze_facial_emotions

    fer_config = config.get("facial_emotion", {})
    processed_video_path = media_info["video_path"]  # MP4 after WebM conversion
    video_basename = os.path.splitext(os.path.basename(processed_video_path))[0]

    facial_data = analyze_facial_emotions(
        video_path=processed_video_path,
        fer_model_path=fer_config.get("fer_model_path", ""),
        yolo_model_path=fer_config.get("yolo_model_path", "yolo11n-face.pt"),
        output_video_path=os.path.join(output_dir, f"{video_basename}_analyzed.mp4"),
        output_csv_path=os.path.join(output_dir, f"{video_basename}_emotions.csv"),
        emotion_classes=fer_config.get("emotion_classes"),
        face_conf_threshold=fer_config.get("face_conf_threshold", 0.45),
        smoothing_window=fer_config.get("smoothing_window", 5),
        process_every_n=fer_config.get("process_every_n_frames", 1),
    )
    results["facial_emotion"] = facial_data

    # ─── STEP 3: Fusion ───
    print("\n🔗 Step 3: Fusion & Scoring...")
    candidate_name = ""
    if results.get("resume", {}).get("full_name"):
        candidate_name = results["resume"]["full_name"]

    evaluation = compute_integrated_score(
        resume_data=results.get("resume"),
        transcript_data=results.get("transcript"),
        speech_emotion_data=results.get("speech_emotion"),
        facial_emotion_data=results.get("facial_emotion"),
        weights=get_weights(interview_type_norm, has_prior_video=has_prior_video),
        candidate_name=candidate_name
    )
    evaluation.processing_time_seconds = round(time.time() - start_time, 2)

    # Save results
    _save_all_results(evaluation, results, output_dir)

    return evaluation


# ╔══════════════════════════════════════════════════════════╗
# ║   MOD B: MCP CLIENT (Production)                         ║
# ╚══════════════════════════════════════════════════════════╝

async def run_pipeline_mcp(
    video_path: str,
    resume_path: str | None = None,
    job_requirements: str = "",
    config: dict | None = None,
    resume_parsed_data: dict | None = None,
) -> dict:
    """
    Connect to servers via MCP Client protocol.
    Each server runs as a subprocess (stdio transport).
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    if config is None:
        config = load_config()

    start_time = time.time()
    output_dir = config.get("general", {}).get("output_dir", "./pipeline_output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  🎭 HR Multimodal Pipeline - MCP Mode")
    print("=" * 60)

    # Media split
    print("\n📦 Step 1: Media Splitting...")
    media_info = split_video(video_path, os.path.join(output_dir, "media"))
    audio_path = media_info.get("audio_path")

    results = {}
    server_dir = Path(__file__).parent.parent / "mcp_servers"

    # Helper: MCP tool call
    async def call_mcp_tool(script_name, tool_name, arguments):
        params = StdioServerParameters(
            command="python",
            args=[str(server_dir / script_name)]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                if result.content:
                    return json.loads(result.content[0].text)
                return {}

    # Stream 1: Resume
    if resume_parsed_data or resume_path:
        print("\n📄 Stream 1: Resume Analysis...")
        if resume_parsed_data:
            print("   ✅ Using pre-parsed CV data (skipping parse_resume)")
            resume_data = dict(resume_parsed_data)
        else:
            resume_data = await call_mcp_tool("resume_server.py", "parse_resume", {
                "pdf_path": resume_path,
                "gemini_api_key": config.get("resume", {}).get("gemini_api_key", ""),
                "model_name": config.get("resume", {}).get("gemini_model", "gemini-2.5-flash")
            })
        if job_requirements and not resume_data.get("error"):
            score_data = await call_mcp_tool("resume_server.py", "score_resume", {
                "resume_data": resume_data,
                "job_requirements": job_requirements,
                "gemini_api_key": config.get("resume", {}).get("gemini_api_key", "")
            })
            resume_data["relevance_score"] = score_data.get("relevance_score", 0.5)
        results["resume"] = resume_data

    # Stream 2: STT
    if audio_path:
        print("\n🎙️ Stream 2: Speech Recognition...")
        stt_cfg = config.get("speech_recognition", {})
        transcript_data = await call_mcp_tool("speech_recognition_server.py", "transcribe_audio", {
            "audio_path": audio_path,
            "model_size": stt_cfg.get("model_size", "large"),
            "language": stt_cfg.get("language", "tr"),
        })
        results["transcript"] = transcript_data

    # Stream 3: SER
    if audio_path:
        print("\n🎵 Stream 3: Speech Emotion...")
        ser_cfg = config.get("speech_emotion", {})
        speech_emo = await call_mcp_tool("speech_emotion_server.py", "analyze_speech_emotion", {
            "audio_path": audio_path,
            "model_dir": ser_cfg.get("model_dir", "./models/HuBERT_SER"),
            "chunk_duration": ser_cfg.get("chunk_duration_sec", 5.0),
        })
        results["speech_emotion"] = speech_emo

    # Stream 4: FER (+ Processed Video)
    print("\n🎭 Stream 4: Facial Emotion...")
    fer_cfg = config.get("facial_emotion", {})
    processed_video_path = media_info["video_path"]  # MP4 after WebM conversion
    video_basename = os.path.splitext(os.path.basename(processed_video_path))[0]
    facial_data = await call_mcp_tool("facial_emotion_server.py", "analyze_facial_emotions", {
        "video_path": processed_video_path,
        "fer_model_path": fer_cfg.get("fer_model_path", ""),
        "yolo_model_path": fer_cfg.get("yolo_model_path", "yolo11n-face.pt"),
        "output_video_path": os.path.join(output_dir, f"{video_basename}_analyzed.mp4"),
        "output_csv_path": os.path.join(output_dir, f"{video_basename}_emotions.csv"),
        "process_every_n": fer_cfg.get("process_every_n_frames", 1),
    })
    results["facial_emotion"] = facial_data

    # Fusion
    print("\n🔗 Step 3: Fusion & Scoring...")
    candidate_name = results.get("resume", {}).get("full_name", "")
    evaluation = compute_integrated_score(
        resume_data=results.get("resume"),
        transcript_data=results.get("transcript"),
        speech_emotion_data=results.get("speech_emotion"),
        facial_emotion_data=results.get("facial_emotion"),
        weights=config.get("fusion", {}).get("weights"),
        candidate_name=candidate_name
    )
    evaluation.processing_time_seconds = round(time.time() - start_time, 2)

    _save_all_results(evaluation, results, output_dir)
    return evaluation


# ╔══════════════════════════════════════════════════════════╗
# ║   SAVE RESULTS                                         ║
# ╚══════════════════════════════════════════════════════════╝

def _save_all_results(evaluation, raw_results: dict, output_dir: str):
    """Save all results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Final evaluation JSON
    eval_path = os.path.join(output_dir, "candidate_evaluation.json")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation.model_dump(), f, ensure_ascii=False, indent=2, default=str)
    print(f"   📊 Evaluation: {eval_path}")

    # 2. Raw stream results JSON
    raw_path = os.path.join(output_dir, "raw_stream_results.json")
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"   📁 Raw results: {raw_path}")

    # 3. HR report TXT
    report_path = os.path.join(output_dir, "hr_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(evaluation.summary)
        f.write(f"\n\n{'='*50}\n")
        f.write(f"Processing time: {evaluation.processing_time_seconds:.1f} seconds\n")
        f.write(f"Pipeline version: {evaluation.pipeline_version}\n")
    print(f"   📝 HR Report: {report_path}")

    # 4. Transcript TXT
    if raw_results.get("transcript", {}).get("full_text"):
        t_path = os.path.join(output_dir, "interview_transcript.txt")
        with open(t_path, 'w', encoding='utf-8') as f:
            f.write(raw_results["transcript"]["full_text"])
        print(f"   🎙️ Transcript: {t_path}")

    # 5. Report processed video and CSV paths
    facial = raw_results.get("facial_emotion", {})
    if facial.get("output_video_path"):
        print(f"   📹 Processed Video: {facial['output_video_path']}")
    if facial.get("output_csv_path"):
        print(f"   📊 Emotion CSV: {facial['output_csv_path']}")
    if facial.get("output_csv_summary_path"):
        print(f"   📊 Minute Summary: {facial['output_csv_summary_path']}")

    print(f"\n{'='*60}")
    print(evaluation.summary)
    print(f"{'='*60}")
