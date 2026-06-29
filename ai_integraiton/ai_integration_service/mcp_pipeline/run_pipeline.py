"""
HR Multimodal Pipeline - Main Entry Point
===========================================
Usage:
    python run_pipeline.py --video interview.mp4 --resume cv.pdf --mode direct
    python run_pipeline.py --video interview.mp4 --mode mcp
    python run_pipeline.py --video interview.mp4
"""

import argparse
import asyncio
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

from agent.orchestrator import run_pipeline_direct, run_pipeline_mcp, load_config


def main():
    parser = argparse.ArgumentParser(
        description="HR Multimodal Pipeline - Candidate Assessment System"
    )
    parser.add_argument("--video", "-v", required=True, help="Interview video file")
    parser.add_argument("--resume", "-r", default=None, help="Resume PDF file (optional)")
    parser.add_argument("--job-req", "-j", default="", help="Job requirements")
    parser.add_argument("--mode", "-m", choices=["direct", "mcp"], default="direct",
                       help="'direct' (local test) or 'mcp' (production)")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file")
    parser.add_argument("--output", "-o", default=None, help="Output directory")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)

    if args.resume and not os.path.exists(args.resume):
        print(f"❌ Resume not found: {args.resume}")
        sys.exit(1)

    config = load_config(args.config)
    if args.output:
        config.setdefault("general", {})["output_dir"] = args.output

    print(f"\n🚀 Starting pipeline ({args.mode} mode)...\n")

    if args.mode == "direct":
        evaluation = run_pipeline_direct(
            video_path=args.video,
            resume_path=args.resume,
            job_requirements=args.job_req,
            config=config
        )
    else:
        evaluation = asyncio.run(run_pipeline_mcp(
            video_path=args.video,
            resume_path=args.resume,
            job_requirements=args.job_req,
            config=config
        ))

    print(f"\n🏆 Pipeline Completed!")
    print(f"   Final Score: {evaluation.final_score:.2f}")
    print(f"   Recommendation: {evaluation.recommendation}")
    print(f"   Duration: {evaluation.processing_time_seconds:.1f}s")


if __name__ == "__main__":
    main()
