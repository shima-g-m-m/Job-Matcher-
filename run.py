#!/usr/bin/env python3
"""
run.py — Job Matcher System
====================================
Main terminal entry point. Runs fully locally — no API key needed.

Usage:
    python run.py --cv uploads/my_cv.pdf
    python run.py --cv uploads/my_cv.pdf --country gb --remote --top 15
    python run.py --cv uploads/my_cv.pdf --export results.json
    python run.py --cv uploads/my_cv.pdf --no-rag
"""

import sys
import argparse
import textwrap
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import JobMatchingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Job Matcher System — CV analysis + live jobs + ranking (fully local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          Examples:
            python run.py --cv uploads/my_cv.pdf
            python run.py --cv uploads/my_cv.pdf --country gb --remote --top 15
            python run.py --cv uploads/my_cv.pdf --export results.json

          Optional env vars (for more job results):
            ADZUNA_APP_ID   — Adzuna API key (free tier at developer.adzuna.com)
            ADZUNA_APP_KEY  — Adzuna API key

          Free job sources (always active, no key needed):
            RemoteOK · The Muse · Remotive · Arbeitnow

          Score guide:
            🟢 65–100  Strong match — apply now
            🟡 40–64   Good match — worth considering
            🔴  0–39   Weak match — skill gap likely
        """)
    )

    parser.add_argument("--cv",      required=True,
                        help="Path to your CV file (PDF, DOCX, or TXT)")
    parser.add_argument("--country", default="us",
                        help="Country code: us / gb / de / au / ca (default: us)")
    parser.add_argument("--remote",  action="store_true",
                        help="Filter for remote jobs only")
    parser.add_argument("--top",     type=int, default=10,
                        help="Number of top matches to display (default: 10)")
    parser.add_argument("--export",  default=None,
                        help="Save full ranked results to this JSON file")
    parser.add_argument("--no-rag",  action="store_true",
                        help="Disable RAG vector index (faster but less accurate)")
    parser.add_argument("--index",   default="outputs/job_index",
                        help="Path prefix for the persistent job vector index")
    parser.add_argument("--eval",    default=None,
                        help="Save evaluation report to this JSON file")
    # Kept for backwards compat — silently ignored
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip cross-encoder re-ranking (faster, skips ~90MB model download)")
    # Kept for backwards compat — silently ignored
    parser.add_argument("--no-claude", action="store_true",
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    if not os.path.exists(args.cv):
        print(f"\n  ✗  CV file not found: {args.cv}")
        sys.exit(1)

    pipeline = JobMatchingPipeline()
    pipeline.run(
        cv_path     = args.cv,
        country     = args.country,
        remote_only = args.remote,
        top_n       = args.top,
        use_rag     = not args.no_rag,
        use_rerank  = not args.no_rerank,
        index_path  = args.index,
        export_path = args.export,
        eval_report = args.eval,
    )


if __name__ == "__main__":
    main()
