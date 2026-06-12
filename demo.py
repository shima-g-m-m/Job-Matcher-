#!/usr/bin/env python3
"""
demo.py — Offline Demo Mode
============================
Runs the full pipeline using a generated sample dataset.
No internet connection or API key required.

Great for:
  - Testing the system works before you have API keys
  - Demonstrating the system to others
  - Development and debugging

Usage:
    python demo.py --cv uploads/my_cv.pdf
    python demo.py --cv uploads/my_cv.pdf --top 5
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cv_reader.reader  import read_cv
from cv_reader.parser  import parse_local
from jobs.dataset_loader import DatasetLoader
from matching.engine   import rank_jobs
from evaluation        import full_report
from utils.display     import (print_banner, print_profile, print_stats,
                                print_results, sub, ok, warn, export_results)


def run_demo(cv_path: str, top_n: int = 10, export: str = None):

    print_banner()
    print("  ⚡  DEMO MODE — using sample dataset (no internet needed)\n")

    # 1. Read CV
    sub("Reading CV")
    cv_text = read_cv(cv_path)
    ok(f"Loaded {len(cv_text):,} characters")

    # 2. Parse CV (local — no API key)
    sub("Parsing CV (local mode)")
    profile = parse_local(cv_text)
    ok("Profile built")
    print_profile(profile)

    # 3. Generate sample dataset
    sub("Generating sample job dataset")
    loader = DatasetLoader()
    jobs = loader.create_sample(n=150, output_path="outputs/demo_sample_jobs.json")
    ok(f"Generated {len(jobs)} sample jobs")

    # 4. Score & rank
    sub("Scoring and ranking")
    ranked = rank_jobs(profile, jobs)
    ok(f"Ranked {len(ranked)} jobs")

    # 5. Display
    print_stats(ranked, len(jobs))
    print_results(ranked, top_n=top_n)

    # 6. Evaluate
    sub("Evaluation Report")
    report = full_report(ranked, profile.get("skills_hard", []))
    dist = report["score_distribution"]
    print(f"  Mean score  : {dist['mean']}/100")
    print(f"  Top score   : {dist['max']}/100")
    print(f"  Strong (≥65): {dist['strong_matches_65plus']} jobs")
    print(f"  Good   (≥40): {dist['good_matches_40plus']} jobs")

    # 7. Export
    if export:
        export_results(profile, ranked, export, top_n=top_n)

    print(f"\n\033[1m\033[92m  ✅  Demo complete!\033[0m")
    print(f"  Sample dataset saved → outputs/demo_sample_jobs.json")
    print(f"\n  To run with LIVE jobs:\n"
          f"    pip install sentence-transformers  # for semantic matching\n"
          f"    python run.py --cv {cv_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline demo — no internet needed")
    parser.add_argument("--cv",     required=True, help="Path to your CV (PDF/DOCX/TXT)")
    parser.add_argument("--top",    type=int, default=10, help="Top N matches to show")
    parser.add_argument("--export", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.cv):
        print(f"\n  ✗  CV not found: {args.cv}")
        sys.exit(1)

    run_demo(args.cv, top_n=args.top, export=args.export)
