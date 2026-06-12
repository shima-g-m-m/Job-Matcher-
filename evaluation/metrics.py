"""
evaluation/metrics.py
---------------------
Evaluation framework for the job matching system.

Metrics:
  - Skill extraction: precision, recall, F1
  - Job matching:     precision, recall, F1, accuracy, confusion matrix
  - Ranking quality:  Precision@K, MRR, NDCG
  - Score accuracy:   MAE, RMSE, Pearson correlation
"""

import json
import math
import numpy as np
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Skill extraction evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_skill_extraction(extracted: list[str], ground_truth: list[str]) -> dict:
    """Compare extracted skills vs ground truth."""
    ext = set(s.lower() for s in extracted)
    gt  = set(s.lower() for s in ground_truth)

    tp = len(ext & gt)
    fp = len(ext - gt)
    fn = len(gt - ext)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision":      round(precision, 3),
        "recall":         round(recall, 3),
        "f1":             round(f1, 3),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "matched_skills": sorted(ext & gt),
        "extra_skills":   sorted(ext - gt),
        "missed_skills":  sorted(gt - ext),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Job matching binary evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_matching(ranked: list[dict], true_job_ids: list,
                      score_threshold: float = 50.0) -> dict:
    """
    Evaluate job matching quality against known-good job IDs.

    Args:
        ranked:          Output of rank_jobs() — list of scored dicts
        true_job_ids:    List of job URLs or titles that are genuinely good matches
        score_threshold: Score cutoff for predicted positives
    """
    predicted_pos = {r["url"] or r["title"] for r in ranked if r["score"] >= score_threshold}
    actual_pos    = set(str(j) for j in true_job_ids)

    tp = len(predicted_pos & actual_pos)
    fp = len(predicted_pos - actual_pos)
    fn = len(actual_pos - predicted_pos)
    tn = max(0, len(ranked) - tp - fp - fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(ranked) if ranked else 0.0

    return {
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
        "accuracy":  round(accuracy, 3),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ranking quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(ranked: list[dict], relevant_ids: list, k: int = 10) -> float:
    """Fraction of top-K results that are relevant."""
    rel = set(str(r) for r in relevant_ids)
    hits = sum(1 for r in ranked[:k] if (r.get("url") or r.get("title")) in rel)
    return round(hits / k, 3)


def mean_reciprocal_rank(ranked: list[dict], relevant_ids: list) -> float:
    """MRR — 1 / rank of first relevant result."""
    rel = set(str(r) for r in relevant_ids)
    for i, r in enumerate(ranked, 1):
        if (r.get("url") or r.get("title")) in rel:
            return round(1.0 / i, 3)
    return 0.0


def ndcg_at_k(ranked: list[dict], relevant_ids: list, k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain @ K."""
    rel = set(str(r) for r in relevant_ids)

    def dcg(items):
        return sum(
            (1.0 / math.log2(i + 2))
            for i, r in enumerate(items)
            if (r.get("url") or r.get("title")) in rel
        )

    ideal = sorted(ranked, key=lambda x: (x.get("url") or x.get("title")) in rel,
                   reverse=True)
    idcg = dcg(ideal[:k])
    return round(dcg(ranked[:k]) / idcg, 3) if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Score accuracy (when ground-truth scores are known)
# ─────────────────────────────────────────────────────────────────────────────

def score_accuracy(predicted: list[float], actual: list[float]) -> dict:
    """MAE, RMSE, Pearson correlation between predicted and actual scores."""
    p = np.array(predicted); a = np.array(actual)
    mae  = float(np.mean(np.abs(p - a)))
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    if len(p) > 1:
        corr = float(np.corrcoef(p, a)[0, 1])
    else:
        corr = 0.0
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "correlation": round(corr, 3)}


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def full_report(ranked: list[dict],
                extracted_skills: list[str],
                ground_truth_skills: Optional[list[str]] = None,
                relevant_job_ids: Optional[list] = None,
                k: int = 10) -> dict:
    """
    Generate a complete evaluation report.

    Args:
        ranked:               Output of rank_jobs()
        extracted_skills:     Skills extracted from CV
        ground_truth_skills:  Known-correct skills (optional)
        relevant_job_ids:     Known-good job URLs/titles (optional)
        k:                    K for Precision@K and NDCG@K
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_jobs_scored": len(ranked),
    }

    # Score distribution
    scores = [r["score"] for r in ranked]
    report["score_distribution"] = {
        "mean":    round(np.mean(scores), 1) if scores else 0,
        "median":  round(float(np.median(scores)), 1) if scores else 0,
        "max":     max(scores) if scores else 0,
        "min":     min(scores) if scores else 0,
        "strong_matches_65plus": sum(1 for s in scores if s >= 65),
        "good_matches_40plus":   sum(1 for s in scores if s >= 40),
    }

    # Skill extraction
    if ground_truth_skills:
        report["skill_extraction"] = evaluate_skill_extraction(
            extracted_skills, ground_truth_skills)

    # Matching & ranking
    if relevant_job_ids:
        report["matching"]      = evaluate_matching(ranked, relevant_job_ids)
        report["precision_at_k"] = precision_at_k(ranked, relevant_job_ids, k)
        report["mrr"]            = mean_reciprocal_rank(ranked, relevant_job_ids)
        report["ndcg_at_k"]      = ndcg_at_k(ranked, relevant_job_ids, k)

    return report


def save_report(report: dict, path: str):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Evaluation report saved → {path}")
