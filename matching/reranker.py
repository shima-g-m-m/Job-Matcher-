"""
matching/reranker.py
--------------------
Cross-encoder re-ranking stage.

After sentence-transformers retrieval produces a candidate pool, this module
re-scores every (query, job_description) pair with a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) which reads *both* texts jointly
and produces a more accurate relevance score than the bi-encoder alone.

The re-ranker is applied on top of the existing composite score so it acts
as a final boosting/demoting pass — it does NOT replace the skill/seniority
components, it refines the ordering of the top-N candidates.

Usage
-----
    from matching.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()          # loads model once
    ranked = reranker.rerank(profile, jobs, top_n=50)

The reranker gracefully degrades:
  - If sentence-transformers / cross-encoder is not installed → returns
    jobs unchanged (no exception).
  - If the model cannot be downloaded (no internet) → same graceful fallback.
"""

import logging
from typing import Optional

log = logging.getLogger("job_matcher")

# Model identifier — small, fast, MIT-licensed
_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How much the cross-encoder score influences the final ranking.
# Blended score = (1 - ALPHA) * original_score + ALPHA * cross_encoder_normalised
BLEND_ALPHA = 0.35

# Maximum characters of JD text fed to the cross-encoder (keeps latency low)
_MAX_JD_CHARS = 512


class CrossEncoderReranker:
    """
    Wraps a cross-encoder model and provides a `rerank()` method that
    blends the cross-encoder signal with the existing composite score.
    """

    def __init__(self, model_id: str = _MODEL_ID):
        self._model_id = model_id
        self._model = None          # lazy-loaded on first call
        self._available: Optional[bool] = None   # None = untested yet


    def _load(self) -> bool:
        """Try to load the cross-encoder. Returns True if successful."""
        if self._available is not None:
            return self._available

        try:
            from sentence_transformers import CrossEncoder
            try:
                self._model = CrossEncoder(self._model_id, local_files_only=True)
                log.info(f"Cross-encoder loaded from local cache: {self._model_id}")
            except Exception:
                log.info(f"Downloading cross-encoder: {self._model_id} …")
                self._model = CrossEncoder(self._model_id)
                log.info("Cross-encoder downloaded and ready")
            self._available = True
        except ImportError:
            log.warning(
                "sentence-transformers not installed — cross-encoder re-ranking disabled. "
                "Run: pip install sentence-transformers"
            )
            self._available = False
        except Exception as exc:
            log.warning(f"Could not load cross-encoder ({exc}) — skipping re-ranking")
            self._available = False

        return self._available


    @staticmethod
    def _build_query(profile: dict) -> str:
        """
        Build a short natural-language query from the CV profile.
        This is the 'question' side of the (question, passage) pair.
        """
        parts = []

        title = profile.get("current_title") or ""
        if title:
            parts.append(title)

        skills = profile.get("skills_hard", [])
        if skills:
            parts.append("Skills: " + ", ".join(skills[:12]))

        seniority = profile.get("seniority", "")
        domain = profile.get("domain", "")
        if seniority and domain:
            parts.append(f"{seniority} level {domain} professional")

        preferred = profile.get("preferred_roles", [])
        if preferred:
            parts.append("Looking for: " + ", ".join(preferred[:4]))

        summary = profile.get("summary_for_search", "")
        if summary:
            parts.append(summary)

        return " | ".join(parts)[:400]

    @staticmethod
    def _build_passage(job: dict) -> str:
        """
        Build the passage (document) side from a job dict.
        Truncated to keep cross-encoder latency acceptable.
        """
        title = job.get("title", "")
        desc  = job.get("description", "")
        tags  = " ".join(job.get("tags", []) or [])
        combined = f"{title}. {tags}. {desc}"
        return combined[:_MAX_JD_CHARS]


    def rerank(self, profile: dict, jobs: list, top_n: int = 50) -> list:
        """
        Re-rank `jobs` using the cross-encoder.

        Only the top `top_n` jobs (by existing composite score) are re-scored —
        anything beyond that is unlikely to surface as a final result, so
        spending cross-encoder inference on them is wasteful.

        Returns the full job list with the top-N portion re-ordered.
        """
        if not jobs:
            return jobs

        if not self._load():
            log.info("Cross-encoder unavailable — skipping re-rank")
            return jobs

        candidates = jobs[:top_n]
        tail       = jobs[top_n:]

        query   = self._build_query(profile)
        pairs   = [(query, self._build_passage(j)) for j in candidates]

        try:
            raw_scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            log.warning(f"Cross-encoder inference failed: {exc} — skipping re-rank")
            return jobs

        # Normalise cross-encoder scores to [0, 100]
        import numpy as np
        raw = np.array(raw_scores, dtype=float)
        rmin, rmax = raw.min(), raw.max()
        if rmax > rmin:
            ce_norm = (raw - rmin) / (rmax - rmin) * 100
        else:
            ce_norm = np.full_like(raw, 50.0)

        # Blend: keep original composite score but boost/demote via cross-encoder
        for job, ce in zip(candidates, ce_norm):
            orig = float(job.get("score", 50))
            blended = (1 - BLEND_ALPHA) * orig + BLEND_ALPHA * float(ce)
            job["score_before_rerank"] = orig
            job["score_ce"]            = round(float(ce), 1)
            job["score"]               = round(blended, 1)

        # Re-sort candidates by blended score
        candidates.sort(key=lambda j: j["score"], reverse=True)

        log.info(
            f"Cross-encoder re-ranked {len(candidates)} jobs "
            f"(α={BLEND_ALPHA}, model={self._model_id})"
        )
        return candidates + tail


    def is_available(self) -> bool:
        """Return True if the cross-encoder loaded successfully."""
        return self._load()
