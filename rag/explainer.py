"""
rag/explainer.py
----------------
Generates per-job match explanations.

Two modes:
  1. Groq LLM explainer — human-readable, recruiter-quality narrative
  2. Local skill-diff explainer — fast, offline, no API key needed (fallback)

The Groq explainer is invoked when GROQ_API_KEY is set (or passed explicitly).
It falls back to the local explainer per-job if the API call fails.
"""

import re
import os
import json
import logging
from typing import Optional

log = logging.getLogger("job_matcher")

EXPLAIN_PROMPT = None   # kept for import compatibility


# Groq helpers

GROQ_EXPLAIN_SYSTEM = """You are an expert technical recruiter helping a candidate evaluate job opportunities.
Respond ONLY with a valid JSON object — no markdown fences, no explanation."""

GROQ_EXPLAIN_PROMPT = """A candidate is evaluating this job.  Produce a concise, honest match analysis.

CANDIDATE PROFILE:
- Title: {title}
- Seniority: {seniority} ({years} years experience)
- Hard skills: {skills_hard}
- Soft skills: {skills_soft}
- Key achievements: {achievements}
- Summary: {summary}

JOB:
- Title: {job_title}
- Company: {company}
- Description (first 600 chars): {jd_snippet}
- Composite match score: {score}/100

Return ONLY this JSON (no markdown):
{{
  "match_summary": "2-3 sentence honest assessment of fit",
  "strengths": ["strength 1 with specific evidence", "strength 2", "strength 3"],
  "gaps": ["gap 1 with suggestion", "gap 2"],
  "transferable": ["transferable skill 1", "transferable skill 2"],
  "advice": "1-2 sentence actionable advice for applying to this specific role",
  "apply_confidence": "high | medium | low",
  "estimated_fit_pct": {score}
}}"""


def _call_groq_explain(profile: dict, job: dict, score: int,
                        api_key: str, model: str = "llama3-8b-8192") -> Optional[dict]:
    """Call Groq to generate a job match explanation. Returns dict or None."""
    import urllib.request

    prompt = GROQ_EXPLAIN_PROMPT.format(
        title       = profile.get("current_title", "Unknown"),
        seniority   = profile.get("seniority", "mid"),
        years       = profile.get("years_experience", 0),
        skills_hard = ", ".join(profile.get("skills_hard", [])[:20]),
        skills_soft = ", ".join(profile.get("skills_soft", [])[:6]),
        achievements= "; ".join(profile.get("key_achievements", [])[:3]) or "Not specified",
        summary     = profile.get("summary_for_search", "")[:300],
        job_title   = job.get("title", "Unknown role"),
        company     = job.get("company", "Unknown company"),
        jd_snippet  = (job.get("description", "") or "")[:600],
        score       = score,
    )

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": GROQ_EXPLAIN_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 600,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            raw = data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        log.warning(f"Groq explain API failed for '{job.get('title', '?')}': {exc}")
        return None

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning(f"Groq explain returned invalid JSON: {exc}\nRaw: {raw[:200]}")
        return None

    # Normalise
    result.setdefault("match_summary", f"{score}/100 match.")
    result.setdefault("strengths", [])
    result.setdefault("gaps", [])
    result.setdefault("transferable", [])
    result.setdefault("advice", "Tailor your CV to mirror the job description keywords.")
    result.setdefault("apply_confidence", "medium" if score >= 40 else "low")
    result.setdefault("estimated_fit_pct", score)
    result["_source"] = "groq"
    return result


# Local skill-diff explainer (offline fallback)

def _local_explain(profile: dict, job: dict, score: int) -> dict:
    """Full local explanation — no API needed."""
    from cv_reader.parser import TECH_SKILLS

    cv_skills = set(s.lower() for s in profile.get("skills_hard", []))
    jd_text   = (job.get("description", "") + " " + job.get("title", "")).lower()
    jd_skills = {s for s in TECH_SKILLS
                 if re.search(r'\b' + re.escape(s) + r'\b', jd_text)}

    matched      = sorted(cv_skills & jd_skills)
    missing      = sorted(jd_skills - cv_skills)
    transferable_pool = sorted(cv_skills - jd_skills)

    _noise = {"word", "excel", "arm", "go", "r", "c", "qa",
              "leadership", "mentoring", "prototyping", "sprint",
              "ms office", "api", "agile", "arduino"}
    missing           = [s for s in missing           if s not in _noise]
    transferable_pool = [s for s in transferable_pool if s not in _noise]

    strengths = [f"Has {s} — directly required by this role" for s in matched[:5]]
    if not strengths:
        soft = profile.get("skills_soft", [])
        soft_matches = [s for s in soft if s.lower() in jd_text][:3]
        strengths = ([f"Soft skill '{s}' aligns with role expectations"
                      for s in soft_matches]
                     or ["General technical background may be relevant"])

    gaps = [f"Role requires {s} — not found in CV" for s in missing[:5]]
    if not gaps:
        gaps = ["No major technical skill gaps detected"]

    trans = [f"{s} is transferable to this role" for s in transferable_pool[:4]]

    if missing:
        top2   = " and ".join(missing[:2])
        advice = (f"Learn {top2} to strengthen your application for "
                  f"'{job.get('title', 'this role')}'. "
                  f"Consider online courses or small projects to demonstrate this.")
    elif matched:
        advice = (f"Highlight your {matched[0]} experience prominently "
                  f"in your cover letter for '{job.get('title', 'this role')}'. "
                  f"Quantify your impact where possible.")
    else:
        advice = ("Review the job description carefully and tailor your CV "
                  "to mirror the exact keywords used.")

    if matched and missing:
        summary = (f"{score}/100 match. "
                   f"Strong on: {', '.join(matched[:3])}. "
                   f"Missing: {', '.join(missing[:2])}.")
    elif matched:
        summary = (f"{score}/100 match. "
                   f"Good alignment on: {', '.join(matched[:4])}. "
                   f"Well-suited for this role.")
    else:
        summary = (f"{score}/100 match. "
                   f"Limited keyword overlap — role may be outside your "
                   f"current skill focus.")

    confidence = "high" if score >= 65 else ("medium" if score >= 40 else "low")

    return {
        "match_summary":     summary,
        "strengths":         strengths,
        "gaps":              gaps,
        "transferable":      trans,
        "advice":            advice,
        "apply_confidence":  confidence,
        "estimated_fit_pct": score,
        "_source":           "local",
    }


# RAGExplainer — public API

class RAGExplainer:
    """
    Generates per-job match explanations.

    When a Groq API key is available (GROQ_API_KEY env var or api_key arg),
    each explanation is generated by the LLM for natural, recruiter-quality
    language.  Falls back to local skill-diff analysis on any API failure.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._cache: dict = {}
        self._groq_calls = 0
        self._local_calls = 0

    @property
    def _using_groq(self) -> bool:
        return bool(self._api_key)

    def explain(self, profile: dict, job: dict, score: int = 0) -> dict:
        cache_key = (job.get("url") or job.get("title", "")) + str(score)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._using_groq:
            result = _call_groq_explain(profile, job, score, api_key=self._api_key)
            if result:
                self._groq_calls += 1
                self._cache[cache_key] = result
                return result
            # Groq failed — fall through to local
            log.debug(f"Groq explain fell back to local for: {job.get('title', '?')}")

        result = _local_explain(profile, job, score)
        self._local_calls += 1
        self._cache[cache_key] = result
        return result

    def explain_batch(self, profile: dict, ranked_jobs: list,
                      top_n: int = 10) -> list:
        candidates = [j for j in ranked_jobs if j.get("url")][:top_n]
        if not candidates:
            candidates = ranked_jobs[:top_n]

        mode   = "Groq LLM" if self._using_groq else "local skill-diff"
        total  = len(candidates)
        log.info(f"Generating explanations via {mode} for {total} jobs …")

        for i, job in enumerate(candidates, 1):
            print(f"  [explain] {i}/{total}: {job.get('title','?')[:45]}"
                  + (" 🤖" if self._using_groq else ""))
            job["rag"] = self.explain(profile, job, score=job.get("score", 0))

        if self._groq_calls or self._local_calls:
            log.info(
                f"Explanations complete — "
                f"Groq: {self._groq_calls}, local: {self._local_calls}"
            )

        return ranked_jobs


def print_best_job_recommendation(ranked: list, profile: dict):
    """Delegates to the main display module for consistent output."""
    from utils.display import print_best_match
    print_best_match(ranked, profile=profile)
