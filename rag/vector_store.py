"""
rag/vector_store.py
--------------------
Builds a dense vector index over all fetched jobs and retrieves the
most semantically similar ones to a candidate CV.

Two backends (auto-selected):
  1. FAISS (facebook/faiss-cpu) — fast, production-grade, recommended
  2. NumPy cosine search         — pure-Python fallback, no extra install

RAG Model — sentence-transformers (all-MiniLM-L6-v2):
  A pre-trained 384-dimensional bi-encoder model that maps job texts and
  the candidate query into the same embedding space. Cosine similarity
  between the query vector and each job vector is used to rank relevance.
  Runs fully locally — no API key, no cost.

Query generation — Prompt-based template:
  A structured prompt template extracts the most discriminative signals
  from the CV profile (role, seniority, domain, hard skills) and formats
  them into a natural-language recruiter search string. The template
  mirrors how a recruiter would phrase a search query, prioritising
  role titles and hard skills over soft skills.
"""

import os
import pickle
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Embedding backend — sentence-transformers (local, free)
# ─────────────────────────────────────────────────────────────────────────────

_st_model = None

def _get_st_model():
    global _st_model
    if _st_model is not None:
        return _st_model
    try:
        from sentence_transformers import SentenceTransformer
        print("  [RAG] Loading sentence-transformers (all-MiniLM-L6-v2)...")
        try:
            _st_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [RAG] sentence-transformers ready")
    except Exception as e:
        print(f"  [RAG] sentence-transformers unavailable: {e}")
        _st_model = "unavailable"
    return _st_model


def _embed_texts(texts: list) -> Optional[np.ndarray]:
    model = _get_st_model()
    if model == "unavailable":
        return None
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True)
    return vecs.astype(np.float32)


def _embed_one(text: str) -> Optional[np.ndarray]:
    model = _get_st_model()
    if model == "unavailable":
        return None
    return model.encode([text], convert_to_numpy=True)[0].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Query generation — rule-based, no API needed
# ─────────────────────────────────────────────────────────────────────────────

"""
Query generation prompt template.

Sent to Groq (llama3-8b-8192) to generate a recruiter-style search string.
Falls back to local template rendering if Groq key is not set.
"""
_QUERY_PROMPT_TEMPLATE = """\
Task: Generate a job-board search query for this candidate.

Candidate profile:
  - Current title : {title}
  - Seniority     : {seniority}
  - Domain        : {domain}
  - Hard skills   : {skills}
  - Target roles  : {roles}

Rules:
  1. Lead with the most specific job title or target role
  2. Include 3-5 of the most important hard skills
  3. Add seniority only if senior/lead/principal/staff
  4. Keep the query under 15 words
  5. Output the query string only — no explanation, no quotes

Query:"""


def _call_groq(prompt: str) -> str:
    """Call Groq API and return the text response."""
    import urllib.request
    import json
    from config import GROQ_API_KEY, GROQ_MODEL

    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 40,
        "temperature": 0.2,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def generate_query(profile: dict) -> str:
    """
    Generate a recruiter-style search query from the CV profile.

    Uses Groq (llama3-8b-8192) to process the structured prompt template
    and produce a natural-language search string. Falls back to local
    template rendering if GROQ_API_KEY is not set.
    """
    from config import GROQ_API_KEY

    title     = profile.get("current_title", "software engineer")
    seniority = profile.get("seniority", "mid")
    domain    = profile.get("domain", "software")
    skills    = ", ".join(profile.get("skills_hard", [])[:8])
    roles     = ", ".join(profile.get("preferred_roles", [])[:3])

    prompt = _QUERY_PROMPT_TEMPLATE.format(
        title=title,
        seniority=seniority,
        domain=domain,
        skills=skills or "not specified",
        roles=roles or title,
    )

    if GROQ_API_KEY:
        try:
            query = _call_groq(prompt)
            # Clean up any accidental quotes or newlines
            query = query.split("\n")[0].strip().strip('"').strip("'")
            print(f"  [RAG] Groq-generated query: {query!r}")
            return query
        except Exception as exc:
            print(f"  [RAG] Groq query generation failed: {exc} — using local fallback")

    # Local fallback: extract the answer slot from the filled template
    query = f"{title} {roles} {skills}".strip()
    print(f"  [RAG] Query (local fallback): {query!r}")
    return query


def generate_query_with_claude(api_key: str, profile: dict) -> str:
    """Alias kept for backwards compatibility with pipeline.py."""
    return generate_query(profile)


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class JobVectorStore:
    """
    Semantic job index with two model components:

    1. RAG model — sentence-transformers (all-MiniLM-L6-v2)
       Pre-trained bi-encoder that embeds job texts into 384-dim vectors.
       At retrieval time the candidate query is embedded with the same model
       and cosine similarity ranks the most relevant jobs.

    2. Query model — structured prompt template (generate_query)
       A prompt-based approach that extracts role, seniority, domain and
       hard skills from the CV profile and renders them into a search string
       following recruiter best-practices (role-first, skill-focused).

    Both run fully locally — no API key or internet connection required.
    """

    def __init__(self, api_key: Optional[str] = None):
        # api_key param kept for backwards-compat with pipeline.py — ignored
        self.jobs         = []
        self.job_texts    = []
        self.embeddings   = None
        self._faiss_index = None
        self._built       = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, jobs: list) -> bool:
        if not jobs:
            return False

        self.jobs      = jobs
        self.job_texts = [self._job_to_text(j) for j in jobs]

        print(f"  [RAG] Embedding {len(jobs)} jobs with sentence-transformers...")
        vecs = _embed_texts(self.job_texts)

        if vecs is not None:
            self.embeddings = vecs
            self._try_build_faiss(vecs)
            self._built = True
            backend = "FAISS" if self._faiss_index else "NumPy cosine"
            print(f"  [RAG] Index built — {len(jobs)} jobs × {vecs.shape[1]} dims ({backend})")
            return True

        # TF-IDF last resort
        print("  [RAG] sentence-transformers unavailable — RAG disabled (install it: pip install sentence-transformers)")
        self._built = False
        return False

    def _try_build_faiss(self, vecs: np.ndarray):
        try:
            import faiss
            dim   = vecs.shape[1]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normed = vecs / norms
            index  = faiss.IndexFlatIP(dim)
            index.add(normed)
            self._faiss_index = index
        except ImportError:
            self._faiss_index = None

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, cv_profile: dict, k: int = 30) -> list:
        k = min(k, len(self.jobs))
        query_text = generate_query(cv_profile)

        if self._built and self.embeddings is not None:
            return self._retrieve_st(query_text, k)
        return self.jobs[:k]

    def _retrieve_st(self, query_text: str, k: int) -> list:
        query_vec = _embed_one(query_text)
        if query_vec is None:
            return self.jobs[:k]

        if self._faiss_index is not None:
            indices, scores = self._faiss_search(query_vec, k)
        else:
            indices, scores = self._numpy_search(query_vec, k)

        results = []
        for idx, score in zip(indices, scores):
            job = dict(self.jobs[idx])
            job["_rag_score"] = round(float(score), 4)
            results.append(job)
        return results

    def _faiss_search(self, query_vec, k):
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        scores, indices = self._faiss_index.search(
            query_vec.reshape(1, -1).astype(np.float32), k
        )
        return indices[0].tolist(), scores[0].tolist()

    def _numpy_search(self, query_vec, k):
        norms = np.linalg.norm(self.embeddings, axis=1)
        qnorm = np.linalg.norm(query_vec)
        if qnorm == 0 or np.all(norms == 0):
            return list(range(k)), [0.0] * k
        sims = self.embeddings @ query_vec / (norms * qnorm + 1e-10)
        top_k_idx = np.argsort(sims)[::-1][:k]
        return top_k_idx.tolist(), sims[top_k_idx].tolist()

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        with open(f"{path_prefix}.jobs.pkl", "wb") as f:
            pickle.dump(self.jobs, f)
        if self.embeddings is not None:
            np.save(f"{path_prefix}.embeddings.npy", self.embeddings)
        print(f"  [RAG] Index saved → {path_prefix}.*")

    def load(self, path_prefix: str) -> bool:
        jobs_path = f"{path_prefix}.jobs.pkl"
        emb_path  = f"{path_prefix}.embeddings.npy"
        if not os.path.exists(jobs_path):
            return False
        with open(jobs_path, "rb") as f:
            self.jobs = pickle.load(f)
        self.job_texts = [self._job_to_text(j) for j in self.jobs]
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path, allow_pickle=False).astype(np.float32)
            self._try_build_faiss(self.embeddings)
            self._built = True
        print(f"  [RAG] Loaded index: {len(self.jobs)} jobs from {path_prefix}.*")
        return True

    # ── Text helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _job_to_text(job: dict) -> str:
        title   = job.get("title", "")
        company = job.get("company", "")
        desc    = (job.get("description") or "")[:300]
        return f"{title} at {company} — {desc}".strip()
