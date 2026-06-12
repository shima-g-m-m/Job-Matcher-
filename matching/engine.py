"""
matching/engine.py  —  Identity-Aware Recruitment Engine  v4.0
──────────────────────────────────────────────────────────────
Score breakdown (100 pts):
  35 pts  semantic similarity  (identity-enriched embeddings, cleaned JD text)
  25 pts  weighted skill match  (specialist > supporting > generic)
  20 pts  seniority + experience alignment  (progressive penalties)
  15 pts  role/title relevance  (taxonomy-aware, soft adjacency)
   5 pts  intent fit + certs

Key design principles:
  - Professional identity drives scoring, not isolated skills
  - Domain family compatibility uses a 5-tier distance table
  - Hard caps enforce recruiter realism (juniors never get lead/manager roles)
  - Intent detection is domain-gated (ai student ≠ frontend developer)
"""

import re
import numpy as np
from typing import Optional
from cv_reader.parser import TECH_SKILLS

# ── Embedding model ────────────────────────────────────────────────────────────

_model = None
_cache: dict[str, np.ndarray] = {}
_CACHE_FILE = "outputs/embed_cache.npz"


def _load_cache():
    import os
    if os.path.exists(_CACHE_FILE):
        try:
            data = np.load(_CACHE_FILE, allow_pickle=True)
            return {str(k): data[k] for k in data.files}
        except Exception:
            pass
    return {}


def _save_cache():
    import os
    if not _cache:
        return
    try:
        os.makedirs("outputs", exist_ok=True)
        np.savez_compressed(_CACHE_FILE, **{k: v for k, v in _cache.items()})
    except Exception:
        pass


_cache.update(_load_cache())
_cache_misses = 0


def _get_model():
    """Load sentence-transformers model once; share with RAG vector store."""
    global _model
    if _model is not None:
        return _model
    # Reuse model already loaded by RAG (avoids double load + network retry)
    try:
        import rag.vector_store as _vs
        shared = getattr(_vs, "_model", None)
        if shared is not None and shared != "tfidf":
            _model = shared
            return _model
    except Exception:
        pass
    # Load fresh (local cache first, then download, then tfidf fallback)
    try:
        from sentence_transformers import SentenceTransformer
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            try:
                _model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                _model = "tfidf"
    except ImportError:
        _model = "tfidf"
    return _model


def _embed(text: str) -> Optional[np.ndarray]:
    global _cache_misses
    m = _get_model()
    if m == "tfidf":
        return None
    key = text[:300]
    if key in _cache:
        return _cache[key]
    _cache_misses += 1
    try:
        v = m.encode([text], show_progress_bar=False)[0]
    except Exception:
        return None
    _cache[key] = v
    if _cache_misses % 50 == 0:
        _save_cache()
    return v


def _cosine(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


# ── JD cleaning ───────────────────────────────────────────────────────────────

_BOILERPLATE_RE = re.compile(
    r"(?:equal opportunity|eeo|diversity|inclusion|"
    r"benefits\s+(?:include|package|offered)|health\s+insurance|dental|"
    r"401k|paid\s+time\s+off|pto|vacation|sick\s+leave|"
    r"we\s+offer|we\s+provide|compensation\s+(?:range|package)|salary\s+range|"
    r"authorized\s+to\s+work|visa\s+sponsor|background\s+check|"
    r"all\s+qualified\s+applicants|employment\s+(?:is|at\s+will)|"
    r"must\s+be\s+located|privacy\s+policy|cookie|gdpr|"
    r"apply\s+(?:now|today|online)|click\s+(?:here|apply)|"
    r"about\s+(?:us|our\s+company)|our\s+mission|our\s+values|"
    r"we\s+are\s+(?:a\s+)?(?:leading|fast-growing|innovative|dynamic))",
    re.I,
)


def _clean_jd(text: str, max_chars: int = 1500) -> str:
    """Strip boilerplate from JD before embedding; keep title/requirements/skills."""
    lines = text.splitlines()
    clean = [l.strip() for l in lines
             if l.strip() and len(l.strip()) > 4
             and not _BOILERPLATE_RE.search(l.strip())]
    return " ".join(clean)[:max_chars]


# ── Skill tiers ───────────────────────────────────────────────────────────────

TIER_A = {   # weight 3.0 — role-defining specialist skills
    # AI/ML
    "pytorch", "tensorflow", "keras", "scikit-learn", "hugging face", "langchain",
    "openai", "llm", "generative ai", "nlp", "computer vision", "deep learning",
    "machine learning", "mlops", "stable diffusion",
    # Frontend
    "react", "angular", "vue", "next.js", "svelte", "typescript", "redux", "graphql",
    "webpack", "vite", "tailwind", "sass", "figma",
    # Backend/infra
    "kubernetes", "terraform", "ansible", "kafka", "spark", "airflow", "dbt",
    "fastapi", "django", "flask", "spring", "nestjs", "grpc", "websockets",
    "ci/cd", "github actions",
    # Mobile
    "flutter", "swift", "kotlin", "dart", "react native",
    # Data
    "tableau", "power bi", "snowflake", "databricks", "bigquery",
    # DB
    "postgresql", "mongodb", "redis", "elasticsearch", "dynamodb", "cassandra",
    # Cloud
    "aws", "azure", "gcp", "docker",
    # Finance
    "sap", "bloomberg", "ifrs", "gaap", "salesforce",
    # Systems
    "rust", "golang", "scala",
    # Embedded specialist
    "autosar", "rtos", "freertos", "zephyr", "can bus", "fpga", "vhdl", "verilog",
    "stm32", "arm cortex", "ecu", "bootloader", "embedded linux", "yocto",
}

TIER_B = {   # weight 1.5 — domain-relevant common skills
    "python", "java", "javascript", "c++", "c#", "go", "ruby", "php",
    "html", "css", "node", "express", "rails", "laravel", "asp.net",
    "sql", "mysql", "sqlite", "firebase", "supabase",
    "jest", "pytest", "cypress", "selenium",
    "android studio", "xcode", "bash", "linux", "nginx",
    "git", "rest api", "api", "jwt", "oauth",
    # Embedded supporting
    "c", "assembly", "microcontroller", "firmware", "embedded", "iot",
    "arduino", "esp32", "raspberry pi", "i2c", "spi", "uart",
}

TIER_C = {   # weight 0.1 — generic/soft skills
    "communication", "teamwork", "collaboration", "agile", "scrum", "kanban",
    "problem solving", "leadership", "presentation", "reporting", "adaptability",
    "microsoft office", "google sheets", "excel", "powerpoint",
    "github", "gitlab", "bitbucket",
}

NOISE_GAPS = {
    "word", "arm", "r", "c", "qa", "sprint", "word processing", "ms office",
    "mentoring", "prototyping", "benefits", "compliance", "onboarding",
    "insurance", "compensation", "management", "strategy",
    "teamwork", "communication", "collaboration",
}

# ── Transferable skill ecosystems ─────────────────────────────────────────────

TRANSFER_MAP: dict[str, set[str]] = {
    "react":          {"next.js", "gatsby", "react native"},
    "vue":            {"nuxt", "vuex"},
    "next.js":        {"react"},
    "typescript":     {"javascript"},
    "javascript":     {"typescript", "node"},
    "tailwind":       {"css", "sass"},
    "django":         {"flask", "fastapi"},
    "flask":          {"django", "fastapi"},
    "fastapi":        {"django", "flask"},
    "node":           {"express", "nestjs"},
    "postgresql":     {"sql", "mysql", "database"},
    "mysql":          {"postgresql", "sql"},
    "mongodb":        {"nosql", "database"},
    "pytorch":        {"tensorflow", "deep learning", "keras"},
    "tensorflow":     {"pytorch", "keras", "deep learning"},
    "hugging face":   {"transformers", "nlp", "llm", "bert"},
    "langchain":      {"llm", "openai"},
    "nlp":            {"natural language", "bert", "transformers"},
    "flutter":        {"dart", "cross-platform", "mobile"},
    "android":        {"kotlin", "mobile"},
    "ios":            {"swift", "mobile"},
    "kotlin":         {"android"},
    "swift":          {"ios"},
    "aws":            {"cloud", "ec2", "s3", "lambda"},
    "azure":          {"cloud"},
    "gcp":            {"cloud", "bigquery"},
    "kubernetes":     {"docker", "helm", "container"},
    "terraform":      {"infrastructure as code", "ansible"},
    # Embedded transferable
    "autosar":        {"ecu", "automotive", "bsw", "rte"},
    "rtos":           {"freertos", "zephyr", "embedded os"},
    "can bus":        {"automotive", "j1939", "canopen"},
    "stm32":          {"arm cortex", "microcontroller", "firmware"},
    "arm cortex":     {"stm32", "embedded", "firmware"},
    "sap":            {"erp", "enterprise"},
    "ifrs":           {"gaap", "accounting standards"},
}


def _expand(skills: set) -> set:
    expanded = set(skills)
    for s in list(skills):
        expanded.update(TRANSFER_MAP.get(s, set()))
    return expanded


# ── Skill normalisation ────────────────────────────────────────────────────────

ALIASES = {
    "js": "javascript", "ts": "typescript",
    "react.js": "react", "reactjs": "react",
    "node.js": "node", "nodejs": "node",
    "next.js": "next.js", "nextjs": "next.js",
    "vue.js": "vue", "vuejs": "vue",
    "k8s": "kubernetes", "tf": "tensorflow",
    "pg": "postgresql", "mongo": "mongodb",
    "genai": "generative ai", "llms": "llm",
    "bert": "nlp", "gpt": "llm", "transformers": "nlp",
    "hf": "hugging face", "tailwindcss": "tailwind", "scss": "sass",
    "restful": "rest api", "restful api": "rest api",
    "freertos": "freertos", "autosar": "autosar",
}


def _norm(skills: set) -> set:
    out = set(skills)
    for s in list(out):
        c = ALIASES.get(s)
        if c:
            out.add(c)
    return out


def _tok(text: str) -> set:
    t = text.lower()
    for a, c in ALIASES.items():
        t = re.sub(r"\b" + re.escape(a) + r"\b", c, t)
    tokens: set = set()
    for sk in TECH_SKILLS:
        if " " in sk and sk in t:
            tokens.add(sk)
    tokens.update(re.findall(r"\b[a-z][a-z0-9#+.\-]{1,}\b", t))
    return tokens


# ── Role family taxonomy ──────────────────────────────────────────────────────

FAMILIES: dict[str, set] = {
    "embedded":  {
        "embedded engineer", "embedded software", "firmware engineer",
        "embedded developer", "autosar", "rtos", "microcontroller",
        "ecu developer", "embedded linux", "iot engineer", "firmware",
        "embedded c", "embedded c++", "stm32", "arm cortex",
        "automotive engineer", "fpga engineer", "can bus",
    },
    "ai_ml":     {
        "machine learning", "ml engineer", "ai engineer", "nlp engineer",
        "deep learning", "computer vision", "data science", "llm",
        "applied ai", "generative", "mlops", "hugging face", "pytorch",
        "tensorflow", "ai researcher", "ml researcher", "nlp",
        "transformers", "bert", "gpt", "diffusion",
    },
    "data":      {
        "data engineer", "data analyst", "business intelligence",
        "analytics engineer", "bi developer", "data warehouse", "etl",
        "dbt developer", "data pipeline", "reporting analyst",
        "spark", "airflow", "databricks",
    },
    "data_sci":  {"data scientist", "statistician", "quantitative analyst",
                  "research scientist"},
    "frontend":  {
        "frontend", "front-end", "ui engineer", "ux engineer",
        "react developer", "vue developer", "angular developer",
        "javascript developer", "web ui", "css engineer", "ui developer",
        "react", "vue", "angular", "svelte", "next.js",
        "typescript developer", "tailwind",
    },
    "fullstack": {
        "full stack", "fullstack", "full-stack", "full stack engineer",
        "full-stack engineer", "full stack developer", "fullstack developer",
    },
    "backend":   {
        "backend", "back-end", "api developer", "python developer",
        "java developer", "node developer", "server-side",
        "microservices engineer", "api engineer",
    },
    "mobile":    {
        "mobile developer", "flutter developer", "android developer",
        "ios developer", "react native developer", "mobile engineer",
        "app developer", "cross-platform developer",
        "flutter", "android", "ios", "kotlin", "swift", "dart",
        "react native", "mobile app",
    },
    "devops":    {
        "devops", "sre", "site reliability", "infrastructure engineer",
        "platform engineer", "cloud engineer", "devsecops",
        "kubernetes engineer", "cloud architect",
        "kubernetes", "terraform", "ansible", "helm",
        "prometheus", "grafana", "jenkins", "ci/cd", "gitops",
    },
    "security":  {
        "security engineer", "cybersecurity", "penetration tester",
        "soc analyst", "infosec", "security analyst", "devsecops engineer",
    },
    "software":  {"software engineer", "software developer"},   # catch-all
    "finance":   {
        "financial analyst", "investment analyst", "banking analyst",
        "treasury analyst", "risk analyst", "credit analyst",
        "quant analyst", "finance manager",
    },
    "accounting":{"accountant", "auditor", "financial accountant",
                  "tax accountant", "management accountant", "bookkeeper"},
    "design":    {"designer", "ux designer", "ui designer", "product designer",
                  "graphic designer", "visual designer"},
}

# 5-tier compatibility table
_FC: dict[tuple, float] = {
    ("frontend", "fullstack"):    0.90,
    ("backend",  "fullstack"):    0.90,
    ("frontend", "software"):     0.82,
    ("backend",  "software"):     0.82,
    ("fullstack","software"):     0.92,
    ("ai_ml",    "data_sci"):     0.88,
    ("ai_ml",    "data"):         0.72,
    ("data_sci", "data"):         0.85,
    ("frontend", "mobile"):       0.62,
    ("mobile",   "software"):     0.75,
    ("mobile",   "fullstack"):    0.65,
    ("devops",   "backend"):      0.55,
    ("devops",   "software"):     0.60,
    ("security", "devops"):       0.55,
    ("ai_ml",    "software"):     0.65,
    ("ai_ml",    "backend"):      0.55,
    ("data",     "software"):     0.62,
    ("backend",  "frontend"):     0.50,
    ("design",   "frontend"):     0.55,
    ("accounting","finance"):     0.85,
    ("data",     "finance"):      0.50,
    ("ai_ml",    "accounting"):   0.12,
    ("ai_ml",    "finance"):      0.18,
    ("frontend", "devops"):       0.28,
    ("mobile",   "devops"):       0.28,
    ("mobile",   "ai_ml"):        0.35,
    ("frontend", "finance"):      0.12,
    ("backend",  "finance"):      0.22,
    ("data",     "accounting"):   0.45,
    # Embedded adjacency
    ("embedded", "software"):     0.65,
    ("embedded", "devops"):       0.25,
    ("embedded", "mobile"):       0.30,
    ("embedded", "backend"):      0.20,
    ("embedded", "data"):         0.20,
    ("embedded", "ai_ml"):        0.35,
}


def _family(text: str) -> Optional[str]:
    t = text.lower()
    scores: dict[str, int] = {}
    for fam, kws in FAMILIES.items():
        s = sum(1 for kw in kws if kw in t)
        if s:
            scores[fam] = s
    if not scores:
        return None
    specific = {f: s for f, s in scores.items() if f != "software"}
    if specific:
        best = max(specific, key=lambda f: specific[f])
        if specific[best] >= 1:
            return best
    return max(scores, key=lambda f: scores[f])


def _compat(f1, f2) -> float:
    if not f1 or not f2:
        return 0.70
    if f1 == f2:
        return 1.0
    key = (min(f1, f2), max(f1, f2))
    return _FC.get(key, 0.15)


# ── Seniority ─────────────────────────────────────────────────────────────────

SENIOR_W = {"senior", "sr", "principal", "staff", "lead", "architect",
             "director", "vp", "head", "chief", "president", "manager"}
JUNIOR_W = {"junior", "jr", "intern", "entry", "graduate", "grad",
             "trainee", "apprentice", "new grad", "early career"}

EXP_RE = [
    re.compile(r"(\d+)\+?\s*years?\s*(?:of\s+)?(?:professional\s+)?experience", re.I),
    re.compile(r"minimum\s+(\d+)\s*years?", re.I),
    re.compile(r"at\s+least\s+(\d+)\s*years?", re.I),
    re.compile(r"(\d+)\+\s*years?", re.I),
]


def _req_years(text: str) -> int:
    yrs = []
    for pat in EXP_RE:
        for m in pat.finditer(text):
            try:
                y = int(m.group(1))
                if 1 <= y <= 20:
                    yrs.append(y)
            except ValueError:
                pass
    return max(yrs) if yrs else 0


# ── Intent detection (domain-gated, avoids false positives) ──────────────────

def _detect_intent(profile: dict) -> dict:
    """
    Domain-gated intent detection.
    An AI/embedded student with 'css' or 'docker' in skills should NOT be
    flagged as frontend/devops focused — domain gates the intent signals.
    """
    title   = (profile.get("current_title") or "").lower()
    summary = (profile.get("summary_for_search") or "").lower()
    roles   = " ".join(profile.get("preferred_roles", [])).lower()
    domain  = profile.get("domain", "software")

    # Use identity specializations if available
    identity  = profile.get("_identity", {})
    specs     = set(identity.get("specializations", []))
    industries = set(identity.get("industry_tags", []))

    text = f"{title} {summary} {roles}"

    return {
        # Only trigger internship intent if EXPLICITLY stated in title/roles
        # (not inferred from skills which may contain intern-era tools)
        "seeking_internship": (any(w in text for w in
            ["internship", "intern", "summer 2025", "summer 2026"])
            and profile.get("seniority","mid") in ("junior",)),
        "seeking_remote":     "remote" in text,
        "research_oriented":  any(w in text for w in
            ["research", "phd", "thesis"]),
        # Domain-gated: only fire if domain actually matches
        "frontend_focus":     (domain in ("frontend", "fullstack", "software") and
                               any(w in text for w in
                                   ["frontend", "front-end", "react", "vue",
                                    "angular", "ui", "css", "tailwind"])),
        "ai_focus":           (domain in ("ai_ml", "data_sci", "data") and
                               any(w in text for w in
                                   ["machine learning", "ai", "nlp", "deep learning",
                                    "computer vision", "pytorch", "tensorflow"])),
        "mobile_focus":       (domain == "mobile" and
                               any(w in text for w in
                                   ["flutter", "mobile", "android", "ios",
                                    "react native", "kotlin", "swift"])),
        "backend_focus":      (domain in ("backend", "fullstack", "software") and
                               any(w in text for w in
                                   ["backend", "back-end", "django", "fastapi",
                                    "node", "api", "postgresql"])),
        "devops_focus":       (domain in ("devops",) and
                               any(w in text for w in
                                   ["devops", "kubernetes", "terraform",
                                    "docker", "ci/cd", "aws", "cloud"])),
        "embedded_focus":     (domain == "embedded" or
                               # Only fire for non-embedded domains if STRONGLY specialized
                               (domain != "ai_ml" and "automotive" in industries) or
                               (domain == "embedded" and
                                bool(specs & {"autosar","rtos","can_bus","arm_cortex","fpga"}))),
        "finance_focus":      (domain in ("finance", "accounting") and
                               any(w in text for w in
                                   ["finance", "accounting", "banking",
                                    "financial", "audit", "sap"])),
    }


# ── Confidence ────────────────────────────────────────────────────────────────

def _conf(score: int) -> str:
    if score >= 80: return "EXCELLENT"
    if score >= 68: return "STRONG"
    if score >= 55: return "MODERATE"
    if score >= 40: return "STRETCH"
    return "WEAK"


# ── Pre-filter (hard exclusions before scoring) ───────────────────────────────

def pre_filter(profile: dict, jobs: list) -> list:
    """
    Hard remove clearly incompatible jobs.
    Keeps the candidate pool clean BEFORE expensive scoring.
    """
    seniority = profile.get("seniority", "mid").lower()
    cv_years  = profile.get("years_experience", 0)
    cv_dom    = profile.get("domain", "software")
    identity  = profile.get("_identity", {})
    industries = set(identity.get("industry_tags", []))

    # Absolute exclusions for non-executives
    HARD_EXCL = {
        "chief executive", "chief technology", "chief operating",
        "vp of engineering", "vice president engineering",
        "general manager", "president",
    }
    # Manager/lead exclusions for junior candidates (< 2 yrs)
    MANAGER_EXCL = {
        "engineering manager", "product manager", "project manager",
        "head of", "team lead", "tech lead", "people manager",
    }
    # Distant domain pairs (suppress entirely)
    DISTANT_PAIRS = {
        ("ai_ml", "accounting"), ("ai_ml", "finance"),
        ("frontend", "accounting"), ("mobile", "accounting"),
        ("devops", "accounting"), ("embedded", "accounting"),
        ("embedded", "finance"), ("frontend", "finance"),
    }

    out = []
    for job in jobs:
        tl  = (job.get("title") or "").lower()
        fl  = tl + " " + (job.get("description") or "")[:400].lower()
        ry  = _req_years(fl)

        # Hard exclude C-suite always
        if any(p in tl for p in HARD_EXCL):
            continue

        # Hard exclude manager/lead for junior/fresh candidates
        if cv_years < 2 and any(p in tl for p in MANAGER_EXCL):
            continue

        # Experience gap > 8 years → skip
        if ry > 0 and ry > cv_years + 8:
            continue

        # Distant domain pair → skip
        jf   = _family(tl + " " + fl[:200])
        cdom = cv_dom.replace("_infra", "")
        pair = (min(cdom, jf), max(cdom, jf)) if jf else None
        if pair and pair in DISTANT_PAIRS:
            continue

        out.append(job)
    return out


# ── Dedup ─────────────────────────────────────────────────────────────────────

def deduplicate_jobs(jobs: list) -> list:
    seen, out = set(), []
    for job in jobs:
        title = re.sub(r"\s*(remote|us|uk|eu|latam|\|.*$)", "",
                       (job.get("title") or "").lower()).strip()
        key   = title + "|" + (job.get("company") or "").lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(job)
    return out


# ── CV embedding text builder ─────────────────────────────────────────────────

def _build_cv_embed_text(profile: dict, intent: dict) -> str:
    """
    Build clean, identity-enriched CV text for embedding.
    Prioritises: role identity → specializations → domain skills → target roles.
    Excludes generic/soft skills that dilute the signal.
    """
    GENERIC = {
        "communication", "teamwork", "collaboration", "agile", "scrum",
        "problem solving", "leadership", "presentation", "adaptability",
        "microsoft office", "google sheets", "excel", "powerpoint",
        "github", "gitlab", "bitbucket",
    }
    identity = profile.get("_identity", {})

    parts = []

    # 1. Identity layer (most specific signal)
    if identity.get("role_identity"):
        parts.append(identity["role_identity"].replace("_", " "))
    if identity.get("specializations"):
        parts.append(" ".join(s.replace("_", " ")
                               for s in identity["specializations"]))
    if identity.get("industry_tags"):
        parts.append(" ".join(identity["industry_tags"]))
    if identity.get("query_roles"):
        parts.append(" ".join(identity["query_roles"][:4]))

    # 2. Current title + job titles held
    title = profile.get("current_title", "")
    if title:
        parts.append(title)
    for jt in profile.get("job_titles_held", [])[:2]:
        parts.append(jt)

    # 3. Domain-specific skills only (no soft skills)
    domain_skills = [s for s in profile.get("skills_hard", [])
                     if s.lower() not in GENERIC]
    if domain_skills:
        parts.append(" ".join(domain_skills[:20]))

    return " ".join(parts)


# ── Core scorer ────────────────────────────────────────────────────────────────

def score_job(profile: dict, job: dict) -> dict:
    cv_raw   = set(s.lower() for s in profile.get("skills_hard", []))
    cv_hard  = _norm(cv_raw)
    cv_exp   = _expand(cv_hard)
    cv_roles = set(t.lower() for t in
                   profile.get("preferred_roles", []) +
                   profile.get("job_titles_held", []))
    cv_certs = set(c.lower() for c in profile.get("certifications", []))
    cv_sen   = profile.get("seniority", "junior").lower()
    cv_yrs   = profile.get("years_experience", 0)
    cv_dom   = profile.get("domain", "software")
    intent   = _detect_intent(profile)
    identity = profile.get("_identity", {})

    jd_full  = (job.get("description") or "") + " " + (job.get("title") or "")
    jd_lower = jd_full.lower()
    jd_tok   = _tok(jd_full)
    jt_lower = (job.get("title") or "").lower()
    jt_words = set(jt_lower.split())

    # ── Family compatibility ──────────────────────────────────────────────────
    cv_fam_text = (" ".join(cv_roles) + " " + cv_dom + " " +
                   " ".join(profile.get("skills_hard", [])[:12]) + " " +
                   " ".join(identity.get("specializations", [])))
    cv_fam  = _family(cv_fam_text)
    job_fam = _family(jd_full[:800])
    fc      = _compat(cv_fam, job_fam)

    # ── 1. Semantic similarity (35 pts) ───────────────────────────────────────
    cv_emb_text = _build_cv_embed_text(profile, intent)
    jd_emb_text = _clean_jd(jd_full)

    cv_vec = _embed(cv_emb_text)
    jv_vec = _embed(jd_emb_text)
    sem_raw = 0.0
    if cv_vec is not None and jv_vec is not None:
        sem_raw   = _cosine(cv_vec, jv_vec)
        fc_sem    = 0.50 + 0.50 * fc
        sem_score = int(sem_raw * 35 * fc_sem)
    else:
        cv_toks   = _tok(cv_emb_text)
        ov        = cv_toks & jd_tok & TECH_SKILLS
        sem_score = min(35, int(len(ov) / max(len(cv_toks & TECH_SKILLS), 1) * 35))

    # Stack-overlap boost (compensates for embedding model limitations)
    jd_skills = _norm(jd_tok & TECH_SKILLS)
    shared_a  = cv_hard & jd_tok & TIER_A
    if len(shared_a) >= 4:   sem_score = max(sem_score, 22)
    elif len(shared_a) >= 3: sem_score = max(sem_score, 17)
    elif len(shared_a) >= 2: sem_score = max(sem_score, 12)
    elif len(shared_a) >= 1 and sem_raw > 0.2: sem_score = max(sem_score, 8)

    # ── 2. Weighted skill match (25 pts) ──────────────────────────────────────
    matched_d  = cv_hard & (jd_tok | jd_skills)
    matched_t  = (cv_exp - cv_hard) & (jd_tok | jd_skills)

    m_a  = matched_d & TIER_A
    m_b  = (matched_d & TIER_B) - TIER_A
    m_c  = matched_d & TIER_C
    m_ta = matched_t & TIER_A
    m_tb = matched_t & TIER_B

    raw_pts = (len(m_a) * 3.0 + len(m_b) * 1.5 + len(m_c) * 0.1 +
               len(m_ta) * 1.5 + len(m_tb) * 0.75)

    jd_a = jd_skills & TIER_A
    jd_b = (jd_skills & TIER_B) - TIER_A
    jd_w = len(jd_a) * 3.0 + len(jd_b) * 1.5
    denom = max(jd_w, 5.0)

    skill_score = min(25, int(raw_pts / denom * 25))
    if len(m_a) >= 3: skill_score = max(skill_score, 14)
    elif len(m_a) >= 2: skill_score = max(skill_score, 10)
    elif len(m_a) >= 1: skill_score = max(skill_score, 6)
    if fc <= 0.20: skill_score = int(skill_score * 0.5)

    missing = sorted(
        (jd_skills - cv_hard - NOISE_GAPS) &
        (TIER_A | (jd_skills - TIER_C - NOISE_GAPS))
    )[:6]
    matched_disp = sorted(m_a | m_b | m_ta) or sorted(matched_d)[:6]

    # ── 3. Seniority + experience (20 pts) ────────────────────────────────────
    t_senior = bool(jt_words & SENIOR_W)
    t_junior = bool(jt_words & JUNIOR_W)
    d_senior = any(w in jd_lower for w in SENIOR_W)
    d_junior = any(w in jd_lower for w in JUNIOR_W)

    j_senior  = t_senior or d_senior
    j_junior  = t_junior or d_junior
    j_neutral = not j_senior and not j_junior

    ry       = _req_years(jd_lower)
    exp_gap  = max(0, ry - cv_yrs) if ry > 0 else 0
    exp_pen  = {0: 0, 1: 1, 2: 4, 3: 8, 4: 12}.get(exp_gap,
               min(18, 12 + (exp_gap - 4) * 2))

    if cv_sen == "junior":
        if t_senior and not t_junior: base_sn = 0
        elif j_junior:                 base_sn = 20
        elif j_neutral:                base_sn = 14
        elif j_senior:                 base_sn = 5
        else:                          base_sn = 9
    elif cv_sen == "mid":
        if t_senior and not t_junior:  base_sn = 13
        elif j_junior and not j_senior: base_sn = 12
        else:                           base_sn = 20
    else:
        if t_junior and not t_senior:  base_sn = 7
        elif j_senior or j_neutral:    base_sn = 20
        else:                          base_sn = 14

    seniority_score = max(0, base_sn - exp_pen)

    # ── 4. Role/title relevance (15 pts) ──────────────────────────────────────
    LEVEL_W = {
        "intern", "junior", "jr", "senior", "sr", "lead", "and", "the",
        "a", "co", "op", "associate", "staff", "principal", "mid", "entry",
        "ii", "iii", "i", "iv", "latam", "us", "uk", "eu", "remote",
        "new", "grad", "early", "career",
    }
    GEN_W   = {"developer", "engineer", "manager", "specialist", "analyst",
               "architect", "consultant", "coordinator", "officer"}
    DOM_STOP = LEVEL_W | GEN_W

    title_score, matched_roles = 0, []
    for role in cv_roles:
        r_dom = set(role.split()) - DOM_STOP
        t_dom = set(jt_lower.split()) - DOM_STOP
        if not (r_dom & t_dom):
            continue
        rw = set(role.split()) - LEVEL_W
        tw = set(jt_lower.split()) - LEVEL_W
        ov = rw & tw
        if ov:
            sc = int(len(ov) / max(len(rw), 1) * 15)
            title_score = max(title_score, sc)
            matched_roles.append(role)

    cur = (profile.get("current_title") or "").lower()
    if cur and cur in jt_lower:
        title_score = min(15, title_score + 5)

    # Soft adjacency bonus for close families
    if title_score < 5  and fc >= 0.85: title_score = max(title_score, 6)
    elif title_score < 3 and fc >= 0.70: title_score = max(title_score, 3)

    if fc < 0.40:   title_score = int(title_score * 0.40)
    elif fc < 0.70: title_score = int(title_score * 0.75)

    # ── 5. Intent + industry + certs (5 pts) ──────────────────────────────────
    bonus = 0

    # Domain-gated intent bonuses
    if intent.get("seeking_internship") and j_junior: bonus += 2
    if intent.get("ai_focus") and any(
            w in jt_lower for w in ["ai", "ml", "machine learning", "nlp",
                                     "deep learning", "data science", "pytorch"]):
        bonus += 2
    if intent.get("mobile_focus") and any(
            w in jt_lower for w in ["mobile", "flutter", "android", "ios",
                                     "kotlin", "swift"]): bonus += 2
    if intent.get("frontend_focus") and any(
            w in jt_lower for w in ["frontend", "react", "vue", "angular",
                                     "ui", "javascript"]): bonus += 2
    if intent.get("backend_focus") and any(
            w in jt_lower for w in ["backend", "api", "server", "django",
                                     "node", "python"]): bonus += 1
    if intent.get("embedded_focus") and any(
            w in jt_lower for w in ["embedded", "firmware", "autosar", "rtos",
                                     "automotive", "ecu", "microcontroller",
                                     "fpga", "iot", "can"]): bonus += 2
    if intent.get("finance_focus") and any(
            w in jt_lower for w in ["finance", "accounting", "banking",
                                     "financial"]): bonus += 2

    # Industry alignment bonus
    industry_bonus = 0
    try:
        from cv_reader.identity import INDUSTRY_SIGNALS
        cv_industries = identity.get("industry_tags", [])
        for ind in cv_industries[:2]:
            signals = INDUSTRY_SIGNALS.get(ind, [])
            if any(sig in jd_lower for sig in signals[:5]):
                industry_bonus += 1
                break
    except Exception:
        pass

    cert_score = min(2, sum(2 for c in cv_certs if len(c) > 4 and c in jd_lower))
    extra = min(5, bonus + industry_bonus + cert_score)

    # ── Raw total ─────────────────────────────────────────────────────────────
    total = sem_score + skill_score + seniority_score + title_score + extra

    # ── Hard caps ─────────────────────────────────────────────────────────────
    if cv_sen == "junior":
        if t_senior and not t_junior: total = min(total, 54)
        if exp_gap >= 5:               total = min(total, 38)
        if intent.get("seeking_internship") and j_senior and not j_junior:
            total = min(total, 46)
    if fc <= 0.18:  total = min(total, 50)
    if ry > 0 and ry > cv_yrs + 7: total = min(total, 42)

    total = max(0, min(100, total))
    conf  = _conf(total)

    return {
        "score": total, "score_semantic": sem_score, "score_skills": skill_score,
        "score_seniority": seniority_score, "score_title": title_score,
        "score_extra": extra, "semantic_pct": round(sem_raw * 100, 1),
        "confidence": conf, "req_years": ry, "family_compat": round(fc, 2),
        "cv_family": cv_fam, "job_family": job_fam,
        "matched_skills": matched_disp[:8], "missing_skills": missing,
        "matched_roles": matched_roles,
        "title": job.get("title", ""), "company": job.get("company", ""),
        "location": job.get("location", ""), "source": job.get("source", ""),
        "url": job.get("url", ""), "salary": job.get("salary", ""),
        "posted": job.get("posted", ""), "geo_flag": job.get("geo_flag", False),
    }


# ── Rank + diversify ──────────────────────────────────────────────────────────

def rank_jobs(profile: dict, jobs: list, diversify: bool = True) -> list:
    jobs   = deduplicate_jobs(jobs)
    intent = _detect_intent(profile)
    scored = [score_job(profile, j) for j in jobs]

    # Domain-aligned internship boost
    if intent.get("seeking_internship"):
        for r in scored:
            tl = r["title"].lower()
            is_intern = any(w in tl for w in
                ["intern", "entry", "graduate", "new grad", "junior", "jr"])
            if is_intern and r["family_compat"] >= 0.60:
                r["score"]      = min(100, r["score"] + 6)
                r["confidence"] = _conf(r["score"])

    scored.sort(key=lambda x: x["score"], reverse=True)

    if not diversify or len(scored) < 10:
        return scored

    # Diversity: max 2 per title-root
    STOP = {"senior", "junior", "lead", "principal", "staff", "mid",
            "associate", "intern", "remote", "us", "uk", "the", "and",
            "a", "of", "for", "ii", "iii"}

    def _root(r):
        words = [w for w in r["title"].lower().split() if w not in STOP]
        return " ".join(words[:2])

    seen: dict[str, int] = {}
    top, rest = [], []
    for r in scored:
        rt = _root(r)
        c  = seen.get(rt, 0)
        if c < 2:
            seen[rt] = c + 1
            top.append(r)
        else:
            rest.append(r)

    _save_cache()
    return top + rest
