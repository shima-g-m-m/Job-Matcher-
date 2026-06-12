"""
config.py
---------
Central configuration for the job matching system.
Edit values here to customise behaviour — no need to touch other files.
"""

import os

# Get free Adzuna keys at: https://developer.adzuna.com/
ADZUNA_APP_ID     = os.getenv("ADZUNA_APP_ID",  "")
ADZUNA_APP_KEY    = os.getenv("ADZUNA_APP_KEY", "")

# Get free Groq key at: https://console.groq.com/
# Used for: RAG query generation (prompt → LLM → search string)
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = "llama3-8b-8192"   # fast, free, good quality

DEFAULT_TOP_N       = 10        # how many top jobs to show
DEFAULT_COUNTRY     = "us"      # adzuna country code: us / gb / de / au / ca
MAX_JOBS_TO_FETCH   = 250       # cap on total jobs fetched per run
SCORE_STRONG        = 65        # score ≥ this → strong match  🟢
SCORE_GOOD          = 40        # score ≥ this → good match    🟡
                                # score < 40   → weak match    🔴

EMBEDDING_MODEL     = "all-MiniLM-L6-v2"   # fast + accurate sentence-transformer
EMBEDDING_CACHE     = True                  # cache vectors in memory per session

# Applied after bi-encoder retrieval to re-score the top candidates jointly.
# Set USE_CROSS_ENCODER = False to skip (saves ~300ms but lowers ranking quality).
USE_CROSS_ENCODER       = True
CROSS_ENCODER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOP_N     = 50      # only re-rank the top N retrieved jobs
CROSS_ENCODER_BLEND     = 0.35    # 0=ignore CE score, 1=use CE score only

# All LLM features use the same GROQ_API_KEY / GROQ_MODEL from above.
# Set USE_LLM_CV_PARSER = False to always use the local regex parser.
# Set USE_LLM_EXPLAIN   = False to always use local skill-diff explanations.
USE_LLM_CV_PARSER       = True   # parse CV with Groq (fallback: local NLP)
USE_LLM_EXPLAIN         = True   # explain matches with Groq (fallback: local)

SKILL_MATCH_THRESHOLD = 0.70    # 0.0–1.0; skills above this count as matched

WEIGHT_SEMANTIC   = 40   # sentence-transformer cosine similarity
WEIGHT_SKILLS     = 20   # hard skill keyword overlap
WEIGHT_SENIORITY  = 15   # junior / mid / senior alignment
WEIGHT_TITLE      = 15   # role / job title relevance
WEIGHT_EXTRA      = 10   # soft skills + certifications

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR  = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs")

API_TIMEOUT        = 15    # seconds per HTTP request
MAX_QUERIES        = 3     # how many search queries to build from profile
REMOTEOK_MAX       = 60    # max results from RemoteOK
THEMUSE_PAGES      = 3     # pages to fetch from The Muse (25 jobs/page)
ADZUNA_RESULTS     = 50    # results per Adzuna page
