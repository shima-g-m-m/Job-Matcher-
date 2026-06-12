"""
jobs/fetcher.py  —  Title-Aware, Role-Expanded Multi-Source Retrieval
──────────────────────────────────────────────────────────────────────
Architecture:
  CV Profile (skills + titles + domain + intent)
      ↓
  Role-Aware Query Expansion  (title aliases + intent modifiers)
      ↓
  Parallel Multi-Source Retrieval  (RemoteOK · Muse · Remotive · Arbeitnow · Adzuna)
      ↓
  Aggregation + Deduplication
      ↓
  Geo Flagging
      ↓
  Final Job Pool (3000–5000 jobs)

Sources (no key required):
  RemoteOK · The Muse · Remotive · Arbeitnow

Optional (free sign-up):
  Adzuna  — set ADZUNA_APP_ID + ADZUNA_APP_KEY
"""

import os
import re
import time
import json
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── HTTP helper ────────────────────────────────────────────────────────────────

def _get(url: str, extra_headers: dict = None, timeout: int = 20):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    for e, r in [("&nbsp;"," "),("&amp;","&"),("&lt;","<"),
                 ("&gt;",">"),("&#39;","'"),("&quot;",'"')]:
        text = text.replace(e, r)
    return re.sub(r"\s{2,}", " ", text).strip()


# ── Role alias expansion ───────────────────────────────────────────────────────
# Maps a canonical role → list of job-board-friendly query variants.
# Used to expand retrieval beyond just the primary role title.

ROLE_ALIASES: dict[str, list[str]] = {
    # AI / ML
    "machine learning engineer": ["ml engineer","ai engineer","applied ml engineer",
                                   "machine learning developer","deep learning engineer"],
    "ai engineer":               ["ml engineer","machine learning engineer","applied ai engineer",
                                   "artificial intelligence engineer"],
    "data scientist":            ["ml researcher","applied scientist","data science engineer",
                                   "research scientist","quantitative analyst"],
    "nlp engineer":              ["natural language processing engineer","nlp researcher",
                                   "computational linguist","llm engineer"],
    "computer vision engineer":  ["cv engineer","vision engineer","image recognition engineer"],
    "mlops engineer":            ["ml platform engineer","ml infrastructure engineer",
                                   "ai infrastructure engineer"],
    # Frontend
    "frontend developer":        ["frontend engineer","ui developer","react developer",
                                   "javascript developer","web developer"],
    "react developer":           ["react engineer","react.js developer","frontend react developer",
                                   "next.js developer"],
    "vue developer":             ["vue.js developer","nuxt developer","frontend vue developer"],
    "angular developer":         ["angular engineer","frontend angular developer"],
    # Backend
    "backend developer":         ["backend engineer","server-side developer","api developer",
                                   "python developer","java developer"],
    "python developer":          ["django developer","fastapi developer","flask developer",
                                   "python backend engineer"],
    # Full stack
    "full stack developer":      ["fullstack developer","full-stack engineer","web developer",
                                   "software developer"],
    # Mobile
    "flutter developer":         ["flutter engineer","mobile developer",
                                   "cross-platform mobile developer","flutter mobile developer",
                                   "dart mobile developer"],
    "android developer":         ["android engineer","kotlin developer","mobile developer",
                                   "android mobile developer"],
    "ios developer":             ["ios engineer","swift developer","apple developer",
                                   "mobile developer"],
    "mobile developer":          ["mobile engineer","app developer","cross-platform developer",
                                   "flutter developer","android developer"],
    # DevOps
    "devops engineer":           ["sre","site reliability engineer","platform engineer",
                                   "cloud engineer","infrastructure engineer"],
    "cloud engineer":            ["aws engineer","azure engineer","gcp engineer",
                                   "cloud architect","cloud developer"],
    # Data
    "data engineer":             ["data pipeline engineer","etl developer","big data engineer",
                                   "data infrastructure engineer"],
    "data analyst":              ["business intelligence analyst","bi analyst",
                                   "analytics engineer","reporting analyst"],
    # Finance/Accounting
    "financial analyst":         ["finance analyst","investment analyst","banking analyst",
                                   "treasury analyst","financial reporting analyst"],
    "accountant":                ["financial accountant","management accountant",
                                   "staff accountant","accounting specialist"],
}

def _expand_role_aliases(role: str) -> list[str]:
    """Return the canonical role + all known aliases. Filters garbage."""
    clean = re.sub(r'\b(intern|junior|jr|senior|sr|lead|entry|graduate)\b', '',
                   role).strip()
    # Reject roles that are too short, single-word non-meaningful, or contain single letters
    if len(clean) < 5 or re.match(r'^[a-z]$', clean):
        return [clean] if len(clean) > 4 else []
    aliases = ROLE_ALIASES.get(clean, [])
    # Filter alias garbage: must be > 5 chars and contain a real role word
    ROLE_WORDS = {'engineer','developer','scientist','analyst','researcher',
                  'architect','manager','specialist','consultant','designer'}
    valid = [a for a in aliases if len(a) > 5 and
             any(w in a for w in ROLE_WORDS)]
    return [clean] + valid[:4]


# ── Intent modifiers ───────────────────────────────────────────────────────────

def _intent_modifier(profile: dict) -> str:
    """Return query prefix based on candidate intent."""
    sen = profile.get("seniority","mid").lower()
    if sen == "junior": return "intern"
    if sen == "mid":    return ""
    return ""   # senior/lead: no prefix


# ── RemoteOK tag map (canonical skill → API tag) ──────────────────────────────

REMOTEOK_TAG_MAP: dict[str, list[str]] = {
    "python":           ["python"],
    "machine learning": ["machine-learning","ai"],
    "deep learning":    ["deep-learning","machine-learning"],
    "ai":               ["ai","machine-learning"],
    "nlp":              ["nlp","ai"],
    "computer vision":  ["computer-vision","ai"],
    "flutter":          ["flutter","mobile"],
    "dart":             ["flutter","dart"],
    "django":           ["django","python"],
    "flask":            ["flask","python"],
    "fastapi":          ["python","backend"],
    "tensorflow":       ["machine-learning","python"],
    "pytorch":          ["machine-learning","python"],
    "scikit-learn":     ["machine-learning","python"],
    "langchain":        ["ai","machine-learning"],
    "llm":              ["ai","machine-learning"],
    "react":            ["react","javascript"],
    "vue":              ["vue","javascript"],
    "angular":          ["angular","javascript"],
    "next.js":          ["react","javascript"],
    "typescript":       ["typescript","javascript"],
    "javascript":       ["javascript"],
    "node":             ["nodejs","javascript"],
    "docker":           ["docker","devops"],
    "kubernetes":       ["kubernetes","devops"],
    "terraform":        ["devops","aws"],
    "aws":              ["aws","cloud"],
    "azure":            ["azure","cloud"],
    "gcp":              ["google-cloud","cloud"],
    "postgresql":       ["postgresql","sql"],
    "mongodb":          ["mongodb"],
    "redis":            ["redis"],
    "golang":           ["golang"],
    "rust":             ["rust"],
    "kotlin":           ["android","mobile"],
    "swift":            ["ios","mobile"],
    "android":          ["android","mobile"],
    "ios":              ["ios","mobile"],
    "mobile":           ["mobile"],
    "flutter developer":["flutter","mobile"],
    "mobile developer": ["mobile","flutter"],
    "devops":           ["devops"],
    "sap":              ["finance"],
    "excel":            ["finance"],
    "financial":        ["finance","fintech"],
    "sql":              ["sql"],
    "java":             ["java"],
    "intern":           ["intern"],
    "junior":           ["junior"],
}


# ── RemoteOK source ────────────────────────────────────────────────────────────

def _fetch_remoteok_tag(tag: str) -> list:
    time.sleep(0.3)   # polite rate limit
    url = f"https://remoteok.io/api?tag={urllib.parse.quote_plus(tag)}"
    data = _get(url, extra_headers={
        "Referer": "https://remoteok.io/",
        "Accept": "application/json",
    })
    if not data or not isinstance(data, list):
        return []
    out = []
    for r in data:
        if not isinstance(r, dict) or not r.get("position"):
            continue
        lo = r.get("salary_min"); hi = r.get("salary_max")
        sal = f"${int(lo):,} – ${int(hi):,}" if lo and hi else (f"${int(lo):,}+" if lo else "")
        out.append({
            "source":      "RemoteOK",
            "title":       r.get("position",""),
            "company":     r.get("company",""),
            "location":    "Remote",
            "description": _strip_html(r.get("description","")),
            "url":         r.get("url",""),
            "salary":      sal,
            "posted":      (r.get("date") or "")[:10],
            "tags":        r.get("tags",[]),
        })
    return out

def fetch_remoteok_parallel(tags: list, max_workers: int = 3) -> list:
    out, seen = [], set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_remoteok_tag, t): t for t in tags}
        for fut in as_completed(futures):
            for job in fut.result():
                key = job.get("url") or (job["title"]+job["company"])
                if key and key not in seen:
                    seen.add(key); out.append(job)
    return out


# ── The Muse source ────────────────────────────────────────────────────────────

def _fetch_muse_page(page: int) -> list:
    url = (f"https://www.themuse.com/api/public/jobs"
           f"?category=Computer+and+IT&page={page}&descending=true")
    data = _get(url)
    if not data: return []
    out = []
    for r in data.get("results",[]):
        levels    = r.get("levels",[])
        locations = r.get("locations",[])
        out.append({
            "source":      "The Muse",
            "title":       r.get("name",""),
            "company":     r.get("company",{}).get("name",""),
            "location":    locations[0].get("name","Remote") if locations else "Remote",
            "description": _strip_html(r.get("contents",""))[:3000],
            "url":         r.get("refs",{}).get("landing_page",""),
            "salary":      "",
            "posted":      (r.get("publication_date") or "")[:10],
            "tags":        [levels[0].get("name","")] if levels else [],
        })
    return out

def fetch_themuse_parallel(pages: int = 20, max_workers: int = 8) -> list:
    out, seen = [], set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_muse_page, p): p for p in range(1, pages+1)}
        for fut in as_completed(futures):
            for job in fut.result():
                key = job["title"] + job["company"]
                if key and key not in seen:
                    seen.add(key); out.append(job)
    return out


# ── Remotive source ────────────────────────────────────────────────────────────

def _fetch_remotive_query(query: str, limit: int = 100) -> list:
    url = (f"https://remotive.com/api/remote-jobs"
           f"?search={urllib.parse.quote_plus(query)}&limit={limit}")
    data = _get(url)
    if not data: return []
    out = []
    for r in data.get("jobs",[]):
        out.append({
            "source":      "Remotive",
            "title":       r.get("title",""),
            "company":     r.get("company_name",""),
            "location":    r.get("candidate_required_location","Remote"),
            "description": _strip_html(r.get("description",""))[:3000],
            "url":         r.get("url",""),
            "salary":      r.get("salary",""),
            "posted":      (r.get("publication_date") or "")[:10],
            "tags":        r.get("tags",[]),
        })
    return out

def fetch_remotive_parallel(queries: list, limit: int = 100,
                             max_workers: int = 5) -> list:
    out, seen = [], set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_remotive_query, q, limit): q for q in queries}
        for fut in as_completed(futures):
            for job in fut.result():
                key = job.get("url") or (job["title"]+job["company"])
                if key and key not in seen:
                    seen.add(key); out.append(job)
    return out


# ── Arbeitnow source ──────────────────────────────────────────────────────────

def _fetch_arbeitnow_page(query: str, page: int) -> list:
    if query:
        url = (f"https://www.arbeitnow.com/api/job-board-api"
               f"?search={urllib.parse.quote_plus(query)}&page={page}")
    else:
        url = f"https://www.arbeitnow.com/api/job-board-api?page={page}"
    data = _get(url)
    if not data: return []
    out = []
    for r in data.get("data",[]):
        posted = str(r.get("created_at",""))[:10]
        out.append({
            "source":      "Arbeitnow",
            "title":       r.get("title",""),
            "company":     r.get("company_name",""),
            "location":    r.get("location","Remote"),
            "description": _strip_html(r.get("description",""))[:3000],
            "url":         r.get("url",""),
            "salary":      "",
            "posted":      posted,
            "tags":        r.get("tags",[]),
        })
    return out

def fetch_arbeitnow_parallel(queries: list, pages_per_query: int = 10,
                              unfiltered_pages: int = 20,
                              max_workers: int = 8) -> list:
    tasks = set()
    for q in (queries or [""]):
        for pg in range(1, pages_per_query+1):
            tasks.add((q, pg))
    for pg in range(1, unfiltered_pages+1):
        tasks.add(("", pg))

    out, seen = [], set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_arbeitnow_page, q, pg): (q,pg)
                   for q, pg in tasks}
        for fut in as_completed(futures):
            for job in fut.result():
                key = (job.get("url","").strip()
                       or job["title"]+job["company"])
                if key and key not in seen:
                    seen.add(key); out.append(job)
    return out


# ── Adzuna source (optional) ──────────────────────────────────────────────────

def fetch_adzuna(query: str, country: str = "us") -> list:
    app_id  = os.getenv("ADZUNA_APP_ID","")
    app_key = os.getenv("ADZUNA_APP_KEY","")
    if not app_id or not app_key: return []
    url = (f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
           f"?app_id={app_id}&app_key={app_key}"
           f"&results_per_page=50&what={urllib.parse.quote_plus(query)}")
    data = _get(url)
    if not data: return []
    out = []
    for r in data.get("results",[]):
        lo = r.get("salary_min"); hi = r.get("salary_max")
        sal = (f"${int(lo):,} – ${int(hi):,}" if lo and hi and lo!=hi
               else (f"${int(lo):,}+" if lo else ""))
        out.append({
            "source":      "Adzuna",
            "title":       r.get("title",""),
            "company":     r.get("company",{}).get("display_name",""),
            "location":    r.get("location",{}).get("display_name",""),
            "description": _strip_html(r.get("description",""))[:3000],
            "url":         r.get("redirect_url",""),
            "salary":      sal,
            "posted":      (r.get("created") or "")[:10],
            "tags":        [],
        })
    return out


# ── Geographic awareness ───────────────────────────────────────────────────────

def _geo_filter(jobs: list) -> list:
    NO_SPONSOR = re.compile(
        r"no visa|must be authorized|us citizen|"
        r"security clearance|must reside in|no sponsorship", re.I)
    REMOTE_PAT = re.compile(r"remote|work from home|wfh", re.I)
    for job in jobs:
        loc  = (job.get("location") or "").lower()
        desc = (job.get("description") or "")[:800]
        job["geo_flag"] = (bool(NO_SPONSOR.search(desc))
                           and not bool(REMOTE_PAT.search(loc+desc)))
    return jobs


# ── Smart query builder ────────────────────────────────────────────────────────

def _build_remoteok_tags(profile: dict) -> list:
    """Identity-aware RemoteOK tags."""
    try:
        from cv_reader.identity import extract_identity
        identity = extract_identity(profile)
        return identity["query_tags"][:10]
    except Exception:
        from cv_reader.parser import DOMAIN_REMOTEOK_TAGS
        domain = profile.get("domain","software")
        tags = list(DOMAIN_REMOTEOK_TAGS.get(domain, ["software-engineer"]))
        if profile.get("seniority","mid") == "junior":
            tags = ["intern","junior"] + tags
        return tags[:10]


def _build_keyword_queries(profile: dict) -> list:
    """
    LLM-aware query generation.

    Priority:
      1. Claude-generated query (profile["_llm_query"]) — prepended as first query
      2. Identity-based role queries (from extract_identity)
      3. Domain keyword fallback
    """
    base_queries: list[str] = []

    try:
        from cv_reader.identity import extract_identity
        identity = extract_identity(profile)
        base_queries = identity["query_roles"][:10]
    except Exception:
        from cv_reader.parser import DOMAIN_KEYWORD_QUERIES
        domain   = profile.get("domain", "software")
        modifier = "intern" if profile.get("seniority", "mid") == "junior" else ""
        qs = DOMAIN_KEYWORD_QUERIES.get(domain, ["software engineer"])
        base_queries = [f"{modifier} {q}".strip() if modifier else q for q in qs[:8]]

    # Prepend the LLM-generated query as the primary search signal
    llm_query = profile.get("_llm_query", "")
    if llm_query and llm_query not in base_queries:
        base_queries = [llm_query] + base_queries

    return base_queries[:10]



# ── Main entry point ──────────────────────────────────────────────────────────

def fetch_jobs(profile: dict, country: str = "us",
               remote_only: bool = False,
               max_jobs: int = 5000) -> list:
    """
    Title-aware, role-expanded, parallel multi-source job retrieval.
    Target: 2000–5000 unique jobs.
    """
    remoteok_tags    = _build_remoteok_tags(profile)
    keyword_queries  = _build_keyword_queries(profile)

    # Display retrieval plan
    print(f"  RemoteOK tags  : {', '.join(remoteok_tags[:10])}")
    print(f"  Keyword queries: {', '.join(keyword_queries[:6])}")

    all_jobs: list[dict] = []
    seen_urls: set[str]  = set()

    def _add(batch: list, label: str):
        added = 0
        for job in batch:
            key = (job.get("url","").strip()
                   or job.get("title","") + job.get("company",""))
            if key and key not in seen_urls:
                seen_urls.add(key); all_jobs.append(job); added += 1
        print(f"  Fetching {label:<22} {len(batch):>5} fetched → {added:>4} new  (total {len(all_jobs)})")

    t0 = time.time()

    # Run all sources concurrently
    with ThreadPoolExecutor(max_workers=4) as outer:
        f_rok     = outer.submit(fetch_remoteok_parallel,  remoteok_tags, max_workers=3)
        f_muse    = outer.submit(fetch_themuse_parallel,   pages=20, max_workers=8)
        f_remot   = outer.submit(fetch_remotive_parallel,  keyword_queries[:6], limit=100, max_workers=5)
        f_arb     = outer.submit(fetch_arbeitnow_parallel, keyword_queries[:4],
                                 pages_per_query=10, unfiltered_pages=20, max_workers=8)

        _add(f_rok.result(),   "RemoteOK")
        _add(f_muse.result(),  "The Muse")
        _add(f_remot.result(), "Remotive")
        _add(f_arb.result(),   "Arbeitnow")

    # Adzuna (optional)
    if os.getenv("ADZUNA_APP_ID"):
        az = []
        for q in keyword_queries[:3]:
            az += fetch_adzuna(q, country=country)
        _add(az, "Adzuna")

    # Geographic awareness
    all_jobs = _geo_filter(all_jobs)

    # Remote filter
    if remote_only:
        all_jobs = [j for j in all_jobs
                    if "remote" in j.get("location","").lower()
                    or j.get("source") in ("RemoteOK","Remotive")]

    elapsed = time.time() - t0
    print(f"  {'─'*51}")
    print(f"  Total unique    : {len(all_jobs)} jobs  ({elapsed:.1f}s)")
    return all_jobs[:max_jobs]
