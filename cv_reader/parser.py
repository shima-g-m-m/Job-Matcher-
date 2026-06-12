"""
cv_reader/parser.py
-------------------
Parses raw CV text into a structured profile dict.

Three modes (in order of preference):
  1. Groq LLM parser  — structured JSON via llama3, highest quality
  2. Local NLP parser — regex + keyword matching, works offline, no key needed
"""

import re
import json
import logging
from typing import Optional

log = logging.getLogger("job_matcher")

SKILLS_BY_DOMAIN = {
    "frontend": {
        "react", "vue", "angular", "svelte", "next.js", "nuxt", "typescript",
        "javascript", "css", "sass", "tailwind", "bootstrap", "webpack", "vite",
        "redux", "graphql", "websockets", "figma", "jest", "cypress",
        "html", "responsive design", "ui design", "ux design",
    },
    "backend": {
        "django","flask","fastapi","express","nestjs","spring","rails","laravel",
        "postgresql","mongodb","redis","mysql","graphql","rest api","grpc",
        "docker","microservices","api","jwt","oauth","node",
    },
    "software": {
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "scala", "kotlin", "swift", "dart", "php", "ruby", "bash", "shell",
        "react", "angular", "vue", "svelte", "next.js", "nuxt", "html", "css",
        "sass", "tailwind", "bootstrap", "webpack", "vite", "redux", "graphql",
        "node", "express", "django", "flask", "fastapi", "spring", "nestjs",
        "fastify", "laravel", "rails", "asp.net", "microservices", "api",
        "rest api", "websockets", "grpc", "oauth", "jwt",
        "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible",
        "ci/cd", "github actions", "linux", "nginx", "kafka", "serverless",
        "devops", "git", "github", "gitlab", "bitbucket", "jira",
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "dynamodb", "sqlite", "oracle", "firebase", "supabase",
        "unit testing", "tdd", "cypress", "jest", "pytest", "selenium",
        "agile", "scrum", "kanban", "figma", "ui design", "ux design",
    },
    "ai_ml": {
        "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
        "scikit-learn", "pandas", "numpy", "scipy", "nlp", "computer vision",
        "llm", "generative ai", "openai", "langchain", "transformers", "bert",
        "stable diffusion", "data analysis", "statistics", "mlops", "mlflow",
        "data visualization", "big data", "spark", "hadoop", "reinforcement learning",
        "feature engineering", "a/b testing", "time series", "hugging face",
        "object detection", "image classification", "recommender systems",
    },
    "mobile": {
        "flutter", "react native", "android", "ios", "kotlin", "swift",
        "firebase", "mobile development", "cross-platform", "material design",
        "android studio", "xcode", "android sdk", "ios sdk", "jetpack compose",
        "swiftui", "expo", "dart",
    },
    "data": {
        "sql", "postgresql", "mysql", "mongodb", "tableau", "power bi", "looker",
        "spark", "hadoop", "data warehouse", "data lake", "dbt", "airflow",
        "snowflake", "bigquery", "databricks", "etl", "data pipeline",
        "data analysis", "statistics", "excel", "r", "data visualization",
        "business intelligence", "bi", "reporting", "sas", "spss",
    },
    "finance": {
        "financial accounting", "auditing", "tax", "corporate finance",
        "financial reporting", "ifrs", "gaap", "budget", "forecasting",
        "financial analysis", "banking", "investment", "risk management",
        "credit analysis", "portfolio management", "derivatives", "equity",
        "fixed income", "treasury", "compliance", "kyc", "aml", "basel",
        "sap", "oracle financials", "quickbooks", "xero", "bloomberg",
        "financial modelling", "valuation", "due diligence", "m&a",
        "retail banking", "corporate banking", "trade finance",
    },
    "accounting": {
        "accounting", "bookkeeping", "accounts payable", "accounts receivable",
        "general ledger", "reconciliation", "payroll", "tax accounting",
        "cost accounting", "management accounting", "audit", "internal audit",
        "external audit", "financial statements", "balance sheet",
        "income statement", "cash flow", "ifrs", "gaap", "excel",
        "sap", "quickbooks", "xero", "sage",
    },
    "marketing": {
        "seo", "sem", "ppc", "google ads", "facebook ads", "social media",
        "content marketing", "email marketing", "crm", "hubspot", "salesforce",
        "marketing analytics", "google analytics", "brand management",
        "market research", "copywriting", "digital marketing", "growth hacking",
        "affiliate marketing", "influencer marketing", "marketing automation",
    },
    "design": {
        "figma", "sketch", "adobe xd", "photoshop", "illustrator", "indesign",
        "after effects", "premiere", "blender", "cinema 4d", "ui design",
        "ux design", "wireframing", "prototyping", "user research",
        "usability testing", "design systems", "typography", "branding",
        "graphic design", "motion design", "3d modeling",
    },
    "project_management": {
        "project management", "pmp", "prince2", "agile", "scrum", "kanban",
        "jira", "confluence", "asana", "trello", "ms project", "stakeholder",
        "risk management", "budget management", "resource planning",
        "waterfall", "program management", "portfolio management",
    },
    "sales": {
        "sales", "business development", "crm", "salesforce", "hubspot",
        "lead generation", "cold calling", "account management", "b2b", "b2c",
        "negotiation", "deal closing", "pipeline management", "quota",
        "customer success", "saas sales", "enterprise sales",
    },
    "hr": {
        "recruitment", "talent acquisition", "onboarding", "performance management",
        "employee relations", "hr policies", "compensation", "benefits",
        "workday", "bamboohr", "applicant tracking", "organizational development",
        "training", "learning development", "succession planning",
    },
    "devops_infra": {
        "docker", "kubernetes", "terraform", "ansible", "jenkins", "ci/cd",
        "aws", "azure", "gcp", "linux", "nginx", "kafka", "prometheus",
        "grafana", "datadog", "helm", "site reliability", "sre", "devops",
        "infrastructure as code", "cloud", "serverless",
    },
    "cybersecurity": {
        "cybersecurity", "penetration testing", "ethical hacking", "kali linux",
        "network security", "cryptography", "siem", "soc", "vulnerability",
        "owasp", "burp suite", "wireshark", "nmap", "firewall", "cissp",
        "ceh", "incident response", "threat intelligence", "zero trust",
    },
    "embedded": {
        "arduino", "raspberry pi", "embedded systems", "fpga", "vhdl", "verilog",
        "iot", "rtos", "freertos", "zephyr", "vxworks", "qnx", "threadx",
        "stm32", "esp32", "arm cortex", "nxp", "can bus", "lin bus", "flexray",
        "i2c", "spi", "uart", "usb", "ethernet", "j1939", "canopen",
        "pcb design", "altium", "kicad", "circuit simulation", "proteus",
        "autosar", "ecu", "bootloader", "embedded linux", "yocto", "buildroot",
        "uboot", "bare metal", "cmsis", "hal", "jtag", "openocd",
        "automotive", "firmware", "microcontroller",
    },
    "general": {
        "communication", "teamwork", "leadership", "problem solving",
        "critical thinking", "collaboration", "presentation", "agile",
        "project management", "time management", "analytical thinking",
        "excel", "powerpoint", "google sheets", "microsoft office",
        "data visualization", "reporting",
    },
}

TECH_SKILLS: set = set().union(*SKILLS_BY_DOMAIN.values())

DOMAIN_ROLES = {
    "frontend":          ["frontend developer", "react developer", "javascript developer",
                          "ui developer", "frontend engineer", "web developer"],
    "backend":           ["backend developer", "back-end developer", "api developer",
                          "backend engineer", "server-side developer",
                          "django", "fastapi", "flask", "express", "nestjs",
                          "rest api", "postgresql", "mongodb", "redis",
                          "microservices", "graphql", "api engineer"],
    "software":          ["software engineer", "full stack developer", "backend developer",
                          "frontend developer", "web developer"],
    "ai_ml":             ["machine learning engineer", "ai engineer", "data scientist",
                          "ml engineer", "ai researcher"],
    "mobile":            ["flutter developer", "mobile developer", "android developer",
                          "ios developer", "react native developer", "kotlin developer",
                          "mobile engineer", "cross-platform developer"],
    "data":              ["data analyst", "data engineer", "business intelligence analyst",
                          "data scientist", "bi developer"],
    "finance":           ["financial analyst", "investment analyst", "banking analyst",
                          "risk analyst", "finance manager", "treasury analyst",
                          "credit analyst"],
    "accounting":        ["accountant", "auditor", "financial accountant",
                          "management accountant", "tax accountant", "bookkeeper"],
    "marketing":         ["digital marketing specialist", "marketing analyst",
                          "seo specialist", "content marketer", "growth marketer"],
    "design":            ["ui designer", "ux designer", "graphic designer",
                          "product designer", "visual designer"],
    "project_management":["project manager", "scrum master", "product manager",
                          "program manager"],
    "sales":             ["sales executive", "business development manager",
                          "account manager", "sales representative"],
    "hr":                ["hr specialist", "recruiter", "talent acquisition specialist",
                          "hr business partner"],
    "devops_infra":      ["devops engineer", "cloud engineer", "site reliability engineer",
                          "infrastructure engineer", "platform engineer"],
    "cybersecurity":     ["security engineer", "penetration tester", "security analyst",
                          "information security analyst"],
    "embedded":          ["embedded systems engineer", "firmware engineer",
                          "hardware engineer", "iot engineer"],
}

DOMAIN_REMOTEOK_TAGS = {
    "frontend":          ["frontend", "react", "javascript", "typescript", "css"],
    "backend":           ["backend", "python", "nodejs", "django", "fastapi"],
    "software":          ["software-engineer", "backend", "fullstack", "javascript", "python"],
    "ai_ml":             ["machine-learning", "ai", "data-science", "python", "deep-learning"],
    "mobile":            ["mobile", "flutter", "android", "ios", "react-native", "kotlin", "swift", "dart"],
    "data":              ["data-science", "sql", "python", "analytics"],
    "finance":           ["finance", "fintech", "accounting"],
    "accounting":        ["accounting", "finance"],
    "marketing":         ["marketing", "seo", "growth"],
    "design":            ["design", "ux", "ui"],
    "project_management":["product", "project-management"],
    "sales":             ["sales", "business-development"],
    "hr":                ["hr", "recruiting"],
    "devops_infra":      ["devops", "aws", "kubernetes", "cloud"],
    "cybersecurity":     ["security", "cybersecurity"],
    "embedded":          ["embedded", "hardware", "iot"],
}

DOMAIN_KEYWORD_QUERIES = {
    "frontend":          ["frontend developer", "react developer", "javascript developer",
                          "ui developer", "frontend engineer"],
    "backend":           ["backend developer", "python developer", "api developer", "django developer", "node developer"],
    "software":          ["software engineer", "web developer", "full stack developer"],
    "ai_ml":             ["machine learning engineer", "data scientist", "ai engineer"],
    "mobile":            ["mobile developer", "flutter developer", "android developer"],
    "data":              ["data analyst", "data engineer", "business intelligence"],
    "finance":           ["financial analyst", "finance analyst", "investment analyst",
                          "banking analyst", "treasury analyst"],
    "accounting":        ["accountant", "auditor", "financial accountant",
                          "management accountant"],
    "marketing":         ["digital marketing", "marketing specialist", "seo specialist"],
    "design":            ["ui ux designer", "graphic designer", "product designer"],
    "project_management":["project manager", "product manager", "scrum master"],
    "sales":             ["sales executive", "business development", "account manager"],
    "hr":                ["hr specialist", "recruiter", "talent acquisition"],
    "devops_infra":      ["devops engineer", "cloud engineer", "sre"],
    "cybersecurity":     ["security engineer", "penetration tester", "security analyst"],
    "embedded":          ["embedded engineer", "firmware engineer", "iot engineer"],
}

# Groq-powered LLM CV parser

GROQ_CV_SYSTEM = """You are a senior technical recruiter with expertise across all industries.
Analyse the CV and return ONLY a valid JSON object — no markdown fences, no explanation, no preamble.
Be precise and extract only what is actually stated in the CV."""

GROQ_CV_PROMPT = """Analyse this CV and return ONLY a valid JSON object with exactly these fields:

{{
  "name": "full name or empty string",
  "current_title": "most recent job title or professional role",
  "years_experience": <integer, 0 if student/unclear>,
  "seniority": "junior | mid | senior | lead | executive",
  "domain": "primary domain: software | ai_ml | mobile | data | finance | accounting | marketing | design | project_management | sales | hr | devops_infra | cybersecurity | embedded | other",
  "skills_hard": ["exact technical skills, tools, languages, frameworks — be exhaustive"],
  "skills_soft": ["communication", "leadership" ...],
  "industries": ["sectors the person has worked in"],
  "education": ["degree · field · institution · year"],
  "certifications": ["certifications held"],
  "languages_spoken": ["English", "Arabic" ...],
  "job_titles_held": ["Job Title at Company (2021-2023)"],
  "key_achievements": ["quantified achievements e.g. Led migration that cut costs 40%"],
  "preferred_roles": ["specific role titles most relevant to this candidate"],
  "summary_for_search": "2-3 sentence recruiter pitch describing this candidate"
}}

CV TEXT:
{cv_text}"""


def _call_groq(prompt: str, api_key: str, model: str = "llama3-8b-8192",
               system: str = "", max_tokens: int = 1500) -> Optional[str]:
    """Call Groq chat completions. Returns text or None on failure."""
    import urllib.request
    import json as _json

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = _json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        log.warning(f"Groq API call failed: {exc}")
        return None


def parse_groq(cv_text: str, api_key: str) -> Optional[dict]:
    """
    Use Groq LLM to extract a rich CV profile.
    Returns a dict on success, None if the API call fails or returns invalid JSON.
    """
    prompt = GROQ_CV_PROMPT.format(cv_text=cv_text[:6000])
    raw = _call_groq(prompt, api_key=api_key, system=GROQ_CV_SYSTEM, max_tokens=1500)
    if not raw:
        return None

    # Strip markdown fences if the model wrapped the output anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        profile = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning(f"Groq returned invalid JSON: {exc}\nRaw: {raw[:300]}")
        return None

    profile.setdefault("name", "")
    profile.setdefault("current_title", "")
    profile.setdefault("years_experience", 0)
    profile.setdefault("seniority", "junior")
    profile.setdefault("domain", "software")
    profile.setdefault("skills_hard", [])
    profile.setdefault("skills_soft", [])
    profile.setdefault("industries", [])
    profile.setdefault("education", [])
    profile.setdefault("certifications", [])
    profile.setdefault("languages_spoken", ["English"])
    profile.setdefault("job_titles_held", [])
    profile.setdefault("key_achievements", [])
    profile.setdefault("preferred_roles", [])
    profile.setdefault("summary_for_search", "")

    # Ensure list types
    for list_field in ("skills_hard", "skills_soft", "industries", "education",
                        "certifications", "languages_spoken", "job_titles_held",
                        "key_achievements", "preferred_roles"):
        if not isinstance(profile[list_field], list):
            profile[list_field] = [str(profile[list_field])]

    # Ensure int
    try:
        profile["years_experience"] = int(profile["years_experience"])
    except (TypeError, ValueError):
        profile["years_experience"] = 0

    # Add preferred_roles from domain table if LLM left them empty
    if not profile["preferred_roles"]:
        domain = profile.get("domain", "software")
        profile["preferred_roles"] = DOMAIN_ROLES.get(domain, DOMAIN_ROLES["software"])[:6]

    log.info("CV parsed via Groq LLM")
    return profile


# Local NLP parser (unchanged — regex + keyword matching)

SENIORITY_PATTERNS = {
    "executive": [r"\bdirector\b", r"\bvp\b", r"\bcto\b", r"\bceo\b", r"\bchief\b"],
    "lead":      [r"\blead\b", r"\bprincipal\b", r"\bstaff\b", r"\barchitect\b"],
    "senior":    [r"\bsenior\b", r"\bsr\.?\b"],
    "mid":       [r"\bmid\b", r"\bintermediate\b", r"\bassociate\b"],
    "junior":    [r"\bjunior\b", r"\bjr\.?\b", r"\bintern\b", r"\bgraduate\b",
                  r"\bentry.level\b", r"\btrainee\b", r"\bstudent\b",
                  r"\bseeking.*internship\b", r"\bsummer internship\b"],
}

DOMAIN_DETECTION_KEYWORDS = {
    "finance":           ["financial accounting", "auditing", "corporate finance",
                          "banking laws", "investment banking", "retail banking",
                          "credit analysis", "risk management", "treasury",
                          "financial modelling", "valuation", "bloomberg",
                          "ifrs", "gaap", "kyc", "aml"],
    "accounting":        ["accounting", "bookkeeping", "accounts payable",
                          "accounts receivable", "general ledger", "reconciliation",
                          "payroll", "tax accounting", "audit", "financial statements",
                          "cost accounting", "management accounting"],
    "ai_ml":             ["machine learning", "deep learning", "tensorflow", "pytorch",
                          "nlp", "computer vision", "data science", "artificial intelligence"],
    "mobile":            ["flutter", "react native", "android development",
                          "ios development", "mobile app", "dart"],
    "data":              ["data engineering", "data pipeline", "etl", "business intelligence",
                          "data warehouse", "tableau", "power bi", "spark", "hadoop"],
    "embedded":          ["embedded software", "firmware engineer", "embedded developer",
                          "autosar", "rtos", "freertos", "microcontroller",
                          "stm32", "arm cortex", "ecu", "can bus", "iot firmware",
                          "embedded linux", "bootloader", "bare metal", "fpga"],
    "devops_infra":      ["devops", "site reliability", "infrastructure as code",
                          "kubernetes", "terraform", "ci/cd pipeline"],
    "cybersecurity":     ["cybersecurity", "penetration testing", "ethical hacking",
                          "network security", "vulnerability assessment", "soc analyst"],
    "marketing":         ["digital marketing", "seo", "google ads", "social media marketing",
                          "content marketing", "email marketing", "growth hacking"],
    "design":            ["ui/ux", "user experience", "user interface", "graphic design",
                          "visual design", "product design", "wireframing"],
    "project_management":["project manager", "pmp", "prince2", "scrum master",
                          "program manager", "delivery manager"],
    "sales":             ["sales manager", "business development", "account executive",
                          "sales representative", "b2b sales", "enterprise sales"],
    "hr":                ["human resources", "talent acquisition", "recruitment",
                          "hr business partner", "people operations"],
    "frontend":          ["frontend developer", "front-end developer", "ui developer",
                          "react developer", "vue developer", "javascript developer",
                          "web ui", "frontend engineer", "next.js developer",
                          "react", "vue.js", "next.js", "angular", "svelte",
                          "css", "tailwind", "sass", "webpack", "vite",
                          "frontend", "front-end"],
    "backend":           ["backend developer", "back-end developer", "api developer",
                          "server-side developer", "backend engineer",
                          "python backend", "node backend", "django developer",
                          "fastapi developer", "flask developer", "rest api developer"],
    "software":          ["software engineer", "web developer", "backend", "frontend",
                          "full stack", "react", "django", "javascript", "python"],
}


def _detect_domain(lower: str, skills_hard: list) -> str:
    scores: dict[str, int] = {d: 0 for d in DOMAIN_DETECTION_KEYWORDS}
    for domain, keywords in DOMAIN_DETECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[domain] += 2
    skills_set = set(skills_hard)
    for domain, domain_skills in SKILLS_BY_DOMAIN.items():
        if domain in scores:
            overlap = len(skills_set & domain_skills)
            scores[domain] += overlap
    if scores.get("finance", 0) > 0 and scores.get("accounting", 0) > 0:
        scores["finance"] = max(scores["finance"], scores["accounting"])
    specific_domains = ["frontend", "backend", "ai_ml", "mobile", "devops", "security",
                        "embedded", "finance", "accounting", "marketing",
                        "design", "hr", "sales", "data"]
    best_specific = max(
        (d for d in specific_domains if d in scores),
        key=lambda d: scores.get(d, 0),
        default=None
    )
    if best_specific and scores.get(best_specific, 0) >= 2:
        return best_specific
    best = max(scores, key=lambda d: scores[d])
    if scores[best] == 0:
        return "software"
    return best


def _seniority_from_experience(years: int, text_seniority: str,
                                explicit_junior: bool = False) -> str:
    if explicit_junior:
        return "junior"
    if text_seniority in ("lead", "executive"):
        return text_seniority
    if years == 0:
        return "junior"
    if years <= 2:
        return "junior"
    if years <= 5:
        return "mid"
    if years <= 10:
        return "senior"
    return "senior"


def _extract_job_titles(cv_text: str, skills_hard: list) -> list:
    lower = cv_text.lower()
    lines = [l.strip() for l in cv_text.splitlines() if l.strip()]

    JOB_MARKERS = re.compile(
        r"\b(intern|engineer|developer|analyst|scientist|researcher|"
        r"designer|manager|consultant|specialist|officer|architect|"
        r"accountant|auditor|coordinator|administrator)\b", re.I)

    EDU_MARKERS = re.compile(
        r"\b(bachelor|master|phd|degree|university|college|gpa|coursework|"
        r"udemy|kaggle|coursera|edx|certificate|online)\b", re.I)

    WORK_SECTION = re.compile(
        r"^(work experience|professional experience|employment|"
        r"internship|internships|career|positions)$", re.I)
    STOP_SECTION = re.compile(
        r"^(education|projects?|skills|certifications?|courses?|"
        r"self.learning|achievements|publications|awards)$", re.I)

    in_work = False
    titles   = []

    for line in lines:
        ll = line.strip()
        if WORK_SECTION.match(ll) and len(ll) < 60:
            in_work = True; continue
        if STOP_SECTION.match(ll) and len(ll) < 60:
            in_work = False; continue
        if not in_work:
            continue
        if EDU_MARKERS.search(ll):
            continue
        if not JOB_MARKERS.search(ll):
            continue

        title_part = re.split(r"\s*[|@–—]\s*|\s+at\s+|\s*,\s*", ll)[0].strip()
        title_part = re.sub(
            r"\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec).*$",
            "", title_part, flags=re.I).strip()
        title_part = re.sub(r"\s+\d{4}.*$", "", title_part).strip()

        if (3 < len(title_part) < 80 and
                JOB_MARKERS.search(title_part) and
                title_part.lower() not in {t.lower() for t in titles}):
            titles.append(title_part)

    return titles[:5]


def parse_local(cv_text: str) -> dict:
    """Domain-aware CV parser using regex + keyword matching. Works offline."""
    lower = cv_text.lower()
    lines = [l.strip() for l in cv_text.splitlines() if l.strip()]

    skills_hard = sorted({
        s for s in TECH_SKILLS
        if re.search(r'\b' + re.escape(s) + r'\b', lower)
    })

    domain = _detect_domain(lower, skills_hard)

    text_seniority = "junior"
    for level in ["executive", "lead", "senior", "mid", "junior"]:
        patterns = SENIORITY_PATTERNS[level]
        if any(re.search(p, lower) for p in patterns):
            text_seniority = level
            break

    import datetime
    current_year = datetime.datetime.now().year

    STUDENT_SIGNALS = [
        r"\bstudent\b", r"seeking.*internship", r"summer internship",
        r"expected.*graduation", r"expected.*jun", r"expected.*may",
        r"\bgpa\b", r"\bcgpa\b", r"undergraduate", r"freshman",
        r"sophomore", r"junior year", r"senior year",
    ]
    is_student = any(re.search(s, lower) for s in STUDENT_SIGNALS)

    explicit_yrs = re.findall(
        r"(\d+)\+?\s*years?\s*(?:of\s+)?(?:professional\s+)?experience", lower)
    explicit_years = max((int(y) for y in explicit_yrs), default=0)

    WORK_SECTION  = re.compile(
        r"^(work experience|professional experience|employment|"
        r"career history|internship|internships|positions held)$", re.I)
    STOP_SECTION  = re.compile(
        r"^(education|academic|university|college|school|courses?|"
        r"self.learning|projects?|certifications?|certificates?|"
        r"technical skills|skills|languages|achievements|publications|"
        r"awards|honors|references|volunteer)$", re.I)
    INLINE_STOP = re.compile(
        r"^(education|skills|certifications?|projects?)[\s:]+", re.I)
    JOB_LINE = re.compile(
        r"\b(intern|engineer|developer|analyst|manager|officer|"
        r"consultant|researcher|assistant|coordinator|specialist)\b", re.I)
    COURSE_LINE = re.compile(
        r"\b(udemy|kaggle|coursera|edx|harvard|online|course|"
        r"project|simulation|certificate|self.learning|bootcamp)\b", re.I)

    in_work_section = False
    work_months = 0.0
    seen_job_ranges: list = []

    for line in lines:
        stripped = line.strip()
        ll = stripped.lower()

        if WORK_SECTION.match(stripped) and len(stripped) < 60:
            in_work_section = True; continue
        if STOP_SECTION.match(stripped) and len(stripped) < 60:
            in_work_section = False; continue
        if INLINE_STOP.match(stripped) and len(stripped) < 60:
            in_work_section = False; continue
        if not in_work_section:
            continue
        if COURSE_LINE.search(ll):
            continue

        week_match = re.search(r"(\d+)\s*weeks?", ll)
        if week_match and JOB_LINE.search(ll):
            work_months += int(week_match.group(1)) / 4.33
            continue

        month_match = re.search(r"(\d+)\s*months?", ll)
        if month_match and JOB_LINE.search(ll):
            work_months += int(month_match.group(1))
            continue

        if JOB_LINE.search(ll):
            ranges = re.findall(
                r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*"
                r"(\d{4})\s*[-–—]\s*"
                r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*"
                r"(\d{4}|present|current|now)", ll)
            for start_s, end_s in ranges:
                start_y = int(start_s)
                end_y = current_year if end_s in ("present","current","now") else int(end_s)
                if 1990 <= start_y <= current_year:
                    months = max(0, (end_y - start_y) * 12)
                    months = min(months, 36)
                    seen_job_ranges.append(months)

    work_months += sum(seen_job_ranges)
    exp_from_roles = round(work_months / 12, 1)

    if is_student and explicit_years == 0 and exp_from_roles < 1:
        years_exp = 0
    else:
        years_exp = max(explicit_years, int(exp_from_roles))
    years_exp = min(years_exp, 40)

    explicit_junior = any(
        re.search(p, lower)
        for p in SENIORITY_PATTERNS["junior"]
    )
    seniority = _seniority_from_experience(years_exp, text_seniority, explicit_junior)

    name = ""
    for line in lines[:6]:
        words = line.split()
        if (2 <= len(words) <= 5
                and all(w[0].isupper() for w in words if w and w[0].isalpha())
                and not any(ch.isdigit() for ch in line)
                and "@" not in line):
            name = line
            break

    title = ""
    title_hints = [
        "engineer", "developer", "scientist", "analyst", "designer",
        "manager", "intern", "student", "researcher", "architect",
        "consultant", "specialist", "officer", "accountant", "auditor",
        "banker", "advisor", "executive", "director", "associate",
        "coordinator", "recruiter", "marketer",
    ]
    for line in lines[1:10]:
        if any(h in line.lower() for h in title_hints) and len(line) < 100:
            title = line
            break

    domain_roles = DOMAIN_ROLES.get(domain, DOMAIN_ROLES["software"])
    all_roles_flat = [r for roles in DOMAIN_ROLES.values() for r in roles]
    detected_roles = [r for r in all_roles_flat if r in lower]
    seen_r: set[str] = set()
    merged_roles: list[str] = []
    for r in domain_roles + detected_roles:
        if r not in seen_r:
            seen_r.add(r)
            merged_roles.append(r)

    if seniority == "junior":
        intern_variants = [f"{r} intern" for r in merged_roles[:3]]
        roles = intern_variants + merged_roles
    else:
        roles = merged_roles

    edu = []
    for line in lines:
        if re.search(r'\b(bachelor|master|phd|b\.?sc|m\.?sc|mba|b\.?eng|b\.?com|'
                     r'diploma|degree|llb|bba|bca|btech)\b', line.lower()):
            edu.append(line[:120])

    cert_kw = ["certified", "certification", "certificate", "aws", "google cloud",
               "azure", "pmp", "cissp", "comptia", "cpa", "acca", "cfa", "cma",
               "ceh", "prince2", "pmi", "itil"]
    certs = [l for l in lines if any(k in l.lower() for k in cert_kw) and len(l) < 120]

    spoken = ["English"]
    for lang in ["Arabic", "French", "German", "Spanish", "Chinese", "Japanese",
                 "Portuguese", "Italian", "Russian", "Turkish", "Hindi"]:
        if lang.lower() in lower:
            spoken.append(lang)

    non_intern_roles = [r for r in roles if "intern" not in r]
    top_roles_str = ", ".join(non_intern_roles[:3])
    level_str = "intern / entry-level" if seniority == "junior" else seniority
    top_skills_str = ", ".join(skills_hard[:20]) if skills_hard else "general professional skills"

    summary = (
        f"Seeking {level_str} position"
        + (f" as {top_roles_str}" if top_roles_str else "")
        + f". Skills: {top_skills_str}"
        + (f". Education: {edu[0]}" if edu else "")
        + (f". {title}" if title else "")
    )

    return {
        "name":             name,
        "current_title":    title,
        "years_experience": years_exp,
        "seniority":        seniority,
        "domain":           domain,
        "skills_hard":      skills_hard,
        "skills_soft":      ["communication", "teamwork", "problem solving"],
        "industries":       [],
        "education":        edu[:3],
        "certifications":   certs[:5],
        "languages_spoken": spoken,
        "job_titles_held":  _extract_job_titles(cv_text, skills_hard),
        "key_achievements": [],
        "preferred_roles":  roles[:8],
        "summary_for_search": summary,
    }


def parse_cv(cv_text: str, api_key: Optional[str] = None) -> dict:
    """
    Main entry point for CV parsing.

    If a Groq API key is available (passed directly or via GROQ_API_KEY env var),
    uses the LLM parser for richer extraction. Falls back to the local NLP parser
    if the API call fails or no key is provided.
    """
    import os
    key = api_key or os.getenv("GROQ_API_KEY", "")

    if key:
        log.info("Attempting Groq LLM CV parsing …")
        result = parse_groq(cv_text, api_key=key)
        if result:
            return result
        log.warning("Groq CV parsing failed — falling back to local NLP parser")

    return parse_local(cv_text)
