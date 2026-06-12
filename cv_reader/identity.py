"""
cv_reader/identity.py
─────────────────────
Professional Identity Extractor.

Converts a parsed CV profile into a rich identity vector:
  - role_identity:   what the person IS professionally
  - domain_identity: which sub-domain they belong to
  - industry_tags:   which industries they've worked in
  - tech_stack:      their core technology fingerprint
  - specializations: specific sub-specializations
  - query_roles:     ordered list of role queries for retrieval
  - query_tags:      RemoteOK/API tags
"""

import re
from typing import Optional


# ── Role taxonomy: domain → role cluster → role variants ─────────────────────

ROLE_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "embedded": {
        "firmware":      ["firmware engineer","firmware developer","embedded firmware engineer",
                          "embedded c engineer","bare metal engineer"],
        "autosar":       ["autosar engineer","autosar developer","autosar software engineer",
                          "automotive software engineer","ecu software engineer"],
        "rtos":          ["rtos engineer","real-time embedded engineer","embedded systems engineer",
                          "embedded software engineer"],
        "iot":           ["iot engineer","iot developer","embedded iot engineer",
                          "connected devices engineer"],
        "embedded_linux":["embedded linux engineer","linux kernel engineer",
                          "yocto engineer","buildroot engineer"],
    },
    "ai_ml": {
        "ml_engineer":   ["machine learning engineer","ml engineer","applied ml engineer",
                          "deep learning engineer"],
        "ai_engineer":   ["ai engineer","artificial intelligence engineer","applied ai engineer"],
        "nlp":           ["nlp engineer","natural language processing engineer","llm engineer",
                          "conversational ai engineer","text ai engineer"],
        "cv_engineer":   ["computer vision engineer","vision engineer","image ai engineer",
                          "perception engineer"],
        "data_scientist":["data scientist","applied scientist","ml researcher","research scientist"],
        "mlops":         ["mlops engineer","ml platform engineer","ml infrastructure engineer",
                          "ai platform engineer"],
    },
    "frontend": {
        "react":         ["react developer","react engineer","next.js developer","frontend react"],
        "vue":           ["vue developer","nuxt developer","vue.js engineer"],
        "angular":       ["angular developer","angular engineer","frontend angular"],
        "general_fe":    ["frontend developer","frontend engineer","ui developer",
                          "javascript developer","web developer"],
        "typescript":    ["typescript developer","typescript frontend engineer"],
    },
    "backend": {
        "python_be":     ["python developer","django developer","fastapi developer",
                          "flask developer","python backend engineer"],
        "java_be":       ["java developer","spring developer","java backend engineer"],
        "node_be":       ["node developer","express developer","nestjs developer",
                          "javascript backend engineer"],
        "general_be":    ["backend developer","backend engineer","api developer",
                          "server-side developer"],
    },
    "fullstack": {
        "js_fullstack":  ["full stack javascript developer","mern developer","mean developer",
                          "full stack react node developer"],
        "python_fullstack":["full stack python developer","django full stack","flask full stack"],
        "general_fs":    ["full stack developer","fullstack engineer","software developer"],
    },
    "mobile": {
        "flutter":       ["flutter developer","flutter engineer","dart developer",
                          "cross-platform mobile developer"],
        "android":       ["android developer","android engineer","kotlin developer",
                          "android mobile developer"],
        "ios":           ["ios developer","swift developer","ios engineer","apple developer"],
        "react_native":  ["react native developer","cross-platform developer","mobile developer"],
        "general_mob":   ["mobile developer","mobile engineer","app developer"],
    },
    "devops": {
        "cloud":         ["cloud engineer","aws engineer","azure engineer","gcp engineer",
                          "cloud architect","cloud developer"],
        "kubernetes":    ["kubernetes engineer","k8s engineer","platform engineer",
                          "container engineer","helm engineer"],
        "ci_cd":         ["devops engineer","ci/cd engineer","build engineer",
                          "release engineer","infrastructure engineer"],
        "sre":           ["site reliability engineer","sre","reliability engineer",
                          "production engineer"],
    },
    "data": {
        "data_eng":      ["data engineer","data pipeline engineer","etl developer",
                          "big data engineer","data infrastructure engineer"],
        "analyst":       ["data analyst","business intelligence analyst","bi analyst",
                          "analytics engineer","reporting analyst"],
        "data_sci":      ["data scientist","analytics scientist","quantitative analyst"],
    },
    "security": {
        "pentester":     ["penetration tester","ethical hacker","red team engineer",
                          "security researcher","offensive security engineer"],
        "soc":           ["soc analyst","security operations analyst","threat analyst",
                          "incident response engineer"],
        "appsec":        ["application security engineer","appsec engineer",
                          "devsecops engineer","product security engineer"],
    },
    "finance": {
        "analyst":       ["financial analyst","investment analyst","banking analyst",
                          "equity analyst","credit analyst"],
        "quant":         ["quantitative analyst","quant developer","quant researcher",
                          "algorithmic trading developer"],
        "accounting":    ["accountant","financial accountant","management accountant",
                          "tax accountant","auditor"],
    },
}

# ── Industry signals: keywords → industry tag ─────────────────────────────────

INDUSTRY_SIGNALS: dict[str, list[str]] = {
    "automotive":   ["autosar","can bus","ecu","adas","vehicle","automotive","obd","lim",
                     "bmw","volkswagen","mercedes","bosch","continental","delphi"],
    "fintech":      ["fintech","payment","banking","trading","blockchain","cryptocurrency",
                     "financial technology","digital payment","neo bank"],
    "healthcare":   ["medical","healthcare","clinical","patient","hospital","pharma",
                     "health tech","medical device","fda","hipaa","ehr","emr"],
    "robotics":     ["robot","robotics","ros","gazebo","manipulation","autonomous",
                     "industrial automation","mechanical","servo","pid"],
    "gaming":       ["game","unity","unreal","gaming","game engine","shader","opengl",
                     "directx","vulkan","game developer"],
    "ai_research":  ["research lab","deepmind","openai","paper","publication","arxiv",
                     "phd","research scientist","academic","university research"],
    "cloud_infra":  ["aws","azure","gcp","iaas","paas","cloud native","serverless",
                     "infrastructure","terraform","kubernetes"],
    "cybersecurity":["pentest","cve","vulnerability","exploit","malware","threat",
                     "siem","soc","red team","blue team","ctf"],
    "data_platform":["data lake","data warehouse","spark","hadoop","databricks",
                     "snowflake","dbt","kafka","data mesh","lakehouse"],
    "ecommerce":    ["ecommerce","e-commerce","shopify","woocommerce","marketplace",
                     "retail tech","checkout","cart","payment gateway"],
}

# ── Technical specialization detection ────────────────────────────────────────

SPECIALIZATION_SIGNALS: dict[str, list[str]] = {
    # Embedded
    "autosar":       ["autosar","arxml","runnable","swc","ecuc","composition","bsw","rte"],
    "rtos":          ["rtos","freertos","zephyr","vxworks","qnx","threadx","cmsis"],
    "can_bus":       ["canopen","j1939","lin bus","flexray","can bus","automotive bus",
                       "can fd","iso 15765","obd-ii"],
    "arm_cortex":    ["stm32","arm cortex","cortex-m","nxp","ti microcontroller","esp32",
                      "arduino","embedded c","bare metal","hal","cmsis"],
    "fpga":          ["fpga","vhdl","verilog","xilinx","intel fpga","altera","pld"],
    # AI
    "transformers":  ["transformer","bert","gpt","llm","langchain","hugging face",
                      "attention mechanism","fine-tuning","rlhf"],
    "computer_vision":["opencv","yolo","object detection","image segmentation","cnn",
                       "feature extraction","point cloud","lidar"],
    "rl":            ["reinforcement learning","q-learning","ppo","dqn","policy gradient",
                      "reward","environment","gym","stable baselines"],
    # Frontend
    "react_ecosystem":["react","next.js","redux","zustand","react query","react native",
                       "styled components","chakra","material ui"],
    "vue_ecosystem": ["vue","nuxt","vuex","pinia","vue router","vite vue"],
    # Backend
    "microservices": ["microservices","service mesh","grpc","event driven","saga pattern",
                      "api gateway","circuit breaker","istio"],
    "data_pipeline": ["apache kafka","flink","spark streaming","airflow","celery",
                      "message queue","rabbitmq","event streaming"],
    # Finance
    "quantitative":  ["quant","alpha","factor","portfolio optimisation","risk model",
                      "monte carlo","black scholes","stochastic"],
    "sap_ecosystem": ["sap","s4hana","abap","sap fi","sap co","sap mm","fiori","bapi"],
}


def extract_identity(profile: dict, cv_text: str = "") -> dict:
    """
    Extract professional identity from a parsed CV profile.

    Returns a rich identity dict with:
      - role_identity:   primary role cluster(s)
      - domain_identity: domain + sub-domain
      - industry_tags:   detected industries
      - tech_stack:      core technology fingerprint
      - specializations: detected specializations
      - query_roles:     ordered retrieval queries (most specific first)
      - query_tags:      API tag keywords
    """
    lower = cv_text.lower() if cv_text else ""
    lower += " " + " ".join(profile.get("skills_hard", [])).lower()
    lower += " " + " ".join(profile.get("preferred_roles", [])).lower()
    lower += " " + (profile.get("current_title") or "").lower()
    lower += " " + " ".join(profile.get("job_titles_held", [])).lower()

    domain     = profile.get("domain", "software")
    seniority  = profile.get("seniority", "mid")
    roles_raw  = profile.get("preferred_roles", [])

    # ── 1. Detect specializations ─────────────────────────────────────────────
    specs = []
    for spec, signals in SPECIALIZATION_SIGNALS.items():
        hits = sum(1 for s in signals if s in lower)
        # Require at least 2 signal matches to confirm specialization
        # (avoids false positives from single coursework mentions)
        # Embedded specs need 2+ hits, AI/frontend specs need 1+
        if spec in ("react_ecosystem","vue_ecosystem","microservices",
                    "transformers","computer_vision","rl",
                    "sap_ecosystem","quantitative"):
            threshold = 1
        else:
            threshold = 2
        if hits >= threshold:
            specs.append((spec, hits))
    specs.sort(key=lambda x: -x[1])
    active_specs = [s for s, _ in specs[:4]]

    # ── 2. Detect industries ──────────────────────────────────────────────────
    industries = []
    for ind, signals in INDUSTRY_SIGNALS.items():
        if sum(1 for s in signals if s in lower) >= 1:
            industries.append(ind)

    # ── 3. Map domain → role taxonomy cluster ─────────────────────────────────
    dom_key = domain.replace("_infra","").replace("data_sci","data")
    taxonomy = ROLE_TAXONOMY.get(dom_key, ROLE_TAXONOMY.get("backend", {}))

    # Score each cluster
    cluster_scores: dict[str, int] = {}
    for cluster, role_variants in taxonomy.items():
        score = sum(1 for rv in role_variants if rv in lower)
        # Also check specializations
        if cluster in active_specs: score += 2
        if score > 0: cluster_scores[cluster] = score

    # Best cluster
    best_cluster = max(cluster_scores, key=lambda c: cluster_scores[c]) \
        if cluster_scores else list(taxonomy.keys())[0] if taxonomy else None

    # ── 4. Build ordered retrieval queries (identity-first) ───────────────────
    query_roles: list[str] = []
    modifier = "intern" if seniority == "junior" else ""

    def _add(roles: list):
        for r in roles:
            q = f"{modifier} {r}".strip() if modifier else r
            if q not in query_roles and len(q) > 5:
                query_roles.append(q)

    # Priority 1: best cluster roles (most specific identity)
    if best_cluster and best_cluster in taxonomy:
        _add(taxonomy[best_cluster][:3])

    # Priority 2: other cluster roles from this domain
    for cluster, roles in taxonomy.items():
        if cluster != best_cluster:
            _add(roles[:2])

    # Priority 3: preferred roles from parser
    non_intern = [r for r in roles_raw if "intern" not in r]
    _add(non_intern[:4])

    # Priority 4: previous job titles
    for title in profile.get("job_titles_held", [])[:2]:
        clean = re.sub(r'\b(senior|lead|principal|staff|intern|junior)\b', '',
                       title, flags=re.I)
        clean = re.sub(r'[&|/\\]', ' ', clean).strip()
        if len(clean) > 5:
            q = f"{modifier} {clean}".strip() if modifier else clean
            if q not in query_roles:
                query_roles.append(q)

    # Priority 5: specialization-specific queries
    spec_queries = {
        "autosar":       ["autosar engineer","automotive software engineer","ecu developer"],
        "rtos":          ["rtos engineer","embedded systems engineer","firmware engineer"],
        "arm_cortex":    ["embedded c developer","stm32 developer","firmware engineer"],
        "fpga":          ["fpga engineer","vhdl developer","digital design engineer"],
        "transformers":  ["llm engineer","nlp engineer","ai engineer"],
        "computer_vision":["computer vision engineer","cv engineer","vision ai engineer"],
        "react_ecosystem":["react developer","frontend react engineer","next.js developer"],
        "microservices": ["microservices engineer","backend engineer","api developer"],
        "sap_ecosystem": ["sap consultant","sap developer","erp consultant"],
        "quantitative":  ["quant developer","quantitative analyst","algo trader"],
    }
    for spec in active_specs[:2]:
        _add(spec_queries.get(spec, []))

    # ── 5. Build API tags ─────────────────────────────────────────────────────
    TAG_MAP: dict[str, list[str]] = {
        "embedded":    ["embedded","firmware","iot"],
        "ai_ml":       ["ai","machine-learning","python","deep-learning"],
        "frontend":    ["frontend","react","javascript","typescript","vue","angular"],
        "backend":     ["backend","python","nodejs","java"],
        "fullstack":   ["fullstack","react","javascript","python"],
        "mobile":      ["mobile","flutter","android","ios","react-native","kotlin"],
        "devops":      ["devops","kubernetes","aws","docker","terraform"],
        "data":        ["data-science","python","sql"],
        "security":    ["cybersecurity","security"],
        "finance":     ["finance","fintech"],
    }
    query_tags = list(TAG_MAP.get(dom_key, ["software-engineer"]))
    if seniority == "junior":
        query_tags = ["intern","junior"] + query_tags

    # Spec-specific tags
    spec_tags = {
        "autosar": ["embedded","automotive"],
        "react_ecosystem": ["react","frontend"],
        "transformers": ["ai","machine-learning","nlp"],
        "computer_vision": ["computer-vision","ai"],
        "rtos": ["embedded","iot"],
        "fpga": ["hardware","embedded"],
    }
    for spec in active_specs[:2]:
        query_tags += spec_tags.get(spec, [])

    return {
        "role_identity":   best_cluster,
        "domain_identity": domain,
        "industry_tags":   industries[:5],
        "tech_stack":      profile.get("skills_hard", [])[:15],
        "specializations": active_specs,
        "query_roles":     query_roles[:12],
        "query_tags":      list(dict.fromkeys(query_tags))[:10],  # dedup
    }
