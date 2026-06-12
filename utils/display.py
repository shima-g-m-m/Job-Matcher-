"""
utils/display.py
----------------
Terminal display helpers for the job matching system.
Handles profile display, job cards, stats, and best match summary.
"""

import os
import textwrap

_NO_COLOR = os.getenv("NO_COLOR") or not os.isatty(1)
def _c(code): return "" if _NO_COLOR else f"\033[{code}m"

RESET  = _c("0");  BOLD  = _c("1");  DIM   = _c("2")
RED    = _c("31"); GREEN = _c("32"); YELLOW = _c("33")
BLUE   = _c("34"); CYAN  = _c("36"); WHITE  = _c("37")


def header(title: str):
    w = 70
    print(f"\n{'═'*w}")
    print(f"  {BOLD}{title}{RESET}")
    print(f"{'═'*w}\n")


def sub(msg: str):
    print(f"\n{BOLD}▶  {msg}{RESET}")


def ok(msg: str):
    print(f"  {GREEN}✓{RESET}  {msg}")


def warn(msg: str):
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def score_bar(score: int, width: int = 28) -> str:
    filled = int(score / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    color  = GREEN if score >= 68 else (YELLOW if score >= 55 else RED)
    return f"[{color}{bar}{RESET}]  {BOLD}{score}/100{RESET}"


def confidence_badge(label: str) -> str:
    colors = {
        "EXCELLENT": GREEN, "STRONG": GREEN,
        "MODERATE": YELLOW, "STRETCH": YELLOW, "WEAK": RED,
    }
    c = colors.get(label, DIM)
    icons = {
        "EXCELLENT": "🟢", "STRONG": "🟢",
        "MODERATE": "🟡", "STRETCH": "🟠", "WEAK": "🔴",
    }
    icon = icons.get(label, "⚪")
    return f"{icon} {c}{BOLD}{label} MATCH{RESET}"


def print_profile(p: dict):
    header("CV ANALYSIS — Your Profile")
    print(f"  {BOLD}{'Name':<14}:{RESET} {p.get('name') or '—'}")
    print(f"  {BOLD}{'Current Title':<14}:{RESET} {p.get('current_title') or '—'}")

    domain = p.get("domain", "")
    if domain:
        print(f"  {BOLD}{'Domain':<14}:{RESET} {domain.replace('_',' ').title()}")

    seniority = p.get("seniority", "—").upper()
    sen_color = (GREEN if seniority == "JUNIOR" else
                 YELLOW if seniority in ("MID",) else
                 RED)
    print(f"  {BOLD}{'Seniority':<14}:{RESET} {sen_color}{seniority}{RESET}")
    print(f"  {BOLD}{'Experience':<14}:{RESET} {p.get('years_experience', 0)} year(s)")

    # Professional identity
    identity = p.get("_identity", {})
    specs = identity.get("specializations", [])
    industries = identity.get("industry_tags", [])
    role_id = identity.get("role_identity", "")
    if role_id:
        print(f"  {BOLD}{'Role Identity':<14}:{RESET} {role_id.replace('_',' ').title()}")
    if specs:
        print(f"  {BOLD}{'Specialization':<14}:{RESET} {', '.join(s.replace('_',' ') for s in specs[:3])}")
    if industries:
        print(f"  {BOLD}{'Industries':<14}:{RESET} {', '.join(industries[:3])}")

    # Intent signals
    from matching.engine import _detect_intent
    intent = _detect_intent(p)
    active_intents = [k.replace("_", " ") for k, v in intent.items() if v]
    if active_intents:
        print(f"  {BOLD}{'Intent':<14}:{RESET} {', '.join(active_intents[:4])}")

    print(f"  {BOLD}{'Languages':<14}:{RESET} {', '.join(p.get('languages_spoken', [])) or '—'}")
    print(f"  {BOLD}{'Industries':<14}:{RESET} {', '.join(p.get('industries', [])[:5]) or '—'}")

    skills = p.get("skills_hard", [])
    if skills:
        print(f"\n  {BOLD}Hard Skills ({len(skills)}):{RESET}")
        cols = 5
        for i in range(0, min(len(skills), 30), cols):
            row = skills[i:i+cols]
            print("    " + "  ".join(f"• {s:<22}" for s in row))

    certs = p.get("certifications", [])
    if certs:
        print(f"\n  {BOLD}Certifications:{RESET}")
        for c in certs[:5]:
            print(f"    ✦ {c}")

    edu = p.get("education", [])
    if edu:
        print(f"\n  {BOLD}Education:{RESET}")
        for e in edu[:3]:
            print(f"    ▸ {e}")

    roles = [r for r in p.get("preferred_roles", []) if "intern" not in r][:4]
    if roles:
        print(f"\n  {BOLD}Target Roles:{RESET}")
        for r in roles:
            print(f"    → {r.title()}")

    print(f"\n  {BOLD}Recruiter Pitch:{RESET}")
    pitch = p.get("summary_for_search", "")
    for line in textwrap.wrap(pitch, 62):
        print(f"    {DIM}{line}{RESET}")


def print_stats(results: list, total_fetched: int = 0):
    header("Match Statistics")
    print(f"  {BOLD}Jobs scored         :{RESET} {len(results)}")
    if total_fetched:
        print(f"  {BOLD}Jobs in index       :{RESET} {total_fetched}")
    with_links = sum(1 for r in results if r.get("url"))
    print(f"  {BOLD}With apply links    :{RESET} {with_links}")
    excellent = sum(1 for r in results if r["score"] >= 80)
    strong    = sum(1 for r in results if 68 <= r["score"] < 80)
    moderate  = sum(1 for r in results if 55 <= r["score"] < 68)
    stretch   = sum(1 for r in results if 40 <= r["score"] < 55)
    weak      = sum(1 for r in results if r["score"] < 40)
    print(f"  {BOLD}Excellent  (≥80)    :{RESET} {GREEN}{excellent}{RESET}")
    print(f"  {BOLD}Strong     (≥68)    :{RESET} {GREEN}{strong}{RESET}")
    print(f"  {BOLD}Moderate   (≥55)    :{RESET} {YELLOW}{moderate}{RESET}")
    print(f"  {BOLD}Stretch    (≥40)    :{RESET} {YELLOW}{stretch}{RESET}")
    print(f"  {BOLD}Weak       (<40)    :{RESET} {RED}{weak}{RESET}")
    if results:
        top20 = results[:20]
        avg = sum(r["score"] for r in top20) / len(top20)
        print(f"  {BOLD}Avg score (top 20)  :{RESET} {avg:.1f}/100")


def _job_card(i: int, r: dict, profile: dict = None):
    """Render a single job card with recruiter-quality explanation."""
    score = r["score"]
    conf  = r.get("confidence", "MODERATE")
    color = GREEN if score >= 70 else (YELLOW if score >= 50 else RED)

    print(f"\n  {BOLD}{color}#{i:02d}  {r['title']}{RESET}")
    print(f"       {BOLD}Company :{RESET} {r.get('company') or '—'}")
    print(f"       {BOLD}Location:{RESET} {r.get('location') or '—'}")
    print(f"       {BOLD}Source  :{RESET} {r.get('source','')}")
    if r.get("salary"):
        print(f"       {BOLD}Salary  :{RESET} {r['salary']}")
    if r.get("posted"):
        print(f"       {BOLD}Posted  :{RESET} {r['posted'][:10]}")

    # Score bar + confidence badge
    print(f"       {BOLD}Score   :{RESET} {score_bar(score)}")
    print(f"       {BOLD}Fit     :{RESET} {confidence_badge(conf)}")

    # Score breakdown (updated to new weights)
    fam_str = ""
    if r.get("cv_family") and r.get("job_family") and r["cv_family"] != r["job_family"]:
        fam_str = f" │ Family {r['cv_family']}→{r['job_family']}"
    print(f"       {DIM}Sem {r.get('score_semantic',0):2d}/30 │ "
          f"Skills {r.get('score_skills',0):2d}/25 │ "
          f"Seniority {r.get('score_seniority',0):2d}/20 │ "
          f"Role {r.get('score_title',0):2d}/15 │ "
          f"Intent {r.get('score_extra',0):2d}/10{fam_str}{RESET}")

    if r.get("semantic_pct"):
        print(f"       {DIM}Semantic similarity: {r['semantic_pct']}%{RESET}")

    # Experience requirement warning
    req = r.get("req_years", 0)
    cv_yrs = profile.get("years_experience", 0) if profile else 0
    if req > 0:
        gap = req - cv_yrs
        if gap > 2:
            print(f"       {YELLOW}⚠  Requires {req}+ yrs experience "
                  f"(you have {cv_yrs}){RESET}")
        else:
            print(f"       {DIM}Requires {req}+ yrs experience{RESET}")

    # Geographic awareness warning
    if r.get("geo_flag"):
        print(f"       {YELLOW}🌍 May require local presence or US work authorisation{RESET}")

    # Matched specialist skills
    matched = r.get("matched_skills", [])
    if matched:
        print(f"\n       {BOLD}✅ Matched Skills:{RESET}")
        print("          " + "  ".join(f"{GREEN}{s}{RESET}" for s in matched[:8])
              + (f"  {DIM}+{len(matched)-8} more{RESET}" if len(matched) > 8 else ""))

    # Missing high-value skills
    missing = r.get("missing_skills", [])
    if missing:
        print(f"\n       {BOLD}⚠  Skill Gaps:{RESET}")
        print("          " + "  ".join(f"{RED}{s}{RESET}" for s in missing[:6]))

    # Why it fits / why it doesn't — recruiter-style explanation
    _print_fit_explanation(r, profile)

    # Role match
    if r.get("matched_roles"):
        print(f"\n       {BOLD}💡 Role match:{RESET} "
              f"{', '.join(r['matched_roles'][:3])}")

    url = r.get("url", "")
    if url:
        print(f"\n       {BOLD}🔗 Apply:{RESET} {CYAN}{url}{RESET}")
    print(f"       {'─'*62}")


def _print_fit_explanation(r: dict, profile: dict = None):
    """Print a recruiter-style why-it-fits / why-it-doesn't block."""
    seniority  = r.get("score_seniority", 0)
    req_years  = r.get("req_years", 0)
    cv_years   = (profile or {}).get("years_experience", 0)
    fam_compat = r.get("family_compat", 1.0)
    cv_fam     = (r.get("cv_family") or "").replace("_", " ")
    job_fam    = (r.get("job_family") or "").replace("_", " ")
    matched    = r.get("matched_skills", [])
    missing    = r.get("missing_skills", [])

    positives, concerns = [], []

    # Seniority
    if seniority >= 14:
        positives.append("Seniority level is a strong fit")
    elif seniority == 0:
        concerns.append("Role is aimed at a much more senior candidate")
    elif seniority <= 5:
        concerns.append("Significant seniority gap — role targets a more experienced hire")

    # Domain/family alignment
    if fam_compat >= 0.85:
        if cv_fam and job_fam and cv_fam != job_fam:
            positives.append(f"Role domains are closely aligned: {cv_fam} → {job_fam}")
        else:
            positives.append("Domain is an exact match for your background")
    elif fam_compat >= 0.55:
        positives.append(f"Adjacent domains with skill overlap: {cv_fam} → {job_fam}")
    elif fam_compat <= 0.25 and cv_fam and job_fam:
        concerns.append(f"Domain mismatch: you focus on {cv_fam}, role is {job_fam}")

    # Skill overlap — use domain-specific language
    # Identity-aware skill explanation
    if len(matched) >= 4:
        tech = [s for s in matched if s not in
                {"communication","teamwork","agile","collaboration"}]
        if tech:
            specs = (profile or {}).get("_identity",{}).get("specializations",[])
            spec_str = f" ({', '.join(s.replace('_',' ') for s in specs[:2])})" if specs else ""
            positives.append(f"Strong technical overlap{spec_str}: {', '.join(tech[:4])}")
    elif len(matched) >= 2:
        positives.append(f"Partial skill match: {', '.join(matched[:3])}")
    elif not matched:
        concerns.append("Very little skill overlap with this role's requirements")

    # Role title
    if r.get("score_title", 0) >= 10:
        positives.append("Job title aligns with your target roles")
    elif r.get("score_title", 0) == 0 and fam_compat < 0.5:
        concerns.append("Role title does not match your target career track")

    # Experience requirement
    if req_years > 0 and req_years > cv_years + 3:
        concerns.append(f"Requires {req_years}+ yrs experience (you have {cv_years}) — stretch role")

    # Missing skills — only high-value ones
    if missing:
        concerns.append(f"Key skills to develop: {', '.join(missing[:3])}")

    if positives:
        print(f"\n       {BOLD}✅ Why it fits:{RESET}")
        for pt in positives[:3]:
            print(f"          {GREEN}+{RESET} {pt}")

    if concerns:
        print(f"\n       {BOLD}⚠  Watch out:{RESET}")
        for ct in concerns[:3]:
            print(f"          {YELLOW}−{RESET} {ct}")


def print_results(ranked: list, top_n: int = 10, profile: dict = None):
    """Print top N job cards."""
    header(f"TOP {top_n} JOB MATCHES")
    shown = [r for r in ranked if r.get("url")][:top_n]
    if not shown:
        shown = ranked[:top_n]
    if not shown:
        warn("No jobs to display. Check your internet connection.")
        return
    for i, r in enumerate(shown, 1):
        _job_card(i, r, profile=profile)


def print_rag_results(ranked: list, top_n: int = 10, profile: dict = None):
    """Print job cards with match explanations."""
    header(f"TOP {top_n} JOB MATCHES")
    shown = [r for r in ranked if r.get("url")][:top_n]
    if not shown:
        shown = ranked[:top_n]
    if not shown:
        warn("No jobs to display.")
        return

    for i, r in enumerate(shown, 1):
        _job_card(i, r, profile=profile)

        rag = r.get("rag", {})
        if not rag:
            continue

        if rag.get("match_summary"):
            print(f"\n       {BOLD}📋 Match Analysis:{RESET}")
            for line in textwrap.wrap(rag["match_summary"], 62):
                print(f"          {line}")

        if rag.get("strengths"):
            print(f"\n       {BOLD}✅ Key Strengths:{RESET}")
            for s in rag["strengths"][:3]:
                print(f"          {GREEN}+{RESET} {s}")

        if rag.get("gaps"):
            print(f"\n       {BOLD}⚠  Skill Gaps:{RESET}")
            for g in rag["gaps"][:3]:
                print(f"          {RED}−{RESET} {g}")

        if rag.get("advice"):
            print(f"\n       {BOLD}💡 Advice:{RESET}")
            for line in textwrap.wrap(rag["advice"], 62):
                print(f"          {DIM}{line}{RESET}")


def print_best_match(ranked: list, profile: dict = None):
    """Print the best match summary with confidence-aware wording."""
    header("BEST RECOMMENDATION FOR YOU")

    if not ranked:
        warn("No matches found.")
        return

    best  = ranked[0]
    score = best["score"]
    conf  = best.get("confidence", "MODERATE")

    # Confidence-aware headline
    if conf == "EXCELLENT":
        headline = "🏆  EXCELLENT MATCH"
    elif conf == "STRONG":
        headline = "🥇  STRONG MATCH"
    elif conf == "MODERATE":
        headline = "🥈  GOOD MATCH"
    elif conf == "STRETCH":
        headline = "🎯  CLOSEST MATCH"
    else:
        headline = "📋  BEST AVAILABLE MATCH"

    color = GREEN if score >= 70 else (YELLOW if score >= 50 else RED)
    print(f"  {BOLD}{color}{headline}{RESET}\n")
    print(f"  {BOLD}{best['title']}{RESET}  @  {best.get('company','')}")
    print(f"  Score: {score}/100  |  {confidence_badge(conf)}")
    print(f"  {best.get('location','Remote')}  |  "
          f"Salary: {best.get('salary','not listed')}")

    matched = best.get("matched_skills", [])
    if matched:
        print(f"\n  Why this fits you:")
        print(f"    ✓ Matched skills: {', '.join(matched[:5])}")
    if best.get("score_seniority", 0) >= 15:
        print(f"    ✓ Seniority level aligns with your profile")
    if best.get("score_title", 0) >= 10:
        print(f"    ✓ Role domain matches your target roles")

    missing = best.get("missing_skills", [])
    if missing:
        print(f"\n  To strengthen your application:")
        for s in missing[:3]:
            print(f"    → Build experience with {s}")

    if best.get("url"):
        print(f"\n  {BOLD}🔗 Apply:{RESET} {CYAN}{best['url']}{RESET}")

    # Runner-ups
    if len(ranked) > 1:
        print(f"\n  {BOLD}Runner-ups:{RESET}")
        for j, r in enumerate(ranked[1:4], 2):
            conf2 = r.get("confidence", "")
            print(f"    #{j}  {r['title']} @ {r.get('company','')}  "
                  f"— {r['score']}/100  {confidence_badge(conf2)}")

    print(f"\n{'═'*70}\n")


def print_banner():
    print(f"""
  {BOLD}╔══════════════════════════════════════════════════════════════╗
  ║              JOB MATCHER SYSTEM                            ║
  ║        Embeddings · Live Jobs · Ranked Results             ║
  ╚══════════════════════════════════════════════════════════════╝{RESET}
""")


def export_results(profile: dict, ranked: list, path: str, top_n: int = 10):
    """Export results to JSON."""
    import json, datetime
    out = {
        "generated":  datetime.datetime.now().isoformat(),
        "candidate":  profile.get("name", ""),
        "domain":     profile.get("domain", ""),
        "seniority":  profile.get("seniority", ""),
        "top_matches": [
            {
                "rank":       i + 1,
                "title":      r["title"],
                "company":    r["company"],
                "location":   r["location"],
                "score":      r["score"],
                "confidence": r.get("confidence", ""),
                "url":        r["url"],
                "salary":     r.get("salary", ""),
                "matched_skills": r.get("matched_skills", []),
                "missing_skills": r.get("missing_skills", []),
            }
            for i, r in enumerate(ranked[:top_n])
        ]
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  {GREEN}✓{RESET}  Results exported → {path}")


# Aliases for backward compatibility
def err(msg: str):
    print(f"  {RED}✗{RESET}  {msg}")
