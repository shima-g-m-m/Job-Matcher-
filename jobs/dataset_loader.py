"""
jobs/dataset_loader.py
-----------------------
Load jobs from static local datasets (CSV, JSON, Excel) as an alternative
to (or supplement of) live API fetching.

Supports the same Kaggle dataset formats as the original system:
  - LinkedIn jobs CSV
  - Indeed-style CSVs
  - Custom JSON job lists

Usage:
    from jobs.dataset_loader import DatasetLoader

    loader = DatasetLoader()
    jobs = loader.load("data/jobs.csv")          # auto-detects format
    jobs = loader.load("data/linkedin.csv", fmt="linkedin")
    jobs = loader.create_sample(n=100)           # generate test data
"""

import json
import random
from pathlib import Path
from typing import Optional


class DatasetLoader:
    """
    Load and normalise job datasets from local files.
    All loaders return the same unified job dict format as the live fetchers.
    """

    UNIFIED_KEYS = {
        "source", "title", "company", "location",
        "description", "url", "salary", "posted", "tags",
    }

    # ── Column presets for popular Kaggle datasets ────────────────────────────
    PRESETS = {
        "linkedin": {
            "title":       "job_title",
            "description": "job_description",
            "company":     "company_name",
            "location":    "location",
            "skills":      "skills",
            "experience":  "experience_level",
            "url":         "job_url",
            "salary":      "salary",
        },
        "indeed": {
            "title":       "title",
            "description": "description",
            "company":     "company",
            "location":    "location",
            "skills":      "required_skills",
            "experience":  "experience",
            "url":         "url",
            "salary":      "salary",
        },
        "glassdoor": {
            "title":       "Job Title",
            "description": "Job Description",
            "company":     "Company Name",
            "location":    "Location",
            "skills":      "Skills",
            "experience":  "Experience",
            "url":         "",
            "salary":      "Salary Estimate",
        },
        "generic": {
            "title":       "title",
            "description": "description",
            "company":     "company",
            "location":    "location",
            "skills":      "skills",
            "experience":  "",
            "url":         "url",
            "salary":      "salary",
        },
    }

    def __init__(self):
        self.jobs: list[dict] = []
        self.source_file: str = ""

    # ── Main loader ───────────────────────────────────────────────────────────

    def load(self, file_path: str, fmt: str = "auto") -> list[dict]:
        """
        Load jobs from a local file. Format is auto-detected from filename
        unless you pass fmt explicitly ('linkedin', 'indeed', 'glassdoor',
        'generic', 'json').
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        self.source_file = file_path
        ext = path.suffix.lower()

        if ext == ".json":
            return self._load_json(file_path)
        elif ext in (".csv",):
            preset = fmt if fmt != "auto" else self._guess_preset(file_path)
            return self._load_csv(file_path, preset)
        elif ext in (".xlsx", ".xls"):
            preset = fmt if fmt != "auto" else "generic"
            return self._load_excel(file_path, preset)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ── CSV loader ────────────────────────────────────────────────────────────

    def _load_csv(self, path: str, preset: str = "generic") -> list[dict]:
        try:
            import pandas as pd
        except ImportError:
            raise SystemExit("Install pandas for CSV support:  pip install pandas")

        cols = self.PRESETS.get(preset, self.PRESETS["generic"])
        df   = pd.read_csv(path, encoding="utf-8", errors="replace")

        jobs = []
        for _, row in df.iterrows():
            title       = str(row.get(cols["title"],       "") or "").strip()
            description = str(row.get(cols["description"], "") or "").strip()
            company     = str(row.get(cols["company"],     "") or "").strip()
            location    = str(row.get(cols["location"],    "") or "").strip()
            url         = str(row.get(cols.get("url",""),  "") or "").strip()
            salary      = str(row.get(cols.get("salary",""),"") or "").strip()

            # Merge skills column into description for better matching
            skill_col = cols.get("skills", "")
            if skill_col and skill_col in df.columns:
                skills_raw = str(row.get(skill_col, "") or "")
                description = f"{description} Skills: {skills_raw}".strip()

            if not title or title.lower() == "nan":
                continue

            jobs.append({
                "source":      f"Dataset ({preset})",
                "title":       title,
                "company":     company if company != "nan" else "",
                "location":    location if location != "nan" else "",
                "description": description,
                "url":         url if url not in ("nan", "") else "",
                "salary":      salary if salary != "nan" else "",
                "posted":      "",
                "tags":        [],
            })

        self.jobs = jobs
        print(f"  Loaded {len(jobs)} jobs from {path} [{preset}]")
        return jobs

    # ── JSON loader ───────────────────────────────────────────────────────────

    def _load_json(self, path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raw = raw.get("jobs", raw.get("results", []))

        jobs = []
        for r in raw:
            # Normalise skills: list[str | dict] → append to description
            skills = r.get("skills", [])
            skill_names = []
            for s in skills:
                if isinstance(s, dict):
                    skill_names.append(s.get("name", ""))
                elif isinstance(s, str):
                    skill_names.append(s)

            desc = r.get("description", "") + " " + " ".join(skill_names)

            jobs.append({
                "source":      "Dataset (JSON)",
                "title":       r.get("title",       r.get("job_title", "")),
                "company":     r.get("company",     r.get("company_name", "")),
                "location":    r.get("location",    ""),
                "description": desc.strip(),
                "url":         r.get("url",         r.get("job_url", "")),
                "salary":      r.get("salary",      ""),
                "posted":      r.get("posted",      r.get("date_posted", ""))[:10],
                "tags":        skill_names,
            })

        self.jobs = jobs
        print(f"  Loaded {len(jobs)} jobs from {path} [JSON]")
        return jobs

    # ── Excel loader ──────────────────────────────────────────────────────────

    def _load_excel(self, path: str, preset: str = "generic") -> list[dict]:
        try:
            import pandas as pd
        except ImportError:
            raise SystemExit("Install pandas for Excel support:  pip install pandas openpyxl")
        # Reuse CSV logic after loading into DataFrame
        import tempfile, os
        df = pd.read_excel(path)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tf:
            df.to_csv(tf.name, index=False)
            tmp_path = tf.name
        jobs = self._load_csv(tmp_path, preset)
        os.unlink(tmp_path)
        return jobs

    # ── Sample dataset generator ──────────────────────────────────────────────

    def create_sample(self, n: int = 100, output_path: str = "outputs/sample_jobs.json") -> list[dict]:
        """Generate a realistic sample dataset for testing (no internet needed)."""
        TITLES = [
            "Machine Learning Engineer", "Data Scientist", "NLP Engineer",
            "Backend Developer", "Full Stack Developer", "DevOps Engineer",
            "Data Analyst", "AI Research Intern", "Computer Vision Engineer",
            "Software Engineer", "Cloud Engineer", "ML Research Intern",
            "Data Engineer", "Python Developer", "Deep Learning Engineer",
        ]
        COMPANIES = [
            "Google", "Amazon", "Microsoft", "Meta", "Apple", "OpenAI",
            "Hugging Face", "DeepMind", "IBM", "NVIDIA", "Stripe",
            "Spotify", "Airbnb", "Uber", "Netflix",
        ]
        LOCATIONS = ["Remote", "New York, US", "London, UK", "San Francisco, US",
                     "Berlin, Germany", "Toronto, Canada", "Amsterdam, Netherlands",
                     "Cairo, Egypt", "Dubai, UAE"]
        SKILL_POOLS = {
            "Machine Learning Engineer":  ["python","pytorch","tensorflow","scikit-learn","mlflow","docker","kubernetes","aws"],
            "Data Scientist":             ["python","pandas","numpy","sql","machine learning","statistics","tableau","r"],
            "NLP Engineer":               ["python","pytorch","transformers","nlp","spacy","bert","fastapi","docker"],
            "Backend Developer":          ["python","django","flask","fastapi","sql","postgresql","docker","redis","git"],
            "Full Stack Developer":       ["python","javascript","react","node","sql","docker","git","html","css"],
            "DevOps Engineer":            ["docker","kubernetes","aws","terraform","ci/cd","linux","jenkins","python"],
            "Data Analyst":               ["sql","python","pandas","tableau","power bi","excel","statistics"],
            "AI Research Intern":         ["python","pytorch","tensorflow","machine learning","deep learning","nlp","research"],
            "Computer Vision Engineer":   ["python","pytorch","tensorflow","computer vision","opencv","deep learning","cuda"],
            "Software Engineer":          ["python","java","git","algorithms","sql","docker","agile","rest api"],
            "Cloud Engineer":             ["aws","azure","gcp","terraform","kubernetes","docker","linux","python"],
            "ML Research Intern":         ["python","pytorch","machine learning","deep learning","research","nlp","transformers"],
            "Data Engineer":              ["python","spark","kafka","sql","aws","docker","airflow","postgresql"],
            "Python Developer":           ["python","django","flask","sql","git","docker","rest api","linux"],
            "Deep Learning Engineer":     ["python","pytorch","tensorflow","deep learning","cuda","machine learning","computer vision"],
        }
        LEVELS = ["Junior", "Mid-Level", "Senior", "Intern", "Entry Level"]

        jobs = []
        for i in range(n):
            title   = random.choice(TITLES)
            company = random.choice(COMPANIES)
            skills  = SKILL_POOLS.get(title, ["python","sql","git"])
            picked  = random.sample(skills, min(len(skills), random.randint(4, 7)))
            level   = random.choice(LEVELS)

            jobs.append({
                "source":      "Sample Dataset",
                "title":       f"{level} {title}" if level != "Intern" else f"{title} Intern",
                "company":     company,
                "location":    random.choice(LOCATIONS),
                "description": (f"We are looking for a {level} {title} to join {company}. "
                                f"You will work on exciting projects involving "
                                f"{', '.join(picked[:3])}. "
                                f"Requirements: {', '.join(picked)}."),
                "url":         f"https://{company.lower().replace(' ','-')}.com/jobs/{i+1}",
                "salary":      random.choice(["$60,000–$80,000","$80,000–$110,000",
                                              "$110,000–$140,000","$140,000–$180,000",""]),
                "posted":      "2026-05-09",
                "tags":        picked,
            })

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(jobs, f, indent=2)
        print(f"  Created {n}-job sample dataset → {output_path}")
        self.jobs = jobs
        return jobs

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _guess_preset(self, path: str) -> str:
        name = Path(path).stem.lower()
        for preset in ("linkedin", "indeed", "glassdoor"):
            if preset in name:
                return preset
        return "generic"

    def stats(self) -> dict:
        """Quick stats about the loaded dataset."""
        if not self.jobs:
            return {"error": "No dataset loaded"}
        return {
            "total_jobs":    len(self.jobs),
            "sources":       list({j["source"] for j in self.jobs}),
            "with_url":      sum(1 for j in self.jobs if j.get("url")),
            "with_salary":   sum(1 for j in self.jobs if j.get("salary")),
            "locations":     list({j["location"] for j in self.jobs if j.get("location")})[:10],
        }
