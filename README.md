# Job Matcher System

A structured, modular terminal-based job matcher.

```
job_matcher/
│
├── run.py                    ← ENTRY POINT — run this
│
├── core/
│   └── pipeline.py           ← Orchestrates the full pipeline
│
├── cv_reader/
│   ├── reader.py             ← Reads PDF / DOCX / TXT
│   └── parser.py             ← LLM or local NLP CV parser
│
├── jobs/
│   └── fetcher.py            ← Fetches live jobs (RemoteOK, The Muse, Adzuna, Arbeitsagentur)
│
├── matching/
│   └── engine.py             ← Sentence-transformer embeddings + cosine similarity + scoring
│
├── evaluation/
│   └── metrics.py            ← Precision, Recall, F1, MRR, NDCG, MAE, RMSE
│
├── utils/
│   └── display.py            ← Coloured terminal output
│
├── uploads/                  ← Put your CV here
└── outputs/                  ← Exported results go here
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# 2. Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY=gsk_...

# 4. Run
python run.py --cv uploads/my_cv.pdf
```

---

## All Options

```bash
python run.py --cv <file> [options]

  --cv PATH         Path to CV (PDF, DOCX, TXT)        [required]
  --country CODE    us / gb / de / au / ca             (default: us)
  --remote          Remote jobs only
  --top N           Show top N matches                 (default: 10)
  --export FILE     Save full results to JSON
  --no-claude       Use local parser (no API key needed)
  --eval FILE       Save evaluation report to JSON
```

---

## Score Breakdown (0–100)

| Component      | Max | What it measures                        |
|----------------|-----|-----------------------------------------|
| Semantic       | 40  | Embedding cosine similarity (deep NLP)  |
| Hard Skills    | 20  | Technical skill keyword overlap         |
| Seniority      | 15  | Junior/Mid/Senior level alignment       |
| Role/Title     | 15  | Job title relevance                     |
| Soft + Certs   | 10  | Soft skills, certifications             |

**Score guide:** 🟢 65+ strong · 🟡 40–64 good · 🔴 <40 weak

---

## Environment Variables

| Variable          | Required | Purpose                        |
|-------------------|----------|--------------------------------|

| ADZUNA_APP_ID     | Optional | More jobs (free at adzuna.com) |
| ADZUNA_APP_KEY    | Optional | More jobs                      |

---

## Examples

```bash
# Basic
python run.py --cv uploads/my_cv.pdf

# UK remote jobs, top 15, export results
python run.py --cv uploads/my_cv.pdf --country gb --remote --top 15 --export outputs/results.json

# Offline mode (no API keys)
python run.py --cv uploads/my_cv.pdf --no-claude

# Full run with evaluation report
python run.py --cv uploads/my_cv.pdf --export outputs/results.json --eval outputs/eval.json
```
