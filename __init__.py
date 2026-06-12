"""
job_matcher — Smart Job Matching System v2.0
---------------------------------------------
Modules:
    core            — pipeline orchestrator
    cv_reader       — CV reading and parsing
    jobs            — live API fetchers + dataset loader
    matching        — embedding-based scoring engine
    evaluation      — metrics (precision, recall, MRR, NDCG …)
    utils           — terminal display helpers
"""

from core.pipeline      import JobMatchingPipeline
from cv_reader.reader   import read_cv
from cv_reader.parser   import parse_cv
from jobs.fetcher       import fetch_jobs
from jobs.dataset_loader import DatasetLoader
from matching.engine    import rank_jobs, score_job
from evaluation.metrics import full_report
