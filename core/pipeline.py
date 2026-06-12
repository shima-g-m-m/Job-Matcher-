"""
core/pipeline.py
----------------
Orchestrates the full job matching pipeline — runs 100% locally, no API key needed.

  1. Read CV           (cv_reader.reader)
  2. Parse CV          (cv_reader.parser — local NLP)
  3. Fetch live jobs   (jobs.fetcher)
  4. RAG retrieval     (rag.vector_store — sentence-transformers, local)
  5. Score & rank      (matching.engine — 5-component composite score)
  6. RAG explanations  (rag.explainer — local skill analysis)
  7. Evaluate          (evaluation.metrics)
  8. Display & export  (utils.display)
"""

import os
import time
import hashlib
import json
import logging
from typing import Optional

log = logging.getLogger('job_matcher')

from cv_reader.reader import read_cv
from cv_reader.parser import parse_cv
from jobs.fetcher     import fetch_jobs
from matching.engine   import rank_jobs, pre_filter
from matching.reranker import CrossEncoderReranker
from rag.vector_store import JobVectorStore, generate_query
from rag.explainer    import RAGExplainer, print_best_job_recommendation
from evaluation       import full_report, save_report
from utils.display    import (print_banner, print_profile, print_stats,
                               print_results, print_rag_results, export_results,
                               sub, ok, warn, err)


class JobMatchingPipeline:
    """
    Single entry point for the job matching system.

    Usage:
        pipeline = JobMatchingPipeline()
        pipeline.run(cv_path="uploads/my_cv.pdf")

        # Persist job index for faster reruns:
        pipeline.run(cv_path="uploads/my_cv.pdf", index_path="outputs/job_index")
    """

    def __init__(self):
        self.cv_text : str  = ""
        self.profile : dict = {}
        self.jobs    : list = []
        self.ranked  : list = []
        self.store   : Optional[JobVectorStore] = None


    def read(self, cv_path: str) -> str:
        sub("Reading CV")
        try:
            self.cv_text = read_cv(cv_path)
        except Exception as exc:
            warn(f"Could not read CV: {exc}")
            self.cv_text = ""
        if not self.cv_text or len(self.cv_text.strip()) < 50:
            warn("CV appears empty or unreadable — results may be limited")
        ok(f"Loaded {len(self.cv_text):,} characters from {cv_path}")
        return self.cv_text

    def parse(self, use_claude: bool = False) -> dict:
        # use_claude param kept for backwards compat — always uses local parser
        sub("Parsing CV (local NLP)")
        self.profile = parse_cv(self.cv_text)
        try:
            from cv_reader.identity import extract_identity
            self.profile["_identity"] = extract_identity(self.profile, self.cv_text)
        except Exception:
            pass
        ok("CV profile built")
        return self.profile

    def fetch(self, country: str = "us", remote_only: bool = False) -> list:
        sub("Fetching live jobs")
        print("  Sources: RemoteOK · The Muse · Remotive · Arbeitnow")

        # Generate rule-based search query and attach to profile
        if not self.profile.get("_llm_query"):
            self.profile["_llm_query"] = generate_query(self.profile)

        t0 = time.time()
        self.jobs = fetch_jobs(self.profile, country=country, remote_only=remote_only)
        elapsed = time.time() - t0
        if not self.jobs:
            err("No jobs fetched — check your internet connection")
            return []
        ok(f"Fetched {len(self.jobs)} jobs in {elapsed:.1f}s")
        return self.jobs


    def _cv_fingerprint(self) -> str:
        return hashlib.md5(self.cv_text[:500].encode()).hexdigest()[:10]

    def _fingerprint_path(self, index_path: str) -> str:
        return index_path + ".cvid"

    def _saved_fingerprint(self, index_path: str) -> str:
        try:
            return open(self._fingerprint_path(index_path)).read().strip()
        except FileNotFoundError:
            return ""

    def _save_fingerprint(self, index_path: str):
        fp = self._fingerprint_path(index_path)
        os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
        open(fp, "w").write(self._cv_fingerprint())


    def build_rag_index(self, index_path: Optional[str] = None) -> JobVectorStore:
        sub("Building RAG vector index")
        self.store = JobVectorStore()

        same_cv = (index_path and
                   self._saved_fingerprint(index_path) == self._cv_fingerprint())

        if index_path and same_cv and self.store.load(index_path):
            ok(f"Loaded existing index ({len(self.store.jobs)} jobs)")
            saved_urls = {j.get("url", "") for j in self.store.jobs}
            new_jobs   = [j for j in self.jobs if j.get("url", "") not in saved_urls]
            if new_jobs:
                self.store.build(self.store.jobs + new_jobs)
                self.store.save(index_path)
                self._save_fingerprint(index_path)
                ok(f"Index updated with {len(new_jobs)} new jobs")
        else:
            if index_path and not same_cv:
                warn("Different CV detected — rebuilding job index from scratch")
            built = self.store.build(self.jobs)
            if built and index_path:
                os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
                self.store.save(index_path)
                self._save_fingerprint(index_path)
            ok(f"Index built over {len(self.jobs)} jobs")

        return self.store

    def rag_retrieve(self, k: int = 50) -> list:
        if self.store and self.store._built:
            sub(f"RAG retrieval — top {k} semantically relevant jobs")
            retrieved = self.store.retrieve(self.profile, k=k)
            ok(f"Retrieved {len(retrieved)} candidate jobs")
            self.jobs = retrieved
        return self.jobs

    def match(self) -> list:
        pre_count  = len(self.jobs)
        self.jobs  = pre_filter(self.profile, self.jobs)
        if len(self.jobs) < pre_count:
            print(f"  Pre-filter removed {pre_count - len(self.jobs)} incompatible jobs "
                  f"({len(self.jobs)} remain)")
        sub("Scoring and ranking")
        self.ranked = rank_jobs(self.profile, self.jobs, diversify=True)
        ok(f"Ranked {len(self.ranked)} jobs")
        return self.ranked

    def rerank(self, top_n: int = 50) -> list:
        """
        Cross-encoder re-ranking pass.
        Re-scores the top `top_n` jobs with a cross-encoder (ms-marco-MiniLM-L-6-v2)
        and blends the result with the existing composite score.
        Gracefully skipped if sentence-transformers is not installed.
        """
        sub(f"Cross-encoder re-ranking top {top_n} jobs")
        reranker = CrossEncoderReranker()
        if reranker.is_available():
            self.ranked = reranker.rerank(self.profile, self.ranked, top_n=top_n)
            ok("Re-ranking complete")
        else:
            warn("Cross-encoder unavailable — skipping re-rank (install sentence-transformers)")
        return self.ranked

    def add_rag_explanations(self, top_n: int = 10) -> list:
        sub(f"RAG explanations — analysing top {top_n} job fits")
        explainer   = RAGExplainer()
        self.ranked = explainer.explain_batch(self.profile, self.ranked, top_n=top_n)
        ok("Explanations added")
        return self.ranked

    def evaluate(self,
                 ground_truth_skills: Optional[list] = None,
                 relevant_job_ids:    Optional[list] = None,
                 save_path:           Optional[str]  = None,
                 k: int = 10) -> dict:
        sub("Evaluating")
        report = full_report(
            ranked=self.ranked,
            extracted_skills=self.profile.get("skills_hard", []),
            ground_truth_skills=ground_truth_skills,
            relevant_job_ids=relevant_job_ids,
            k=k,
        )
        dist = report.get("score_distribution", {})
        print(f"  Score mean  : {dist.get('mean', 0)}/100")
        print(f"  Score median: {dist.get('median', 0)}/100")
        print(f"  Strong (≥65): {dist.get('strong_matches_65plus', 0)} jobs")
        if "skill_extraction" in report:
            se = report["skill_extraction"]
            print(f"  Skill F1    : {se['f1']}  (P={se['precision']} R={se['recall']})")
        if "mrr" in report:
            print(f"  MRR         : {report['mrr']}")
            print(f"  NDCG@{k}     : {report['ndcg_at_k']}")
        if save_path:
            save_report(report, save_path)
        return report


    def run(self,
            cv_path:      str,
            country:      str  = "us",
            remote_only:  bool = False,
            top_n:        int  = 10,
            use_rag:      bool = True,
            use_rerank:   bool = True,
            rerank_top_n: int  = 50,
            index_path:   Optional[str] = "outputs/job_index",
            export_path:  Optional[str] = None,
            no_claude:    bool = False,       # kept for CLI compat, ignored
            eval_report:  Optional[str] = None) -> list:
        """Run the complete pipeline end-to-end (fully local, no API key needed)."""
        print_banner()

        self.read(cv_path)
        self.parse()
        print_profile(self.profile)

        self.fetch(country=country, remote_only=remote_only)
        if not self.jobs:
            return []

        if use_rag:
            self.build_rag_index(index_path=index_path)
            self.rag_retrieve(k=min(200, len(self.jobs)))

        self.match()

        if use_rerank:
            self.rerank(top_n=rerank_top_n)

        self.add_rag_explanations(top_n=top_n)

        print_stats(self.ranked, len(self.jobs))
        if any("rag" in j for j in self.ranked[:top_n]):
            print_rag_results(self.ranked, top_n=top_n, profile=self.profile)
        else:
            print_results(self.ranked, top_n=top_n, profile=self.profile)

        self.evaluate(save_path=eval_report)
        print_best_job_recommendation(self.ranked, self.profile)

        if export_path:
            export_results(self.profile, self.ranked, export_path, top_n=top_n)

        print(f"\n\033[1m\033[92m  ✅  Done!\033[0m\n")
        return self.ranked
