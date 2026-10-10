"""Policy protocol and feature extraction."""

from __future__ import annotations
import numpy as np
from typing import Protocol, runtime_checkable
from ..strategies.base import QueryContext, StrategyResult

STRATEGY_NAMES = ["passthrough", "compaction", "retrieval", "rlm"]


@runtime_checkable
class Policy(Protocol):
    def select(self, qc: QueryContext) -> str:
        """Return the strategy name to use for this query+context pair."""
        ...

    def update(self, qc: QueryContext, chosen: str, result: StrategyResult):
        """Optional: update policy based on observed outcome."""
        ...


def extract_features(qc: QueryContext) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a QueryContext.
    These features are inputs to both the classifier and bandit policies.

    Features:
        0:  log(context_chars)          — primary size signal
        1:  log(query_chars)            — query complexity proxy
        2:  context_chars / 1e6         — raw scale (not log)
        3:  query_chars / 500           — normalized query length
        4:  is_question                 — does query end with '?'
        5:  has_list_query              — "list all", "enumerate", etc.
        6:  has_specific_lookup         — "what is the X", "find the Y"
        7:  context_entropy_proxy       — vocabulary diversity of first 2000 chars
        8:  query_keyword_density       — overlap between query words and context sample
        9:  estimated_answer_locality   — fraction of query words near context start
       10:  is_count_query              — "how many" in query (aggregation)
       11:  is_sum_query                — "total" / "sum" in query (aggregation)
       12:  is_majority_query           — "most frequently" / "majority" in query
       13:  n_records_estimate          — est. records in context (ctx_len / avg_line_len)
       14:  requires_full_scan          — 1 if count/sum (must visit all), 0 otherwise
    """
    ctx_len  = len(qc.context)
    qry_len  = len(qc.query)
    ctx_sample = qc.context[:2000].lower()
    qry_lower  = qc.query.lower()

    f0 = np.log1p(ctx_len)
    f1 = np.log1p(qry_len)
    f2 = ctx_len / 1e6
    f3 = qry_len / 500
    f4 = float(qry_lower.strip().endswith("?"))
    f5 = float(any(kw in qry_lower for kw in ["list", "enumerate", "all of", "every"]))
    f6 = float(any(kw in qry_lower for kw in ["what is", "find the", "who is", "when did"]))

    # Vocabulary diversity of context sample (unique words / total words)
    words = ctx_sample.split()
    f7 = len(set(words)) / max(len(words), 1)

    # Query keyword density in context sample
    qry_words = set(qry_lower.split()) - {"the", "a", "an", "is", "what", "who", "when", "where"}
    matches   = sum(1 for w in qry_words if w in ctx_sample)
    f8 = matches / max(len(qry_words), 1)

    # Answer locality: are query keywords concentrated in first 20% of context?
    first_fifth = qc.context[:max(ctx_len // 5, 1)].lower()
    early_matches = sum(1 for w in qry_words if w in first_fifth)
    f9 = early_matches / max(len(qry_words), 1)

    # Query-type features — strong predictors of routing strategy
    f10 = float("how many" in qry_lower)
    f11 = float(any(kw in qry_lower for kw in ["total", "sum", "add up"]))
    f12 = float(any(kw in qry_lower for kw in ["most frequent", "majority", "most common"]))

    # Estimated record count (assumes ~110 chars per JSON line)
    avg_line_len = 110
    f13 = ctx_len / avg_line_len

    # Full-scan requirement: count/sum queries must visit every record
    f14 = float(f10 or f11)

    return np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
                     f10, f11, f12, f13, f14], dtype=np.float32)
