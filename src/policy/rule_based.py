"""
Rule-based baseline policy: simple length thresholds.
This is the "no learning" baseline that any learned policy must beat.
"""

from ..strategies.base import QueryContext, StrategyResult

# Context length thresholds (in characters)
SHORT_THRESHOLD  = 16_000    # ~4K tokens — passthrough is fine
MEDIUM_THRESHOLD = 100_000   # ~25K tokens — use compaction
LONG_THRESHOLD   = 400_000   # ~100K tokens — use retrieval or RLM


class RuleBasedPolicy:
    """
    Decision tree:
      ctx < 16K  chars → passthrough
      ctx < 100K chars → compaction
      ctx < 400K chars → retrieval
      ctx >= 400K      → rlm
    """
    name = "rule_based"

    def select(self, qc: QueryContext) -> str:
        n = len(qc.context)
        if n < SHORT_THRESHOLD:
            return "passthrough"
        elif n < MEDIUM_THRESHOLD:
            return "compaction"
        elif n < LONG_THRESHOLD:
            return "retrieval"
        else:
            return "rlm"

    def update(self, qc: QueryContext, chosen: str, result: StrategyResult):
        pass  # rule-based policy doesn't learn
