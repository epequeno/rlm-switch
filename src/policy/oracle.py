"""
Oracle policy: runs ALL strategies, picks the winner.

Used in Phase 1 to:
  1. Measure oracle gain (upper bound on switching benefit)
  2. Generate supervised labels for the classifier policy
  3. Characterize which features correlate with each strategy winning
"""

from __future__ import annotations
import time
from ..strategies.base import QueryContext, StrategyResult, Strategy
from .base import STRATEGY_NAMES


class OraclePolicy:
    """
    Runs every strategy on every example and returns the one with the
    best accuracy. If ground truth is not available, returns all results
    for human/LLM-based grading.

    Cost: 4x per example. Only for Phase 1 analysis.
    """
    name = "oracle"

    def __init__(self, strategies: dict[str, Strategy], judge=None):
        """
        strategies: dict of name -> Strategy instance
        judge: optional callable(answer, ground_truth) -> bool
               defaults to exact match if None
        """
        self.strategies = strategies
        self.judge = judge or self._exact_match

    def run_all(self, qc: QueryContext) -> dict[str, StrategyResult]:
        """Run all strategies and return their results."""
        results = {}
        for name, strategy in self.strategies.items():
            try:
                results[name] = strategy.run(qc)
            except Exception as e:
                results[name] = StrategyResult(
                    strategy_name=name,
                    answer=f"[ERROR: {e}]",
                    metadata={"error": str(e)},
                )
        return results

    def select(self, qc: QueryContext) -> str:
        """
        NOTE: Oracle is not usable online (needs ground truth).
        This method is a placeholder — use run_all() + grade() offline.
        """
        raise NotImplementedError(
            "Oracle policy requires ground truth. Use run_all() + grade() offline."
        )

    def grade(
        self, results: dict[str, StrategyResult], ground_truth: str
    ) -> dict[str, bool]:
        """Grade each strategy's answer against ground truth."""
        grades = {}
        for name, result in results.items():
            grades[name] = self.judge(result.answer, ground_truth)
            result.correct = grades[name]
        return grades

    def best_strategy(
        self, results: dict[str, StrategyResult], ground_truth: str
    ) -> str:
        """Return the name of the strategy that got the right answer (lowest cost if tie)."""
        grades = self.grade(results, ground_truth)
        correct = [name for name, ok in grades.items() if ok]
        if not correct:
            # All wrong: pick cheapest
            return min(results, key=lambda n: results[n].cost_usd)
        # Correct: pick cheapest correct one
        return min(correct, key=lambda n: results[n].cost_usd)

    @staticmethod
    def _exact_match(answer: str, ground_truth: str) -> bool:
        return answer.strip().lower() == ground_truth.strip().lower()
