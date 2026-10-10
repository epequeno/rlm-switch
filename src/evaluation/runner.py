"""
Evaluation runner: apply a policy to a task, collect results.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

from ..strategies.base import QueryContext, StrategyResult
from ..policy.base import Policy
from ..tasks.base import Task, Example


@dataclass
class RunRecord:
    example_idx: int
    query: str
    ground_truth: str
    chosen_strategy: str
    result: StrategyResult
    correct: bool | None = None


@dataclass
class EvalResults:
    policy_name: str
    task_name: str
    records: list[RunRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        graded = [r for r in self.records if r.correct is not None]
        return sum(r.correct for r in graded) / max(len(graded), 1)

    @property
    def avg_cost_usd(self) -> float:
        return sum(r.result.cost_usd for r in self.records) / max(len(self.records), 1)

    @property
    def avg_tokens(self) -> float:
        return sum(r.result.total_tokens for r in self.records) / max(len(self.records), 1)

    @property
    def strategy_distribution(self) -> dict[str, float]:
        total = len(self.records)
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r.chosen_strategy] = counts.get(r.chosen_strategy, 0) + 1
        return {k: v / total for k, v in counts.items()}

    def summary(self) -> dict:
        return {
            "policy": self.policy_name,
            "task": self.task_name,
            "n_examples": len(self.records),
            "accuracy": round(self.accuracy, 4),
            "avg_cost_usd": round(self.avg_cost_usd, 6),
            "avg_tokens": round(self.avg_tokens, 1),
            "strategy_distribution": self.strategy_distribution,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "summary": self.summary(),
                "records": [
                    {
                        "idx": r.example_idx,
                        "strategy": r.chosen_strategy,
                        "correct": r.correct,
                        "cost_usd": r.result.cost_usd,
                        "tokens": r.result.total_tokens,
                        "n_calls": r.result.n_llm_calls,
                    }
                    for r in self.records
                ],
            }, f, indent=2)


def run_evaluation(
    policy: Policy,
    task: Task,
    strategies: dict,
    judge,
    n_examples: int | None = None,
    verbose: bool = True,
) -> EvalResults:
    """
    Run policy on task examples, collect accuracy + cost metrics.

    Args:
        policy:     Policy instance (select + update)
        task:       Task instance (iterable of Examples)
        strategies: dict[str, Strategy] — the strategy implementations
        judge:      callable(answer, ground_truth) -> bool
        n_examples: limit to first N examples (None = all)
    """
    results = EvalResults(
        policy_name=policy.name if hasattr(policy, "name") else type(policy).__name__,
        task_name=task.name,
    )
    t0 = time.time()

    examples = list(task)
    if n_examples is not None:
        examples = examples[:n_examples]

    for i, ex in enumerate(tqdm(examples, desc=f"Eval [{results.policy_name}]", disable=not verbose)):
        qc = QueryContext(
            query=ex.query,
            context=ex.context,
            ground_truth=ex.ground_truth,
            task_type=ex.task_type,
            source=ex.source,
        )

        chosen = policy.select(qc)
        strategy = strategies[chosen]
        result = strategy.run(qc)
        correct = judge(result.answer, ex.ground_truth)
        result.correct = correct

        policy.update(qc, chosen, result)

        results.records.append(RunRecord(
            example_idx=i,
            query=ex.query,
            ground_truth=ex.ground_truth,
            chosen_strategy=chosen,
            result=result,
            correct=correct,
        ))

    results.elapsed_seconds = time.time() - t0
    return results
