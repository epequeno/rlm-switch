"""
Phase 1: Oracle study.

Runs all 4 strategies on every example and records which wins.
Outputs oracle labels + per-example statistics for Phase 2.

Run:
    uv run python experiments/phase1_oracle.py --task dense_aggregation --n_examples 50
    uv run python experiments/phase1_oracle.py --task needle --n_examples 100
    uv run python experiments/phase1_oracle.py --task dense_aggregation --n_examples 100 --context_lengths 4000 32000 128000
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.strategies.passthrough import PassthroughStrategy
from src.strategies.compaction  import CompactionStrategy
from src.strategies.retrieval   import RetrievalStrategy
from src.strategies.rlm         import RLMStrategy
from src.policy.oracle          import OraclePolicy
from src.tasks.long_qa          import SyntheticNeedleTask
from src.tasks.dense_aggregation import DenseAggregationTask
from src.policy.base            import extract_features
from tqdm import tqdm


def _default_judge(answer: str, ground_truth: str) -> bool:
    return ground_truth.strip().lower() in answer.strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",            type=str,   default="dense_aggregation",
                        choices=["needle", "dense_aggregation"],
                        help="Benchmark task to run (default: dense_aggregation)")
    parser.add_argument("--n_examples",      type=int,   default=50)
    parser.add_argument("--context_lengths", type=int,   nargs="+",
                        default=[4_000, 16_000, 64_000])
    parser.add_argument("--model",           type=str,   default="openai/gpt-4o-mini")
    parser.add_argument("--rlm_model",       type=str,   default="anthropic/claude-3-haiku",
                        help="Root model for RLM (should be capable of agentic code execution)")
    parser.add_argument("--rlm_sub_model",   type=str,   default="openai/gpt-4o-mini",
                        help="Sub-model for RLM recursive llm_query() calls")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--out",             type=str,   default="results/phase1_oracle.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    strategies = {
        "passthrough": PassthroughStrategy(model=args.model),
        "compaction":  CompactionStrategy(model=args.model),
        "retrieval":   RetrievalStrategy(model=args.model),
        "rlm":         RLMStrategy(model=args.rlm_model, sub_model=args.rlm_sub_model),
    }

    # Select task and its judge
    if args.task == "needle":
        task = SyntheticNeedleTask(
            n_examples=args.n_examples,
            context_lengths=args.context_lengths,
            seed=args.seed,
        )
    else:
        task = DenseAggregationTask(
            n_examples=args.n_examples,
            context_lengths=args.context_lengths,
            seed=args.seed,
        )
    judge_fn = getattr(task, "judge", _default_judge)
    oracle = OraclePolicy(strategies, judge=judge_fn)

    records = []
    strategy_wins = {k: 0 for k in strategies}
    all_wrong = 0

    for ex in tqdm(task, desc="Oracle study"):
        from src.strategies.base import QueryContext
        qc = QueryContext(query=ex.query, context=ex.context,
                          ground_truth=ex.ground_truth)
        all_results = oracle.run_all(qc)
        grades      = oracle.grade(all_results, ex.ground_truth)
        best        = oracle.best_strategy(all_results, ex.ground_truth)

        if not any(grades.values()):
            all_wrong += 1

        strategy_wins[best] += 1
        records.append({
            "query":          ex.query[:100],
            "context_len":    len(ex.context),
            "ground_truth":   ex.ground_truth,
            "grades":         grades,
            "best_strategy":  best,
            "features":       extract_features(qc).tolist(),
            "costs":          {k: v.cost_usd for k, v in all_results.items()},
            "answers":        {k: v.answer[:200] for k, v in all_results.items()},
        })

    # Compute oracle gain
    total = len(records)
    oracle_acc = sum(any(r["grades"].values()) for r in records) / total
    avg_acc    = {k: sum(r["grades"].get(k, False) for r in records) / total
                  for k in strategies}
    mean_single_strategy_acc = sum(avg_acc.values()) / len(avg_acc)
    oracle_gain = oracle_acc - mean_single_strategy_acc

    summary = {
        "task":               args.task,
        "n_examples":         total,
        "oracle_accuracy":    round(oracle_acc, 4),
        "per_strategy_acc":   {k: round(v, 4) for k, v in avg_acc.items()},
        "strategy_wins":      strategy_wins,
        "oracle_gain":        round(oracle_gain, 4),
        "all_wrong_count":    all_wrong,
        "context_lengths":    args.context_lengths,
    }

    print("\n=== PHASE 1 RESULTS ===")
    print(f"Oracle accuracy (best possible):     {oracle_acc:.3f}")
    print(f"Per-strategy accuracy:")
    for k, v in avg_acc.items():
        print(f"  {k:<15} {v:.3f}  ({strategy_wins[k]} wins)")
    print(f"Oracle gain over avg strategy:       {oracle_gain:.3f}")
    print(f"Examples where ALL strategies fail:  {all_wrong}")

    output = {"summary": summary, "records": records}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
