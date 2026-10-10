"""
Phase 3: Online routing policy — offline replay evaluation.

Uses the Phase 1 oracle JSON as a logged dataset. For each example, all 4
strategy results are already known. Each policy's routing decision just
selects which pre-computed result to "use" — no new API calls needed.

This is the standard offline policy evaluation (OPE) approach for
contextual bandits: replay from logged data instead of re-running.

Policies compared:
  - oracle:            upper bound (always picks the best strategy)
  - always_<strategy>: 4 fixed-strategy baselines
  - rule_based:        length-threshold baseline (no learning)
  - bandit:            epsilon-greedy contextual bandit (learns online)
  - classifier:        Phase 2 classifier (warm-started, static)

Run:
    uv run python experiments/phase3_bandit.py \
        --oracle results/phase1_dense_agg_200ex.json \
        --warm_start results/phase2_200ex/classifier.pkl
"""

import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.policy.rule_based import RuleBasedPolicy
from src.policy.classifier import ClassifierPolicy
from src.policy.bandit     import BanditPolicy
from src.policy.base       import STRATEGY_NAMES
from src.strategies.base   import QueryContext, StrategyResult
from tqdm import tqdm


class LoggedExample:
    """A logged example from Phase 1 with pre-computed features.

    Duck-types as a QueryContext: policies call extract_features(qc) which
    reads qc.context and qc.query. We intercept by providing a context
    that produces the right features. But since we stored the features,
    we monkey-patch extract_features instead (see main loop).
    """
    def __init__(self, record: dict):
        self.query = record["query"]
        self.context_len = record["context_len"]
        self.features = np.array(record["features"], dtype=np.float32)
        self.ground_truth = record["ground_truth"]
        self.grades = record["grades"]
        self.answers = record["answers"]
        self.costs = record["costs"]
        self.best_strategy = record["best_strategy"]

    def make_qc(self) -> QueryContext:
        """Build a QueryContext with a dummy context of the right length.

        extract_features will be monkey-patched to return our stored features.
        """
        return QueryContext(
            query=self.query,
            context="x" * self.context_len,
            ground_truth=self.ground_truth,
        )

    def outcome(self, strategy: str) -> dict:
        return {
            "correct": self.grades[strategy],
            "answer": self.answers[strategy],
            "cost_usd": self.costs[strategy],
        }


class FixedPolicy:
    """Always picks the same strategy — a fixed-strategy baseline."""
    def __init__(self, strategy_name: str):
        self.name = f"always_{strategy_name}"
        self._strategy = strategy_name

    def select(self, qc: QueryContext) -> str:
        return self._strategy

    def update(self, qc: QueryContext, chosen: str, result: StrategyResult):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle",      type=str, required=True,
                        help="Path to phase1 oracle JSON (logged data)")
    parser.add_argument("--warm_start",  type=str, default="",
                        help="Path to Phase 2 classifier .pkl")
    parser.add_argument("--epsilon",     type=float, default=0.2)
    parser.add_argument("--out_dir",     type=str, default="results/phase3")
    parser.add_argument("--seed",        type=int,  default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Load logged Phase 1 data
    with open(args.oracle) as f:
        data = json.load(f)
    records = data["records"]
    n = len(records)
    print(f"Loaded {n} logged examples from {args.oracle}")

    label_dist = {s: sum(1 for r in records if r["best_strategy"] == s) for s in STRATEGY_NAMES}
    print(f"Oracle label distribution: {label_dist}")

    examples = [LoggedExample(r) for r in records]

    # ---- Monkey-patch extract_features to use stored features ----
    import src.policy.base as policy_base
    original_extract = policy_base.extract_features
    feature_map = {}  # id(qc) -> features

    def patched_extract(qc: QueryContext) -> np.ndarray:
        qc_id = id(qc)
        if qc_id in feature_map:
            return feature_map[qc_id]
        return original_extract(qc)

    policy_base.extract_features = patched_extract

    # Also patch in bandit and classifier modules (they imported the name)
    import src.policy.bandit as bandit_mod
    import src.policy.classifier as clf_mod
    bandit_mod.extract_features = patched_extract
    clf_mod.extract_features = patched_extract

    # ---- Define policies ----
    policies: dict[str, object] = {
        "rule_based": RuleBasedPolicy(),
        "bandit":     BanditPolicy(epsilon=args.epsilon, seed=args.seed),
    }
    if args.warm_start and os.path.exists(args.warm_start):
        policies["classifier"] = ClassifierPolicy.load(args.warm_start)
        print(f"Loaded classifier from {args.warm_start}")

    for s in STRATEGY_NAMES:
        policies[f"always_{s}"] = FixedPolicy(s)

    # ---- Offline replay ----
    all_results = {}
    all_cumulative = {}

    for policy_name, policy in policies.items():
        print(f"\n--- Replaying {policy_name} ---")
        cumulative_correct = 0
        cumulative_cost = 0.0
        cumulative_accs = []
        strat_choices = {s: 0 for s in STRATEGY_NAMES}

        for i, ex in enumerate(tqdm(examples, desc=f"  {policy_name}", disable=False)):
            qc = ex.make_qc()
            # Register features for this qc so patched extract_features returns them
            feature_map[id(qc)] = ex.features

            chosen = policy.select(qc)
            outcome = ex.outcome(chosen)

            result = StrategyResult(
                strategy_name=chosen,
                answer=outcome["answer"],
                correct=outcome["correct"],
                cost_usd=outcome["cost_usd"],
            )

            policy.update(qc, chosen, result)

            # Clean up feature_map to avoid memory leak
            del feature_map[id(qc)]

            cumulative_correct += int(outcome["correct"])
            cumulative_cost += outcome["cost_usd"]
            cumulative_accs.append(cumulative_correct / (i + 1))
            strat_choices[chosen] += 1

        accuracy = cumulative_correct / max(n, 1)
        avg_cost = cumulative_cost / max(n, 1)
        strat_dist = {s: strat_choices[s] / max(n, 1) for s in STRATEGY_NAMES}

        all_results[policy_name] = {
            "accuracy": round(accuracy, 4),
            "avg_cost_usd": round(avg_cost, 6),
            "strategy_distribution": {k: round(v, 4) for k, v in strat_dist.items()},
            "n_examples": n,
        }
        all_cumulative[policy_name] = cumulative_accs

        print(f"  accuracy={accuracy:.3f}  avg_cost=${avg_cost:.5f}")
        print(f"  strategy dist: {strat_dist}")

    # Add oracle (upper bound — always picks best)
    oracle_acc = sum(ex.grades[ex.best_strategy] for ex in examples) / max(n, 1)
    oracle_cost = sum(ex.costs[ex.best_strategy] for ex in examples) / max(n, 1)
    all_results["oracle"] = {
        "accuracy": round(oracle_acc, 4),
        "avg_cost_usd": round(oracle_cost, 6),
        "n_examples": n,
    }
    # Oracle cumulative (for plot)
    cum = 0
    oracle_cum = []
    for ex in examples:
        cum += int(ex.grades[ex.best_strategy])
        oracle_cum.append(cum / len(oracle_cum + [1]))
    all_cumulative["oracle"] = oracle_cum

    # ---- Comparison table ----
    order = (["oracle"] +
             [f"always_{s}" for s in STRATEGY_NAMES] +
             ["rule_based", "bandit"] +
             (["classifier"] if "classifier" in all_results else []))
    print(f"\n=== PHASE 3 COMPARISON (offline replay, n={n}) ===")
    print(f"{'Policy':<20} {'Accuracy':>10} {'Avg Cost':>12}")
    print("-" * 44)
    for name in order:
        if name not in all_results:
            continue
        s = all_results[name]
        print(f"{name:<20} {s['accuracy']:>10.3f} ${s['avg_cost_usd']:>11.5f}")

    with open(os.path.join(args.out_dir, "comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out_dir}/comparison.json")

    # ---- Convergence plot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "oracle":             "green",
        "bandit":             "steelblue",
        "classifier":         "darkblue",
        "rule_based":         "gray",
        "always_passthrough": "coral",
        "always_compaction":  "gold",
        "always_retrieval":   "orange",
        "always_rlm":         "darkgreen",
    }
    for name, cum_accs in all_cumulative.items():
        if name not in colors:
            continue
        linestyle = "--" if name.startswith("always_") or name == "oracle" else "-"
        ax.plot(cum_accs, label=name, color=colors[name], linewidth=1.5, linestyle=linestyle)
    ax.set_xlabel("Examples seen")
    ax.set_ylabel("Cumulative accuracy")
    ax.set_title(f"Phase 3: Policy Convergence (offline replay, n={n})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "convergence.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Convergence plot saved to {fig_path}")

    # ---- Bandit arm stats ----
    if isinstance(policies.get("bandit"), BanditPolicy):
        stats = policies["bandit"].arm_stats()
        print(f"\n=== BANDIT ARM STATS ===")
        for s, st in stats.items():
            print(f"  {s:<15} pulls={st['count']:>4}  avg_reward={st['avg_reward']:.3f}  avg_correct={st['avg_correct']:.3f}")

    # ---- Cost-accuracy tradeoff plot ----
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for name in order:
        if name not in all_results:
            continue
        s = all_results[name]
        ax2.scatter(s["avg_cost_usd"], s["accuracy"], s=100, zorder=5)
        ax2.annotate(name, (s["avg_cost_usd"], s["accuracy"]),
                     textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax2.set_xlabel("Avg cost per example (USD)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Phase 3: Cost-Accuracy Tradeoff")
    plt.tight_layout()
    fig2_path = os.path.join(args.out_dir, "cost_accuracy.png")
    plt.savefig(fig2_path, dpi=150)
    print(f"Cost-accuracy plot saved to {fig2_path}")

    # Restore original
    policy_base.extract_features = original_extract


if __name__ == "__main__":
    main()
