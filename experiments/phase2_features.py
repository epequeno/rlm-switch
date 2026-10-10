"""
Phase 2: Train a classifier on oracle labels and analyze feature importance.

Uses the pre-computed feature vectors stored in Phase 1 records (not
reconstructed dummy contexts) to ensure content-dependent features
(entropy, density, locality) are accurate.

Compares the learned classifier against:
  - Majority-class baseline (always predict the most common strategy)
  - Rule-based length-threshold baseline (from rule_based.py)
  - Oracle upper bound

Run:
    uv run python experiments/phase2_features.py --oracle results/phase1_dense_agg_50ex.json
"""

import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.policy.classifier import ClassifierPolicy
from src.policy.base       import STRATEGY_NAMES, extract_features
from src.policy.rule_based import RuleBasedPolicy
from src.strategies.base   import QueryContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle",  type=str, required=True,
                        help="Path to phase1 oracle JSON")
    parser.add_argument("--out_dir", type=str, default="results/phase2")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    with open(args.oracle) as f:
        data = json.load(f)

    records = data["records"]
    n = len(records)

    # Use pre-computed features from Phase 1 (computed on real context)
    X = np.array([r["features"] for r in records])
    y_str = [r["best_strategy"] for r in records]
    y = np.array([STRATEGY_NAMES.index(s) for s in y_str])

    # Context lengths for per-bucket analysis and rule-based baseline
    ctx_lens = np.array([r["context_len"] for r in records])

    print(f"Loaded {n} examples")
    label_dist = {s: y_str.count(s) for s in STRATEGY_NAMES}
    print(f"Label distribution: {label_dist}")

    # ---- Train/val split (stratified-ish by shuffling) ----
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    ctx_val = ctx_lens[val_idx]

    # ---- Train classifier on pre-computed features ----
    policy = ClassifierPolicy()
    policy.clf.fit(X_train, y_train)
    policy._fitted = True

    print(f"\nClassifier trained on {len(train_idx)} examples, "
          f"validated on {len(val_idx)}")
    print(f"Classes: {policy.clf.classes_}")

    # ---- Predictions ----
    clf_preds = policy.clf.predict(X_val)
    clf_acc = np.mean(clf_preds == y_val)

    # Majority-class baseline
    majority = np.bincount(y_train).argmax()
    maj_preds = np.full(len(val_idx), majority)
    maj_acc = np.mean(maj_preds == y_val)

    # Rule-based baseline (needs QueryContext for len())
    rule = RuleBasedPolicy()
    rule_preds = []
    for i in val_idx:
        r = records[i]
        qc = QueryContext(query=r["query"], context="x" * r["context_len"])
        rule_preds.append(STRATEGY_NAMES.index(rule.select(qc)))
    rule_preds = np.array(rule_preds)
    rule_acc = np.mean(rule_preds == y_val)

    # Oracle upper bound (on val set)
    oracle_acc = 1.0  # by definition, oracle picks the best

    print(f"\n=== VALIDATION ACCURACY (n={len(val_idx)}) ===")
    print(f"  Majority-class:    {maj_acc:.3f}  (always '{STRATEGY_NAMES[majority]}')")
    print(f"  Rule-based:        {rule_acc:.3f}  (length thresholds)")
    print(f"  Classifier:        {clf_acc:.3f}  (logistic regression)")
    print(f"  Oracle (upper):    {oracle_acc:.3f}")

    # ---- 5-fold cross-validation for robust accuracy ----
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3),
    }

    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        cv_results[name] = (scores.mean(), scores.std())

    cv_acc, cv_std = cv_results["LogisticRegression"]
    gbm_acc, gbm_std = cv_results["GradientBoosting"]

    # Rule-based via cross-val (deterministic, no training needed)
    rule_preds_all = []
    for r in records:
        qc = QueryContext(query=r["query"], context="x" * r["context_len"])
        rule_preds_all.append(STRATEGY_NAMES.index(rule.select(qc)))
    rule_preds_all = np.array(rule_preds_all)
    rule_acc_cv = np.mean(rule_preds_all == y)

    # Majority-class
    maj_global = np.bincount(y).argmax()
    maj_acc_cv = np.mean(np.full(n, maj_global) == y)

    print(f"\n=== 5-FOLD CROSS-VALIDATION (n={n}) ===")
    print(f"  Majority-class:      {maj_acc_cv:.3f}  (always '{STRATEGY_NAMES[maj_global]}')")
    print(f"  Rule-based:          {rule_acc_cv:.3f}  (length thresholds)")
    for name, (acc, std) in cv_results.items():
        print(f"  {name:<20} {acc:.3f} ± {std:.3f}")
    print(f"  Oracle (upper):      1.000")

    # ---- Per-context-length breakdown ----
    print(f"\n=== BY CONTEXT LENGTH (validation set) ===")
    buckets = {"4K": (0, 8000), "16K": (8000, 24000), "32K": (24000, 48000), "64K": (48000, 96000), "128K+": (96000, float("inf"))}
    for bname, (lo, hi) in buckets.items():
        mask = (ctx_val >= lo) & (ctx_val < hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        print(f"  {bname} (n={n_b}): "
              f"clf={np.mean(clf_preds[mask] == y_val[mask]):.3f}  "
              f"rule={np.mean(rule_preds[mask] == y_val[mask]):.3f}  "
              f"maj={np.mean(maj_preds[mask] == y_val[mask]):.3f}")

    # ---- Confusion matrix ----
    print(f"\n=== CONFUSION MATRIX (classifier, val set) ===")
    print(f"  {'':>15}  " + "  ".join(f"{s:>8}" for s in STRATEGY_NAMES))
    for i, s in enumerate(STRATEGY_NAMES):
        row = np.bincount(clf_preds[y_val == i], minlength=len(STRATEGY_NAMES))
        print(f"  {s:>15}  " + "  ".join(f"{v:>8}" for v in row))

    # ---- Feature importance ----
    importance = policy.feature_importance()
    feature_names = [
        "log_ctx_len", "log_qry_len", "ctx_scale", "norm_qry_len",
        "is_question", "is_list", "is_lookup", "ctx_entropy",
        "qry_density", "answer_locality",
    ]

    print(f"\n=== FEATURE IMPORTANCE (logistic regression coefficients) ===")
    for strat in STRATEGY_NAMES:
        if strat not in importance:
            continue
        coefs = importance[strat]
        sorted_feats = sorted(coefs.items(), key=lambda x: abs(float(x[1])), reverse=True)
        print(f"\n  {strat}:")
        for fname, coef in sorted_feats[:5]:
            print(f"    {fname:>20}: {float(coef):+.4f}")

    # ---- Feature importance plot ----
    n_strat = len(STRATEGY_NAMES)
    fig, axes = plt.subplots(1, n_strat, figsize=(4 * n_strat, 6), sharey=True)
    for ax, strat in zip(axes, STRATEGY_NAMES):
        if strat not in importance:
            ax.set_visible(False)
            continue
        coefs = importance[strat]
        names = list(coefs.keys())
        vals = [float(v) for v in coefs.values()]
        colors = ["steelblue" if c >= 0 else "coral" for c in vals]
        ax.barh(names, vals, color=colors)
        ax.set_title(strat)
        ax.axvline(0, color="black", linewidth=0.5)
    plt.suptitle("Feature Importance per Strategy (Logistic Regression Coefficients)")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "feature_importance.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nFeature importance plot saved to {fig_path}")

    # ---- Accuracy bar chart (uses CV metrics) ----
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    methods = ["Majority", "Rule-based", "LogReg (CV)", "GBM (CV)", "Oracle"]
    accs = [maj_acc_cv, rule_acc_cv, cv_acc, gbm_acc, 1.0]
    bars = ax2.bar(methods, accs, color=["coral", "gold", "steelblue", "darkblue", "green"])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Phase 2: Routing Policy Accuracy vs Baselines (5-fold CV)")
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    fig2_path = os.path.join(args.out_dir, "accuracy_comparison.png")
    plt.savefig(fig2_path, dpi=150)
    print(f"Accuracy comparison plot saved to {fig2_path}")

    # ---- Save model + results ----
    policy.save(os.path.join(args.out_dir, "classifier.pkl"))

    results = {
        "n_total": n,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "val_accuracy": round(clf_acc, 4),
        "cv_accuracy_lr": round(cv_acc, 4),
        "cv_std_lr": round(cv_std, 4),
        "cv_accuracy_gbm": round(gbm_acc, 4),
        "cv_std_gbm": round(gbm_std, 4),
        "majority_accuracy_cv": round(maj_acc_cv, 4),
        "rule_based_accuracy_cv": round(rule_acc_cv, 4),
        "oracle_accuracy": 1.0,
        "label_distribution": label_dist,
        "majority_class": STRATEGY_NAMES[maj_global],
        "feature_importance": {
            strat: {fname: round(float(coef), 4) for fname, coef in imp.items()}
            for strat, imp in importance.items()
        },
    }
    with open(os.path.join(args.out_dir, "phase2_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {os.path.join(args.out_dir, 'phase2_results.json')}")


if __name__ == "__main__":
    main()
