# rlm-switch

**Can a learned policy decide *per query* whether to pass through, compact, retrieve, or recurse
(RLM) — and does it beat any single fixed strategy?**

Short answer from this pilot: **not on accuracy.** It does cut cost substantially.

**Status: a completed pilot, one task family, 200 examples, one run.** Not a leaderboard.

---

## What was built

Four context strategies (`passthrough`, `compaction`, `retrieval`, `rlm`), five policies
(`rule_based`, `oracle`, `classifier`, `bandit`, plus always-one-strategy baselines), one
evaluation harness, and a three-phase experiment: an oracle study, a supervised router trained on
oracle labels, and a contextual bandit.

## Headline result

On 200 dense-aggregation examples (a task requiring aggregation over many in-context records):

| Policy | Accuracy | Cost / query | Cost vs always-RLM |
|---|---|---|---|
| **oracle** (hindsight) | **0.84** | $0.003553 | 24% |
| always-RLM | 0.76 | $0.014880 | — |
| **classifier router** | **0.72** | **$0.008631** | **58%** |
| bandit router | 0.33 | $0.005580 | 38% |
| always-passthrough | 0.23 | $0.002082 | 14% |
| rule-based heuristic | 0.20 | $0.002390 | 16% |
| always-compaction | 0.20 | $0.002372 | 16% |
| always-retrieval | 0.17 | $0.002358 | 16% |

**The router is 42% cheaper than always-RLM for a 4-point accuracy drop (0.72 vs 0.76).** It does
not beat the best fixed strategy on accuracy, and the 8-point gap between always-RLM and the
oracle is not captured.

## Why it doesn't win

The mechanism, not just the number: **the learned router collapses to two actions.**

Its strategy distribution over the 200 examples is `rlm` 0.685, `passthrough` 0.315,
`compaction` **0.0**, `retrieval` **0.0**.

That is a consequence of the oracle label distribution — `rlm` 110, `passthrough` 68,
`compaction` 17, `retrieval` 5. The two minority strategies are too rare to learn from 160
training examples, so the router degenerates into "RLM usually, passthrough sometimes" and can
only lose accuracy where it wrongly chooses passthrough. A router cannot capture an oracle gap
that lives in strategies it never selects.

Supporting numbers: router label accuracy 0.66 (5-fold CV, ±0.03) against a 0.55 majority-class
baseline. The length-threshold heuristic manages 0.095 — much better at appearing plausible than
at predicting the oracle.

## Structure

```
src/
  strategies/   base.py, passthrough.py, compaction.py, retrieval.py, rlm.py, _client.py
  policy/       base.py, rule_based.py, oracle.py, classifier.py, bandit.py
  tasks/        base.py, long_qa.py, dense_aggregation.py
  evaluation/   runner.py
experiments/
  phase1_oracle.py      # run all strategies per example; measure the oracle gap
  phase2_features.py    # feature extraction + supervised router training
  phase3_bandit.py      # online contextual bandit
results/
  phase1_oracle.json, phase1_dense_agg_{50,200}ex.json, ...
  phase2{,_200ex}/      # classifier, feature importance, results
  phase3/               # comparison.json (the table above), convergence, cost/accuracy
```

## Quickstart

```bash
uv sync
cp .env.example .env          # OPENROUTER_API_KEY, or OPENAI_API_KEY

# Phase 1 — oracle study: which strategy wins per example?
uv run python experiments/phase1_oracle.py --task dense_aggregation --n_examples 200

# Phase 2 — train a router on the oracle labels
uv run python experiments/phase2_features.py --oracle results/phase1_dense_agg_200ex.json

# Phase 3 — online bandit
uv run python experiments/phase3_bandit.py --n_examples 200 --warm_start results/phase2_200ex/classifier.pkl
```

Each script takes `--seed`; `--help` lists the rest.

## Limitations

- **One task family.** The headline table is dense aggregation only. `long_qa` runs exist
  (`results/phase1_oracle.json`) but the router was not evaluated across task families.
- **200 examples, one run, one seed.** No run-to-run variance is estimated.
- **Two of four strategies are effectively unlearned** (5 and 17 positive labels out of 200).
  The router's ceiling here is really "RLM vs passthrough".
- **Cost accounting needs checking.** The oracle's cost ($0.003553) is *lower* than
  always-RLM's, which is only possible if oracle cost is recorded as the cost of the *winning*
  strategy rather than the sum of all four runs. If so, that is a hindsight lower bound and
  should be labelled as such. Worth confirming before any of these numbers are quoted.
- **Costs are API prices at the time of the run**, not compute-normalised.

## What would move this forward

1. Rebalance the label distribution — more `compaction`/`retrieval`-optimal examples (or
   class weighting) — since the router's failure is a label-support problem, not an architecture
   one.
2. Evaluate across task families and report per-family, so the router is not tuned to one.
3. Confirm and label the oracle cost basis.
