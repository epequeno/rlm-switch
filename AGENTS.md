# Agent Instructions — rlm-switch

## Documentation

**Long-form design notes live in a private vault** (not published here):

```
<private research vault>/rlm-switch/
```

The vault is shared across multiple research projects. This project's space is
isolated under the `rlm-switch/` folder.

### Vault layout for this project

| Note | Contents |
|------|----------|
| `_Index.md` | Project home — links to all notes |
| `Overview.md` | Research question, hypotheses, three-phase approach |
| `Architecture.md` | Module structure, class relationships |
| `Strategies.md` | The four context strategies (passthrough, compaction, retrieval, rlm) |
| `Policies.md` | Oracle, rule-based, classifier, and bandit policies |
| `Feature Engineering.md` | The 10-dimensional feature vector used for routing |
| `Experiments.md` | Phase 1 / 2 / 3 experiment scripts and expected outputs |
| `Implementation Notes.md` | Code-level details, gotchas, and design decisions |

### Keeping docs in sync

- When you add, rename, or significantly change a source file, update the
  corresponding vault note.
- When you complete an experiment run, record summary statistics in
  `Experiments.md` under the appropriate phase.
- New design decisions or non-obvious trade-offs belong in
  `Implementation Notes.md`.
- Cross-link liberally using Obsidian `[[Note Name]]` syntax.

## Project overview (quick ref)

**Goal:** Train a meta-policy that decides *per query* whether to use passthrough,
compaction, retrieval, or RLM recursion — and beat any single fixed strategy.

**Three phases:**
1. Oracle study — run all strategies, measure oracle gain
2. Feature analysis — supervised classifier on oracle labels
3. Online bandit — epsilon-greedy contextual bandit, updates in real-time

**Stack:** Python 3.11+, uv, OpenAI/Anthropic APIs, scikit-learn, wandb

```bash
uv sync
uv run python experiments/phase1_oracle.py --n_examples 50
uv run python experiments/phase2_features.py --oracle results/phase1_oracle.json
uv run python experiments/phase3_bandit.py --n_examples 200
```
