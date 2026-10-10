"""
Contextual bandit policy (Phase 3).

Uses epsilon-greedy with a linear reward model per strategy.
Updates online as outcomes come in — no need for an offline oracle phase.

Can also serve as a warm-start for full RL if you later want PPO/GRPO.
"""

from __future__ import annotations
import numpy as np
from .base import extract_features, STRATEGY_NAMES
from ..strategies.base import QueryContext, StrategyResult


class BanditPolicy:
    """
    Epsilon-greedy contextual bandit.

    For each strategy k, maintains a linear model: r_k = w_k · features.
    On each step:
      - With prob epsilon: pick uniformly (explore)
      - Otherwise:        pick argmax_k (w_k · features) (exploit)
    After observing reward, update w_k with a gradient step.

    Reward signal: binary correctness (1.0 / 0.0) - cost_penalty * cost_usd
    """
    name = "bandit"

    def __init__(
        self,
        epsilon: float = 0.2,
        lr: float = 0.01,
        cost_penalty: float = 10.0,   # penalty per $0.01 USD spent
        decay: float = 0.995,         # epsilon decay per step
        seed: int = 42,
    ):
        self.epsilon   = epsilon
        self.lr        = lr
        self.cost_penalty = cost_penalty
        self.decay     = decay
        self.rng       = np.random.default_rng(seed)
        n_feat = len(extract_features(QueryContext(query="test", context="test")))
        n_strategies = len(STRATEGY_NAMES)
        self.weights   = np.zeros((n_strategies, n_feat))   # reward models
        self.counts    = np.zeros(n_strategies, dtype=int)  # pulls per arm
        self.step      = 0
        self._history: list[dict] = []

    def select(self, qc: QueryContext) -> str:
        feat = extract_features(qc)
        if self.rng.random() < self.epsilon:
            idx = int(self.rng.integers(len(STRATEGY_NAMES)))
        else:
            scores = self.weights @ feat
            idx    = int(np.argmax(scores))
        return STRATEGY_NAMES[idx]

    def update(self, qc: QueryContext, chosen: str, result: StrategyResult):
        """Update the reward model for the chosen strategy."""
        feat     = extract_features(qc)
        k        = STRATEGY_NAMES.index(chosen)
        correct  = float(result.correct or 0.0)
        reward   = correct - self.cost_penalty * result.cost_usd

        # Gradient step on linear model: loss = (r_hat - r)^2
        r_hat    = self.weights[k] @ feat
        grad     = (r_hat - reward) * feat
        self.weights[k] -= self.lr * grad
        self.counts[k]  += 1
        self.epsilon     = max(0.05, self.epsilon * self.decay)
        self.step       += 1

        self._history.append({
            "step": self.step,
            "strategy": chosen,
            "reward": reward,
            "correct": result.correct,
            "cost_usd": result.cost_usd,
            "epsilon": self.epsilon,
        })

    def arm_stats(self) -> dict:
        """Summary of how often each strategy was selected and its average reward."""
        stats = {}
        for k, name in enumerate(STRATEGY_NAMES):
            relevant = [h for h in self._history if h["strategy"] == name]
            stats[name] = {
                "count": self.counts[k],
                "avg_reward": np.mean([h["reward"] for h in relevant]) if relevant else 0.0,
                "avg_correct": np.mean([h["correct"] for h in relevant if h["correct"] is not None])
                               if relevant else 0.0,
            }
        return stats

    def predicted_best(self, qc: QueryContext) -> dict[str, float]:
        """Return predicted reward scores per strategy (no exploration)."""
        feat = extract_features(qc)
        return {name: float(self.weights[k] @ feat)
                for k, name in enumerate(STRATEGY_NAMES)}
