"""Core types shared across all strategies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import time


@dataclass
class StrategyResult:
    strategy_name: str
    answer: str
    correct: bool | None = None      # filled in by evaluation
    tokens_input: int = 0
    tokens_output: int = 0
    n_llm_calls: int = 1
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output


@dataclass
class QueryContext:
    """A single (query, context) pair with optional ground truth."""
    query: str
    context: str
    ground_truth: str | None = None
    task_type: str = "qa"            # "qa" | "summarization" | "needle" | ...
    source: str = ""                 # dataset/task name

    # Derived features (computed lazily)
    _context_tokens: int | None = None
    _query_tokens: int | None = None

    def context_tokens(self, tokenizer=None) -> int:
        if self._context_tokens is None:
            if tokenizer:
                self._context_tokens = len(tokenizer.encode(self.context))
            else:
                self._context_tokens = len(self.context) // 4  # rough
        return self._context_tokens

    def query_tokens(self, tokenizer=None) -> int:
        if self._query_tokens is None:
            if tokenizer:
                self._query_tokens = len(tokenizer.encode(self.query))
            else:
                self._query_tokens = len(self.query) // 4
        return self._query_tokens


@runtime_checkable
class Strategy(Protocol):
    name: str

    def run(self, qc: QueryContext) -> StrategyResult:
        """Execute the strategy on a query+context pair."""
        ...
