"""Passthrough: send query + full context to the model in one shot."""

from __future__ import annotations
import time
from .base import Strategy, QueryContext, StrategyResult
from ._client import make_client


class PassthroughStrategy:
    name = "passthrough"

    def __init__(self, model: str = "gpt-4o-mini", backend: str = "openai",
                 max_context_chars: int = 800_000):
        self.model = model
        self.backend = backend
        self.max_context_chars = max_context_chars

    def run(self, qc: QueryContext) -> StrategyResult:
        context = qc.context[:self.max_context_chars]
        prompt = f"Context:\n{context}\n\nQuestion: {qc.query}\n\nAnswer:"

        t0 = time.time()
        response, usage = self._call(prompt)
        latency = time.time() - t0

        return StrategyResult(
            strategy_name=self.name,
            answer=response,
            tokens_input=usage.get("input", 0),
            tokens_output=usage.get("output", 0),
            n_llm_calls=1,
            latency_seconds=latency,
            cost_usd=self._cost(usage),
        )

    def _call(self, prompt: str) -> tuple[str, dict]:
        if self.backend == "openai":
            client = make_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            usage = {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            }
            return resp.choices[0].message.content, usage
        raise NotImplementedError(f"Backend {self.backend} not implemented")

    def _cost(self, usage: dict) -> float:
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        return usage.get("input", 0) * 0.15e-6 + usage.get("output", 0) * 0.60e-6
