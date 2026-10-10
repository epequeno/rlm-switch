"""Compaction: summarize the context into a budget, then answer."""

from __future__ import annotations
import time
from .base import Strategy, QueryContext, StrategyResult
from ._client import make_client


class CompactionStrategy:
    name = "compaction"

    def __init__(self, model: str = "gpt-4o-mini", target_chars: int = 8_000,
                 backend: str = "openai"):
        self.model = model
        self.target_chars = target_chars
        self.backend = backend

    def run(self, qc: QueryContext) -> StrategyResult:
        t0 = time.time()
        total_input, total_output = 0, 0

        # Step 1: Compress context
        if len(qc.context) > self.target_chars:
            summary, usage1 = self._summarize(qc.context, qc.query)
            total_input  += usage1.get("input", 0)
            total_output += usage1.get("output", 0)
        else:
            summary = qc.context
            usage1  = {"input": 0, "output": 0}

        # Step 2: Answer from summary
        prompt = f"Context summary:\n{summary}\n\nQuestion: {qc.query}\n\nAnswer:"
        answer, usage2 = self._call(prompt)
        total_input  += usage2.get("input", 0)
        total_output += usage2.get("output", 0)

        return StrategyResult(
            strategy_name=self.name,
            answer=answer,
            tokens_input=total_input,
            tokens_output=total_output,
            n_llm_calls=2 if len(qc.context) > self.target_chars else 1,
            latency_seconds=time.time() - t0,
            cost_usd=self._cost(total_input, total_output),
            metadata={"summary_len": len(summary)},
        )

    def _summarize(self, context: str, query: str) -> tuple[str, dict]:
        prompt = (
            f"Summarize the following text to roughly {self.target_chars // 4} words, "
            f"preserving information relevant to this question: {query}\n\n"
            f"Text:\n{context[:400_000]}"
        )
        return self._call(prompt, max_tokens=self.target_chars // 4)

    def _call(self, prompt: str, max_tokens: int = 512) -> tuple[str, dict]:
        if self.backend == "openai":
            client = make_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            usage = {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            }
            return resp.choices[0].message.content, usage
        raise NotImplementedError

    def _cost(self, inp: int, out: int) -> float:
        return inp * 0.15e-6 + out * 0.60e-6
