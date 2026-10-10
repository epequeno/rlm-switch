"""
Retrieval: chunk context, embed chunks, retrieve top-k, answer from retrieved.
Uses OpenAI embeddings. No external vector DB — keeps everything in-memory.
"""

from __future__ import annotations
import time
import numpy as np
from .base import QueryContext, StrategyResult
from ._client import make_client


class RetrievalStrategy:
    name = "retrieval"

    def __init__(self, model: str = "gpt-4o-mini",
                 embed_model: str = "openai/text-embedding-3-small",
                 chunk_size: int = 1_000,
                 top_k: int = 5,
                 backend: str = "openai"):
        self.model = model
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.backend = backend

    def run(self, qc: QueryContext) -> StrategyResult:
        t0 = time.time()
        total_input, total_output = 0, 0

        # Step 1: chunk the context
        chunks = self._chunk(qc.context)

        # Step 2: embed query + chunks
        query_emb, chunk_embs, embed_usage = self._embed(qc.query, chunks)
        total_input += embed_usage

        # Step 3: retrieve top-k by cosine similarity
        sims = self._cosine(query_emb, chunk_embs)
        top_idx = np.argsort(sims)[::-1][:self.top_k]
        retrieved = "\n\n---\n\n".join(chunks[i] for i in sorted(top_idx))

        # Step 4: answer from retrieved chunks
        prompt = f"Relevant context:\n{retrieved}\n\nQuestion: {qc.query}\n\nAnswer:"
        answer, usage = self._call(prompt)
        total_input  += usage.get("input", 0)
        total_output += usage.get("output", 0)

        return StrategyResult(
            strategy_name=self.name,
            answer=answer,
            tokens_input=total_input,
            tokens_output=total_output,
            n_llm_calls=2,  # embed call + answer call
            latency_seconds=time.time() - t0,
            cost_usd=self._cost(total_input, total_output, len(chunks)),
            metadata={"n_chunks": len(chunks), "top_k": self.top_k},
        )

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks, buf = [], []
        for w in words:
            buf.append(w)
            if len(" ".join(buf)) >= self.chunk_size:
                chunks.append(" ".join(buf))
                buf = []
        if buf:
            chunks.append(" ".join(buf))
        return chunks

    def _embed(self, query: str, chunks: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
        client = make_client()
        texts = [query] + chunks
        resp = client.embeddings.create(model=self.embed_model, input=texts)
        vecs = np.array([d.embedding for d in resp.data])
        total_tokens = resp.usage.total_tokens
        return vecs[0], vecs[1:], total_tokens

    def _cosine(self, q: np.ndarray, docs: np.ndarray) -> np.ndarray:
        q_norm   = q / (np.linalg.norm(q) + 1e-10)
        d_norm   = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-10)
        return d_norm @ q_norm

    def _call(self, prompt: str) -> tuple[str, dict]:
        client = make_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content, {
            "input": resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
        }

    def _cost(self, inp: int, out: int, n_chunks: int) -> float:
        embed_cost = n_chunks * 100 * 0.02e-6   # text-embedding-3-small: $0.02/1M
        return inp * 0.15e-6 + out * 0.60e-6 + embed_cost
