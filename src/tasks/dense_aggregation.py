"""
Dense aggregation task — synthetic OOLONG-style benchmark for rlm-switch.

The context is a long sequence of labeled examples (like an ICL prompt):

    Example 1: <short_text> -> Label: <category>, Score: <value>
    Example 2: <short_text> -> Label: <category>, Score: <value>
    ...
    Example N: <short_text> -> Label: <category>, Score: <value>

Queries require aggregation across ALL examples — not finding a single
needle:

    - Count:    "How many examples have the label 'sports'?"
    - Sum:      "What is the total score of all examples labeled 'business'?"
    - Majority: "Which label appears most frequently among the examples?"

Why this replaces the synthetic needle task for Phase 1:

The needle task was invalidated because passthrough achieves 100% at all
tested context lengths — there is no context rot in a single-needle
regime within the context window. This task requires processing MANY
records and aggregating, so passthrough degrades as context grows because
the model misses or miscounts records. REPL wins because it can write
regex + count programmatically, which is length-invariant.

The regime where each strategy should win:
    - Passthrough:  short contexts (few records, easy to count by eye)
    - Compaction:   loses specific records in the summary
    - Retrieval:    retrieves chunks by similarity but may miss scattered records
    - REPL:         regex extraction + programmatic aggregation — exact at any length
"""

from __future__ import annotations
import random
import re
from collections import Counter
from .base import Task, Example


# Generic text snippets — label assignment is RANDOM and independent of text
# content. This prevents the model from shortcutting by associating text
# patterns with labels. The model must read each example's explicit label
# field to determine its category.
_TEXT_SNIPPETS = [
    "The quarterly report showed improved performance across multiple departments",
    "Local officials announced a new initiative to improve community infrastructure",
    "Researchers published findings that challenge the conventional understanding",
    "The committee voted unanimously to approve the revised proposal after deliberation",
    "Market analysts predict steady growth in the coming months despite uncertainties",
    "A recent survey revealed changing attitudes among younger demographics",
    "The development team completed the migration ahead of schedule and under budget",
    "Community leaders gathered to discuss strategies for addressing regional challenges",
    "New regulations will take effect next month affecting operations across the sector",
    "The review process identified several areas requiring immediate attention",
    "Preliminary results indicate a significant shift in public opinion",
    "The board approved a comprehensive restructuring plan at yesterday's meeting",
    "Industry experts highlighted emerging trends during the annual conference",
    "The assessment concluded that current measures are insufficient",
    "Officials confirmed that implementation will begin in phases starting next quarter",
    "The investigation revealed previously unknown factors contributing to the situation",
    "Stakeholders expressed mixed reactions to the proposed policy changes",
    "The analysis covered multiple dimensions including efficiency and sustainability",
    "Recent developments have prompted a reassessment of long-term strategies",
    "The evaluation highlighted both strengths and areas needing improvement",
]


class DenseAggregationTask(Task):
    """OOLONG-style dense aggregation benchmark."""

    name = "dense_aggregation"

    def __init__(
        self,
        n_examples: int = 50,
        context_lengths: list[int] | None = None,
        categories: list[str] | None = None,
        query_types: list[str] | None = None,
        seed: int = 42,
    ):
        self.n_examples = n_examples
        self.context_lengths = context_lengths or [4_000, 16_000, 64_000]
        self.categories = categories or ["world", "sports", "business", "technology"]
        self.query_types = query_types or ["count", "sum", "majority"]
        self.rng = random.Random(seed)
        self._examples = list(self._generate())

    def _generate(self) -> list[Example]:
        examples = []
        for _ in range(self.n_examples):
            ctx_len = self.rng.choice(self.context_lengths)
            q_type = self.rng.choice(self.query_types)

            context, records = self._make_context(ctx_len)
            query, ground_truth = self._make_query(q_type, records)

            examples.append(Example(
                query=query,
                context=context,
                ground_truth=ground_truth,
                task_type="aggregation",
                source="dense_aggregation",
                metadata={
                    "context_len": len(context),
                    "query_type": q_type,
                    "n_records": len(records),
                },
            ))
        return examples

    # -- Context generation ---------------------------------------------------

    def _make_context(self, target_len: int) -> tuple[str, list[dict]]:
        """
        Build a context of labeled examples as JSON-lines at approximately
        target_len chars. Each line is a JSON object:
            {"id": 1, "text": "...", "label": "sports", "score": 42}

        JSON-lines format is chosen deliberately: it's unambiguous for
        programmatic processing (the RLM model writes json.loads() instead
        of guessing regex patterns), isolating the routing question from
        model code-generation ability. The number of records scales with
        target_len so longer contexts have more records to aggregate —
        this is what causes passthrough to degrade (counting more items
        is harder for attention, but trivial for code).
        """
        import json as _json
        avg_line_len = 110  # approximate chars per JSON line
        n_records = max(10, target_len // avg_line_len)

        records: list[dict] = []
        lines: list[str] = []
        for i in range(n_records):
            text = self.rng.choice(_TEXT_SNIPPETS)
            label = self.rng.choice(self.categories)
            score = self.rng.randint(1, 100)
            record = {"id": i + 1, "text": text, "label": label, "score": score}
            records.append({"label": label, "score": score, "text": text})
            lines.append(_json.dumps(record))

        context = "\n".join(lines)

        # Pad to at least 90% of target if we fell short due to text variance
        while len(context) < target_len * 0.9:
            i = len(records)
            text = self.rng.choice(_TEXT_SNIPPETS)
            label = self.rng.choice(self.categories)
            score = self.rng.randint(1, 100)
            record = {"id": i + 1, "text": text, "label": label, "score": score}
            records.append({"label": label, "score": score, "text": text})
            context += "\n" + _json.dumps(record)

        return context, records

    # -- Query generation -----------------------------------------------------

    def _make_query(self, q_type: str, records: list[dict]) -> tuple[str, str]:
        if q_type == "count":
            # Only pick labels that actually appear (avoids trivial count=0)
            available = [l for l in self.categories
                         if any(r["label"] == l for r in records)]
            target_label = self.rng.choice(available)
            count = sum(1 for r in records if r["label"] == target_label)
            query = f"How many examples have the label '{target_label}'?"
            return query, str(count)

        elif q_type == "sum":
            available = [l for l in self.categories
                         if any(r["label"] == l for r in records)]
            target_label = self.rng.choice(available)
            total = sum(r["score"] for r in records if r["label"] == target_label)
            query = f"What is the total score of all examples labeled '{target_label}'?"
            return query, str(total)

        elif q_type == "majority":
            label_counts = Counter(r["label"] for r in records)
            winner = label_counts.most_common(1)[0][0]
            query = "Which label appears most frequently among the examples?"
            return query, winner

        else:
            raise ValueError(f"Unknown query type: {q_type}")

    # -- Task-specific judge --------------------------------------------------

    @staticmethod
    def judge(answer: str, ground_truth: str) -> bool:
        """
        Numeric-aware judge for aggregation answers.

        For numeric ground truth, extracts all integers from the answer and
        checks if the target is among them — prevents '7' matching '17' or
        '70', which the naive contains judge would accept.

        For non-numeric ground truth (e.g. majority label name), uses
        word-boundary regex matching.
        """
        gt = ground_truth.strip()
        ans = answer.strip()

        if gt.isdigit():
            numbers = re.findall(r'\b(\d+)\b', ans)
            return gt in numbers

        pattern = r'\b' + re.escape(gt.lower()) + r'\b'
        return bool(re.search(pattern, ans.lower()))

    # -- Iteration ------------------------------------------------------------

    def __iter__(self):
        return iter(self._examples)

    def __len__(self):
        return len(self._examples)
