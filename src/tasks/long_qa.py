"""
Synthetic long-context QA task.

Generates examples at controlled context lengths by:
1. Taking a "fact" (the needle)
2. Padding with random distractor text to reach target length
3. Query asks for the fact; ground_truth is the fact value

This gives full control over:
  - Context length (sweep from 1K to 1M chars)
  - Needle position (start, middle, end, random)
  - Needle density (single fact vs. multiple interleaved facts)
  - Distractor type (random words vs. coherent paragraphs)
"""

from __future__ import annotations
import random
from .base import Task, Example

# Distractor text pool — replace with a real corpus for more realistic experiments
_DISTRACTORS = [
    "The mitochondria is the powerhouse of the cell. ",
    "Water freezes at zero degrees Celsius. ",
    "The capital of France is Paris, a city known for the Eiffel Tower. ",
    "Photosynthesis converts sunlight into chemical energy. ",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second. ",
    "Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet. ",
    "The Great Wall of China stretches over 21,000 kilometers. ",
    "DNA carries genetic information in living organisms. ",
]

_FACTS = [
    ("secret code",      "ZEPHYR-7742"),
    ("hidden password",  "BlueMoon_99!"),
    ("meeting time",     "3:47 PM on Thursday"),
    ("treasure location","under the old oak tree by the river bend"),
    ("answer to quiz",   "forty-two"),
    ("special token",    "TOKEN-ALPHA-9"),
    ("override command", "ENGAGE_OVERRIDE_DELTA"),
    ("target value",     "1.618033988"),
]


class SyntheticNeedleTask(Task):
    name = "synthetic_needle"

    def __init__(
        self,
        n_examples: int = 100,
        context_lengths: list[int] = None,  # in chars
        positions: list[str] = None,
        seed: int = 42,
    ):
        self.n_examples = n_examples
        self.context_lengths = context_lengths or [4_000, 16_000, 64_000, 256_000]
        self.positions = positions or ["start", "middle", "end", "random"]
        self.rng = random.Random(seed)
        self._examples = list(self._generate())

    def _generate(self) -> list[Example]:
        examples = []
        for _ in range(self.n_examples):
            key, value = self.rng.choice(_FACTS)
            ctx_len    = self.rng.choice(self.context_lengths)
            position   = self.rng.choice(self.positions)

            needle_text = f"The {key} is: {value}."
            filler      = self._make_filler(ctx_len - len(needle_text))
            context     = self._insert(needle_text, filler, position)

            examples.append(Example(
                query=f"What is the {key}?",
                context=context,
                ground_truth=value,
                task_type="needle",
                source="synthetic_needle",
                metadata={
                    "context_len": len(context),
                    "needle_position": position,
                    "key": key,
                },
            ))
        return examples

    def _make_filler(self, target_len: int) -> str:
        parts = []
        while sum(len(p) for p in parts) < target_len:
            parts.append(self.rng.choice(_DISTRACTORS))
        return "".join(parts)[:target_len]

    def _insert(self, needle: str, filler: str, position: str) -> str:
        n = len(filler)
        if position == "start":
            return needle + " " + filler
        elif position == "end":
            return filler + " " + needle
        elif position == "middle":
            mid = n // 2
            return filler[:mid] + " " + needle + " " + filler[mid:]
        else:  # random
            pos = self.rng.randint(0, n)
            return filler[:pos] + " " + needle + " " + filler[pos:]

    def __iter__(self):
        return iter(self._examples)

    def __len__(self):
        return len(self._examples)
