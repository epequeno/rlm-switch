"""Task and example types."""

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Example:
    query: str
    context: str
    ground_truth: str
    task_type: str = "qa"
    source: str = ""
    metadata: dict = field(default_factory=dict)


class Task:
    """Base class for benchmark tasks."""
    name: str = ""

    def __iter__(self) -> Iterator[Example]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @staticmethod
    def judge(answer: str, ground_truth: str) -> bool:
        """Default judge: case-insensitive contains. Override for task-specific judging."""
        return ground_truth.strip().lower() in answer.strip().lower()
