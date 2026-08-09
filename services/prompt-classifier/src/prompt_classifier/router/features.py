from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable

import torch


TASK_TYPES = (
    "Extraction",
    "Closed QA",
    "Text Generation",
    "Open QA",
    "Summarization",
    "Classification",
    "Code Generation",
    "Chatbot",
    "Rewrite",
    "Brainstorming",
    "Other",
)
TASK_TO_ID = {name.casefold(): index for index, name in enumerate(TASK_TYPES)}

# Kept intentionally small: these are useful signals without assuming the
# task type is perfectly supplied by callers.
COMPLEXITY_PATTERNS = (
    r"\b(prove|proof|derive|theorem)\b",
    r"\b(debug|bug|stack trace|root cause)\b",
    r"\b(architecture|refactor|distributed|concurrent)\b",
    r"\b(security|threat model|exploit|vulnerability)\b",
    r"\b(legal|medical|financial|contract)\b",
    r"\b(compare|trade[- ]?off|evaluate alternatives)\b",
    r"\b(step by step|multi[- ]?step|carefully reason)\b",
    r"```|\b(function|class|sql|regex|api)\b",
)
COMPLEXITY_RE = tuple(re.compile(pattern, re.IGNORECASE) for pattern in COMPLEXITY_PATTERNS)
WORD_RE = re.compile(r"[A-Za-z0-9_+#.-]+")


@dataclass(frozen=True)
class RouterInput:
    prompt: str
    token_count: int
    context_provided: bool
    context_window: int
    task_type: str

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.token_count < 1:
            raise ValueError("token_count must be at least 1")
        if self.context_window < self.token_count:
            raise ValueError("context_window must be at least token_count")
        if not self.task_type.strip():
            raise ValueError("task_type must not be empty")
        normalize_task_type(self.task_type)


def stable_hash(value: str, modulus: int) -> int:
    """Return a process-independent hash, unlike Python's built-in hash()."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus


def tokenize(prompt: str, max_tokens: int) -> list[str]:
    return WORD_RE.findall(prompt.lower())[:max_tokens] or ["<empty>"]


def normalize_task_type(task_type: str) -> str:
    """Return the canonical task type, accepting case/whitespace variation."""
    normalized = " ".join(task_type.split()).casefold()
    task_index = TASK_TO_ID.get(normalized)
    if task_index is None:
        valid = ", ".join(TASK_TYPES)
        raise ValueError(f"unknown task_type {task_type!r}; choose one of: {valid}")
    return TASK_TYPES[task_index]


def task_id(task_type: str) -> int:
    return TASK_TO_ID[normalize_task_type(task_type).casefold()]


def numeric_features(item: RouterInput) -> list[float]:
    """Return normalized, bounded features so model behaviour is stable."""
    prompt = item.prompt
    complexity_hits = sum(bool(regex.search(prompt)) for regex in COMPLEXITY_RE)
    question_count = min(prompt.count("?"), 5) / 5.0
    line_count = min(prompt.count("\n") + 1, 30) / 30.0
    # A large history can make retrieval/synthesis harder even if the new turn
    # is short. log1p keeps the range well behaved.
    return [
        min(math.log1p(item.token_count) / math.log1p(128_000), 1.5),
        min(math.log1p(item.context_window) / math.log1p(1_000_000), 1.5),
        float(item.context_provided),
        complexity_hits / len(COMPLEXITY_RE),
        question_count,
        line_count,
    ]


def collate_inputs(
    items: Iterable[RouterInput],
    *,
    hash_buckets: int,
    max_prompt_tokens: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Create tensors for a variable-length bag-of-words batch."""
    materialized = list(items)
    if not materialized:
        raise ValueError("at least one router input is required")
    for item in materialized:
        item.validate()

    flattened: list[int] = []
    offsets: list[int] = []
    task_ids: list[int] = []
    numbers: list[list[float]] = []
    for item in materialized:
        offsets.append(len(flattened))
        flattened.extend(stable_hash(token, hash_buckets) for token in tokenize(item.prompt, max_prompt_tokens))
        task_ids.append(task_id(item.task_type))
        numbers.append(numeric_features(item))

    return {
        "word_ids": torch.tensor(flattened, dtype=torch.long, device=device),
        "offsets": torch.tensor(offsets, dtype=torch.long, device=device),
        "task_ids": torch.tensor(task_ids, dtype=torch.long, device=device),
        "numeric": torch.tensor(numbers, dtype=torch.float32, device=device),
    }
