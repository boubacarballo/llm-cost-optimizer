from __future__ import annotations

import torch
from torch import nn

from prompt_classifier.router.features import TASK_TYPES

FEATURE_SCHEMA_VERSION = 2


class TierRouter(nn.Module):
    def __init__(
        self,
        hash_buckets: int = 16_384,
        word_embedding_dim: int = 32,
        task_embedding_dim: int = 8,
        hidden_dim: int = 64,
        max_prompt_tokens: int = 384,
    ) -> None:
        super().__init__()
        self.config = {
            "hash_buckets": hash_buckets,
            "word_embedding_dim": word_embedding_dim,
            "task_embedding_dim": task_embedding_dim,
            "hidden_dim": hidden_dim,
            "max_prompt_tokens": max_prompt_tokens,
        }
        self.word_bag = nn.EmbeddingBag(hash_buckets, word_embedding_dim, mode="mean")
        self.task_embedding = nn.Embedding(len(TASK_TYPES), task_embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(word_embedding_dim + task_embedding_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self, word_ids: torch.Tensor, offsets: torch.Tensor, task_ids: torch.Tensor, numeric: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat((self.word_bag(word_ids, offsets), self.task_embedding(task_ids), numeric), dim=1)
        return self.classifier(features)

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device | str) -> "TierRouter":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "checkpoint uses a different task-type schema; retrain the router with the current code"
            )
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        return model.to(device).eval()
