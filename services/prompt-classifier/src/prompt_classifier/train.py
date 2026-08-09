from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path
from dotenv import load_dotenv
import os

import torch
from torch import nn

from prompt_classifier.router.features import RouterInput, collate_inputs
from prompt_classifier.router.model import FEATURE_SCHEMA_VERSION, TierRouter

load_dotenv()


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    return torch.device(value)


def load_examples(path: Path) -> list[tuple[RouterInput, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"prompt", "token_count", "context_provided", "context_window", "task_type", "tier"}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(expected))}")
        examples = []
        for line, row in enumerate(reader, start=2):
            try:
                item = RouterInput(
                    prompt=row["prompt"],
                    token_count=int(row["token_count"]),
                    context_provided=row["context_provided"].strip().lower() in {"1", "true", "yes"},
                    context_window=int(row["context_window"]),
                    task_type=row["task_type"],
                )
                item.validate()
                tier = int(row["tier"])
                if tier not in {1, 2, 3}:
                    raise ValueError("tier must be 1, 2, or 3")
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid row {line}: {exc}") from exc
            examples.append((item, tier - 1))
    if len(examples) < 20:
        raise ValueError("provide at least 20 labelled examples; a few hundred is a useful starting point")
    return examples


def accuracy(model: TierRouter, examples: list[tuple[RouterInput, int]], device: torch.device) -> float:
    if not examples:
        return 0.0
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            batch = examples[start : start + 128]
            tensors = collate_inputs(
                [item for item, _ in batch],
                hash_buckets=model.config["hash_buckets"],
                max_prompt_tokens=model.config["max_prompt_tokens"],
                device=device,
            )
            predictions = model(**tensors).argmax(dim=1).cpu().tolist()
            correct += sum(prediction == label for prediction, (_, label) in zip(predictions, batch))
    return correct / len(examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=os.getenv("DATASET_URL"))
    parser.add_argument("--output", default="artifacts/tier_router.pt", type=Path)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=2e-3, type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--seed", default=7, type=int)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if not args.data:
        raise RuntimeError("Failed to get dataset")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_from_arg(args.device)
    examples = load_examples(args.data)
    random.shuffle(examples)
    split = max(1, round(len(examples) * 0.15))
    valid, train = examples[:split], examples[split:]
    model = TierRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        running_loss = 0.0
        for start in range(0, len(train), args.batch_size):
            batch = train[start : start + args.batch_size]
            tensors = collate_inputs(
                [item for item, _ in batch],
                hash_buckets=model.config["hash_buckets"],
                max_prompt_tokens=model.config["max_prompt_tokens"],
                device=device,
            )
            labels = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = criterion(model(**tensors), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch)
        print(
            f"epoch={epoch:02d} train_loss={running_loss / len(train):.4f} "
            f"validation_accuracy={accuracy(model, valid, device):.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config,
            "model_state": model.cpu().state_dict(),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
        args.output,
    )
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
