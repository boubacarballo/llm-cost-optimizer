from __future__ import annotations

import argparse
import json

import torch

from prompt_classifier.router.features import RouterInput, collate_inputs
from prompt_classifier.router.model import TierRouter
from prompt_classifier.train import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/tier_router.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--token-count", required=True, type=int)
    parser.add_argument("--context-provided", action="store_true")
    parser.add_argument("--context-window", required=True, type=int)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--minimum-confidence", type=float, default=0.0,
        help="If confidence is below this, increase the selected tier by one (max 3).",
    )
    args = parser.parse_args()
    if not 0.0 <= args.minimum_confidence <= 1.0:
        raise ValueError("minimum-confidence must be between 0 and 1")

    device = device_from_arg(args.device)
    model = TierRouter.from_checkpoint(args.checkpoint, device)
    item = RouterInput(
        prompt=args.prompt,
        token_count=args.token_count,
        context_provided=args.context_provided,
        context_window=args.context_window,
        task_type=args.task_type,
    )
    tensors = collate_inputs(
        [item],
        hash_buckets=model.config["hash_buckets"],
        max_prompt_tokens=model.config["max_prompt_tokens"],
        device=device,
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(**tensors), dim=1)[0].cpu().tolist()
    raw_tier = int(torch.tensor(probabilities).argmax().item()) + 1
    confidence = probabilities[raw_tier - 1]
    tier = min(raw_tier + 1, 3) if confidence < args.minimum_confidence else raw_tier
    print(
        json.dumps(
            {
                "tier": tier,
                "raw_tier": raw_tier,
                "confidence": round(confidence, 4),
                "probabilities": {f"tier_{index + 1}": round(value, 4) for index, value in enumerate(probabilities)},
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
