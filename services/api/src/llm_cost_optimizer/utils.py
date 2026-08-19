from enum import Enum
import json
import hashlib
import os
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Model constants
OPENAI_MODELS = (
    "gpt-5.6-luna",
    "gpt-5.4-nano",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5.6-terra",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.6-sol",
    "gpt-5.5-pro",
    "gpt-5.4-pro",
)

ANTHROPIC_MODELS = (
    "claude-haiku-4-5-20251001",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
)

TIER_1_TASKS = ("Extraction", "Closed QA", "Text Generation", "Open QA")
TIER_2_TASKS = ("Summarization", "Classification")
TIER_3_TASKS = ("Code Generation", "Chatbot", "Rewrite", "Brainstorming", "Other")

MILLION_TOKENS = 1_000_000


class TaskType(Enum):
    EXTRACTION = "Extraction"
    CLOSED_QA = "Closed QA"
    TEXT_GENERATION = "Text Generation"
    OPEN_QA = "Open QA"
    SUMMARIZATION = "Summarization"
    CLASSIFICATION = "Classification"
    CODE_GENERATION = "Code Generation"
    CHATBOT = "Chatbot"
    REWRITE = "Rewrite"
    BRAINSTORMING = "Brainstorming"
    OTHER = "Other"


def breakdown_results(result: Dict) -> Tuple[List[int], Optional[float], Optional[float]]:
    """Extract tier flags, contextual knowledge and prompt complexity from classifier result.

    Returns (tier_flags, contextual_knowledge, prompt_complexity).
    """
    try:
        task_types = result.get("task_type_1", []) + result.get("task_type_2", [])
        tier_flags = [0, 0, 0]
        contextual_knowledge = None
        prompt_complexity = None

        if "contextual_knowledge" in result and result["contextual_knowledge"]:
            contextual_knowledge = result["contextual_knowledge"][0]
        if "prompt_complexity_score" in result and result["prompt_complexity_score"]:
            prompt_complexity = result["prompt_complexity_score"][0]

        for task in task_types:
            if task in TIER_1_TASKS:
                tier_flags[0] = 1
            elif task in TIER_2_TASKS:
                tier_flags[1] = 1
            elif task in TIER_3_TASKS:
                tier_flags[2] = 1

        return tier_flags, contextual_knowledge, prompt_complexity
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Error breaking down classifier results: {exc}")


def load_models_configs(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError as err:
        raise FileNotFoundError(f"File not found or invalid path: {err}")


# attempt to load config from environment path at import time (may be None)
config = None
try:
    config = load_models_configs(os.getenv("MODEL_CONFIGS_PATH"))
except FileNotFoundError:
    config = None


def get_candidate_models(flags: List[int], cfg: Optional[Dict]) -> List[Dict]:
    """Return candidate models from config given tier flags.

    flags is [tier1, tier2, tier3]
    """
    if not cfg:
        return []
    try:
        if flags[0]:
            return cfg.get("tiers", {}).get("tier_1", [])
        elif flags[1]:
            return cfg.get("tiers", {}).get("tier_2", [])
        elif flags[2]:
            return cfg.get("tiers", {}).get("tier_3", [])
        return []
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Error loading candidate models: {exc}")


def select_model_and_provider(models: List[Dict], context_window: int) -> Optional[Dict]:

    try:
        
        sorted_models = sorted(
            models,
            key=lambda m: (
                m.get("context_window", 0),
                m.get("pricing_per_million_tokens", {}).get("input", 0)
                + m.get("pricing_per_million_tokens", {}).get("output", 0),
            ),
        )

        for model in sorted_models:
            if model.get("context_window", 0) >= context_window:
                return model
        return None
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Something went wrong selecting model and provider: {exc}")


def parse_token_pricing(model_config: Dict) -> Tuple[float, float]:
    million_token_pricing = model_config["pricing_per_million_tokens"]
    input_token_pricing = million_token_pricing["input"]
    output_token_pricing = million_token_pricing["output"]
    return input_token_pricing, output_token_pricing


def compute_request_cost(model_config: Dict, input_tokens: int, output_tokens: int) -> float:
    input_token_pricing, output_token_pricing = parse_token_pricing(model_config)
    input_cost = input_tokens * (input_token_pricing / MILLION_TOKENS)
    output_cost = output_tokens * (output_token_pricing / MILLION_TOKENS)
    return input_cost + output_cost


def hash_text(text: str) -> str:
    encoded_bytes = text.encode("utf-8")
    return hashlib.sha256(encoded_bytes).hexdigest()


def generate_messages_template(role: str, message: str, messages: Optional[List[Dict]] = None) -> List[Dict]:
    if messages is None:
        return [{"role": role, "content": message}]
    messages.append({"role": role, "content": message})
    return messages


def get_model(data: dict, context_window: int) -> dict:
    config = load_models_configs(os.getenv("MODEL_CONFIGS_PATH"))
    model_tier = "tier_" + str(data["results"]["tier"])
    tier_models = config["tiers"][model_tier]
    model_config = select_model_and_provider(
        models=tier_models,
        context_window=context_window
    )
    return model_config
    
    


########## Verification Model Configs ##########


class ModelConfig(BaseModel):
    provider: Literal["openai", "anthropic"]
    model_id: str = Field(..., min_length=1)


class SummaryJudgeClaims(BaseModel):
    claim: str
    status: Literal["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]


class SummaryJudgeKeyPoints(BaseModel):
    point: str
    status: Literal["PRESENT", "MISSING"]


class SummaryJudgeResponse(BaseModel):
    claims: List[SummaryJudgeClaims]
    points: List[SummaryJudgeKeyPoints]


class SummaryJudgeResult(BaseModel):
    precision: float
    recall: float
    raw: Dict


class SummaryJudgeModelConfig(BaseModel):
    provider: str = "openai"
    model_id: str = "gpt-5.6-luna"


class Verification(BaseModel):
    task_type: str
    score: float
