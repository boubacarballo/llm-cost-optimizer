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
TIER_1_TASKS = (
    "Extraction",
    "Closed QA",
    "Text Generation",
    "Open QA",
)
TIER_2_TASKS = (
    "Summarization",
    "Classification",
)
TIER_3_TASKS = (
    "Code Generation",
    "Chatbot",
    "Rewrite",
    "Brainstorming",
    "Other",
)
MILLION_TOKENS = 1000000

from enum import Enum
import yaml
from pydantic import BaseModel, Field
from typing import Literal
from typing import Literal
import json
import hashlib

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
    
    


def breakdown_results(result):
    
    try:
        task_types = result["task_type_1"] + result["task_type_2"]
        tier_flags = [0, 0, 0]
        contextual_knowledge = result["contextual_knowledge"][0]
        prompt_complexity = result["prompt_complexity_score"][0]
        
        
        for task in task_types:
            if task in TIER_1_TASKS:
                tier_flags[0] = 1
            elif task in TIER_2_TASKS:
                tier_flags[1] = 1
            elif task in TIER_3_TASKS:
                tier_flags[2] = 1
                
        return tier_flags, contextual_knowledge, prompt_complexity
    
    except Exception as exc:
        raise f"Error breaking down classifier results: {exc}"



def load_models_configs(path):
    
    try:
        with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                
        return config
        
    except FileNotFoundError as err:
        print(f"File not found or invalid path: {err}")
        

def get_candidate_models(flags, config):
    try:
        if flags[0]:
            return config.get("tiers", {}).get("tier_1", [])
        
        elif flags[1]:
            return config.get("tiers", {}).get("tier_2", [])
        
        elif flags[2]:
            return config.get("tiers", {}).get("tier_2", [])
    except Exception as exc:
        raise f"Error loading candidate models: {exc}"
    
def select_model_and_provider(models, context_window):
    # take the cheapest that satisfies the context window
    try:
        models = sorted(
        models,
        key=lambda m: (
            m["context_window"],
            m["pricing_per_million_tokens"]["input"] + m["pricing_per_million_tokens"]["output"]
        )
        )
        
        for model in models:
            if model["context_window"] > context_window:
                return model
    except Exception as exc:
        raise f"Something went wrong selecting model and provider: {exc}"
    
        
def parse_token_pricing(model_config):
    million_token_pricing = model_config["pricing_per_million_tokens"]
    input_token_pricing = million_token_pricing["input"]
    output_token_pricing = million_token_pricing["output"]
    return input_token_pricing, output_token_pricing
    
def compute_request_cost(model_config, input_tokens, output_tokens):
    
    input_token_pricing, output_token_pricing = parse_token_pricing(model_config)
    input_cost = input_tokens * (input_token_pricing / MILLION_TOKENS)
    output_cost = output_tokens * (output_token_pricing / MILLION_TOKENS)
    return input_cost + output_cost


    
        
def hash_text(text):
    encoded_bytes = text.encode("utf-8")
    return hashlib.sha256(encoded_bytes).hexdigest()
        
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
    claims: list[SummaryJudgeClaims]
    points: list[SummaryJudgeKeyPoints]


class SummaryJudgeResult(BaseModel):
    precision: float
    recall: float
    raw: dict
    
class SummaryJudgeModelConfig(BaseModel):
    provider: str = "openai"
    model_id: str = "gpt-5.6-luna"