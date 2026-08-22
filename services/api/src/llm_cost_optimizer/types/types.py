from pydantic import BaseModel, Field
from typing import Literal, Optional
from decimal import Decimal
from llm_cost_optimizer.utils import TaskType



class TokenPricing(BaseModel):
    input: float = Field(..., description="Price (in USD) per 1,000,000 input tokens")
    output: float = Field(..., description="Price (in USD) per 1,000,000 output tokens")


class ModelConfig(BaseModel):
    id: str
    provider: str
    pricing_per_million_tokens: TokenPricing
    reasoning_model: bool
    context_window: int
    max_output_tokens: int
    best_for: str

class VerificationResult(BaseModel):
    verdict: Literal["Good", "Bad"]

        

class RoutingContext(BaseModel):
    """Everything about the original request that a routing failure needs to be
    interpretable later: the prompt itself, what the classifiers measured, and
    which tier we ended up routing to."""
    prompt: str
    token_count: int
    context_provided: bool
    context_window: int
    task_type: TaskType
    task_tier: int
