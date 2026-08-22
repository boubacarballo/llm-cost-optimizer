import json
from pydantic import BaseModel, Field
from typing import List
from llm_cost_optimizer.chat import Chat
from llm_cost_optimizer.types.types import VerificationResult
from llm_cost_optimizer.prompts import GENERATION_JUDGE_PROMPT_TEMPLATE
from llm_cost_optimizer.verifiers.utils import handle_alternate_verification
import asyncio

chat = Chat()


class GenerationJudgeResponse(BaseModel):
    instruction_adherence: float = Field(..., ge=0.0, le=1.0)
    coherence: float = Field(..., ge=0.0, le=1.0)
    relevance: float = Field(..., ge=0.0, le=1.0)
    violated_constraints: List[str] = []
    unsupported_claims: List[str] = []


class GenerationJudgeResult(VerificationResult):
    score: float
    raw: dict


def build_generation_judge_prompt(prompt: str, generated: str) -> str:
    return GENERATION_JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        generated=generated,
    )


def get_generation_judge_result(judge_response: GenerationJudgeResponse) -> GenerationJudgeResult:
    data = json.loads(judge_response.model_dump_json()) if hasattr(judge_response, "model_dump_json") else judge_response.dict()

    adherence = float(data.get("instruction_adherence", 0.0))
    coherence = float(data.get("coherence", 0.0))
    relevance = float(data.get("relevance", 0.0))
    violated = data.get("violated_constraints", []) or []
    unsupported = data.get("unsupported_claims", []) or []

    # doing what was asked matters more than reading nicely while doing something else
    base = 0.4 * adherence + 0.3 * coherence + 0.3 * relevance

    penalty = 0.0
    if violated:
        # an explicitly broken constraint is a harder failure than a shaky claim
        penalty += min(0.35, 0.15 * len(violated))
    if unsupported:
        penalty += min(0.2, 0.05 * len(unsupported))

    final = max(0.0, min(1.0, base - penalty))

    return GenerationJudgeResult(
        score=final,
        raw=data,
        verdict="Good" if final >= 0.7 else "Bad"
    )


async def verify_generation(prompt: str, response: str, model_config: dict) -> GenerationJudgeResult:

    judge_prompt = build_generation_judge_prompt(
        prompt=prompt,
        generated=response
    )

    messages = [
        {
            "role": "user",
            "content": judge_prompt
        }
    ]

    judge_response = await chat.send_request(messages, model_config=model_config, responseFormat=GenerationJudgeResponse)

    return get_generation_judge_result(judge_response.output_text)


async def handle_generation_verification(prompt, response, model_config, routing_context):
    initial_res = await verify_generation(
        prompt=prompt,
        response=response,
        model_config=model_config
    )
    if initial_res.verdict == "Bad":  # in this branch we log routing failure
        await handle_alternate_verification(
            verification_function=verify_generation,
            initial_response=initial_res,
            prompt=prompt,
            model_config=model_config,
            routing_context=routing_context
        )
    return initial_res


if __name__ == "__main__":

    prompt = (
        "Write a product blurb for a stainless steel water bottle. "
        "Exactly two sentences, no exclamation marks, aimed at commuters."
    )
    # four sentences, an exclamation mark, and drifts to hikers
    response = (
        "Stay hydrated on the trail! Our bottle keeps drinks cold for 24 hours. "
        "It is perfect for hikers scaling remote peaks. Built from tough steel that lasts."
    )

    print(asyncio.run(verify_generation(prompt, response, {"provider": "openai", "id": "gpt-5-mini"})))
