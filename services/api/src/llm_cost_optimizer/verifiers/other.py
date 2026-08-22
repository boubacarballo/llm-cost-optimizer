import json
from pydantic import BaseModel, Field
from typing import List
from llm_cost_optimizer.chat import Chat
from llm_cost_optimizer.types.types import VerificationResult
from llm_cost_optimizer.prompts import GENERIC_JUDGE_PROMPT_TEMPLATE
from llm_cost_optimizer.verifiers.utils import handle_alternate_verification
import asyncio

chat = Chat()


class GenericJudgeResponse(BaseModel):
    helpfulness: float = Field(..., ge=0.0, le=1.0)
    correctness: float = Field(..., ge=0.0, le=1.0)
    instruction_adherence: float = Field(..., ge=0.0, le=1.0)
    unsupported_claims: List[str] = []


class GenericJudgeResult(VerificationResult):
    score: float
    raw: dict


def build_generic_judge_prompt(prompt: str, reply: str) -> str:
    return GENERIC_JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        reply=reply,
    )


def get_generic_judge_result(judge_response: GenericJudgeResponse) -> GenericJudgeResult:
    data = json.loads(judge_response.model_dump_json()) if hasattr(judge_response, "model_dump_json") else judge_response.dict()

    helpfulness = float(data.get("helpfulness", 0.0))
    correctness = float(data.get("correctness", 0.0))
    adherence = float(data.get("instruction_adherence", 0.0))
    unsupported = data.get("unsupported_claims", []) or []

    base = 0.4 * helpfulness + 0.35 * correctness + 0.25 * adherence

    penalty = min(0.3, 0.1 * len(unsupported)) if unsupported else 0.0

    final = max(0.0, min(1.0, base - penalty))

    return GenericJudgeResult(
        score=final,
        raw=data,
        verdict="Good" if final >= 0.7 else "Bad"
    )


async def verify_other(prompt: str, response: str, model_config: dict) -> GenericJudgeResult:

    judge_prompt = build_generic_judge_prompt(
        prompt=prompt,
        reply=response
    )

    messages = [
        {
            "role": "user",
            "content": judge_prompt
        }
    ]

    judge_response = await chat.send_request(messages, model_config=model_config, responseFormat=GenericJudgeResponse)

    return get_generic_judge_result(judge_response.output_text)


async def handle_other_verification(prompt, response, model_config, routing_context):
    initial_res = await verify_other(
        prompt=prompt,
        response=response,
        model_config=model_config
    )
    if initial_res.verdict == "Bad":  # in this branch we log routing failure
        await handle_alternate_verification(
            verification_function=verify_other,
            initial_response=initial_res,
            prompt=prompt,
            model_config=model_config,
            routing_context=routing_context
        )
    return initial_res


if __name__ == "__main__":

    prompt = "My sourdough starter smells like acetone. In two sentences, what do I do?"
    # evasive, ignores the two-sentence instruction, adds a dubious claim
    response = (
        "Sourdough is a fascinating craft with a long history. There are many possible causes. "
        "Acetone smell always means the starter is dead and must be thrown away. "
        "You may want to consult a professional baker."
    )

    print(asyncio.run(verify_other(prompt, response, {"provider": "openai", "id": "gpt-5-mini"})))
