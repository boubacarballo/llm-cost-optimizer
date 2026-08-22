import json
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from llm_cost_optimizer.chat import Chat, Response
from llm_cost_optimizer.utils import Verification
from llm_cost_optimizer.types.types import VerificationResult
from llm_cost_optimizer.verifiers.utils import handle_alternate_verification, log_routing_failure
import asyncio

chat = Chat()

REWRITE_JUDGE_PROMPT_TEMPLATE = """
You are an exacting evaluator that checks whether a REWRITTEN TEXT preserves the original
meaning and follows any rewrite instructions. Do NOT rewrite the text — only evaluate it.

Inputs:
- PROMPT: The task + the original source text that needed to be rewritten
- REWRITTEN: The rewritten output produced by a model.

Tasks (follow exactly):
1) Meaning preservation: decide whether the REWRITTEN text PRESERVED, PARTIALLY_PRESERVED, or NOT_PRESERVED the meaning of the text that needed rewriting.
2) Added information: list any facts or claims present in REWRITTEN that are NOT supported by the PROMPT (may be empty array).
3) Missing information: list any important facts or details present in PROMPT that are missing from REWRITTEN (may be empty array).
4) Adherence score: a number between 0.0 and 1.0 indicating how well REWRITTEN follows the instructions inside PROMPT (higher is better).
5) Fluency score: a number between 0.0 and 1.0 indicating writing quality and readability.


PROMPT:
{prompt}

REWRITTEN:
{rewritten}
"""


class RewriteJudgeResponse(BaseModel):
    meaning: Literal["PRESERVED", "PARTIALLY_PRESERVED", "NOT_PRESERVED"]
    added_information: List[str] = []
    missing_information: List[str] = []
    adherence: float = Field(..., ge=0.0, le=1.0)
    fluency: float = Field(..., ge=0.0, le=1.0)


class RewriteJudgeResult(VerificationResult):
    score: float
    raw: dict


def build_rewrite_judge_prompt(prompt: str, rewritten: str, instructions: Optional[str] = "") -> str:
    return REWRITE_JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        rewritten=rewritten,
    )


def get_rewrite_judge_result(judge_response: RewriteJudgeResponse) -> RewriteJudgeResult:
    data = json.loads(judge_response.model_dump_json()) if hasattr(judge_response, "model_dump_json") else judge_response.dict()

    meaning = data.get("meaning")
    adherence = float(data.get("adherence", 0.0))
    fluency = float(data.get("fluency", 0.0))
    added = data.get("added_information", []) or []
    missing = data.get("missing_information", []) or []

    meaning_score = 0.0
    if meaning == "PRESERVED":
        meaning_score = 1.0
    elif meaning == "PARTIALLY_PRESERVED":
        meaning_score = 0.5
    else:
        meaning_score = 0.0

    # combine scores: meaning has highest weight
    base = 0.6 * meaning_score + 0.2 * adherence + 0.2 * fluency

    # penalize presence of added or missing information
    penalty = 0.0
    if len(added) > 0:
        penalty += min(0.3, 0.05 * len(added))
    if len(missing) > 0:
        penalty += min(0.3, 0.05 * len(missing))

    final = max(0.0, min(1.0, base - penalty))

    return RewriteJudgeResult(score=final,
                              raw=data,
                              verdict="Good" if final >= 0.7 else "Bad"
                            )



async def verify_rewrite(prompt: str, response: str, model_config: dict) -> RewriteJudgeResult:

    judge_prompt = build_rewrite_judge_prompt(
        prompt=prompt,
        rewritten=response
    )

    messages = [
        {
            "role": "user",
            "content": judge_prompt
        }
    ]

    # ask model to output structured JSON parsed into RewriteJudgeResponse
    judge_response = await chat.send_request(messages, model_config=model_config, responseFormat=RewriteJudgeResponse)

    result = get_rewrite_judge_result(judge_response.output_text)
        
    return result
    

async def handle_rewrite_verification(
    prompt,
    response,
    model_config,
    routing_context
):
    initial_res = await verify_rewrite(
        prompt=prompt,
        response=response,
        model_config=model_config
    )
    if initial_res.verdict == "Bad": # in this branch we log routing failure
        await handle_alternate_verification(
            verification_function=verify_rewrite,
            initial_response=initial_res,
            prompt=prompt,
            model_config=model_config,
            routing_context=routing_context
        )
    return initial_res


if __name__ == "__main__":
    
    prompt = "The cat sat on the mat and looked at the window. It was raining outside."
    response = "A cat rested on a rug, staring out the window at the rain."

    asyncio.run(verify_rewrite(prompt, response, {"provider": "openai", "id": "gpt-5-mini"}))
