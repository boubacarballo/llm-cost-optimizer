import json
from llm_cost_optimizer.chat import Chat    
from llm_cost_optimizer.types.response_models import BrainstormingJudgeOutput
from llm_cost_optimizer.types.types import VerificationResult
from llm_cost_optimizer.prompts import BRAINSTORMING_JUDGE_PROMPT_TEMPLATE
from llm_cost_optimizer.utils import generate_messages_template
from llm_cost_optimizer.verifiers.utils import handle_alternate_verification
import asyncio

chat = Chat()

# the brainstorming rubric is scored 1-5 per dimension; everything else in the
# library is 0-1, so normalize on the way out
RUBRIC_WEIGHTS = {
    "relevance": 0.30,
    "novelty": 0.20,
    "diversity": 0.20,
    "feasibility": 0.20,
    "framing": 0.10,
}


class BrainstormingJudgeResult(VerificationResult):
    score: float
    raw: dict


def build_brainstorming_judge_prompt(original_prompt: str, brainstorming_output: str) -> str:
    return BRAINSTORMING_JUDGE_PROMPT_TEMPLATE.format(
        original_prompt=original_prompt,
        brainstorm_output=brainstorming_output,
        constraints=""
    )


def _normalize(score) -> float:
    """1-5 rubric point -> 0.0-1.0, clamped."""
    try:
        return max(0.0, min(1.0, (float(score) - 1.0) / 4.0))
    except (TypeError, ValueError):
        return 0.0


def get_brainstorming_judge_result(judge_response: BrainstormingJudgeOutput) -> BrainstormingJudgeResult:
    data = json.loads(judge_response.model_dump_json()) if hasattr(judge_response, "model_dump_json") else judge_response.dict()

    rubric = 0.0
    for dimension, weight in RUBRIC_WEIGHTS.items():
        entry = data.get(dimension) or {}
        rubric += weight * _normalize(entry.get("score"))

    overall = _normalize(data.get("overall_score"))
    weak_ideas = data.get("weak_ideas_to_cut", []) or []

    # the rubric leads, but the judge's holistic call gets a real say
    base = 0.75 * rubric + 0.25 * overall

    penalty = min(0.2, 0.05 * len(weak_ideas)) if weak_ideas else 0.0

    final = max(0.0, min(1.0, base - penalty))

    return BrainstormingJudgeResult(
        score=final,
        raw=data,
        verdict="Good" if final >= 0.7 else "Bad"
    )


async def verify_brainstorming(prompt: str, response: str, model_config) -> BrainstormingJudgeResult:
    brainstorming_judge_prompt = build_brainstorming_judge_prompt(
        prompt,
        response,
    )
    messages = generate_messages_template(
        role="user",
        message=brainstorming_judge_prompt
    )
    judge_response = await chat.send_request(
        messages,
        model_config=model_config,
        responseFormat=BrainstormingJudgeOutput
    )
    return get_brainstorming_judge_result(judge_response.output_text)


async def handle_brainstorming_verification(prompt, response, model_config, routing_context):
    initial_res = await verify_brainstorming(
        prompt=prompt,
        response=response,
        model_config=model_config
    )
    if initial_res.verdict == "Bad":  # in this branch we log routing failure
        await handle_alternate_verification(
            verification_function=verify_brainstorming,
            initial_response=initial_res,
            prompt=prompt,
            model_config=model_config,
            routing_context=routing_context
        )
    return initial_res


async def main():
    print(await verify_brainstorming(
            prompt="give me 2 date spot ideas",
            response="1. Go to a restaurant. 2. Go to a different restaurant.",
            model_config={
                "provider": "openai",
                "id": "gpt-5-mini"
            } 
        ))

if __name__ == "__main__":
    
    asyncio.run(main())
