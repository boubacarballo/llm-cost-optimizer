import json
from pydantic import BaseModel, Field
from typing import Literal
from llm_cost_optimizer.chat import Chat
from llm_cost_optimizer.types.types import VerificationResult
from llm_cost_optimizer.prompts import CLASSIFICATION_JUDGE_PROMPT_TEMPLATE
from llm_cost_optimizer.verifiers.utils import handle_alternate_verification
import asyncio

chat = Chat()


class ClassificationJudgeResponse(BaseModel):
    label_valid: bool
    correctness: Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]
    format_adherence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""


class ClassificationJudgeResult(VerificationResult):
    score: float
    raw: dict


def build_classification_judge_prompt(prompt: str, label: str) -> str:
    return CLASSIFICATION_JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        label=label,
    )


def get_classification_judge_result(judge_response: ClassificationJudgeResponse) -> ClassificationJudgeResult:
    data = json.loads(judge_response.model_dump_json()) if hasattr(judge_response, "model_dump_json") else judge_response.dict()

    label_valid = bool(data.get("label_valid", False))
    correctness = data.get("correctness")
    format_adherence = float(data.get("format_adherence", 0.0))

    if correctness == "CORRECT":
        correctness_score = 1.0
    elif correctness == "AMBIGUOUS":
        correctness_score = 0.6
    else:
        correctness_score = 0.0

    base = 0.75 * correctness_score + 0.25 * format_adherence

    # a label outside the allowed set is unusable downstream no matter how well
    # reasoned it is -- that is a routing failure regardless of the other scores
    final = 0.0 if not label_valid else max(0.0, min(1.0, base))

    return ClassificationJudgeResult(
        score=final,
        raw=data,
        verdict="Good" if final >= 0.7 else "Bad"
    )


async def verify_classification(prompt: str, response: str, model_config: dict) -> ClassificationJudgeResult:

    judge_prompt = build_classification_judge_prompt(
        prompt=prompt,
        label=response
    )

    messages = [
        {
            "role": "user",
            "content": judge_prompt
        }
    ]

    judge_response = await chat.send_request(messages, model_config=model_config, responseFormat=ClassificationJudgeResponse)

    return get_classification_judge_result(judge_response.output_text)


async def handle_classification_verification(prompt, response, model_config, routing_context):
    initial_res = await verify_classification(
        prompt=prompt,
        response=response,
        model_config=model_config
    )
    if initial_res.verdict == "Bad":  # in this branch we log routing failure
        await handle_alternate_verification(
            verification_function=verify_classification,
            initial_response=initial_res,
            prompt=prompt,
            model_config=model_config,
            routing_context=routing_context
        )
    return initial_res


if __name__ == "__main__":

    prompt = (
        "Classify the sentiment of this review as exactly one of: positive, negative, neutral. "
        "Reply with the label only.\n\n"
        "REVIEW: The battery died after two days and support never replied."
    )
    # invents a label outside the allowed set, and pads it with commentary
    response = "mostly_negative (the reviewer seems frustrated)"

    print(asyncio.run(verify_classification(prompt, response, {"provider": "openai", "id": "gpt-5-mini"})))
