import json
from pydantic import BaseModel, Field
from typing import Literal, List
from llm_cost_optimizer.chat import Chat
from llm_cost_optimizer.types.types import VerificationResult
from llm_cost_optimizer.prompts import QA_JUDGE_PROMPT_TEMPLATE
from llm_cost_optimizer.verifiers.utils import handle_alternate_verification
import asyncio

chat = Chat()

# Closed QA must stay inside the context the asker supplied; open QA may draw on
# world knowledge. Same judge, same scoring -- only what counts as "supported" moves.
CLOSED_QA_GROUNDING_RULE = (
    "This is CLOSED-BOOK question answering. The ANSWER must be supported entirely by the "
    "context supplied inside the PROMPT. Any claim that is true in the world but absent from "
    "that context is an unsupported claim, and an answer built on such claims is not CORRECT."
)

OPEN_QA_GROUNDING_RULE = (
    "This is OPEN-BOOK question answering. The ANSWER may draw on general world knowledge. "
    "A claim counts as unsupported only if it is dubious, unverifiable, or contradicted by "
    "well-established fact -- not merely because the PROMPT did not state it."
)


class QAJudgeResponse(BaseModel):
    correctness: Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]
    unsupported_claims: List[str] = []
    missing_aspects: List[str] = []
    directness: float = Field(..., ge=0.0, le=1.0)


class QAJudgeResult(VerificationResult):
    score: float
    raw: dict


def build_qa_judge_prompt(prompt: str, answer: str, grounding_rule: str) -> str:
    return QA_JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        answer=answer,
        grounding_rule=grounding_rule,
    )


def get_qa_judge_result(judge_response: QAJudgeResponse) -> QAJudgeResult:
    data = json.loads(judge_response.model_dump_json()) if hasattr(judge_response, "model_dump_json") else judge_response.dict()

    correctness = data.get("correctness")
    directness = float(data.get("directness", 0.0))
    unsupported = data.get("unsupported_claims", []) or []
    missing = data.get("missing_aspects", []) or []

    if correctness == "CORRECT":
        correctness_score = 1.0
    elif correctness == "PARTIALLY_CORRECT":
        correctness_score = 0.5
    else:
        correctness_score = 0.0

    # a wrong answer is a wrong answer, however well phrased -- correctness dominates
    base = 0.7 * correctness_score + 0.3 * directness

    penalty = 0.0
    if unsupported:
        penalty += min(0.3, 0.1 * len(unsupported))
    if missing:
        penalty += min(0.2, 0.05 * len(missing))

    final = max(0.0, min(1.0, base - penalty))

    return QAJudgeResult(
        score=final,
        raw=data,
        verdict="Good" if final >= 0.7 else "Bad"
    )


async def _verify_qa(prompt: str, response: str, model_config: dict, grounding_rule: str) -> QAJudgeResult:
    judge_prompt = build_qa_judge_prompt(
        prompt=prompt,
        answer=response,
        grounding_rule=grounding_rule
    )

    messages = [
        {
            "role": "user",
            "content": judge_prompt
        }
    ]

    judge_response = await chat.send_request(messages, model_config=model_config, responseFormat=QAJudgeResponse)

    return get_qa_judge_result(judge_response.output_text)


async def verify_closed_qa(prompt: str, response: str, model_config: dict) -> QAJudgeResult:
    return await _verify_qa(prompt, response, model_config, CLOSED_QA_GROUNDING_RULE)


async def verify_open_qa(prompt: str, response: str, model_config: dict) -> QAJudgeResult:
    return await _verify_qa(prompt, response, model_config, OPEN_QA_GROUNDING_RULE)


async def handle_closed_qa_verification(prompt, response, model_config, routing_context):
    initial_res = await verify_closed_qa(
        prompt=prompt,
        response=response,
        model_config=model_config
    )
    if initial_res.verdict == "Bad":  # in this branch we log routing failure
        await handle_alternate_verification(
            verification_function=verify_closed_qa,
            initial_response=initial_res,
            prompt=prompt,
            model_config=model_config,
            routing_context=routing_context
        )
    return initial_res


async def handle_open_qa_verification(prompt, response, model_config, routing_context):
    initial_res = await verify_open_qa(
        prompt=prompt,
        response=response,
        model_config=model_config
    )
    if initial_res.verdict == "Bad":  # in this branch we log routing failure
        await handle_alternate_verification(
            verification_function=verify_open_qa,
            initial_response=initial_res,
            prompt=prompt,
            model_config=model_config,
            routing_context=routing_context
        )
    return initial_res


if __name__ == "__main__":

    context_prompt = (
        "Using only the passage below, answer: how long did the Aswan High Dam take to build?\n\n"
        "PASSAGE: Construction of the Aswan High Dam began in 1960 and finished in 1970. "
        "The reservoir it created was named Lake Nasser."
    )
    # correct span, but pads with a fact the passage never states
    grounded_answer = "It took ten years, from 1960 to 1970, and cost roughly one billion dollars."

    print(asyncio.run(verify_closed_qa(context_prompt, grounded_answer, {"provider": "openai", "id": "gpt-5-mini"})))
