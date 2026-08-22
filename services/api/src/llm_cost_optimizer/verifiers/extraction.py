import json
from pydantic import BaseModel, Field
from typing import Literal, List
from llm_cost_optimizer.chat import Chat
from llm_cost_optimizer.types.types import VerificationResult
from llm_cost_optimizer.prompts import EXTRACTION_JUDGE_PROMPT_TEMPLATE
from llm_cost_optimizer.verifiers.utils import handle_alternate_verification
import asyncio

chat = Chat()


class ExtractionJudgeItem(BaseModel):
    field: str
    value: str
    status: Literal["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]


class ExtractionJudgeResponse(BaseModel):
    extracted_items: List[ExtractionJudgeItem] = []
    missing_items: List[str] = []
    format_adherence: float = Field(..., ge=0.0, le=1.0)


class ExtractionJudgeResult(VerificationResult):
    score: float
    raw: dict


def build_extraction_judge_prompt(prompt: str, extracted: str) -> str:
    return EXTRACTION_JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        extracted=extracted,
    )


def get_extraction_judge_result(judge_response: ExtractionJudgeResponse) -> ExtractionJudgeResult:
    data = json.loads(judge_response.model_dump_json()) if hasattr(judge_response, "model_dump_json") else judge_response.dict()

    items = data.get("extracted_items", []) or []
    missing = data.get("missing_items", []) or []
    format_adherence = float(data.get("format_adherence", 0.0))

    supported = sum(1 for i in items if i.get("status") == "SUPPORTED")
    contradicted = sum(1 for i in items if i.get("status") == "CONTRADICTED")

    # precision: of what we pulled out, how much is actually grounded in the source.
    # nothing extracted and nothing missing is a vacuous success, not a divide by zero.
    precision = supported / len(items) if items else 1.0

    # recall: of what was there to find, how much did we get.
    found = len(items)
    recall = found / (found + len(missing)) if (found + len(missing)) > 0 else 1.0

    # faithfulness carries the most weight: a hallucinated field value is the
    # failure mode that actually matters for extraction.
    base = 0.45 * precision + 0.35 * recall + 0.20 * format_adherence

    # a contradiction is a harder error than an omission, so it is penalized separately
    penalty = min(0.3, 0.15 * contradicted) if contradicted else 0.0

    final = max(0.0, min(1.0, base - penalty))

    return ExtractionJudgeResult(
        score=final,
        raw=data,
        verdict="Good" if final >= 0.7 else "Bad"
    )


async def verify_extraction(prompt: str, response: str, model_config: dict) -> ExtractionJudgeResult:

    judge_prompt = build_extraction_judge_prompt(
        prompt=prompt,
        extracted=response
    )

    messages = [
        {
            "role": "user",
            "content": judge_prompt
        }
    ]

    judge_response = await chat.send_request(messages, model_config=model_config, responseFormat=ExtractionJudgeResponse)

    result = get_extraction_judge_result(judge_response.output_text)

    return result


async def handle_extraction_verification(
    prompt,
    response,
    model_config,
    routing_context
):
    initial_res = await verify_extraction(
        prompt=prompt,
        response=response,
        model_config=model_config
    )
    if initial_res.verdict == "Bad":  # in this branch we log routing failure
        await handle_alternate_verification(
            verification_function=verify_extraction,
            initial_response=initial_res,
            prompt=prompt,
            model_config=model_config,
            routing_context=routing_context
        )
    return initial_res


if __name__ == "__main__":

    prompt = (
        "Extract the customer name, order date, and total amount from this receipt. "
        "Return JSON with keys: customer_name, order_date, total.\n\n"
        "RECEIPT\nBrightleaf Coffee Roasters\nSold to: Amara Okonkwo\n"
        "Date: 2026-03-14\nItems: 2x Ethiopia Guji 12oz, 1x ceramic dripper\n"
        "Subtotal: $58.00\nTax: $4.64\nTotal: $62.64\n"
    )
    # deliberately wrong: total is invented, order_date is dropped entirely
    response = '{"customer_name": "Amara Okonkwo", "total": "$58.00"}'

    print(asyncio.run(verify_extraction(prompt, response, {"provider": "openai", "id": "gpt-5-mini"})))
