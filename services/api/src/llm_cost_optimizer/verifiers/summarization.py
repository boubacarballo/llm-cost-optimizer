import asyncio
from llm_cost_optimizer.chat import Response, Chat
from llm_cost_optimizer.utils import TaskType, ModelConfig, SummaryJudgeResponse, SummaryJudgeResult, SummaryJudgeModelConfig
from llm_cost_optimizer.types.types import VerificationResult
import json
from llm_cost_optimizer.prompts import SUMMARIZATION_JUDGE_PROMPT_TEMPLATE
from llm_cost_optimizer.verifiers.utils import handle_alternate_verification

chat = Chat()


class SummarizationJudgeResult(VerificationResult):
    # precision/recall are kept alongside score so the older callers that read
    # them off this result (verification.py) keep working unchanged
    precision: float
    recall: float
    score: float
    raw: dict


def build_summarization_judge_prompt(source_document: str, summary: str):
    return SUMMARIZATION_JUDGE_PROMPT_TEMPLATE.format(
        source_document=source_document,
        summary=summary
    )
    
def get_summary_judge_result(summary_response: SummaryJudgeResponse) -> SummarizationJudgeResult:
    data = json.loads(summary_response.model_dump_json()) if hasattr(summary_response, "model_dump_json") else summary_response.dict()

    claims = data.get("claims", []) or []
    points = data.get("points", []) or []

    # a summary that asserts nothing cannot be unfaithful, and one judged against
    # no key points cannot have missed any -- both are 1.0, never None
    supported = sum(1 for c in claims if c.get("status") == "SUPPORTED")
    faithfulness = supported / len(claims) if claims else 1.0

    present = sum(1 for p in points if p.get("status") == "PRESENT")
    coverage = present / len(points) if points else 1.0

    # inventing facts is worse than leaving one out, so faithfulness carries more
    base = 0.6 * faithfulness + 0.4 * coverage

    # a contradicted claim is a harder error than one merely unsupported
    contradicted = sum(1 for c in claims if c.get("status") == "CONTRADICTED")
    penalty = min(0.3, 0.15 * contradicted) if contradicted else 0.0

    final = max(0.0, min(1.0, base - penalty))

    return SummarizationJudgeResult(
        precision=faithfulness,
        recall=coverage,
        score=final,
        raw=data,
        verdict="Good" if final >= 0.7 else "Bad"
    )
    
async def verify_summarization(prompt: str, response: str, model_config) -> SummarizationJudgeResult:
    
    summary_judge_sys_prompt = build_summarization_judge_prompt(
        prompt, 
        response
    )
    messages = [
        {
            "role": "user",
            "content": summary_judge_sys_prompt
        },
    ]

    judge_response = await chat.send_request(
        messages, 
        model_config=model_config,
        responseFormat=SummaryJudgeResponse
    )

    return get_summary_judge_result(judge_response.output_text)


async def handle_summarization_verification(prompt, response, model_config, routing_context):
    initial_res = await verify_summarization(
        prompt=prompt,
        response=response,
        model_config=model_config
    )
    if initial_res.verdict == "Bad":  # in this branch we log routing failure
        await handle_alternate_verification(
            verification_function=verify_summarization,
            initial_response=initial_res,
            prompt=prompt,
            model_config=model_config,
            routing_context=routing_context
        )
    return initial_res


if __name__ == "__main__":
    prompt = "The old lighthouse keeper had watched a thousand storms roll in from the same window, but tonight felt different. The wind carried a strange stillness beneath its howling, as if the sea itself was holding its breath. He climbed the spiral stairs one last time, counting each step out of habit rather than need, his hand tracing the worn groove in the railing left by decades of the same gesture. At the top, the great lamp turned steadily, throwing its beam across water that had swallowed ships and returned nothing but driftwood and silence. He thought of his daughter, who had begged him to retire years ago, and wondered if tonight would finally be the night he listened. But the light needed tending, and some habits outlive their reasons. He settled into his chair, wrapped his coat tighter, and waited for the storm to decide what it wanted from him."
    response = "An old lighthouse keeper notices something odd about a storm coming in one night. He walks up to the lighthouse and turns on the light, thinking about how his daughter wants him to quit his job. He decides to stay and face the storm because he feels like he has to."
    print(asyncio.run(verify_summarization(
        prompt,
        response,
        {"provider": "openai", "id": "gpt-5-mini"}
    )))
