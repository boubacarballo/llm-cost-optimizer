import asyncio
from llm_cost_optimizer.chat import Response, Chat
from llm_cost_optimizer.utils import TaskType, ModelConfig, SummaryJudgeResponse, SummaryJudgeResult, SummaryJudgeModelConfig
import json
from llm_cost_optimizer.prompts import SUMMARIZATION_JUDGE_PROMPT_TEMPLATE

chat = Chat()
def build_summarization_judge_prompt(source_document: str, summary: str):
    return SUMMARIZATION_JUDGE_PROMPT_TEMPLATE.format(
        source_document=source_document,
        summary=summary
    )
    
def get_summary_judge_result(summary_response: SummaryJudgeResponse):
    data = json.loads(summary_response.model_dump_json())
    
    
    claims = data.get("claims")
    points = data.get("points")
    if claims:
        supported = sum(1 for c in claims if c.get("status") == "SUPPORTED")
        faithfulness = supported / len(claims) if claims else 1.0
        
    if points:
        present = sum(1 for p in points if p.get("status") == "PRESENT")
        coverage = present / len(points) if points else 1.0
        
    return SummaryJudgeResult(
        precision=faithfulness,
        recall=coverage,
        raw=data
    )
    
async def verify_summarization(prompt: str, response: Response, judge_model_config: SummaryJudgeModelConfig):
    
    # we need to define the judge here
    
    summary_judge_sys_prompt = build_summarization_judge_prompt(
        prompt, 
        response.output_text
    )
    messages = [
        {
            "role": "user",
            "content": summary_judge_sys_prompt
        },
    ]
    
    response = chat.send_request(
        messages, 
        judge_model_config.provider,
        judge_model_config.model_id,
        SummaryJudgeResponse
    )
    
    
    results = get_summary_judge_result(response.output_text)
    
    print(results)
    
if __name__ == "__main__":
    prompt = "The old lighthouse keeper had watched a thousand storms roll in from the same window, but tonight felt different. The wind carried a strange stillness beneath its howling, as if the sea itself was holding its breath. He climbed the spiral stairs one last time, counting each step out of habit rather than need, his hand tracing the worn groove in the railing left by decades of the same gesture. At the top, the great lamp turned steadily, throwing its beam across water that had swallowed ships and returned nothing but driftwood and silence. He thought of his daughter, who had begged him to retire years ago, and wondered if tonight would finally be the night he listened. But the light needed tending, and some habits outlive their reasons. He settled into his chair, wrapped his coat tighter, and waited for the storm to decide what it wanted from him."
    response = Response(
        output_text="An old lighthouse keeper notices something odd about a storm coming in one night. He walks up to the lighthouse and turns on the light, thinking about how his daughter wants him to quit his job. He decides to stay and face the storm because he feels like he has to.",
        input_tokens=0,
        output_tokens=0,
        latency=0,
        cost=1.0,
        model_id="gpt-5.4"
        
    )
    asyncio.run(verify_summarization(
        prompt,
        response,
        SummaryJudgeModelConfig(
            provider="openai",
            model_id="gpt-5.6-luna",
        )
    ))