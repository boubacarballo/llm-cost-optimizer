import asyncio
from chat import Response
from utils import TaskType, ModelConfig, SummaryJudgeResponse, SummaryJudgeResult
from chat import Chat
from prompts import SUMMARIZATION_JUDGE_PROMPT_TEMPLATE

chat = Chat()



def build_summarization_judge_prompt(source_document: str, summary: str):
    return SUMMARIZATION_JUDGE_PROMPT_TEMPLATE.format(
        source_document=source_document,
        summary=summary
    )
    
def get_summary_judge_result(summary_response: SummaryJudgeResponse):
    data = summary_response.model_dump_json()
    
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
async def verify_extraction(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_closed_qa(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_text_generation(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_open_qa(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_summarization(prompt: str, response: Response, judge_model_config: ModelConfig):
    
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
        "openai",
        "gpt-5.4",
        SummaryJudgeResponse
    )
    
    
    
    pass

async def verify_classification(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_code_generation(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_chatbot(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_rewrite(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_brainstorming(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_other(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify(task_type: TaskType, prompt: str, response: Response, model_config):
    # this is essentially where your evals live depending on the task type

    match task_type:
        case TaskType.EXTRACTION:
            await verify_extraction(task_type, prompt, response, model_config)
        case TaskType.CLOSED_QA:
            await verify_closed_qa(task_type, prompt, response, model_config)
        case TaskType.TEXT_GENERATION:
            await verify_text_generation(task_type, prompt, response, model_config)
        case TaskType.OPEN_QA:
            await verify_open_qa(task_type, prompt, response, model_config)
        case TaskType.SUMMARIZATION:
            await verify_summarization(task_type, prompt, response, model_config)
        case TaskType.CLASSIFICATION:
            await verify_classification(task_type, prompt, response, model_config)
        case TaskType.CODE_GENERATION:
            await verify_code_generation(task_type, prompt, response, model_config)
        case TaskType.CHATBOT:
            await verify_chatbot(task_type, prompt, response, model_config)
        case TaskType.REWRITE:
            await verify_rewrite(task_type, prompt, response, model_config)
        case TaskType.BRAINSTORMING:
            await verify_brainstorming(task_type, prompt, response, model_config)
        case TaskType.OTHER:
            await verify_other(task_type, prompt, response, model_config)
        case _:
            await verify_other(task_type, prompt, response, model_config)

