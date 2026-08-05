import asyncio
from llm_cost_optimizer.chat import Response, Chat
from llm_cost_optimizer.utils import TaskType, ModelConfig, SummaryJudgeResponse, SummaryJudgeResult, SummaryJudgeModelConfig, Verification
import json
from llm_cost_optimizer.prompts import SUMMARIZATION_JUDGE_PROMPT_TEMPLATE
from llm_cost_optimizer.verifiers.summarization import verify_summarization

async def verify_extraction(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_closed_qa(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_text_generation(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_open_qa(task_type: TaskType, prompt: str, response: Response, model_config):
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


async def verify(task_type: TaskType, prompt: str, response: str, model_config):
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
            summarization_result = await verify_summarization(prompt, response, model_config)
            score = 0.5 * summarization_result.precision + 0.5 * summarization_result.recall
            return Verification(
                task_type=task_type.value,
                score=score
            )
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

    return Verification(task_type=task_type.value if isinstance(task_type, TaskType) else str(task_type), score=0.0)

