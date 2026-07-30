import asyncio
from chat import Response
from utils import TaskType


async def verify_extraction(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_closed_qa(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_text_generation(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_open_qa(task_type: TaskType, prompt: str, response: Response, model_config):
    pass


async def verify_summarization(task_type: TaskType, prompt: str, response: Response, model_config):
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

