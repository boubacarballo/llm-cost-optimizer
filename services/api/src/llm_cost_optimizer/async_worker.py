import asyncio
from arq import run_worker, create_pool
from arq.connections import RedisSettings
from llm_cost_optimizer.utils import TaskType
from typing import Literal
from llm_cost_optimizer.verifiers.rewrite import handle_rewrite_verification
from llm_cost_optimizer.verifiers.extraction import handle_extraction_verification
from llm_cost_optimizer.verifiers.qa import handle_closed_qa_verification, handle_open_qa_verification
from llm_cost_optimizer.verifiers.generation import handle_generation_verification
from llm_cost_optimizer.verifiers.classification import handle_classification_verification
from llm_cost_optimizer.verifiers.code_generation import handle_code_generation_verification
from llm_cost_optimizer.verifiers.summarization import handle_summarization_verification
from llm_cost_optimizer.verifiers.brainstorming import handle_brainstorming_verification
from llm_cost_optimizer.verifiers.other import handle_other_verification
from llm_cost_optimizer.types.types import RoutingContext
from llm_cost_optimizer.utils import load_models_configs
import os
REDIS_SETTINGS = RedisSettings()
config = load_models_configs(os.getenv("MODEL_CONFIGS_PATH"))


# One handler per task type. Every handler takes
# (prompt, response, model_config, routing_context) and returns a VerificationResult.
VERIFIERS = {
    TaskType.EXTRACTION:      handle_extraction_verification,
    TaskType.REWRITE:         handle_rewrite_verification,
    TaskType.CLOSED_QA:       handle_closed_qa_verification,
    TaskType.OPEN_QA:         handle_open_qa_verification,
    TaskType.TEXT_GENERATION: handle_generation_verification,
    TaskType.CLASSIFICATION:  handle_classification_verification,
    TaskType.CODE_GENERATION: handle_code_generation_verification,
    TaskType.SUMMARIZATION:   handle_summarization_verification,
    TaskType.BRAINSTORMING:   handle_brainstorming_verification,
    # conversational and uncategorized traffic share one general-purpose judge
    TaskType.CHATBOT:         handle_other_verification,
    TaskType.OTHER:           handle_other_verification,
}



            

async def verify_response(
    task_type: TaskType,
    prompt: str,
    response: str,
    model_config: dict,
    routing_context: RoutingContext
):
    # handle verifications by task type

    handler = VERIFIERS.get(task_type)

    if handler is None:
        print(f"No verifier implemented for task type {task_type}, skipping")
        return None

    result = await handler(
        prompt=prompt,
        response=response,
        model_config=model_config,
        routing_context=routing_context
    )

    print(f"Verification logged for {task_type}: {getattr(result, 'verdict', 'unknown')}")
    return result
        
    

async def handle_verification(
    ctx, 
    task_type: TaskType,
    prompt: str,
    response: str,
    model_config: dict,
    routing_context: RoutingContext
    
):

    return await verify_response(
        task_type=task_type,
        prompt=prompt,
        response=response,
        model_config=model_config,
        routing_context=routing_context
        
    )
            

async def startup(ctx):
    pass

async def shutdown(ctx):
    pass


async def main():
    redis = await create_pool(RedisSettings())
    for prompt in ["hello how are you", "I am good and you?", "Not bad either"]:
        await redis.enqueue_job('handle_verification', prompt)
    

class WorkerSettings:
    functions = [handle_verification]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings()
    max_jobs = 20
    
    
if __name__ == "__main__":
    asyncio.run(main())
    
    