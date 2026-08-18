import asyncio
from arq import run_worker, create_pool
from arq.connections import RedisSettings
from llm_cost_optimizer.utils import TaskType
from typing import Literal
from llm_cost_optimizer.verifiers.summarization import verify_summarization
from llm_cost_optimizer.verifiers.rewrite import handle_rewrite_verification
from llm_cost_optimizer.utils import load_models_configs
import os
REDIS_SETTINGS = RedisSettings()
config = load_models_configs(os.getenv("MODEL_CONFIGS_PATH"))



            

async def verify_response(
    task_type: TaskType,
    prompt: str,
    response: str,
    model_config: dict
):
    # handle verifications by task type
    
    if task_type.value == "Rewrite":
            
        await handle_rewrite_verification(
            prompt=prompt,
            response=response,
            judge_model_config=model_config
        )
        
    

async def handle_verification(
    ctx, 
    task_type: TaskType,
    prompt: str,
    response: str,
    model_config: dict
    
):

    await verify_response(
        task_type=task_type,
        prompt=prompt,
        response=response,
        model_config=model_config
        
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
    
    