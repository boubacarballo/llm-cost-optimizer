import asyncio
from arq import run_worker, create_pool
from arq.connections import RedisSettings
from llm_cost_optimizer.utils import TaskType
from llm_cost_optimizer.verifiers.summarization import verify_summarization
REDIS_SETTINGS = RedisSettings()
async def handle_verification(
    ctx, 
    task_type: TaskType,
    prompt: str,
    response: str,
    model_config: dict
    
):
    match task_type:
        case TaskType.SUMMARIZATION:
            summarization_result = await verify_summarization(prompt, response, model_config)
            score = 0.5 * summarization_result.precision + 0.5 * summarization_result.recall
            


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
    
    