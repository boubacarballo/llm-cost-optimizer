import asyncio
from arq import run_worker, create_pool
from arq.connections import RedisSettings
from llm_cost_optimizer.utils import TaskType
from llm_cost_optimizer.verifiers.summarization import verify_summarization
from llm_cost_optimizer.verifiers.rewrite import verify_rewrite
REDIS_SETTINGS = RedisSettings()

async def verify_response(
    task_type: TaskType,
    prompt: str,
    response: str,
    model_config: dict
):
    
    response = None
    if task_type.value == "Rewrite":
            
        response = await verify_rewrite(
            prompt=prompt,
            response=response,
            judge_model_config=model_config
        )
        
        
    
    return response
        
    

async def handle_verification(
    ctx, 
    task_type: TaskType,
    prompt: str,
    response: str,
    model_config: dict
    
):

    response = await verify_response(
        task_type=task_type,
        prompt=prompt,
        response=response,
        model_config=model_config
        
    )
    
    if not response:
        raise Exception
    
    if response.verdict == "Bad":
        pass
        alternate_models = get_alternate_models(model_config)
        
        for cfg in alternate_models:
            #async run for each
        #if its bad, then its a ROUTING FAILURE , we take a model that's 1 tier higher (if 1 then 2, if 2 then 3, if 3 then we take 3 but another provider's highest tier model)
            # we also take another provider of the same tier
            
            #run the same prompt with that "higher-tier" model (this we will make async so its fast)
            
            #run the prompt-response pair through the verifier again
            
            # take the 3-4 prompt-response pairs and make then go through judge and pick the best and cheapest
            #LOG the routing failure so we can retrain the model down the line
        
    else: #when the response is good
        
        pass
        
            


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
    
    