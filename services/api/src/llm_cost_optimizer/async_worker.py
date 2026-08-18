import asyncio
from arq import run_worker, create_pool
from arq.connections import RedisSettings
from llm_cost_optimizer.utils import TaskType
from typing import Literal
from llm_cost_optimizer.verifiers.summarization import verify_summarization
from llm_cost_optimizer.verifiers.rewrite import verify_rewrite
from llm_cost_optimizer.utils import load_models_configs
import os
REDIS_SETTINGS = RedisSettings()
config = load_models_configs(os.getenv("MODEL_CONFIGS_PATH"))


async def select_alternate_model(model_config: dict, tier: Literal["similar", "higher"]):
    if tier == "similar" or model_config["tier"] == "tier_3":
        alternate_models = [
                        model
                        for model in config["tiers"][model_config["tier"]]
                        if model["id"] != model_config["id"]
                        or model["provider"] != model_config["provider"]
                    ]
        new_model_config = alternate_models[0] if alternate_models else None
        return new_model_config
        
    elif tier == "higher": 
        
        current_tier = int(model_config["tier"][-1])
        new_tier = "tier_" + str(current_tier + 1)
        higher_tier_models = config["tiers"][new_tier]
        return higher_tier_models[0] if higher_tier_models else None
            

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

    response_result = await verify_response(
        task_type=task_type,
        prompt=prompt,
        response=response,
        model_config=model_config
        
    )
    
    if not response_result:
        raise Exception
    
    if response_result.verdict == "Bad":
        pass
        alternate_similar_tier_model = await select_alternate_model(
            model_config=model_config,
            tier="similar"
        )
        alternate_higher_tier_model = await select_alternate_model(
            model_config=model_config,
            tier="higher"
        )
                                                                 
        
        
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
    
    