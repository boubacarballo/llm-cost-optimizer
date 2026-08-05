import asyncio
from arq import run_worker, create_pool
from arq.connections import RedisSettings


REDIS_SETTINGS = RedisSettings()
async def handle_verification(ctx, prompt):
    print(f"Current handling verification for response to the following prompt: {prompt}")
    print("Treatment...")
    await asyncio.sleep(2)
    print("Verification for prompt done!")


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
    
    