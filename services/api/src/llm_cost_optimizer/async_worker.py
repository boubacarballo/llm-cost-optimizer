import asyncio
from arq import run_worker
from arq.connections import RedisSettings



async def handle_verification(ctx, prompt):
    print(f"Current handling verification for response to the following prompt: {prompt}")
    print("Treatment...")
    await asyncio.speep(2)
    print("Verification for prompt done!")


async def startup(ctx):
    pass

async def shutdown(ctx):
    pass


async def main():
    
    pass

class WorkerSettings:
    functions = [handle_verification]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings()
    max_jobs = 20
    
    
    