from llm_cost_optimizer.chat import Chat, Response
from llm_cost_optimizer.utils import TaskType, load_models_configs, breakdown_results, get_candidate_models, select_model_and_provider, compute_request_cost, hash_text, get_model
import sys
import requests
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Literal
import time
import uvicorn
from llm_cost_optimizer.database import SessionDep, create_db_and_tables
from llm_cost_optimizer.models import RequestEvent
from llm_cost_optimizer.types.types import RoutingContext
from llm_cost_optimizer.verification import verify
from arq.connections import RedisSettings
from arq import create_pool
import httpx
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    await create_db_and_tables()
    global redis
    global async_client
    async_client = httpx.AsyncClient()
    redis = await create_pool(RedisSettings())
    global prompt_classifier_url
    global task_classifier_url
    global model_configs_path
    task_classifier_url = os.getenv("TASK_CLASSIFIER_SERVER_URL")
    prompt_classifier_url = os.getenv("PROMPT_CLASSIFIER_SERVER_URL")
    model_configs_path = os.getenv("MODEL_CONFIGS_PATH")
    global config
    config = load_models_configs(model_configs_path)
    
    if not prompt_classifier_url:
        raise RuntimeError("Unable to get prompt classifier URL")
        
    if not model_configs_path:
        raise RuntimeError("Unable to load model configurations")
    
    global chat
    chat = Chat()
    
    yield
    
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def home():
    return {"message": "Hello world"}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
class ChatRequest(BaseModel):
    messages: list[Message]
    
@app.post("/chat")
async def chat(req: ChatRequest, session: SessionDep):
    
    prompt_text = req.messages[-1].content
    # rough char/4 estimate, reused for both the classifier call and the routing context
    token_count = int(len(prompt_text) // 4)
    context_provided = True if len(req.messages) else False

    params = {
        "prompt": prompt_text,
    }
    response = await async_client.post(
        url=task_classifier_url,
        json=params
    )
    result = response.json()
    print("Got request back from task classifier")
    task_type = TaskType(result["task_type_1"][0])
    res = await async_client.post(
        url=prompt_classifier_url,
        json={
            "prompt": str(prompt_text),
            "token_count": token_count,
            "context_window": token_count,
            "context_provided": context_provided,
            "task_type": task_type.value,
            "device": "auto"
        }
    )
    data = res.json()
    print("Got response back from prompt classifier")
    model_config = get_model(
        data=data,
        context_window=120000
    )
    model_response = await chat.send_request(
        messages=req.messages,
        model_config=model_config
    )
    
    model_response.cost = compute_request_cost(model_config, model_response.input_tokens, model_response.output_tokens)
    request_event = RequestEvent(
            prompt_hash=hash_text(prompt_text),
            model=model_config["id"],
            provider=model_config["provider"],
            cost=model_response.cost,
            latency=0.0, 
            escalated=False,
    )
    
    session.add(request_event)
    await session.commit()
    await session.refresh(request_event)

    # everything the verification worker needs to make a routing failure interpretable
    routing_context = RoutingContext(
        prompt=prompt_text,
        token_count=token_count,
        context_provided=context_provided,
        context_window=token_count,
        task_type=task_type,
        task_tier=int(data["results"]["tier"]),
    )
    await redis.enqueue_job('handle_verification', task_type, prompt_text, model_response.output_text, model_config, routing_context)

    return {
        "output_text": model_response.output_text,
        "input_tokens": model_response.input_tokens,
        "output_tokens": model_response.output_tokens,
        "latency": model_response.latency,
        "cost": model_response.cost,
        "model_id": model_response.model_id,
        "prompt_classification": data,
    }
    
def start():
    uvicorn.run("llm_cost_optimizer.main:app", host="127.0.0.1", port=8080, reload=True)
    
    
    
