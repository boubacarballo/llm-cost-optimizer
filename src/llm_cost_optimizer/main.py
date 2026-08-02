from llm_cost_optimizer.chat import Chat, Response
from llm_cost_optimizer.utils import load_models_configs, breakdown_results, get_candidate_models, select_model_and_provider, compute_request_cost
import sys
import requests
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Literal
import time
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global prompt_classifier_url
    global model_configs_path
    prompt_classifier_url = os.getenv("PROMPT_CLASSIFIER_SERVER_URL")
    model_configs_path = os.getenv("MODEL_CONFIGS_PATH")
    global config
    config = load_models_configs(model_configs_path)
    
    if not prompt_classifier_url:
        raise "Unable to ge prompt classifier URL"
        
    if not model_configs_path:
        raise "Unable to load model configurations"
    
    global chat
    chat = Chat()
    
    yield
    
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def home():
    return {"message": "Hello world"}


class Message(BaseModel):
    role: str = Literal["user", "assistant"]
    content: str
class ChatRequest(BaseModel):
    messages: list[Message]
    
@app.post("/chat")
async def chat(req: ChatRequest):
    start = time.perf_counter()
    
    params = {
        "prompt": req.messages[-1].content,
    }
    response = requests.post(
        url=prompt_classifier_url,
        json=params
    )
    result = response.json()
        
    tier_flags, contextual_knowledge, complexity = breakdown_results(result)
    print(f"tier flags {tier_flags}")
        
    candidate_models = get_candidate_models(tier_flags, config)
    print(f"candidates: {candidate_models}")
        
    model_config = select_model_and_provider(candidate_models, context_window=12700)
   
    print(model_config)
    response = chat.send_request(
        messages=req.messages,
        provider=model_config["provider"],
        model_id=model_config["id"],
    )
    print(time.perf_counter() - start)
    response.cost = compute_request_cost(model_config, response.input_tokens, response.output_tokens)
    return response
    
    
    
    
    
    
    
