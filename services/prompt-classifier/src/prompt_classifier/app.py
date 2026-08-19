import asyncio
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
from typing import Literal, Optional
from prompt_classifier.infer import infer_prompt_tier

@asynccontextmanager
async def lifespan(app: FastAPI):
    # perform any startup initialization here (if needed)
    yield
    # perform any shutdown/cleanup here (if needed)

app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str
    token_count: int
    context_provided: bool
    context_window: int
    task_type: str
    device: Literal["auto", "cpu", "cuda", "mps"]
    

@app.get("/health")
async def health():
    return {"message": "The server is running"}

@app.get("/")
async def home():
    return {"message", "prompt-classifier server"}

@app.post("/classify")
async def classify_prompt(req: PromptRequest):

    response = await asyncio.to_thread(
        infer_prompt_tier,
        prompt=req.prompt,
        token_count=req.token_count,
        context_provided=req.context_provided,
        context_window=req.context_window,
        task_type=req.task_type,
        device=req.device
    )

    return {"results": response}


def main() -> None:
    uvicorn.run("prompt_classifier.app:app", host="0.0.0.0", port=3000, reload=False)