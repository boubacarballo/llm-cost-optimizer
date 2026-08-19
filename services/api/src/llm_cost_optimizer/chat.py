import json
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from pydantic import BaseModel
load_dotenv()
import time
import asyncio


def get_request_latency(start_time):
    return time.perf_counter() - start_time

class Response:
    
    def __init__(self, output_text: str, input_tokens: int, output_tokens: int, latency: float, cost: float, model_id: str):
        self.output_text = output_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency = latency
        self.cost = cost
        self.model_id = model_id
        
        
    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

class Chat:
    
    def __init__(self):
        print("Creating a chat instance")
        # we check if the keys are present, we only initialize clients for the present keys
        if os.getenv("ANTHROPIC_API_KEY"):
            print("loading anthropic")
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            self.anthropic_client = AsyncAnthropic(api_key=self.anthropic_api_key)
            
        if os.getenv("OPENAI_API_KEY"):
            print("loading openai")
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            
        
        
        
    async def send_request(self, messages, model_config: dict, responseFormat: BaseModel=None):
        start_time = time.perf_counter()
        provider = model_config.get("provider") if model_config else None
        model_id = model_config.get("id") if model_config else None
        
        match provider:
            
            
            case "openai":
                response = None
                if responseFormat:
                    response = await self.openai_client.responses.parse(
                            model=model_id,
                            input=messages,
                            text_format=responseFormat                
                    )
                else:
                    response = await self.openai_client.responses.create(
                        model=model_id,
                        input=messages,
                    )

                latency = get_request_latency(start_time)
                data = json.loads(response.model_dump_json())

                return Response(
                    output_text=response.output_parsed if responseFormat else response.output_text,
                    input_tokens=data["usage"]["input_tokens"],
                    output_tokens=data["usage"]["output_tokens"],
                    latency=latency,
                    cost=0,
                    model_id=model_id
                )
            
            
            case "anthropic":
                
                response = None
                
                if responseFormat:
                    response = await self.anthropic_client.messages.parse(
                        model=model_id,
                        messages=messages,
                        max_tokens=1024
                    )
                else:
                    response = await self.anthropic_client.messages.create(
                            model=model_id,
                            messages=messages,
                            max_tokens=1024
                        )
                latency = get_request_latency(start_time)
                    
                    
                return Response(
                        output_text=response.parsed_output if responseFormat else response.content[0].text,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        latency=latency,
                        cost=1.0,
                        model_id=model_id
                    )

                
                
                
                
if __name__ == "__main__":
    chat = Chat()
    class Temperature(BaseModel):
        city: str
        temperature: float

    response = asyncio.run(chat.send_request(
        messages=[
            {
                "role": "user",
                "content": "What's the average temperature in New York typically"
            }
        ],
        model_config={"provider": "openai", "id": "gpt-5.4"},
        responseFormat=Temperature
    ))

    print(response.output_text)

        
        

        