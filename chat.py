import json
from dotenv import load_dotenv
import os
from openai import OpenAI
from anthropic import Anthropic
load_dotenv()
import time


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
        # we check if the keys are present, we only initialize clients for the present keys
        if os.getenv("ANTHROPIC_API_KEY"):
            print("loading anthropic")
            self.anthropic_api_key = os.getenv("ANTRHOPIC_API_KEY")
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
            
        if os.getenv("OPENAI_API_KEY"):
            print("loading openai")
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            
        self.messages = [
            {
                "role": "assistant",
                "content": "You are a helpful assistant"
            }
        ]
        
        
    def send_request(self, prompt, provider, model_id):
        
        self.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        start_time = time.perf_counter()
        
        match provider:
            
            
            case "openai":
                response = self.openai_client.responses.create(
                    model=model_id,
                    input=self.messages
                    )
                latency = get_request_latency(start_time)
                self.messages.append({
                    "role": "assistant",
                    "content": response.output_text
                })

                
                return Response(
                    output_text=response.output_text,
                    input_tokens=0,
                    output_tokens=0,
                    latency=latency,
                    cost=0,
                    model_id=model_id
                )
            
            
            case "anthropic":
                
                     
                response = self.anthropic_client.messages.create(
                        model=model_id,
                        messages=self.messages,
                        max_tokens=1024
                    )
                latency = get_request_latency(start_time)
                    
                self.messages.append({
                        "role": "assistant",
                        "content": response.content[0].text
                    })
                    
                return Response(
                        output_text=response.content[0].text,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        latency=latency,
                        cost=1.0,
                        model_id=model_id
                    )

                
                
                

        
        

        