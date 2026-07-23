import json
from dotenv import load_dotenv
import os
from openai import OpenAI
from anthropic import Anthropic
load_dotenv()

class Response:
    
    def __init__(self, output_text: str, input_tokens: int, output_tokens: int, latency: float, cost: float, model_id: str):
        self.output_text = output_text
        self.tokens_used = tokens_used
        self.latency = latency
        self.cost = cost
        self.model_id = model_id
        
        
    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

class Chat:
    
    def __init__(self):
        # we check if the keys are present, we only initialize clients for the present keys
        if os.getenv("ANTRHOPIC_API_KEY"):
            self.anthropic_api_key = os.getenv("ANTRHOPIC_API_KEY")
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
            
        if os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            
        self.messages = [
            {
                "role": "assistent",
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
        
        match self.provider:
            
            case "openai":
                response = self.openai_client.responses.create(
                    model=model_id,
                    messages=self.messages
                    )
                
                self.messages.append(response)
                
                return Response(
                    
                )
            
            
            case "anthropic":
                response = self.anthropic_client.messages.create(
                    model=model_id,
                    messages=self.messages
                )
                
                self.messages.append({
                    "role": "assistant",
                    "content": response.content[0].text
                })
                
                return Response(
                    output_text=response.content[0].text,
                    intput_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency=0.0,
                    cost=1.0,
                    model_id=model_id
                )
                
                
                

        
        

        