import json

class ModelConfig:
    
    def __init__(self):
        # there should be providers, and we should have quick maps to the available models
        
        
        self.providers = {
            "openai": {
                "gpt-4o-mini": 0.6,
                "gpt-5": 0.5,
                "gpt-5.6-terra": 0.2,
                "gpt": 0.001,
            },
            "anthropic": {
                "opus-4.6": 1,
                "sonnet-4.6": 0.5,
                "fable-5": 3
            }
        }
        
        
class Response:
    
    def __init__(self, output_text: str, tokens_used: int, latency: float, cost: float, model_id: str):
        self.output_text = output_text
        self.tokens_used = tokens_used
        self.latency = latency
        self.cost = cost
        self.model_id = model_id
        
        
    def to_json(self):
        return json.dumps(self.__dict__, indent=4)
        
        
        