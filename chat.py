import json

class Response:
    
    def __init__(self, output_text: str, tokens_used: int, latency: float, cost: float, model_id: str):
        self.output_text = output_text
        self.tokens_used = tokens_used
        self.latency = latency
        self.cost = cost
        self.model_id = model_id
        
        
    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

class Chat:
    
    def __init__(self):
        pass
        
        
    def send_request(self, prompt, provider, model_id):
        
        match self.provider:
            
            case "openai":
                return # use the openai api
            
            
            case "anthropic":
                return # use the anthropic api

        
        

        