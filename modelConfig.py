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
        
        

        
        