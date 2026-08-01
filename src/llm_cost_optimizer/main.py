from chat import Chat
from utils import load_models_configs, breakdown_results, get_candidate_models, select_model_and_provider
import sys
import requests
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
chat = Chat()

def send_request(prompt: str, model_config):

    response = chat.send_request(
        prompt,
        model_config["provider"],
        model_config["model_id"]
    )
    
    return response
    
    
    
if __name__ == "__main__":
    
    prompt_classifier_url = os.getenv("PROMPT_CLASSIFIER_SERVER_URL")
    model_configs_path = os.getenv("MODEL_CONFIGS_PATH")
    
    if not prompt_classifier_url:
        raise "Unable to ge prompt classifier URL"
    
    if not model_configs_path:
        raise "Unable to load model configurations"
    
    config = load_models_configs(model_configs_path)
    messages = [
        {
            "role": "assistant",
            "content": "You are a helpful assistant",
        }
    ]
    
    while True: # each full loop corresponds to one session
        
        prompt = input("Please enter your prompt: ")
        
        if prompt == "q":
            print("Goodbye")
            #save messages
            break
        
        # TODO: logic to determine model and provider
        
        # get the complexity here
        params = {
            "prompt": prompt,
        }
        response = requests.post(
            url=prompt_classifier_url,
            json=params
        )
        
        result = response.json()
        print(f"result: {result}")
        
        tier_flags, contextual_knowledge, complexity = breakdown_results(result)
        print(f"tier flags {tier_flags}")
        
        candidate_models = get_candidate_models(tier_flags, config)
        print(f"candidates: {candidate_models}")
        
        model_config = select_model_and_provider(candidate_models, context_window=12700)
        
        #append to the messages array
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        print(model_config)
        response = chat.send_request(
            messages=messages,
            provider=model_config["provider"],
            model_id=model_config["model_id"],
        )
        
        print(response.output_text)
        
        #save what the model returned
        messages.append({
            "role": "assistant",
            "content": response.output_text
        })
        
        
        
    
    
    
    
