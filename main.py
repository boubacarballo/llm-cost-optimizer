from chat import Chat
import sys
import requests
import os
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
    
    if not prompt_classifier_url:
        raise "Unable to ge prompt classifier URL"
    
    while True: # each full loop corresponds to one session
        
        prompt = input("Please enter your prompt: ")
        
        if prompt == "q":
            print("Goodbye")
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
        
        print(result)
        
        exit(1)
        
        provider = "anthropic"
        model = "claude-haiku-4-5-20251001"
        model_config = {
            "provider": provider,
            "model_id": model,
        }
        
        response = send_request(prompt, model_config)
        
        print(response.output_text)
        print(response.latency)
        
        
    
    
    
    
