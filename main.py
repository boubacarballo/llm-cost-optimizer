from .chat import Chat


chat = Chat()

def send_request(prompt: str, model_config):

    response = chat.send_request(
        prompt,
        model_config["provider"],
        model_config["model_id"]
    )
    
    return response
    
    
    
if __name__ == "__main__":
    
    while True: # each full loop corresponds to one session
        
        prompt = input("Please enter your prompt: ")
        
        if prompt == "q":
            print("Goodbye")
            break
        
        # TODO: logic to determine model and provider
        
        provider = "openai"
        model = "gpt-5.6-terra"
        model_config = {
            "provider": provider,
            "model": model,
        }
        
        response = send_request(prompt, model_config)
        
        print(response["output_text"])
        
        
    
    
    
    
