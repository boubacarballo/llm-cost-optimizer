
from typing import Literal
from llm_cost_optimizer.utils import config
from llm_cost_optimizer.chat import Chat
chat = Chat()

async def select_alternate_model(model_config: dict, tier: Literal["similar", "higher"]):
    if tier == "similar" or model_config["tier"] == "tier_3":
        alternate_models = [
                        model
                        for model in config["tiers"][model_config["tier"]]
                        if model["id"] != model_config["id"]
                        or model["provider"] != model_config["provider"]
                    ]
        new_model_config = alternate_models[0] if alternate_models else None
        return new_model_config
        
    elif tier == "higher": 
        
        current_tier = int(model_config["tier"][-1])
        new_tier = "tier_" + str(current_tier + 1)
        higher_tier_models = config["tiers"][new_tier]
        return higher_tier_models[0] if higher_tier_models else None
    
async def handle_alternate_verification(
    verification_function,
    prompt,
    model_config
):
    
    alternate_similar_tier_model = await select_alternate_model(
                model_config=model_config,
                tier="similar"
            )
    alternate_higher_tier_model = await select_alternate_model(
                model_config=model_config,
                tier="higher"
            )
    
    similar_res = await chat.send_request(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model_config=alternate_similar_tier_model,
    )
    higher_res = await chat.send_request(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model_config=alternate_higher_tier_model
    )
    similar_model_alternate_response = await verification_function(
        prompt=prompt,
        response=similar_res.output_text,
        model_config=alternate_similar_tier_model
    )
    
    higher_model_alternate_response = await verification_function(
        prompt=prompt,
        response=higher_res.output_text,
        model_config=alternate_higher_tier_model
    )
    
    return similar_model_alternate_response, higher_model_alternate_response
    


    
    