
from typing import Literal
from llm_cost_optimizer.utils import config
from llm_cost_optimizer.chat import Chat
from pydantic import BaseModel
from typing import Literal, Optional
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
    
    
class VerificationRound(BaseModel):
    round: int
    similar_res: Optional[any]
    higher_res: Optional[any]
    
    
class AlternateVerification(BaseModel):
    rounds: int
    satisfactory_model: Optional[Literal["similar", "higher"]]
    rounds: Optional[list[VerificationRound]]
    

async def log_routing_failure(initial_model_config, alternate_model_config, initial_res, alternate_res, verification_log: AlternateVerification):
    pass
        
    
    
async def handle_alternate_verification(
    verification_function,
    initial_response,
    prompt,
    model_config
):
    alternate_verification_rounds = 0
    rounds = []
    similar_model_alternate_response = None
    higher_model_alternate_response = None
    
    while True:
        alternate_verification_rounds += 1
        verification = VerificationRound(round=alternate_verification_rounds)
        
    
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
        verification.similar_res=similar_model_alternate_response
        verification.higher_res=higher_model_alternate_response
        rounds.append(verification)
        
        if similar_model_alternate_response.verdict == "Bad" and higher_model_alternate_response.verdict == "Bad":
            continue
        else:
            break

        
    for alternate_tier, alternate_model, alternate_response in (
    ("similar", alternate_similar_tier_model, similar_model_alternate_response),
    ("higher", alternate_higher_tier_model, higher_model_alternate_response),):
        if alternate_response is not None and alternate_response.verdict == "Good":
            verification_log = AlternateVerification(
                rounds=alternate_verification_rounds,
                satisfactory_model=alternate_tier,
                rounds=rounds
            )
            await log_routing_failure(
                initial_model_config=model_config,
                alternate_model_config=alternate_model,
                initial_res=initial_response,
                alternate_res=alternate_response,
                verification_log=verification_log
                
            )
            break


        