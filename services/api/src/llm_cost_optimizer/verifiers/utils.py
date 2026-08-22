
from typing import Literal
from llm_cost_optimizer.utils import config, hash_text, TaskType
from llm_cost_optimizer.chat import Chat
from llm_cost_optimizer.database import SessionLocal
from llm_cost_optimizer.models import RoutingFailure
from llm_cost_optimizer.types.types import RoutingContext
from pydantic import BaseModel
from typing import Literal, Optional, Any
chat = Chat()

# Both alternates are re-selected deterministically each round, so a third pass
# would re-ask the same two models the same question. Cap it and record the loss.
MAX_ALTERNATE_ROUNDS = 2


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
    similar_res: Optional[dict] = None
    higher_res: Optional[dict] = None
    
    
class VerificationLog(BaseModel):
    round_count: int
    satisfactory_model: Optional[Literal["similar", "higher"]]
    rounds: Optional[list[VerificationRound]]
    

async def log_routing_failure(
    initial_model_config,
    alternate_model_config,
    initial_res,
    alternate_res,
    verification_log: VerificationLog,
    routing_context: Optional[RoutingContext] = None,
):
    """Persist one routing failure. Called both when an alternate rescued the
    request (alternate_* populated) and when none did (alternate_* all None)."""
    if routing_context is None:
        print("log_routing_failure: no routing context supplied, skipping DB write")
        return

    try:
        log_data = verification_log.model_dump(mode="json")
        task_type = routing_context.task_type

        routing_failure = RoutingFailure(
            prompt=routing_context.prompt,
            prompt_hash=hash_text(routing_context.prompt),
            token_count=routing_context.token_count,
            context_provided=routing_context.context_provided,
            context_window=routing_context.context_window,
            task_type=task_type.value if isinstance(task_type, TaskType) else str(task_type),
            task_tier=routing_context.task_tier,

            initial_model=str(initial_model_config.get("id")),
            initial_provider=str(initial_model_config.get("provider")),
            initial_tier=str(initial_model_config.get("tier") or "unknown"),
            initial_verdict=getattr(initial_res, "verdict", "Bad"),
            initial_score=getattr(initial_res, "score", None),

            alternate_model=str(alternate_model_config.get("id")) if alternate_model_config else None,
            alternate_provider=str(alternate_model_config.get("provider")) if alternate_model_config else None,
            alternate_tier=str(alternate_model_config.get("tier")) if alternate_model_config else None,
            winning_tier=verification_log.satisfactory_model,
            alternate_verdict=getattr(alternate_res, "verdict", None),
            alternate_score=getattr(alternate_res, "score", None),

            round_count=verification_log.round_count,
            rounds=log_data.get("rounds", []) or [],
        )

        async with SessionLocal() as session:
            session.add(routing_failure)
            await session.commit()

        print(
            f"Logged routing failure: {routing_failure.initial_model} -> "
            f"{routing_failure.alternate_model or 'no satisfactory alternate'} "
            f"after {routing_failure.round_count} round(s)"
        )
    except Exception as exc:
        # a bad DB write must not take the arq job down with it
        print(f"Failed to persist routing failure: {exc}")


async def run_alternate_model(verification_function, prompt, alternate_model_config):
    """Re-run the prompt on one alternate and verify it. Returns None when there
    is no such alternate (empty tier, or tier_3 with nothing above it)."""
    if not alternate_model_config:
        return None

    alternate_res = await chat.send_request(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model_config=alternate_model_config,
    )
    if alternate_res is None:
        return None

    return await verification_function(
        prompt=prompt,
        response=alternate_res.output_text,
        model_config=alternate_model_config
    )


def _is_unsatisfactory(result) -> bool:
    return result is None or result.verdict == "Bad"


async def handle_alternate_verification(
    verification_function,
    initial_response,
    prompt,
    model_config,
    routing_context: Optional[RoutingContext] = None,
):
    alternate_verification_rounds = 0
    rounds = []
    similar_model_alternate_response = None
    higher_model_alternate_response = None
    alternate_similar_tier_model = None
    alternate_higher_tier_model = None
    
    while alternate_verification_rounds < MAX_ALTERNATE_ROUNDS:
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

        similar_model_alternate_response = await run_alternate_model(
            verification_function, prompt, alternate_similar_tier_model
        )
        higher_model_alternate_response = await run_alternate_model(
            verification_function, prompt, alternate_higher_tier_model
        )

        verification.similar_res = (
            similar_model_alternate_response.model_dump()
            if similar_model_alternate_response is not None else None
        )
        verification.higher_res = (
            higher_model_alternate_response.model_dump()
            if higher_model_alternate_response is not None else None
        )
        rounds.append(verification)
        
        if _is_unsatisfactory(similar_model_alternate_response) and _is_unsatisfactory(higher_model_alternate_response):
            continue
        else:
            break

    # first satisfactory alternate wins, cheaper tier checked first
    satisfactory_model = None
    winning_model_config = None
    winning_response = None

    for alternate_tier, alternate_model, alternate_response in (
    ("similar", alternate_similar_tier_model, similar_model_alternate_response),
    ("higher", alternate_higher_tier_model, higher_model_alternate_response),):
        if alternate_response is not None and alternate_response.verdict == "Good":
            satisfactory_model = alternate_tier
            winning_model_config = alternate_model
            winning_response = alternate_response
            break

    # logged either way: "nothing we tried was good enough" is itself a finding
    verification_log = VerificationLog(
        round_count=alternate_verification_rounds,
        satisfactory_model=satisfactory_model,
        rounds=rounds
    )
    await log_routing_failure(
        initial_model_config=model_config,
        alternate_model_config=winning_model_config,
        initial_res=initial_response,
        alternate_res=winning_response,
        verification_log=verification_log,
        routing_context=routing_context,
    )
