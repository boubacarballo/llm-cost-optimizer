from llm_cost_optimizer.chat import Chat    
from llm_cost_optimizer.types.response_models import BrainstormingJudgeOutput
from llm_cost_optimizer.prompts import BRAINSTORMING_JUDGE_PROMPT_TEMPLATE
from llm_cost_optimizer.utils import generate_messages_template
import asyncio
chat = Chat()

def build_brainstorming_judge_prompt(original_prompt: str, brainstorming_output: str) -> str:
    return BRAINSTORMING_JUDGE_PROMPT_TEMPLATE.format(
        original_prompt=original_prompt,
        brainstorm_output=brainstorming_output,
        constraints=""
    )
    
async def verify_brainstorming(prompt: str, response:str, judge_model_config) -> BrainstormingJudgeOutput:
    brainstorming_judge_prompt = build_brainstorming_judge_prompt(
        prompt,
        response,
    )
    messages = generate_messages_template(
        role="user",
        message=brainstorming_judge_prompt
    )
    response = await chat.send_request(
        messages,
        model_config=judge_model_config,
        responseFormat=BrainstormingJudgeOutput
    )
    return response.output_text


async def main():
    print(await verify_brainstorming(
            prompt="give me 2 date spot ideas",
            response="""Here are two solid date spot ideas:
    1. Sunset picnic by the water
    Pick a scenic spot near the coast or a lake, bring a blanket, some good food, and a bottle of wine. It's relaxed, low-pressure, and gives you space to actually talk. Works especially well if you're somewhere with a nice sunset view.
    2. A cozy café or wine bar with something to do
    Somewhere with good ambiance where you can order small plates or drinks, ideally with a bit of activity thrown in — live music, a board game on the table, or a spot known for a specific specialty (good coffee, tapas, etc.). Gives you built-in conversation starters if things go quiet.
    If you tell me your city, the vibe you're going for (low-key vs. adventurous), or whether it's a first date or long-term partner, I can get more specific.""",
            judge_model_config={
                "provider": "openai",
                "id": "gpt-5.6-luna"
            } 
        ))

if __name__ == "__main__":
    
    asyncio.run(main())
    