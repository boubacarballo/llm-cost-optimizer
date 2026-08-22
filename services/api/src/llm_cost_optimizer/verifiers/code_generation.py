import json
from pydantic import BaseModel, Field
from typing import Literal, List
from llm_cost_optimizer.chat import Chat
from llm_cost_optimizer.types.types import VerificationResult
from llm_cost_optimizer.prompts import CODE_GENERATION_JUDGE_PROMPT_TEMPLATE
from llm_cost_optimizer.verifiers.utils import handle_alternate_verification
import asyncio

chat = Chat()

# what each severity costs the final score, per issue
ISSUE_WEIGHTS = {
    "CRITICAL": 0.25,
    "MAJOR": 0.10,
    "MINOR": 0.03,
}
MAX_ISSUE_PENALTY = 0.5
# one CRITICAL defect can never score as passing, whatever else is right
CRITICAL_ISSUE_CEILING = 0.5


class CodeIssue(BaseModel):
    severity: Literal["CRITICAL", "MAJOR", "MINOR"]
    description: str


class CodeGenerationJudgeResponse(BaseModel):
    requirements_met: Literal["MET", "PARTIALLY_MET", "NOT_MET"]
    syntax_valid: bool
    issues: List[CodeIssue] = []
    format_adherence: float = Field(..., ge=0.0, le=1.0)


class CodeGenerationJudgeResult(VerificationResult):
    score: float
    raw: dict


def build_code_generation_judge_prompt(prompt: str, code: str) -> str:
    return CODE_GENERATION_JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        code=code,
    )


def get_code_generation_judge_result(judge_response: CodeGenerationJudgeResponse) -> CodeGenerationJudgeResult:
    data = json.loads(judge_response.model_dump_json()) if hasattr(judge_response, "model_dump_json") else judge_response.dict()

    requirements = data.get("requirements_met")
    syntax_valid = bool(data.get("syntax_valid", False))
    issues = data.get("issues", []) or []
    format_adherence = float(data.get("format_adherence", 0.0))

    if requirements == "MET":
        requirements_score = 1.0
    elif requirements == "PARTIALLY_MET":
        requirements_score = 0.5
    else:
        requirements_score = 0.0

    base = 0.55 * requirements_score + 0.25 * (1.0 if syntax_valid else 0.0) + 0.20 * format_adherence

    # severity-weighted, so ten nits never outweigh one thing that does not run
    penalty = min(
        MAX_ISSUE_PENALTY,
        sum(ISSUE_WEIGHTS.get(i.get("severity"), 0.0) for i in issues)
    )

    final = max(0.0, min(1.0, base - penalty))

    # two hard gates: weighted scoring alone let these through as "Good".
    # code that will not parse is unusable no matter how well it meets the spec,
    # and one CRITICAL defect (wrong primary case, or a security hole) is a
    # routing failure even when everything else about the answer is clean.
    critical = sum(1 for i in issues if i.get("severity") == "CRITICAL")
    if not syntax_valid:
        final = 0.0
    elif critical:
        final = min(final, CRITICAL_ISSUE_CEILING)

    return CodeGenerationJudgeResult(
        score=final,
        raw=data,
        verdict="Good" if final >= 0.7 else "Bad"
    )


async def verify_code_generation(prompt: str, response: str, model_config: dict) -> CodeGenerationJudgeResult:

    judge_prompt = build_code_generation_judge_prompt(
        prompt=prompt,
        code=response
    )

    messages = [
        {
            "role": "user",
            "content": judge_prompt
        }
    ]

    judge_response = await chat.send_request(messages, model_config=model_config, responseFormat=CodeGenerationJudgeResponse)

    return get_code_generation_judge_result(judge_response.output_text)


async def handle_code_generation_verification(prompt, response, model_config, routing_context):
    initial_res = await verify_code_generation(
        prompt=prompt,
        response=response,
        model_config=model_config
    )
    if initial_res.verdict == "Bad":  # in this branch we log routing failure
        await handle_alternate_verification(
            verification_function=verify_code_generation,
            initial_response=initial_res,
            prompt=prompt,
            model_config=model_config,
            routing_context=routing_context
        )
    return initial_res


if __name__ == "__main__":

    prompt = (
        "Write a Python function `median(nums: list[float]) -> float` that returns the median. "
        "Raise ValueError on an empty list."
    )
    # wrong for even-length input, and never raises on empty
    response = (
        "def median(nums):\n"
        "    nums.sort()\n"
        "    return nums[len(nums) // 2]\n"
    )

    print(asyncio.run(verify_code_generation(prompt, response, {"provider": "openai", "id": "gpt-5-mini"})))
