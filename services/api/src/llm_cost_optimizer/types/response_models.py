from pydantic import BaseModel, Field
from typing import Literal


class BrainstormingJudgeScoring(BaseModel):
    score: int = Field(..., description="A score between 1 and 5")
    justification: str = Field(..., description="A text justiciation for the score")
    
class BrainstormingJudgeOutput(BaseModel):
    relevance: BrainstormingJudgeScoring
    novelty: BrainstormingJudgeScoring
    diversity: BrainstormingJudgeScoring
    feasibility: BrainstormingJudgeScoring
    framing: BrainstormingJudgeScoring
    weak_ideas_to_cut: list[str] = Field(..., description="Ideas that should be removed from the brainstorm")
    overall_score: int
    verdict: Literal["GOOD", "BAD"]
    