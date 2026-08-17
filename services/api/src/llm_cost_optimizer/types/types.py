from pydantic import BaseModel, Field
from typing import Literal, Optional


class VerificationResult(BaseModel):
    verdict: Literal["Good", "Bad"]
    
        
