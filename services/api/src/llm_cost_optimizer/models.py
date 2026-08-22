from sqlmodel import Field, SQLModel, create_engine
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from datetime import datetime, timezone

class RequestEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    prompt_hash: str
    model: str
    provider: str
    cost: float
    latency: float
    quality_score: float | None = Field(default=None, nullable=True)
    escalated: bool | None = Field(default=None, nullable=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    
class RoutingFailure(SQLModel, table=True):
    """One row per verification failure that triggered an escalation.

    Written by the arq worker (verifiers/utils.log_routing_failure) whenever the
    model we routed to fails its judge. `prompt_hash` uses the same hash_text()
    helper as RequestEvent, so a failure joins back to the request that caused it.
    """
    id: int | None = Field(default=None, primary_key=True)

    # request context, captured at routing time in main.py
    prompt: str
    prompt_hash: str
    token_count: int
    context_provided: bool
    context_window: int
    task_type: str
    task_tier: int

    # the model we routed to, which failed verification
    initial_model: str
    initial_provider: str
    initial_tier: str
    initial_verdict: str
    initial_score: float | None = Field(default=None, nullable=True)

    # the model that passed; all NULL when no alternate satisfied the judge
    alternate_model: str | None = Field(default=None, nullable=True)
    alternate_provider: str | None = Field(default=None, nullable=True)
    alternate_tier: str | None = Field(default=None, nullable=True)
    winning_tier: str | None = Field(default=None, nullable=True)
    alternate_verdict: str | None = Field(default=None, nullable=True)
    alternate_score: float | None = Field(default=None, nullable=True)

    round_count: int
    rounds: list = Field(default_factory=list, sa_column=Column(JSONB))

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
