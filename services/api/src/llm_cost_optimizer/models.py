from sqlmodel import Field, SQLModel, create_engine
from sqlalchemy import Column, DateTime
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
    