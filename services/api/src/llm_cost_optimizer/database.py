from sqlmodel import Field, SQLModel, create_engine, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from llm_cost_optimizer.models import RequestEvent
import os
from dotenv import load_dotenv
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Query
load_dotenv()
engine = create_async_engine(os.getenv("DATABASE_URL"), echo=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
async def get_session():
    with SessionLocal() as session:
        yield session
        
SessionDep = Annotated[AsyncSession, Depends(get_session)]
