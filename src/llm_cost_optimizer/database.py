from sqlmodel import Field, SQLModel, create_engine, Session
from llm_cost_optimizer.models import RequestEvent
import os
from dotenv import load_dotenv
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Query
load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"), echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    
def get_session():
    with Session(engine) as session:
        yield session
        
SessionDep = Annotated[Session, Depends(get_session)]
