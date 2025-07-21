from sqlalchemy import Column, String, Integer, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String, nullable=False)
    input_data = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    reasoning_chain = Column(Text, nullable=True)
    reflection = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)
