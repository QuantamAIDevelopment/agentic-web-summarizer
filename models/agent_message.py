"""
Agent Message - Database model for storing agent communications.

This module defines:
1. The AgentMessage model for storing inter-agent communications
2. Functions for logging and retrieving agent messages
"""

from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import json
import logging
from typing import List, Dict, Any, Optional

from config.database import SessionLocal

Base = declarative_base()
logger = logging.getLogger(__name__)

class AgentMessage(Base):
    """
    Database model for storing messages between agents.
    """
    __tablename__ = "agent_messages"

    id = Column(String, primary_key=True)
    sender = Column(String, nullable=False)
    recipient = Column(String, nullable=False)
    message_type = Column(String, nullable=False)
    content = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)
    in_reply_to = Column(String, nullable=True)
    status = Column(String, default="sent")  # sent, delivered, processed

def get_messages_for_agent(agent_name: str, limit: int = 10, status: str = None) -> List[AgentMessage]:
    """
    Get messages for a specific agent.
    
    Args:
        agent_name: The name of the agent
        limit: Maximum number of messages to retrieve
        status: Filter by message status
        
    Returns:
        List of agent messages
    """
    try:
        with SessionLocal() as session:
            query = session.query(AgentMessage).filter(
                AgentMessage.recipient == agent_name
            ).order_by(AgentMessage.timestamp.desc())
            
            if status:
                query = query.filter(AgentMessage.status == status)
            
            messages = query.limit(limit).all()
            return messages
    except Exception as e:
        logger.error(f"Failed to get messages for agent {agent_name}: {e}")
        return []

def mark_message_as_processed(message_id: str) -> bool:
    """
    Mark a message as processed.
    
    Args:
        message_id: The ID of the message
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with SessionLocal() as session:
            message = session.query(AgentMessage).filter(
                AgentMessage.id == message_id
            ).first()
            
            if message:
                message.status = "processed"
                session.commit()
                return True
            return False
    except Exception as e:
        logger.error(f"Failed to mark message {message_id} as processed: {e}")
        return False

def get_conversation_thread(message_id: str) -> List[AgentMessage]:
    """
    Get a conversation thread starting from a message.
    
    Args:
        message_id: The ID of the starting message
        
    Returns:
        List of messages in the conversation thread
    """
    try:
        with SessionLocal() as session:
            # Get the initial message
            initial_message = session.query(AgentMessage).filter(
                AgentMessage.id == message_id
            ).first()
            
            if not initial_message:
                return []
            
            # Get all replies to this message
            replies = session.query(AgentMessage).filter(
                AgentMessage.in_reply_to == message_id
            ).all()
            
            # Recursively get replies to replies
            thread = [initial_message] + replies
            for reply in replies:
                thread.extend(get_conversation_thread(reply.id))
            
            return thread
    except Exception as e:
        logger.error(f"Failed to get conversation thread for message {message_id}: {e}")
        return []