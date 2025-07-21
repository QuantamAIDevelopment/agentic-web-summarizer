from config.database import SessionLocal
from models.agent_decisions import AgentDecision
from datetime import datetime
import logging
import json
import uuid
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def log_agent_decision(
    agent_name: str,
    input_data: str,
    output_summary: str,
    reflection: str,
    reasoning_chain,
    timestamp=None
) -> int:
    """Log an agent's decision to the database.
    
    Args:
        agent_name: Name of the agent making the decision
        input_data: Input data that led to the decision
        output_summary: Summary of the output/decision
        reflection: Agent's reflection on the decision
        reasoning_chain: List or string of reasoning steps
        timestamp: Optional timestamp (defaults to now)
        
    Returns:
        ID of the created decision record
    """
    if isinstance(reasoning_chain, list):
        reasoning_chain = json.dumps(reasoning_chain)
    if timestamp is None:
        timestamp = datetime.now()
    try:
        with SessionLocal() as session:
            decision = AgentDecision(
                agent_name=agent_name,
                input_data=input_data,
                output_summary=output_summary,
                reflection=reflection,
                reasoning_chain=reasoning_chain,
                timestamp=timestamp
            )
            session.add(decision)
            session.commit()
            logger.info(f"Agent decision saved for {agent_name}")
            return decision.id
    except Exception as e:
        logger.error(f"Failed to log agent decision: {e}")
        return -1

def log_agent_communication(
    sender: str,
    recipient: str,
    message_type: str,
    content: Dict[str, Any],
    in_reply_to: Optional[str] = None,
    timestamp=None
) -> str:
    """Log communication between agents.
    
    Args:
        sender: Name of the sending agent
        recipient: Name of the receiving agent
        message_type: Type of message (e.g., "request", "response")
        content: Message content as a dictionary
        in_reply_to: Optional ID of the message this is replying to
        timestamp: Optional timestamp (defaults to now)
        
    Returns:
        ID of the created message record
    """
    from models.agent_message import AgentMessage
    
    if timestamp is None:
        timestamp = datetime.now()
    
    message_id = str(uuid.uuid4())
    
    try:
        with SessionLocal() as session:
            message = AgentMessage(
                id=message_id,
                sender=sender,
                recipient=recipient,
                message_type=message_type,
                content=content,
                timestamp=timestamp,
                in_reply_to=in_reply_to,
                status="sent"
            )
            session.add(message)
            session.commit()
            logger.info(f"Agent communication logged: {sender} -> {recipient} ({message_type})")
            return message_id
    except Exception as e:
        logger.error(f"Failed to log agent communication: {e}")
        return message_id  # Return the generated ID even if storage failed