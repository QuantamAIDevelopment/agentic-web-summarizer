"""
Agent Interaction Viewer.

This script displays recent agent interactions from the database.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import json

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.database import SessionLocal
from models.agent_message import AgentMessage
from models.agent_decisions import AgentDecision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def view_recent_messages(hours=24, limit=20):
    """View recent agent messages."""
    try:
        with SessionLocal() as session:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            messages = session.query(AgentMessage).filter(
                AgentMessage.timestamp >= cutoff_time
            ).order_by(AgentMessage.timestamp.desc()).limit(limit).all()
            
            if not messages:
                print(f"No agent messages found in the last {hours} hours.")
                return
            
            print(f"\n=== Recent Agent Messages (Last {hours} hours) ===\n")
            for msg in messages:
                print(f"ID: {msg.id}")
                print(f"Time: {msg.timestamp}")
                print(f"From: {msg.sender} → To: {msg.recipient}")
                print(f"Type: {msg.message_type}")
                print(f"Status: {msg.status}")
                if msg.in_reply_to:
                    print(f"In reply to: {msg.in_reply_to}")
                
                # Pretty print content if it's a dict
                if isinstance(msg.content, dict):
                    print("Content:")
                    print(json.dumps(msg.content, indent=2))
                else:
                    print(f"Content: {msg.content}")
                print("-" * 50)
    
    except Exception as e:
        logger.error(f"Error viewing messages: {e}")

def view_recent_decisions(hours=24, limit=20):
    """View recent agent decisions."""
    try:
        with SessionLocal() as session:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            decisions = session.query(AgentDecision).filter(
                AgentDecision.timestamp >= cutoff_time
            ).order_by(AgentDecision.timestamp.desc()).limit(limit).all()
            
            if not decisions:
                print(f"No agent decisions found in the last {hours} hours.")
                return
            
            print(f"\n=== Recent Agent Decisions (Last {hours} hours) ===\n")
            for decision in decisions:
                print(f"ID: {decision.id}")
                print(f"Time: {decision.timestamp}")
                print(f"Agent: {decision.agent_name}")
                print(f"Input: {decision.input_data[:100]}..." if len(decision.input_data or "") > 100 else f"Input: {decision.input_data}")
                print(f"Output: {decision.output_summary[:100]}..." if len(decision.output_summary or "") > 100 else f"Output: {decision.output_summary}")
                print(f"Reflection: {decision.reflection}")
                
                # Parse and pretty print reasoning chain if it's JSON
                try:
                    reasoning = json.loads(decision.reasoning_chain)
                    print("Reasoning:")
                    for i, step in enumerate(reasoning):
                        print(f"  {i+1}. {step}")
                except:
                    print(f"Reasoning: {decision.reasoning_chain}")
                print("-" * 50)
    
    except Exception as e:
        logger.error(f"Error viewing decisions: {e}")

def main():
    """Main function."""
    hours = 24
    limit = 20
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            hours = int(sys.argv[1])
        except ValueError:
            print(f"Invalid hours value: {sys.argv[1]}. Using default: {hours}")
    
    if len(sys.argv) > 2:
        try:
            limit = int(sys.argv[2])
        except ValueError:
            print(f"Invalid limit value: {sys.argv[2]}. Using default: {limit}")
    
    view_recent_messages(hours, limit)
    view_recent_decisions(hours, limit)

if __name__ == "__main__":
    main()