"""
Orchestrator Agent - Coordinates the workflow between multiple agents.

This agent:
1. Manages the overall workflow
2. Delegates tasks to specialized agents
3. Collects and integrates results from different agents
4. Makes final decisions based on collective agent inputs
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from langchain.chat_models.base import BaseChatModel

from agents.summarization_agent import SummarizationAgent
from agents.research_agent import ResearchAgent
from agents.critique_agent import CritiqueAgent
from models.agent_log_utils import log_agent_decision, log_agent_communication
from models.agent_message import AgentMessage, get_messages_for_agent
from tools.vectorstore import store_summary, create_faiss_store, get_embeddings

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """
    An agent that coordinates the workflow between multiple specialized agents.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        embedding_provider: str = "openai",
        verbose: bool = False
    ):
        """
        Initialize the orchestrator agent.
        
        Args:
            llm: The language model to use
            embedding_provider: The embedding provider for vector storage
            verbose: Whether to show detailed output
        """
        self.llm = llm
        self.verbose = verbose
        self.agent_name = "orchestrator_agent"
        
        # Initialize sub-agents
        self.summarization_agent = SummarizationAgent(llm, embedding_provider, verbose)
        self.research_agent = ResearchAgent(llm, verbose)
        self.critique_agent = CritiqueAgent(llm, verbose)
        
        # Initialize vector store for final results
        self.embed_model = get_embeddings(embedding_provider)
        self.vectorstore = create_faiss_store(self.embed_model)
    
    def process_url(self, url: str, summary_style: str = "default", max_chars: int = 4000) -> Dict[str, Any]:
        """
        Process a URL through the multi-agent workflow.
        
        Args:
            url: The URL to process
            summary_style: The style of summary to generate
            max_chars: Maximum characters to process
            
        Returns:
            Results including summary, research, and evaluation
        """
        workflow_start = datetime.now()
        
        # Step 1: Get initial summary from summarization agent
        logger.info(f"[Orchestrator] Delegating to summarization agent: {url}")
        summary_result = self.summarization_agent.run(url, {"style": summary_style})
        
        raw_text = summary_result["raw_text"]
        initial_summary = summary_result["summary"]
        
        # Step 2: Send to research agent for topic extraction and research
        logger.info("[Orchestrator] Delegating to research agent for content enrichment")
        research_message_id = self.send_message_to_agent(
            recipient_agent="research_agent",
            message_type="research_request",
            content={
                "url": url,
                "content": raw_text[:10000]  # Limit content size
            }
        )
        
        # Step 3: Send to critique agent for evaluation
        logger.info("[Orchestrator] Delegating to critique agent for summary evaluation")
        critique_message_id = self.send_message_to_agent(
            recipient_agent="critique_agent",
            message_type="evaluate_summary",
            content={
                "original_content": raw_text[:10000],  # Limit content size
                "summary": initial_summary,
                "summary_id": "initial"
            }
        )
        
        # Step 4: Process research results
        research_results = self.wait_for_agent_response("research_agent", research_message_id)
        enriched_content = {}
        if research_results:
            enriched_content = research_results.get("content", {})
            logger.info(f"[Orchestrator] Received research results with {len(enriched_content.get('topics', []))} topics")
        
        # Step 5: Process critique results
        critique_results = self.wait_for_agent_response("critique_agent", critique_message_id)
        evaluation = {}
        if critique_results:
            evaluation = critique_results.get("content", {}).get("evaluation", {})
            logger.info(f"[Orchestrator] Received critique with overall score: {evaluation.get('scores', {}).get('overall', 'N/A')}")
        
        # Step 6: Request summary improvement if score is below threshold
        final_summary = initial_summary
        if evaluation and evaluation.get("scores", {}).get("overall", 10) < 7:
            logger.info("[Orchestrator] Requesting summary improvement based on feedback")
            improve_message_id = self.send_message_to_agent(
                recipient_agent="critique_agent",
                message_type="improve_summary",
                content={
                    "original_content": raw_text[:10000],
                    "summary": initial_summary,
                    "feedback": evaluation.get("feedback", ""),
                    "summary_id": "improved"
                }
            )
            
            # Wait for improved summary
            improvement_results = self.wait_for_agent_response("critique_agent", improve_message_id)
            if improvement_results:
                improved_summary = improvement_results.get("content", {}).get("improved_summary")
                if improved_summary:
                    final_summary = improved_summary
                    logger.info("[Orchestrator] Received improved summary")
        
        # Step 7: Store final results in vector database
        metadata = {
            "source_url": url,
            "timestamp": datetime.now().isoformat(),
            "workflow_duration": (datetime.now() - workflow_start).total_seconds(),
            "agents_involved": ["summarization_agent", "research_agent", "critique_agent", "orchestrator_agent"],
            "summary_style": summary_style,
            "evaluation_scores": json.dumps(evaluation.get("scores", {}))
        }
        
        store_summary(self.vectorstore, final_summary, metadata)
        
        # Log the orchestration decision
        reflection = f"Orchestrated multi-agent workflow for URL: {url}"
        reasoning_chain = [
            f"Step 1: Delegated to summarization agent to process URL",
            f"Step 2: Delegated to research agent for topic extraction and research",
            f"Step 3: Delegated to critique agent for summary evaluation",
            f"Step 4: Processed research results with {len(enriched_content.get('topics', []))} topics",
            f"Step 5: Processed critique with overall score: {evaluation.get('scores', {}).get('overall', 'N/A')}",
            f"Step 6: {'Requested and received improved summary' if final_summary != initial_summary else 'Kept original summary'}",
            f"Step 7: Stored final results in vector database"
        ]
        
        log_agent_decision(
            agent_name=self.agent_name,
            input_data=url,
            output_summary=final_summary,
            reflection=reflection,
            reasoning_chain=reasoning_chain
        )
        
        # Return comprehensive results
        return {
            "url": url,
            "raw_text": raw_text,
            "initial_summary": initial_summary,
            "final_summary": final_summary,
            "research": enriched_content,
            "evaluation": evaluation,
            "metadata": metadata
        }
    
    def send_message_to_agent(self, recipient_agent: str, message_type: str, content: Dict[str, Any]) -> str:
        """
        Send a message to another agent.
        
        Args:
            recipient_agent: The name of the recipient agent
            message_type: The type of message
            content: The message content
            
        Returns:
            Message ID
        """
        message_id = log_agent_communication(
            sender=self.agent_name,
            recipient=recipient_agent,
            message_type=message_type,
            content=content
        )
        
        return message_id
    
    def wait_for_agent_response(self, agent_name: str, request_message_id: str, max_attempts: int = 10) -> Optional[Dict[str, Any]]:
        """
        Wait for a response from an agent.
        
        In a real system, this would use a message queue or event system.
        For this demo, we'll simulate by checking the database.
        
        Args:
            agent_name: The name of the agent to wait for
            request_message_id: The ID of the request message
            max_attempts: Maximum number of attempts to check for response
            
        Returns:
            Dictionary containing the response data, if found
        """
        # In a real system, this would be implemented with a proper message queue
        # For this demo, we'll simulate by directly processing the message
        
        if agent_name == "research_agent":
            # Simulate research agent processing
            content = {}
            if request_message_id:  # Use message ID to retrieve the original request
                messages = get_messages_for_agent(agent_name, limit=5)
                for msg in messages:
                    if msg.message_type == "research_request":
                        raw_content = msg.content.get("content", "")
                        content = self.research_agent.enrich_content(raw_content)
                        break
            
            # Log the response
            response_id = log_agent_communication(
                sender=agent_name,
                recipient=self.agent_name,
                message_type="research_results",
                content=content,
                in_reply_to=request_message_id
            )
            
            return {
                "id": response_id,
                "sender": agent_name,
                "recipient": self.agent_name,
                "message_type": "research_results",
                "content": content,
                "timestamp": datetime.now(),
                "in_reply_to": request_message_id
            }
        
        elif agent_name == "critique_agent":
            # Simulate critique agent processing
            messages = get_messages_for_agent(agent_name, limit=5)
            for msg in messages:
                if msg.in_reply_to == request_message_id:
                    return {
                        "id": msg.id,
                        "sender": msg.sender,
                        "recipient": msg.recipient,
                        "message_type": msg.message_type,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "in_reply_to": msg.in_reply_to
                    }
                
                if msg.message_type == "evaluate_summary" and request_message_id:
                    # Process evaluation request
                    original_content = msg.content.get("original_content", "")
                    summary = msg.content.get("summary", "")
                    evaluation = self.critique_agent.evaluate_summary(original_content, summary)
                    
                    content = {
                        "evaluation": evaluation,
                        "summary_id": msg.content.get("summary_id")
                    }
                    
                    # Log the response
                    response_id = log_agent_communication(
                        sender=agent_name,
                        recipient=self.agent_name,
                        message_type="summary_evaluation",
                        content=content,
                        in_reply_to=request_message_id
                    )
                    
                    return {
                        "id": response_id,
                        "sender": agent_name,
                        "recipient": self.agent_name,
                        "message_type": "summary_evaluation",
                        "content": content,
                        "timestamp": datetime.now(),
                        "in_reply_to": request_message_id
                    }
                
                elif msg.message_type == "improve_summary" and request_message_id:
                    # Process improvement request
                    original_content = msg.content.get("original_content", "")
                    summary = msg.content.get("summary", "")
                    feedback = msg.content.get("feedback", "")
                    improved_summary = self.critique_agent.improve_summary(original_content, summary, feedback)
                    
                    content = {
                        "improved_summary": improved_summary,
                        "summary_id": msg.content.get("summary_id")
                    }
                    
                    # Log the response
                    response_id = log_agent_communication(
                        sender=agent_name,
                        recipient=self.agent_name,
                        message_type="improved_summary",
                        content=content,
                        in_reply_to=request_message_id
                    )
                    
                    return {
                        "id": response_id,
                        "sender": agent_name,
                        "recipient": self.agent_name,
                        "message_type": "improved_summary",
                        "content": content,
                        "timestamp": datetime.now(),
                        "in_reply_to": request_message_id
                    }
        
        return None