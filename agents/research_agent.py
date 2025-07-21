"""
Research Agent - Responsible for gathering additional information about topics in the content.

This agent can:
1. Extract key topics from content
2. Search for additional information about those topics
3. Provide enriched context to other agents
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models.base import BaseChatModel

from tools.fetcher import fetch_url_content
from tools.vectorstore import store_summary
from models.agent_log_utils import log_agent_decision, log_agent_communication
from models.agent_message import AgentMessage

logger = logging.getLogger(__name__)

class ResearchAgent:
    """
    An agent that extracts key topics from content and researches them further.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        verbose: bool = False
    ):
        """
        Initialize the research agent.
        
        Args:
            llm: The language model to use for research and reasoning
            verbose: Whether to show the agent's reasoning steps
        """
        self.llm = llm
        self.verbose = verbose
        self.agent_name = "research_agent"
        
        # Create the agent with tools
        self.agent = self._create_agent()
        
    def _create_agent(self):
        """
        Create a LangChain agent with the necessary tools.
        """
        tools = [
            Tool(
                name="FetchWebContent",
                func=fetch_url_content,
                description="Fetches and extracts the main content from a web page. Input should be a valid URL."
            )
        ]
        
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose
        )
    
    def extract_topics(self, content: str, max_topics: int = 5) -> List[str]:
        """
        Extract key topics from the content.
        
        Args:
            content: The text content to analyze
            max_topics: Maximum number of topics to extract
            
        Returns:
            A list of key topics
        """
        prompt = f"""
        Extract the {max_topics} most important topics or concepts from the following content.
        Return them as a comma-separated list.
        
        CONTENT:
        {content[:5000]}  # Limit content length for prompt
        
        KEY TOPICS:
        """
        
        try:
            response = self.llm.predict(prompt)
            
            # Handle different formats of responses
            if ',' in response:
                # Comma-separated list
                topics = [topic.strip() for topic in response.split(',')]
            elif '\n' in response:
                # Line-separated list
                topics = [topic.strip() for topic in response.split('\n') if topic.strip()]
            else:
                # Single topic or unknown format
                topics = [response.strip()]
            
            # Filter out empty topics and ensure we have at least one topic
            topics = [t for t in topics if t]
            if not topics:
                topics = ["main subject"]
                
            # Limit to max_topics
            topics = topics[:max_topics]
            
            # Log the decision
            reflection = f"Extracted {len(topics)} topics from content"
            reasoning_chain = [
                "Analyzed content to identify key topics",
                f"Selected top {len(topics)} topics based on relevance"
            ]
            
            log_agent_decision(
                agent_name=self.agent_name,
                input_data=content[:200] + "...",  # Truncate for logging
                output_summary=", ".join(topics),
                reflection=reflection,
                reasoning_chain=reasoning_chain
            )
            
            return topics
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    def research_topic(self, topic: str) -> str:
        """
        Research additional information about a topic.
        
        Args:
            topic: The topic to research
            
        Returns:
            Additional information about the topic
        """
        prompt = f"""
        Provide a brief but informative summary about the topic: {topic}
        Include key facts, context, and why this topic might be significant.
        Keep the response under 200 words.
        """
        
        try:
            response = self.llm.predict(prompt)
            
            # Log the decision
            reflection = f"Researched additional information about '{topic}'"
            reasoning_chain = [
                f"Received request to research topic: {topic}",
                "Generated informative summary based on knowledge"
            ]
            
            log_agent_decision(
                agent_name=self.agent_name,
                input_data=topic,
                output_summary=response,
                reflection=reflection,
                reasoning_chain=reasoning_chain
            )
            
            return response
        except Exception as e:
            logger.error(f"Error researching topic: {e}")
            return f"Unable to research topic '{topic}' due to an error."
    
    def send_message_to_agent(self, recipient_agent: str, message_type: str, content: Dict[str, Any]) -> str:
        """
        Send a message to another agent.
        
        Args:
            recipient_agent: The name of the recipient agent
            message_type: The type of message (e.g., "research_results")
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
    
    def process_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """
        Process a message from another agent.
        
        Args:
            message: The message to process
            
        Returns:
            Optional response data
        """
        if message.message_type == "research_request":
            # Handle research request
            if "topic" in message.content:
                topic = message.content["topic"]
                research_results = self.research_topic(topic)
                
                return {
                    "topic": topic,
                    "research": research_results
                }
        
        return None
    
    def enrich_content(self, content: str) -> Dict[str, Any]:
        """
        Enrich content with additional research.
        
        Args:
            content: The content to enrich
            
        Returns:
            Dictionary with original content and enriched information
        """
        # Extract topics
        topics = self.extract_topics(content)
        
        # Research each topic
        research_results = {}
        for topic in topics:
            research_results[topic] = self.research_topic(topic)
        
        return {
            "original_content": content,
            "topics": topics,
            "research": research_results
        }