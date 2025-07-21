import logging
from typing import List, Optional
from datetime import datetime
import json

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models.base import BaseChatModel

from config.database import SessionLocal
from models.agent_decisions import AgentDecision
from models.agent_log_utils import log_agent_decision

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WebAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Tool],
        agent_type: str = "zero-shot-react-description",
        agent_name: Optional[str] = None,
        verbose: bool = True
    ):
        supported_agent_types = {
            "zero-shot-react-description": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            "openai-functions": AgentType.OPENAI_FUNCTIONS,
            "chat-conversational-react-description": AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        }

        if agent_type not in supported_agent_types:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        self.agent_name = agent_name or "WebAgent"
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type=supported_agent_types[agent_type],
            verbose=verbose
        )

    def run(self, query: str) -> str:
        logger.info(f"[WebAgent] Running agent with query: {query}")
        response = self.agent.run(query)

        reflection = self.generate_reflection(response)
        reasoning_chain = [
            f"Agent executed query: {query}",
            f"Generated response: {response}",
            f"Reflection: {reflection}"
        ]

        self.log_agent_decision(query, response, reasoning_chain, reflection)
        return response

    def generate_reflection(self, response: str) -> str:
        if not response or "error" in response.lower():
            return "Response indicates possible failure or missing tool output."
        elif len(response.split()) < 20:
            return "Response was brief. Could indicate a simple task or lack of detail."
        else:
            return "Agent responded with a meaningful and coherent output."

    def log_agent_decision(self, input_data: str, output_data: str,
                           reasoning_chain: List[str], reflection: str) -> None:
        log_agent_decision(
            agent_name=self.agent_name,
            input_data=input_data,
            output_summary=output_data,
            reflection=reflection,
            reasoning_chain=reasoning_chain
        )
