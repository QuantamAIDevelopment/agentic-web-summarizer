"""
Critique Agent - Responsible for evaluating and improving summaries.

This agent can:
1. Evaluate the quality of a summary
2. Suggest improvements
3. Provide feedback to other agents
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.chat_models.base import BaseChatModel

from models.agent_log_utils import log_agent_decision, log_agent_communication
from models.agent_message import AgentMessage

logger = logging.getLogger(__name__)

class CritiqueAgent:
    """
    An agent that evaluates and improves summaries.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        verbose: bool = False
    ):
        """
        Initialize the critique agent.
        
        Args:
            llm: The language model to use for evaluation
            verbose: Whether to show detailed output
        """
        self.llm = llm
        self.verbose = verbose
        self.agent_name = "critique_agent"
    
    def evaluate_summary(self, original_content: str, summary: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a summary.
        
        Args:
            original_content: The original content
            summary: The summary to evaluate
            
        Returns:
            Evaluation results with scores and feedback
        """
        # Truncate content if too long for prompt
        if len(original_content) > 5000:
            original_content = original_content[:5000] + "..."
        
        prompt = f"""
        Evaluate the quality of the following summary based on the original content.
        Score each criterion from 1-10 and provide brief feedback.
        
        ORIGINAL CONTENT:
        {original_content}
        
        SUMMARY:
        {summary}
        
        EVALUATION:
        - Accuracy (1-10): 
        - Completeness (1-10): 
        - Conciseness (1-10): 
        - Clarity (1-10): 
        - Overall (1-10): 
        
        FEEDBACK:
        """
        
        try:
            response = self.llm.predict(prompt)
            
            # Parse scores from response
            scores = {
                "accuracy": 0,
                "completeness": 0,
                "conciseness": 0,
                "clarity": 0,
                "overall": 0
            }
            
            for line in response.split('\n'):
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        # Extract criterion name
                        criterion = parts[0].lower()
                        criterion = criterion.replace('-', '').replace('(', '').replace(')', '').strip()
                        
                        # Common criterion names
                        if "accuracy" in criterion:
                            criterion = "accuracy"
                        elif "complete" in criterion:
                            criterion = "completeness"
                        elif "concise" in criterion:
                            criterion = "conciseness"
                        elif "clarity" in criterion or "clear" in criterion:
                            criterion = "clarity"
                        elif "overall" in criterion:
                            criterion = "overall"
                        
                        # Extract score
                        try:
                            # Find numbers in the value part
                            import re
                            numbers = re.findall(r'\d+', parts[1])
                            if numbers:
                                score = int(numbers[0])
                                if 1 <= score <= 10:  # Validate score range
                                    scores[criterion] = score
                        except (ValueError, IndexError):
                            pass
            
            # Extract feedback
            feedback = response.split('FEEDBACK:')[-1].strip() if 'FEEDBACK:' in response else ""
            
            result = {
                "scores": scores,
                "feedback": feedback,
                "raw_evaluation": response
            }
            
            # Log the decision
            reflection = f"Evaluated summary with overall score: {scores.get('overall', 'N/A')}/10"
            reasoning_chain = [
                "Compared summary against original content",
                "Assessed accuracy, completeness, conciseness, and clarity",
                f"Provided feedback: {feedback[:100]}..."
            ]
            
            log_agent_decision(
                agent_name=self.agent_name,
                input_data=f"Original: {original_content[:100]}...\nSummary: {summary[:100]}...",
                output_summary=response,
                reflection=reflection,
                reasoning_chain=reasoning_chain
            )
            
            return result
        except Exception as e:
            logger.error(f"Error evaluating summary: {e}")
            return {
                "scores": {},
                "feedback": f"Error evaluating summary: {str(e)}",
                "raw_evaluation": ""
            }
    
    def improve_summary(self, original_content: str, summary: str, feedback: str) -> str:
        """
        Improve a summary based on feedback.
        
        Args:
            original_content: The original content
            summary: The summary to improve
            feedback: Feedback on the summary
            
        Returns:
            Improved summary
        """
        # Truncate content if too long for prompt
        if len(original_content) > 5000:
            original_content = original_content[:5000] + "..."
        
        prompt = f"""
        Improve the following summary based on the feedback provided.
        
        ORIGINAL CONTENT:
        {original_content}
        
        CURRENT SUMMARY:
        {summary}
        
        FEEDBACK:
        {feedback}
        
        IMPROVED SUMMARY:
        """
        
        try:
            improved_summary = self.llm.predict(prompt)
            
            # Log the decision
            reflection = "Generated improved summary based on feedback"
            reasoning_chain = [
                "Analyzed original content and current summary",
                f"Applied feedback: {feedback[:100]}...",
                "Generated improved version"
            ]
            
            log_agent_decision(
                agent_name=self.agent_name,
                input_data=f"Original: {original_content[:100]}...\nSummary: {summary[:100]}...\nFeedback: {feedback}",
                output_summary=improved_summary,
                reflection=reflection,
                reasoning_chain=reasoning_chain
            )
            
            return improved_summary
        except Exception as e:
            logger.error(f"Error improving summary: {e}")
            return summary
    
    def send_message_to_agent(self, recipient_agent: str, message_type: str, content: Dict[str, Any]) -> str:
        """
        Send a message to another agent.
        
        Args:
            recipient_agent: The name of the recipient agent
            message_type: The type of message (e.g., "summary_feedback")
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
        if message.message_type == "evaluate_summary":
            # Handle summary evaluation request
            if "original_content" in message.content and "summary" in message.content:
                evaluation = self.evaluate_summary(
                    message.content["original_content"],
                    message.content["summary"]
                )
                
                return {
                    "evaluation": evaluation,
                    "summary_id": message.content.get("summary_id")
                }
        
        elif message.message_type == "improve_summary":
            # Handle summary improvement request
            if all(k in message.content for k in ["original_content", "summary", "feedback"]):
                improved_summary = self.improve_summary(
                    message.content["original_content"],
                    message.content["summary"],
                    message.content["feedback"]
                )
                
                return {
                    "improved_summary": improved_summary,
                    "summary_id": message.content.get("summary_id")
                }
        
        return None