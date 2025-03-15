from core.base import BaseAgent
from agents.chat_agent import ChatAgent
from agents.summarize_agent import SummarizeDocAgent
from agents.codespace_guru import CodespaceGuru
from tools.llm_chat_tool import LLMChat
from tools.summarize_tool import SummarizeTool
from tools.codespace_tool import CodespaceTool

class AgentFactory:
    """
    Factory class for creating agents with their required tools.
    """
    @staticmethod
    def create_agent(agent_type: str, name: str, description: str, settings: dict = None) -> BaseAgent:
        """
        Creates and returns an agent of the specified type with the given configuration.
        
        Args:
            agent_type: The type of agent to create ('ChatAgent', 'SummarizeDocAgent', etc.)
            name: The name of the agent
            description: The description of the agent
            settings: Settings for the agent and its tools
            
        Returns:
            An initialized agent instance
        """
        settings = settings or {}
        
        if agent_type == "ChatAgent":
            model_name = settings.get('model_name', 'llama3.2')
            base_url = settings.get('base_url', 'http://localhost:11434')
            
            llm_tool = LLMChat(
                name="LLMChat",
                description="Chats with desired LLM",
                model_name=model_name,
                base_url=base_url
            )
            
            return ChatAgent(name, description, llm_tool, settings)
            
        elif agent_type == "SummarizeDocAgent":
            model_name = settings.get('model_name', 'llama3.2')
            base_url = settings.get('base_url', 'http://localhost:11434')
            
            summarization_tool = SummarizeTool(
                name="Summarization Tool",
                description="Summarizes documents",
                model_name=model_name,
                base_url=base_url
            )
            
            return SummarizeDocAgent(name, description, summarization_tool, settings)
        
        # Add new agent types/tools here
        elif agent_type == "CodespaceGuru":
            model_name = settings.get('model_name', 'llama3.2')
            base_url = settings.get('base_url', 'http://localhost:11434')
            
            codespace_tool = CodespaceTool(
                name="Codespace Guru",
                description="Learns codespaces",
                model_name=model_name,
                base_url=base_url
            )
            
            return CodespaceGuru(name, description, codespace_tool, settings)
            
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")