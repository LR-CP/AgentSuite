from core.base import BaseAgent
from tools.llm_chat_tool import LLMChat

class ChatAgent(BaseAgent):
    """
    A simple agent that uses an LLM to chat.
    """
    def __init__(
        self,
        name: str,
        description: str,
        llmchat_tool: LLMChat,
        settings: dict = None,
    ):
        super().__init__(name, description, settings)
        self.llmchat_tool = llmchat_tool

    def run(self, input_data: str) -> str:
        return self.llmchat_tool.execute(input_data)