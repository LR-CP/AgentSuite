from core.base import BaseAgent
from tools.summarization_tool import SummarizationTool

class SummarizeDocAgent(BaseAgent):
    """
    An agent that uses a SummarizationTool to summarize uploaded documents.
    """
    def __init__(
        self,
        name: str,
        description: str,
        summarization_tool: SummarizationTool,
        settings: dict = None,
    ):
        super().__init__(name, description, settings)
        self.summarization_tool = summarization_tool

    def run(self, file_obj) -> str:
        return self.summarization_tool.execute(file_obj)