from core.base import BaseAgent
from tools.codespace_tool import CodespaceTool

class CodespaceGuru(BaseAgent):
    """
    An agent that uses a SummarizeDocumentTool to summarize uploaded documents.
    """
    def __init__(
        self,
        name: str,
        description: str,
        codespace_tool: CodespaceTool,
        settings: dict = None,
    ):
        super().__init__(name, description, settings)
        self.codespace_tool = codespace_tool

    def run(self, file_obj, query=None) -> str:
        return self.codespace_tool.execute(file_obj, query)