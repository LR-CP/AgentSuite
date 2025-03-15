import os
import tempfile
from core.base import BaseTool
from tools.document_utils import query_folder

class CodespaceTool(BaseTool):
    """
    A tool to learn codespaces.
    """
    def __init__(
        self,
        name: str,
        description: str,
        model_name="llama3.2",
        base_url="http://localhost:11434",
        settings: dict = None,
    ):
        super().__init__(name, description, settings)
        self.model_name = model_name
        self.base_url = base_url

    def execute(self, folder_obj, query=None) -> str:
        """
        Executes summarization by reading the contents of the given folder.
        """

        if query is None:
            query = "Provide an explanation as to what this codespace does"

        try:
            answer = query_folder(query, folder_path=folder_obj)
        except:
            print("Error opening folder.")
        return answer