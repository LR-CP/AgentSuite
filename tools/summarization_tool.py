import os
import tempfile
from core.base import BaseTool
from tools.document_utils import load_document, summarize_text

class SummarizationTool(BaseTool):
    """
    A tool to summarize documents.
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

    def execute(self, file_obj) -> str:
        """
        Executes summarization by writing the uploaded file to a temporary file,
        then running the summarization pipeline.
        """
        suffix = self._get_suffix(file_obj)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_obj.read())
            tmp_file_path = tmp_file.name

        try:
            text = load_document(tmp_file_path)
            summary = summarize_text(
                text, model_name=self.model_name, base_url=self.base_url
            )
        finally:
            os.remove(tmp_file_path)
        return summary

    def _get_suffix(self, file_obj):
        if hasattr(file_obj, "name") and file_obj.name:
            return os.path.splitext(file_obj.name)[-1]
        return ".txt"