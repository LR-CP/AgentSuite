from ollama import Client
from core.base import BaseTool

class LLMChat(BaseTool):
    """
    A tool to chat with LLM's.
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

    def execute(self, query: str) -> str:
        """
        Executes the LLM chat with the given query.
        """
        client = Client(host=self.base_url)
        response = client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"{query}",
                },
            ],
        )
        return response["message"]["content"]