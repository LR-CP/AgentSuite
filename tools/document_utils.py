import os
import re
import tempfile
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
)
from langchain.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

def load_document(file_path: str) -> str:
    """
    Loads and extracts text from a document using LangChain loaders.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext in [".csv"]:
        loader = CSVLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()
    content = "\n".join([doc.page_content for doc in docs])
    # Clean up excess newlines and spaces.
    content = re.sub(r"\n{2,}", "\n", content.strip())
    return content


def summarize_text(text: str, model_name: str = "llama3.2", base_url: str = "http://localhost:11434") -> str:
    """
    Summarizes text using a locally hosted Ollama model via LangChain.
    """
    llm = Ollama(model=model_name, base_url=base_url)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    # LangChain chains expect Document objects.
    docs = [Document(page_content=text)]
    summary = chain.run(docs)
    return summary