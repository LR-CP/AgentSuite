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

def load_folder(folder_path: str, supported_extensions: list = None) -> dict:
    """
    Loads and extracts text from all supported documents in a folder.
    
    Args:
        folder_path: Path to the folder containing documents
        supported_extensions: List of extensions to process (default: ['.pdf', '.docx', '.doc', '.csv', '.txt'])
        
    Returns:
        Dictionary with filenames as keys and extracted content as values
    """
    if supported_extensions is None:
        supported_extensions = ['.pdf', '.docx', '.doc', '.csv', '.txt']
    
    results = {}
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path {folder_path} is not a valid directory")
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip directories and unsupported files
        if os.path.isdir(file_path):
            continue
            
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in supported_extensions:
            continue
        
        try:
            content = load_document(file_path)
            results[filename] = content
        except Exception as e:
            results[filename] = f"Error loading document: {str(e)}"
    
    return results

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

def summarize_folder(folder_path: str, model_name: str = "llama3.2", base_url: str = "http://localhost:11434") -> dict:
    """
    Summarizes all supported documents in a folder.
    
    Args:
        folder_path: Path to the folder containing documents
        model_name: Name of the Ollama model to use
        base_url: Base URL for the Ollama API
        
    Returns:
        Dictionary with filenames as keys and summaries as values
    """
    documents = load_folder(folder_path)
    results = {}
    
    for filename, content in documents.items():
        if content.startswith("Error loading document:"):
            results[filename] = content
        else:
            try:
                summary = summarize_text(content, model_name, base_url)
                results[filename] = summary
            except Exception as e:
                results[filename] = f"Error summarizing document: {str(e)}"
    
    return results