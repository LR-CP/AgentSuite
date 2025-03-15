import os
import re
from langchain.prompts import PromptTemplate
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain.llms import Ollama
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

def load_document(file_path: str) -> str:
    """
    Loads and extracts text from a document using LangChain loaders.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
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
        supported_extensions: List of extensions to process (default: ['.txt', '.py', '.c', '.cpp'])
        
    Returns:
        Dictionary with filenames as keys and extracted content as values
    """
    if supported_extensions is None:
        supported_extensions = ['.txt', '.py', '.c', '.cpp', '.html', '.js', '.css']
    
    results = {}
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path {folder_path} is not a valid directory")
    
    for root, _, files in os.walk(folder_path):
        # Skip any directory named 'venv'
        if "venv" in root.split(os.sep):
            continue

        for filename in files:
            file_path = os.path.join(root, filename)

            ext = os.path.splitext(filename)[-1].lower()
            if ext not in supported_extensions:
                continue

            try:
                content = load_document(file_path)
                results[filename] = content
            except Exception as e:
                results[filename] = f"Error loading document: {str(e)}"

    return results


def summarize_text(
    text: str, 
    query: str = "Please summarize the document.", 
    model_name: str = "llama3.2", 
    base_url: str = "http://localhost:11434"
) -> str:
    """
    Summarizes text using retrieval augmented generation (RAG) with a locally hosted Ollama model.
    
    The function performs the following steps:
      1. Splits the text into manageable chunks.
      2. Creates Document objects for each chunk.
      3. Builds a FAISS vector store from the documents using local embeddings (via HuggingFaceEmbeddings).
      4. Retrieves relevant chunks using a retriever.
      5. Uses a RetrievalQA chain to generate a summary based on the provided query.
    """
    llm = Ollama(model=model_name, base_url=base_url)
    
    # Split the text into chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Create embeddings and vector store for retrieval using a local model.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    prompt_template = """
        You are a helpful assistant with expert knowledge about code and documentation.
        
        Below is context information from various files:
        {context}
        
        Based on this information, please answer the following question:
        {question}
        
        Instructions:
        1. Only use information from the provided context
        2. If you don't know the answer based on the context, say so
        3. Include specific details and explain your reasoning
        4. For code-related questions, explain how the code works
        5. Format code snippets with appropriate markdown
        
        Answer:
        """

    # Use "stuff" for better processing of small context chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PromptTemplate.from_template(prompt_template),
        }
    )
    
    # Create the RetrievalQA chain using the retriever and LLM.
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    # Run the chain with the summarization query.
    summary = qa_chain.run(query)
    return summary

def build_folder_vectorstore(folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Loads all documents in a folder, splits them into chunks, and builds a FAISS vector store
    using local embeddings from HuggingFaceEmbeddings.

    Args:
        folder_path (str): Path to the folder containing documents.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        FAISS vector store built from the document chunks.
    """

    # Load documents from the folder using the existing load_folder function.
    documents_dict = load_folder(folder_path)
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    for filename, content in documents_dict.items():
        # Skip files that failed to load.
        if content.startswith("Error loading document:"):
            continue
        # Split each document's content into manageable chunks.
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            # Preserve source filename in metadata.
            all_docs.append(Document(page_content=chunk, metadata={"source": filename}))
    
    # Build vector store with local embeddings.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    return vectorstore


def query_folder(
    query: str,
    folder_path: str,
    model_name: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    top_k: int = 5  # Limits retrieved documents
) -> str:
    """
    Queries a folder of documents using retrieval augmented generation (RAG).

    Args:
        query (str): The query string.
        folder_path (str): Path to the folder containing documents.
        model_name (str): Name of the local LLM model to use (via Ollama).
        base_url (str): Base URL for the local LLM API.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
        top_k (int): Number of top relevant chunks to retrieve.

    Returns:
        The answer generated by the RetrievalQA chain.
    """

    # Build the vector store from the folder
    vectorstore = build_folder_vectorstore(folder_path, chunk_size, chunk_overlap)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Check if retrieval works
    retrieved_docs = retriever.get_relevant_documents(query)
    if not retrieved_docs:
        return "No relevant documents found."

    # Initialize the local LLM
    llm = Ollama(model=model_name, base_url=base_url)

    # Custom prompt to guide the LLM
    prompt_template = """
        You are a helpful assistant with expert knowledge about code and documentation.
        
        Below is context information from various files:
        {context}
        
        Based on this information, please answer the following question:
        {question}
        
        Instructions:
        1. Only use information from the provided context
        2. If you don't know the answer based on the context, say so
        3. Include specific details and explain your reasoning
        4. For code-related questions, explain how the code works
        5. Format code snippets with appropriate markdown
        
        Answer:
        """

    # Use "stuff" for better processing of small context chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate.from_template(prompt_template),
        }
    )

    # Run the query
    response = qa_chain.invoke(query)
    
    # Extract answer safely
    return response.get("result", "No response generated by the model.")


