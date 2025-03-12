import os
import re
import tempfile
import streamlit as st
from abc import ABC, abstractmethod
from ollama import Client

# ------------------------------------------------------------------------------
# Base Classes
# ------------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Generic base class for agents.
    """

    def __init__(self, name: str, description: str, settings: dict = None):
        self.name = name
        self.description = description
        self.settings = settings or {}

    @abstractmethod
    def run(self, input_data):
        """
        Runs the agent with given input data.
        """
        pass


class BaseTool(ABC):
    """
    Generic base class for tools.
    """

    def __init__(self, name: str, description: str, settings: dict = None):
        self.name = name
        self.description = description
        self.settings = settings or {}

    @abstractmethod
    def execute(self, input_data):
        """
        Executes the tool with given input data.
        """
        pass


# ------------------------------------------------------------------------------
# Agent Manager
# ------------------------------------------------------------------------------


class AgentManager:
    """
    A simple registry to add, delete, and retrieve agents.
    """

    def __init__(self):
        self.agents = {}

    def add_agent(self, agent_id: str, agent: BaseAgent):
        self.agents[agent_id] = agent

    def delete_agent(self, agent_id: str):
        if agent_id in self.agents:
            del self.agents[agent_id]

    def get_agent(self, agent_id: str):
        return self.agents.get(agent_id)

    def list_agents(self):
        return list(self.agents.keys())


# ------------------------------------------------------------------------------
# Helper Functions for Document Summarization
# ------------------------------------------------------------------------------

# We use LangChain components for document loading and summarization.
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
)
from langchain.llms import Ollama
from langchain.chains.summarize import load_summarize_chain


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


def summarize_text(
    text: str, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"
) -> str:
    """
    Summarizes text using a locally hosted Ollama model via LangChain.
    """
    llm = Ollama(model=model_name, base_url=base_url)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    # LangChain chains expect Document objects.
    from langchain.schema import Document

    docs = [Document(page_content=text)]
    summary = chain.run(docs)
    return summary


# ------------------------------------------------------------------------------
# Tool Classes
# ------------------------------------------------------------------------------


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
        Executes summarization by writing the uploaded file to a temporary file,
        then running the summarization pipeline.
        """
        client = Client(
            host="http://localhost:11434"
        )
        response = client.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": f"{query}",
                },
            ],
        )
        return response["message"]["content"]


# ------------------------------------------------------------------------------
# Sample Agent Implementations
# ------------------------------------------------------------------------------


class ChatAgent(BaseAgent):
    """
    A simple agent that echoes the input text.
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
            print(f"Type of llmchat_tool: {type(self.llmchat_tool)}")  # Debug line

    def run(self, input_data: str) -> str:
        settings_str = (
            ", ".join([f"{k}: {v}" for k, v in self.settings.items()])
            if self.settings
            else "No settings"
        )
        return self.llmchat_tool.execute(input_data)


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


# ------------------------------------------------------------------------------
# Streamlit UI for Agent Manager
# ------------------------------------------------------------------------------


def agent_manager_interface():
    st.title("Generic Agent Manager Interface")

    # Initialize the agent manager in session_state.
    if "agent_manager" not in st.session_state:
        st.session_state.agent_manager = AgentManager()
        # Add a default ChatAgent with LLMChat tool.
        llmchat_tool = LLMChat(
            name="LLMChat",
            description="Chats with desired LLM",
            model_name="llama3.2",
            base_url="http://localhost:11434",
        )
        st.session_state.agent_manager.add_agent(
            "chat_agent",
            ChatAgent("Chat Agent", "Simple LLM to answer questions", llmchat_tool),
        )
        # Add a default SummarizeDocAgent with a SummarizationTool.
        summarization_tool = SummarizationTool(
            "Summarization Tool",
            "Summarizes documents",
            model_name="llama3.2",
            base_url="http://localhost:11434",
        )
        st.session_state.agent_manager.add_agent(
            "summarize_agent",
            SummarizeDocAgent(
                "Summarize Doc Agent",
                "Summarizes uploaded documents",
                summarization_tool,
            ),
        )

    agent_manager: AgentManager = st.session_state.agent_manager
    agent_ids = agent_manager.list_agents()

    st.sidebar.header("Manage Agents")
    if not agent_ids:
        st.sidebar.info("No agents available. Please add one below.")
    else:
        selected_agent_id = st.sidebar.selectbox(
            "Select an Agent", agent_ids, key="selected_agent"
        )
        selected_agent = agent_manager.get_agent(selected_agent_id)
        st.sidebar.markdown("### Selected Agent Details")
        st.sidebar.write("**Name:**", selected_agent.name)
        st.sidebar.write("**Description:**", selected_agent.description)
        st.sidebar.write("**Type:**", type(selected_agent).__name__)
        # Display settings based on agent type.
        if isinstance(selected_agent, SummarizeDocAgent):
            st.sidebar.write(
                "**Settings:**",
                {
                    "model_name": selected_agent.summarization_tool.model_name,
                    "base_url": selected_agent.summarization_tool.base_url,
                },
            )
        else:
            st.sidebar.write("**Settings:**", selected_agent.settings)

        if st.sidebar.button("Delete Selected Agent"):
            agent_manager.delete_agent(selected_agent_id)
            st.experimental_rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.header("Run Agent")
    if agent_ids:
        selected_agent = agent_manager.get_agent(st.session_state.selected_agent)
        # Detect if selected agent requires file upload
        is_summary_agent = selected_agent.__class__.__name__ == "SummarizeDocAgent"
        is_chat_agent = selected_agent.__class__.__name__ == "ChatAgent"

        if is_summary_agent:
            st.info(
                "This agent requires a document to summarize. Please upload a file:"
            )
            file_input = st.file_uploader(
                "Upload File", type=["pdf", "docx", "doc", "csv", "txt"]
            )
            if st.button("Run Agent"):
                if file_input is None:
                    st.error("Please upload a file before running the agent.")
                else:
                    output = selected_agent.run(file_input)
                    st.success("Agent Output:")
                    st.write(output)
        elif is_chat_agent:
            st.info("Chat with the LLM below:")

            # Display Chat History
            for entry in st.session_state.chat_history:
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])

            # User Input
            user_input = st.chat_input("Type your message here...")
            if user_input:
                # Add user's message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                # Run LLMChat Agent
                response = selected_agent.run(user_input)

                # Add agent's response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

                # Display Agent's Response
                with st.chat_message("assistant"):
                    st.markdown(response)
        else:
            input_text = st.text_area("Enter input for the agent:", height=150)
            if st.button("Run Agent"):
                if not input_text.strip():
                    st.error("Please enter some text for the agent to process.")
                else:
                    output = selected_agent.run(input_text)
                    st.success("Agent Output:")
                    st.write(output)
    else:
        st.info("Please add an agent to run.")

    st.sidebar.markdown("---")
    st.sidebar.header("Add New Agent")
    with st.sidebar.form("add_agent_form", clear_on_submit=True):
        new_name = st.text_input("Agent Name")
        new_description = st.text_input("Agent Description")
        agent_type = st.selectbox("Agent Type", ["ChatAgent", "SummarizeDocAgent"])
        new_settings = {}
        if agent_type == "SummarizeDocAgent":
            new_model = st.text_input("Ollama Model Name", value="llama3.2")
            new_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
            new_settings["model_name"] = new_model
            new_settings["base_url"] = new_url
        else:
            new_settings["example"] = st.text_input("Example Setting", value="value")
        submitted = st.form_submit_button("Add Agent")
        if submitted:
            if not new_name.strip():
                st.sidebar.error("Agent name is required.")
            else:
                agent_id = new_name.lower().replace(" ", "_")
                if agent_type == "ChatAgent":
                    new_agent = ChatAgent(new_name, new_description, new_settings)
                elif agent_type == "SummarizeDocAgent":
                    summarization_tool = SummarizationTool(
                        "Summarization Tool",
                        "Summarizes documents",
                        model_name=new_settings["model_name"],
                        base_url=new_settings["base_url"],
                    )
                    new_agent = SummarizeDocAgent(
                        new_name, new_description, summarization_tool
                    )
                agent_manager.add_agent(agent_id, new_agent)
                st.sidebar.success(f"Agent '{new_name}' added!")
                st.experimental_rerun()


if __name__ == "__main__":
    agent_manager_interface()
