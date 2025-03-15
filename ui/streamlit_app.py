import os
import streamlit as st
import tkinter as tk
import json
from tkinter import filedialog
from factory.agent_factory import AgentFactory
from factory.agent_storage import save_agents, load_agents

def initialize_agent_manager():
    """Initialize the agent manager with saved agents or defaults."""
    if "agent_manager" not in st.session_state:
        # Try to load saved agents
        manager = load_agents()
        agent_ids = manager.list_agents()
        
        # If no saved agents, create defaults
        if not agent_ids:
            # Create and add a default ChatAgent
            chat_agent = AgentFactory.create_agent(
                "ChatAgent",
                "Chat Agent",
                "Simple LLM to answer questions",
                {"model_name": "llama3.2", "base_url": "http://localhost:11434"}
            )
            manager.add_agent("chat_agent", chat_agent)
            
            # Create and add a default SummarizeDocAgent
            summary_agent = AgentFactory.create_agent(
                "SummarizeDocAgent",
                "Summarize Doc Agent",
                "Summarizes uploaded documents",
                {"model_name": "llama3.2", "base_url": "http://localhost:11434"}
            )
            manager.add_agent("summarize_agent", summary_agent)

             # Create and add a default CodespaceGuru Agent
            codespace_agent = AgentFactory.create_agent(
                "CodespaceGuru",
                "Codespace Knowledge Base",
                "Summarizes contents of uploaded folders",
                {"model_name": "llama3.2", "base_url": "http://localhost:11434"}
            )
            manager.add_agent("codespace_guru", codespace_agent)
        
        st.session_state.agent_manager = manager
        
        # Initialize selected_agent with the first agent
        if agent_ids:
            st.session_state.selected_agent = agent_ids[0]
        else:
            st.session_state.selected_agent = "chat_agent"

def agent_manager_interface():
    st.title("Agent Suite")
    
    # Initialize the agent manager
    initialize_agent_manager()
    agent_manager = st.session_state.agent_manager
    agent_ids = agent_manager.list_agents()

    # Initialize selected_agent if not already set or if it's invalid
    if "selected_agent" not in st.session_state or st.session_state.selected_agent not in agent_ids:
        if agent_ids:
            st.session_state.selected_agent = agent_ids[0]
        else:
            st.session_state.selected_agent = None

    # Sidebar for agent management
    display_agent_sidebar(agent_manager, agent_ids)
    
    # Chat history initialization
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Main panel for running agents
    display_agent_runner(agent_manager, agent_ids)

def display_agent_sidebar(agent_manager, agent_ids):
    st.sidebar.header("Manage Agents")
    
    # Save button
    if st.sidebar.button("💾 Save All Agents"):
        save_agents(agent_manager)
        st.sidebar.success("Agents saved successfully!")
    
    # Display agent selection or info message
    if not agent_ids:
        st.sidebar.info("No agents available. Please add one below.")
    else:
        # Update the selected_agent in session_state when changed in the UI
        selected_agent_id = st.sidebar.selectbox(
            "Select an Agent", 
            agent_ids, 
            index=agent_ids.index(st.session_state.selected_agent) if st.session_state.selected_agent in agent_ids else 0
        )
        st.session_state.selected_agent = selected_agent_id
        
        selected_agent = agent_manager.get_agent(selected_agent_id)
        
        # Display agent details
        st.sidebar.markdown("### Selected Agent Details")
        st.sidebar.write("**Name:**", selected_agent.name)
        st.sidebar.write("**Description:**", selected_agent.description)
        st.sidebar.write("**Type:**", type(selected_agent).__name__)
        
        # Display agent settings based on type
        if hasattr(selected_agent, 'summarization_tool'):
            st.sidebar.write(
                "**Settings:**",
                {
                    "model_name": selected_agent.summarization_tool.model_name,
                    "base_url": selected_agent.summarization_tool.base_url,
                },
            )
        elif hasattr(selected_agent, 'llmchat_tool'):
            st.sidebar.write(
                "**Settings:**",
                {
                    "model_name": selected_agent.llmchat_tool.model_name,
                    "base_url": selected_agent.llmchat_tool.base_url,
                },
            )
        elif hasattr(selected_agent, 'codespace_tool'):
            st.sidebar.write(
                "**Settings:**",
                {
                    "model_name": selected_agent.codespace_tool.model_name,
                    "base_url": selected_agent.codespace_tool.base_url,
                },
            )
        else:
            st.sidebar.write("**Settings:**", selected_agent.settings)

        # Delete agent button
        if st.sidebar.button("Delete Selected Agent"):
            agent_manager.delete_agent(selected_agent_id)
            # Reset selected_agent if we deleted the currently selected one
            if selected_agent_id == st.session_state.selected_agent:
                remaining_agents = agent_manager.list_agents()
                if remaining_agents:
                    st.session_state.selected_agent = remaining_agents[0]
                else:
                    st.session_state.selected_agent = None
            st.rerun()
    
    # Form for adding new agents
    display_add_agent_form(agent_manager)

def display_add_agent_form(agent_manager):
    st.sidebar.markdown("---")
    st.sidebar.header("Add New Agent")
    
    with st.sidebar.form("add_agent_form", clear_on_submit=True):
        new_name = st.text_input("Agent Name")
        new_description = st.text_input("Agent Description")
        agent_type = st.selectbox("Agent Type", ["ChatAgent", "SummarizeDocAgent","CodespaceGuru"])
        
        # Custom settings based on agent type
        settings = {}
        if agent_type in ["ChatAgent", "SummarizeDocAgent", "CodespaceGuru"]:
            settings["model_name"] = st.text_input("Ollama Model Name", value="llama3.2")
            settings["base_url"] = st.text_input("Ollama Base URL", value="http://localhost:11434")
        
        submitted = st.form_submit_button("Add Agent")
        
        if submitted:
            if not new_name.strip():
                st.sidebar.error("Agent name is required.")
            else:
                agent_id = new_name.lower().replace(" ", "_")
                new_agent = AgentFactory.create_agent(
                    agent_type, new_name, new_description, settings
                )
                agent_manager.add_agent(agent_id, new_agent)
                # Set the newly created agent as the selected one
                st.session_state.selected_agent = agent_id
                st.sidebar.success(f"Agent '{new_name}' added!")
                st.rerun()

def display_agent_runner(agent_manager, agent_ids):
    st.header("Run Agent")
    
    if not agent_ids:
        st.info("Please add an agent to run.")
        return
    
    if st.session_state.selected_agent is None:
        st.warning("No agent selected. Please select or add an agent from the sidebar.")
        return
    
    selected_agent = agent_manager.get_agent(st.session_state.selected_agent)
    agent_type = selected_agent.__class__.__name__
    
    # Different UI based on agent type
    if agent_type == "SummarizeDocAgent":
        handle_summarize_doc_agent(selected_agent)
    elif agent_type == "ChatAgent":
        handle_chat_agent(selected_agent)
    elif agent_type == "CodespaceGuru":
        handle_codespace_guru_agent(selected_agent)
    else:
        handle_generic_agent(selected_agent)

def select_folder():
   root = tk.Tk()
   root.withdraw()
   folder_path = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path

def handle_summarize_doc_agent(agent):
    st.info("This agent requires a document to summarize. Please upload a file:")
    file_input = st.file_uploader(
        "Upload File", type=["pdf", "docx", "doc", "csv", "txt"]
    )
    
    if st.button("Run Agent"):
        if file_input is None:
            st.error("Please upload a file before running the agent.")
        else:
            output = agent.run(file_input)
            st.success("Agent Output:")
            st.write(output)

def handle_codespace_guru_agent(agent):
    st.info("This agent requires a codespace. Please select a folder of your codespace/repo:")
    
    selected_folder_path = st.session_state.get("folder_path", None)
    folder_select_button = st.button("Select Folder")
    if folder_select_button:
        selected_folder_path = select_folder()
        st.session_state.folder_path = selected_folder_path
    
    if selected_folder_path:
        st.write("Selected folder path:", selected_folder_path)
  
    if st.button("Run Agent"):
        if selected_folder_path is None:
            st.error("Please select a folder before running the agent.")
        else:
            output = agent.run(selected_folder_path)
            st.success("Agent Output:")
            st.write(output)
    
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Run LLM agent
        output = agent.run(selected_folder_path, user_input)
        st.success("Agent Output:")
        st.write(output)
        
        # Add response to history
        st.session_state.chat_history.append({"role": "assistant", "content": output})

def save_conversation(title):
    conversation_data = {}
    conversation = []
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            # print("Entry: ", entry["content"])
            conversation.append(entry["content"])
    
    conversation_data[title] = {
        "title": title,
        "convo": conversation
    }

    with open(f"conversations/{title}.json", "w") as f:
        json.dump(conversation_data, f, indent=2)

def load_conversation(agent, title):
    with open(f"conversations/{title}.json", "r") as f:
        conversation_data = json.load(f)
    
    for item in conversation_data[title]["convo"]: # Load convo into display
        st.session_state.chat_history.append({"role": "user", "content": item})
    
    formatted_messages = []
    for i, message in enumerate(conversation_data[title]["convo"]): # Load convo into context of llm
        role = "user" if i % 2 == 0 else "assistant"
        formatted_messages.append({"role": role, "content": message})
    
    response = agent.run(f"These were our last conversations. Reply 'GOT IT' if you received this: {formatted_messages}")
    print(response)

def handle_chat_agent(agent):
    st.info("Chat with the LLM below:")
    
    # Get list of conversation files from the conversations folder
    conversation_files = []
    conversations_dir = "conversations"  # Change this to your folder path
    
    # Check if the directory exists
    if os.path.exists(conversations_dir) and os.path.isdir(conversations_dir):
        # List all JSON files in the directory
        conversation_files = [f for f in os.listdir(conversations_dir) if f.endswith('.json')]
    
    # Create dropdown for file selection
    selected_file = st.selectbox(
        "Select a saved conversation:",
        [""] + conversation_files,  # Empty option at the beginning
        index=0  # Default to empty selection
    )

    # Text input for new conversation name
    title = st.text_input(label="Conversation Name:", value=selected_file.replace('.json', '') if selected_file else "")
    
    # Load button
    if st.button("Load Conversation"):
        load_conversation(agent, title)
        st.success("Conversation loaded successfully!")
    

    if st.button("💾 Save Conversation"):
        save_conversation(title)
        st.success("Conversation saved successfully!")

    # User input field
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Run LLM agent
        response = agent.run(user_input)
        
        # Add response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display response
        # with st.chat_message("assistant"):
        #     st.markdown(response)
    
    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

def handle_generic_agent(agent):
    input_text = st.text_area("Enter input for the agent:", height=150)
    
    if st.button("Run Agent"):
        if not input_text.strip():
            st.error("Please enter some text for the agent to process.")
        else:
            output = agent.run(input_text)
            st.success("Agent Output:")
            st.write(output)