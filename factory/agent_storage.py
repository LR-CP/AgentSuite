import json
import os
from core.manager import AgentManager
from factory.agent_factory import AgentFactory

SAVE_FILE_PATH = "saved_agents.json"

def save_agents(agent_manager: AgentManager):
    """Save all agents in the manager to a JSON file."""
    agents_data = {}
    
    for agent_id, agent in agent_manager.agents.items():
        # Extract the agent type
        agent_type = agent.__class__.__name__
        
        # Extract settings based on agent type
        settings = {}
        if hasattr(agent, 'summarization_tool'):
            settings = {
                "model_name": agent.summarization_tool.model_name,
                "base_url": agent.summarization_tool.base_url,
            }
        elif hasattr(agent, 'llmchat_tool'):
            settings = {
                "model_name": agent.llmchat_tool.model_name,
                "base_url": agent.llmchat_tool.base_url,
            }
        else:
            settings = agent.settings
        
        # Store agent data
        agents_data[agent_id] = {
            "type": agent_type,
            "name": agent.name,
            "description": agent.description,
            "settings": settings
        }
    
    # Write to file
    with open(SAVE_FILE_PATH, 'w') as f:
        json.dump(agents_data, f, indent=2)

def load_agents() -> AgentManager:
    """Load agents from a JSON file into a new AgentManager."""
    manager = AgentManager()
    
    if not os.path.exists(SAVE_FILE_PATH):
        return manager
    
    try:
        with open(SAVE_FILE_PATH, 'r') as f:
            agents_data = json.load(f)
        
        for agent_id, agent_data in agents_data.items():
            agent = AgentFactory.create_agent(
                agent_data["type"],
                agent_data["name"],
                agent_data["description"],
                agent_data["settings"]
            )
            manager.add_agent(agent_id, agent)
    except Exception as e:
        print(f"Error loading agents: {e}")
    
    return manager